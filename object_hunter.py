#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge

# ROS2 Message Types
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

class ObjectHunterNode(Node):
    def __init__(self):
        super().__init__('object_hunter_node')
        
        # ----------------------------------------------------------------------
        # 1. State Machine Settings
        # ----------------------------------------------------------------------
        self.state = "SEARCHING"  # States: SEARCHING, TRACKING, COMPLETED
        self.target_object = ""
        self.target_depth = 0.0
        self.target_x_offset = 0.0
        self.center_x = None
        self.center_y = None
        self.image_width = 640  # Default fallback width
        
        # ----------------------------------------------------------------------
        # 2. Control Constants & Tuning Params
        # ----------------------------------------------------------------------
        self.SAFE_STOP_DISTANCE = 0.6   # Stage 6: Safe threshold in meters
        self.KP_YAW = 0.004              # P-gain constant for centering steering
        self.LINEAR_SPEED = 0.2          # Forward translation velocity (m/s)
        self.ROTATION_SPEED = 0.35       # Search scanning speed (rad/s)
        
        self.bridge = CvBridge()
        
        # Stage 1: Get the target item from the user immediately at startup
        self.get_target_input()

        # ----------------------------------------------------------------------
        # 3. ROS2 Infrastructure Setup
        # ----------------------------------------------------------------------
        self.velocity_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriptions mapping to your perception stack pipeline topics
        self.detection_sub = self.create_subscription(
            Detection2DArray, 
            '/yolo_detections', 
            self.detection_callback, 
            10
        )
        self.depth_sub = self.create_subscription(
            Image, 
            '/camera/depth/image_raw', 
            self.depth_callback, 
            10
        )
        
        # Main processing controller loop running at 10Hz (every 100ms)
        self.timer = self.create_timer(0.1, self.control_loop)

    def get_target_input(self):
        """Stage 1: Prompt the user dynamically to declare the objective target."""
        print("\n" + "="*40)
        self.target_object = input("Enter target object to find (e.g., bottle, chair, person): ").strip().lower()
        print(f"Target selected: {self.target_object}")
        print("="*40 + "\n")
        self.state = "SEARCHING"

    def detection_callback(self, msg):
        """Stage 1 & 3: Filters incoming detection streams to lock onto the specified target."""
        if self.state == "COMPLETED":
            return
            
        target_found_in_frame = False
        
        for detection in msg.detections:
            # Safely extract class label string from vision hypotheses arrays
            if len(detection.results) > 0:
                label = detection.results[0].hypothesis.class_id.strip().lower()
                
                # Check for target matching equality
                if label == self.target_object:
                    target_found_in_frame = True
                    # Stage 2: Track center values from bounding box
                    self.center_x = int(detection.bbox.center.position.x)
                    self.center_y = int(detection.bbox.center.position.y)
                    
                    # Measure error vector relative to view window frame center point
                    self.target_x_offset = self.center_x - (self.image_width / 2)
                    break
                    
        if target_found_in_frame:
            if self.state == "SEARCHING":
                self.get_logger().info("Target Found!")
            self.state = "TRACKING"
        else:
            if self.state == "TRACKING":
                self.get_logger().warn("Target lost! Falling back to scanning loop...")
            self.state = "SEARCHING"

    def depth_callback(self, msg):
        """Stage 2 & Bonus 1: Tracks depth distance via a robust 5x5 spatial median window filter."""
        if self.state != "TRACKING" or self.center_x is None:
            return
            
        try:
            # Extract depth data matrix (32-bit float matrix mapping pixels to meters)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.image_width = depth_image.shape[1]
            h, w = depth_image.shape
            
            # --- Bonus 1: 5x5 Neighborhood Filter ---
            ymin, ymax = max(0, self.center_y - 2), min(h, self.center_y + 3)
            xmin, xmax = max(0, self.center_x - 2), min(w, self.center_x + 3)
            
            depth_roi = depth_image[ymin:ymax, xmin:xmax]
            
            # Eliminate invalid elements, noise dropouts, or NaN returns
            valid_depths = depth_roi[~np.isnan(depth_roi) & (depth_roi > 0)]
            
            if len(valid_depths) > 0:
                # Apply median calculation to filter out extreme sensor outlier spikes
                self.target_depth = float(np.median(valid_depths))
                self.get_logger().info(f"Target Locked | Distance to {self.target_object}: {self.target_depth:.2f} m")
            else:
                self.target_depth = 0.0
                
        except Exception as e:
            self.get_logger().error(f"Error handling depth image conversion: {str(e)}")

    def control_loop(self):
        """FSM execution routine managing navigation based on current target status."""
        cmd_vel = Twist()
        
        if self.state == "SEARCHING":
            # Stage 3: Scan environment via pure axial rotation in place
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.ROTATION_SPEED
            self.velocity_pub.publish(cmd_vel)
            
        elif self.state == "TRACKING":
            # Stage 6: Stopping criteria check
            if 0.0 < self.target_depth <= self.SAFE_STOP_DISTANCE:
                self.state = "COMPLETED"
                return
                
            # Stage 5: Proportional Closed-Loop Visual Servoing Control
            # Steer dynamically to bring horizontal tracking offset to zero
            cmd_vel.angular.z = - (self.KP_YAW * self.target_x_offset)
            
            # Move forward safely only when a valid tracking depth is computed
            if self.target_depth > self.SAFE_STOP_DISTANCE:
                cmd_vel.linear.x = self.LINEAR_SPEED
            else:
                cmd_vel.linear.x = 0.0
                
            self.velocity_pub.publish(cmd_vel)
            
        elif self.state == "COMPLETED":
            # Stage 6: Clean, collision-free mission wrap up
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.velocity_pub.publish(cmd_vel)
            self.get_logger().info("Mission Completed! Target Reached Successfully.")
            
            # --- Bonus 3: Continuous Assistant Mode Reset Loop ---
            self.center_x = None
            self.center_y = None
            self.target_depth = 0.0
            self.get_target_input()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectHunterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Safety force-stop block on shutdown command
        stop_vel = Twist()
        node.velocity_pub.publish(stop_vel)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()