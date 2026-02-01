#!/usr/bin/env python3

import rospy
from yolo_v8_detector.msg import Skeletons
from geometry_msgs.msg import Point
from sensor_msgs.msg import RegionOfInterest
from std_msgs.msg import Bool

class WaveCustomerDetect:
    def __init__(self):
        rospy.init_node("wave_customer_detect")

        # Parameters
        self.skeleton_topic = rospy.get_param("~skeleton_topic", "/yolo_v8_detector/skeletons")
        self.waving_threshold = rospy.get_param("~waving_threshold", 80)  # Height difference threshold (pixels)
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)  # Minimum keypoint confidence
        self.debug = rospy.get_param("~debug", True)  # Enable debug output
        
        # Image parameters
        self.image_width = rospy.get_param("~image_width", 640)  # Camera image width
        self.image_height = rospy.get_param("~image_height", 480)  # Camera image height
        
        # Cooldown to prevent continuous publishing
        self.cooldown_time = rospy.get_param("~cooldown_time", 2.0)  # Seconds between detections
        self.last_publish_time = rospy.Time(0)

        # 检测激活控制
        from std_msgs.msg import Bool
        self.active = False
        rospy.Subscriber("/startcustomerdetection", Bool, self.control_callback)

        # Subscriber
        self.skeleton_sub = rospy.Subscriber(self.skeleton_topic, Skeletons, self.skeleton_callback)

        # Publisher for person position in image coordinates
        self.position_pub = rospy.Publisher('/wave_customer_detect/person_position', Point, queue_size=1)
        
        # Publisher for bounding box of waving person
        self.bbox_pub = rospy.Publisher('/wave_customer_detect/waving_person_bbox', RegionOfInterest, queue_size=1)

        rospy.loginfo("[WaveCustomerDetect] Node initialized - Detection waits for /startcustomerdetection.")

    def control_callback(self, msg):
        if msg.data and not self.active:
            self.active = True
            rospy.loginfo("[WaveCustomerDetect] Detection started.")
        elif not msg.data and self.active:
            self.active = False
            rospy.loginfo("[WaveCustomerDetect] Detection stopped.")

    def skeleton_callback(self, msg):
        if not self.active:
            return
        for skeleton in msg.skeletons:
            keypoints = {kp.type: kp for kp in skeleton.keypoints}

            left_shoulder = keypoints.get("left_shoulder")
            right_shoulder = keypoints.get("right_shoulder")
            left_wrist = keypoints.get("left_wrist")
            right_wrist = keypoints.get("right_wrist")

            # Check if all required keypoints exist
            if not (left_shoulder and right_shoulder and left_wrist and right_wrist):
                continue

            # Check confidence of keypoints
            if (left_shoulder.confidence < self.confidence_threshold or 
                right_shoulder.confidence < self.confidence_threshold or
                left_wrist.confidence < self.confidence_threshold or 
                right_wrist.confidence < self.confidence_threshold):
                if self.debug:
                    rospy.loginfo_throttle(2.0, "Low confidence keypoints, skipping...")
                continue

            # Calculate height differences (y increases downward in image coordinates)
            left_diff = left_shoulder.y - left_wrist.y
            right_diff = right_shoulder.y - right_wrist.y

            if self.debug:
                rospy.loginfo_throttle(1.0, 
                    f"Left: shoulder_y={left_shoulder.y:.1f}, wrist_y={left_wrist.y:.1f}, diff={left_diff:.1f} | "
                    f"Right: shoulder_y={right_shoulder.y:.1f}, wrist_y={right_wrist.y:.1f}, diff={right_diff:.1f} | "
                    f"Threshold={self.waving_threshold}")

            # Check if either wrist is higher than the corresponding shoulder
            # (wrist.y < shoulder.y means hand is raised)
            if left_diff > self.waving_threshold or right_diff > self.waving_threshold:
                # Check cooldown - don't publish too frequently
                current_time = rospy.Time.now()
                time_since_last = (current_time - self.last_publish_time).to_sec()
                
                if time_since_last < self.cooldown_time:
                    if self.debug:
                        rospy.loginfo_throttle(1.0, f"Waving detected but in cooldown ({time_since_last:.1f}s < {self.cooldown_time}s)")
                    continue
                
                rospy.logwarn(f"🙋 Detected waving person! Left_diff={left_diff:.1f}, Right_diff={right_diff:.1f}")

                # Calculate the position of the waving person in image coordinates
                person_center_x = (left_shoulder.x + right_shoulder.x) / 2
                person_center_y = (left_shoulder.y + right_shoulder.y) / 2
                
                # Calculate bounding box from skeleton keypoints
                # Get all valid keypoint coordinates
                all_x = [kp.x for kp in skeleton.keypoints if kp.confidence > self.confidence_threshold]
                all_y = [kp.y for kp in skeleton.keypoints if kp.confidence > self.confidence_threshold]
                
                if len(all_x) > 0 and len(all_y) > 0:
                    bbox_xmin = int(min(all_x))
                    bbox_ymin = int(min(all_y))
                    bbox_xmax = int(max(all_x))
                    bbox_ymax = int(max(all_y))
                    bbox_width = bbox_xmax - bbox_xmin
                    bbox_height = bbox_ymax - bbox_ymin
                else:
                    # Fallback: use shoulder positions with padding
                    bbox_xmin = int(min(left_shoulder.x, right_shoulder.x) - 50)
                    bbox_ymin = int(min(left_shoulder.y, right_shoulder.y) - 100)
                    bbox_xmax = int(max(left_shoulder.x, right_shoulder.x) + 50)
                    bbox_ymax = int(max(left_shoulder.y, right_shoulder.y) + 150)
                    bbox_width = bbox_xmax - bbox_xmin
                    bbox_height = bbox_ymax - bbox_ymin
                
                # Determine direction: left, center, or right
                # In image coordinates: x=0 is LEFT, x=640 is RIGHT
                if person_center_x < self.image_width / 3:
                    direction = "left"
                elif person_center_x > 2 * self.image_width / 3:
                    direction = "right"
                else:
                    direction = "center"
                
                rospy.logwarn(f"📍 Person position in image: x={person_center_x:.1f}/{self.image_width}, y={person_center_y:.1f}/{self.image_height}")
                rospy.logwarn(f"📦 Bounding box: [{bbox_xmin}, {bbox_ymin}, {bbox_width}, {bbox_height}]")
                rospy.logwarn(f"   Direction: {direction}")
                rospy.logwarn(f"   Left: 0-{self.image_width/3:.0f}, Center: {self.image_width/3:.0f}-{2*self.image_width/3:.0f}, Right: {2*self.image_width/3:.0f}-{self.image_width}")
                
                # Publish person position as Point message (x, y in image coords, z unused)
                position_msg = Point()
                position_msg.x = person_center_x
                position_msg.y = person_center_y
                position_msg.z = 0.0  # Reserved for future use (e.g., depth)
                self.position_pub.publish(position_msg)
                
                # Publish bounding box as RegionOfInterest message
                bbox_msg = RegionOfInterest()
                bbox_msg.x_offset = bbox_xmin
                bbox_msg.y_offset = bbox_ymin
                bbox_msg.width = bbox_width
                bbox_msg.height = bbox_height
                bbox_msg.do_rectify = False
                self.bbox_pub.publish(bbox_msg)
                
                # Update last publish time
                self.last_publish_time = current_time
                
                rospy.loginfo(f"✅ Published person position and bounding box")

if __name__ == "__main__":
    WaveCustomerDetect()
    rospy.spin()