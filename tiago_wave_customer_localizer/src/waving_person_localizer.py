#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np

from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
from cv_bridge import CvBridge

import tf2_ros
import tf2_geometry_msgs

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class WavingPersonLocalizer(object):
    """
    Localizes a waving person in 3D using depth and bounding box information, then publishes the target position for robot navigation.
    Optionally, sends navigation goals to move_base so the robot approaches the customer and faces them at a specified distance.
    """
    def __init__(self):
        """
        Initializes ROS node, parameters, subscribers, publishers, and optionally the move_base action client.
        """
        # --- Initialize node ---
        rospy.init_node("waving_person_localizer")

        # Parameters
        self.depth_topic = rospy.get_param("~depth_topic", "/xtion/depth_registered/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/xtion/rgb/camera_info")
        self.bbox_topic = rospy.get_param("~bbox_topic", "/wave_customer_detect/waving_person_bbox")
        self.camera_frame = rospy.get_param("~camera_frame", "xtion_rgb_optical_frame")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # Desired distance between robot and customer (meters)
        self.target_distance = rospy.get_param("~target_distance", 1.2)
        
        # Whether to use move_base for navigation
        self.use_move_base = rospy.get_param("~use_move_base", False)
        
        # Cooldown mechanism: avoid sending navigation goals too frequently
        self.navigation_cooldown = rospy.get_param("~navigation_cooldown", 5.0)  # 秒
        self.last_navigation_time = rospy.Time(0)

        # Internal state
        self.bridge = CvBridge()
        self.depth_img = None
        self.fx = self.fy = self.cx = self.cy = None

        # TF buffer for coordinate transforms
        self.tfb = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfb)

        # Subscribers
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.depth_cb, queue_size=1
        )
        self.caminfo_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.caminfo_cb, queue_size=1
        )
        self.bbox_sub = rospy.Subscriber(
            self.bbox_topic, RegionOfInterest, self.bbox_cb, queue_size=1
        )

        # Publisher: compatible with original author's 'coordinates' topic (goal point)
        self.coord_pub = rospy.Publisher("coordinates", Point, queue_size=10)
        # Publisher: person's position in map frame
        self.person_pub = rospy.Publisher("/wave_customer_detect/person_point_map", Point, queue_size=10)
        
        # move_base action client (optional)
        self.move_base_client = None
        if self.use_move_base:
            rospy.loginfo("[waving_person_localizer] Initializing move_base action client...")
            self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            if self.move_base_client.wait_for_server(timeout=rospy.Duration(5.0)):
                rospy.loginfo("[waving_person_localizer] Connected to move_base action server.")
            else:
                rospy.logwarn("[waving_person_localizer] move_base action server not available.")
                self.move_base_client = None

        rospy.loginfo("[waving_person_localizer] Node started.")
        rospy.loginfo("  depth_topic:        %s", self.depth_topic)
        rospy.loginfo("  camera_info_topic:  %s", self.camera_info_topic)
        rospy.loginfo("  bbox_topic:         %s", self.bbox_topic)
        rospy.loginfo("  target_distance:    %.2f m", self.target_distance)
        rospy.loginfo("  use_move_base:      %s", self.use_move_base)

    # --- Callback functions ---

    def depth_cb(self, msg):
        """
        Callback for depth image topic. Converts and stores the latest depth image for later use in localization.
        """
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[localizer] Failed to convert depth image: %s", e)

    def caminfo_cb(self, msg):
        """
        Callback for camera info topic. Stores camera intrinsic parameters for 3D point calculation.
        """
        K = msg.K  # row-major 3x3
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]

    def bbox_cb(self, roi):
        """
        Callback for bounding box topic. When a waving customer is detected:
        1. Computes the 3D position of the customer using depth and camera info.
        2. Publishes the position for visualization and navigation.
        3. Calculates a goal position for the robot to approach and face the customer at a safe distance.
        4. Optionally sends a navigation goal to move_base.
        """
        if self.depth_img is None or self.fx is None:
            rospy.logwarn_throttle(2.0, "[localizer] No depth image or camera info yet.")
            return

        # 1) Center pixel coordinates of bbox
        u_center = roi.x_offset + roi.width / 2.0
        v_center = roi.y_offset + roi.height / 2.0

        # 2) Take a small window around bbox center, calculate average depth (median is more stable)
        h, w = self.depth_img.shape[:2]

        u_min = int(max(0, min(w - 1, u_center - 5)))
        u_max = int(max(0, min(w - 1, u_center + 5)))
        v_min = int(max(0, min(h - 1, v_center - 5)))
        v_max = int(max(0, min(h - 1, v_center + 5)))

        window = self.depth_img[v_min:v_max + 1, u_min:u_max + 1].astype(np.float32)
        valid = window[np.isfinite(window) & (window > 0.1)]
        if valid.size == 0:
            rospy.logwarn_throttle(1.0, "[localizer] No valid depth in bbox region.")
            return

        d = float(np.median(valid))  # Depth (meters)

        # 3) Pixel + depth -> 3D point in camera coordinate frame
        X_c = (u_center - self.cx) * d / self.fx
        Y_c = (v_center - self.cy) * d / self.fy
        Z_c = d

        pt_cam = PointStamped()
        pt_cam.header.stamp = rospy.Time(0)
        pt_cam.header.frame_id = self.camera_frame
        pt_cam.point.x = X_c
        pt_cam.point.y = Y_c
        pt_cam.point.z = Z_c

        # 4) Use TF to transform to map frame (get customer's position in map)
        try:
            pt_map = self.tfb.transform(pt_cam, self.map_frame, rospy.Duration(0.2))
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[localizer] TF transform to map failed: %s", e)
            return

        person_x = pt_map.point.x
        person_y = pt_map.point.y

        # Publish person point (for RViz debugging)
        person_point = Point()
        person_point.x = person_x
        person_point.y = person_y
        person_point.z = 0.0
        self.person_pub.publish(person_point)

        # 5) Get robot's current position (base_link in map)
        try:
            origin = PointStamped()
            origin.header.stamp = rospy.Time(0)
            origin.header.frame_id = self.base_frame
            origin.point.x = 0.0
            origin.point.y = 0.0
            origin.point.z = 0.0

            base_in_map = self.tfb.transform(origin, self.map_frame, rospy.Duration(0.2))
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[localizer] TF transform base_link->map failed: %s", e)
            return

        base_x = base_in_map.point.x
        base_y = base_in_map.point.y

        # 6) Calculate vector from robot to person
        dx = person_x - base_x
        dy = person_y - base_y
        dist = np.hypot(dx, dy)

        if dist < 1e-3:
            rospy.logwarn_throttle(1.0, "[localizer] Person pose almost identical to base, skip.")
            return

        d_target = self.target_distance

        # ---- Key logic: On the circle with radius d_target from person, select the point closest to robot ----
        if dist <= d_target:
            # Already inside the circle: closest point is current base position (do not move forward)
            goal_x = base_x
            goal_y = base_y
        else:
            # Outside the circle: take the point on the circle closest to robot = person - u * d_target
            ux = dx / dist
            uy = dy / dist
            goal_x = person_x - ux * d_target
            goal_y = person_y - uy * d_target

        # Print debug info: distance relations among three points
        dist_goal_person = np.hypot(goal_x - person_x, goal_y - person_y)
        dist_goal_base   = np.hypot(goal_x - base_x,  goal_y - base_y)

        rospy.loginfo_throttle(
            0.5,
            "[localizer] base=(%.2f, %.2f), person=(%.2f, %.2f), goal=(%.2f, %.2f), "
            "dist_base_person=%.2f, dist_goal_person=%.2f, dist_goal_base=%.2f",
            base_x, base_y, person_x, person_y, goal_x, goal_y,
            dist, dist_goal_person, dist_goal_base
        )

        # 7) Publish to 'coordinates' topic (compatible with original base_controller)
        goal_point = Point()
        goal_point.x = goal_x
        goal_point.y = goal_y
        goal_point.z = 0.0
        self.coord_pub.publish(goal_point)
        rospy.loginfo("[localizer] Published goal to 'coordinates' topic")

        # 8) Optional: if move_base is enabled, send navigation goal directly
        if self.use_move_base and self.move_base_client is not None:
            # Check cooldown again (prevent repeated sending)
            current_time = rospy.Time.now()
            time_since_last = (current_time - self.last_navigation_time).to_sec()
            if time_since_last >= self.navigation_cooldown:
                self.send_move_base_goal(goal_x, goal_y, person_x, person_y)
                # Update last navigation time
                self.last_navigation_time = current_time
                rospy.logwarn("[localizer] Navigation goal sent! Next goal in %.1fs", self.navigation_cooldown)
            else:
                rospy.loginfo("[localizer] Skipping navigation goal (cooldown: %.1fs remaining)", 
                              self.navigation_cooldown - time_since_last)

    def send_move_base_goal(self, goal_x, goal_y, person_x, person_y):
        """
        Sends a navigation goal to move_base so the robot moves to the goal position and faces the customer.
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        
        goal.target_pose.pose.position.x = goal_x
        goal.target_pose.pose.position.y = goal_y
        goal.target_pose.pose.position.z = 0.0
        
        # Use direction from goal to person as target orientation
        yaw = np.arctan2(person_y - goal_y, person_x - goal_x)
        goal.target_pose.pose.orientation = self.yaw_to_quaternion(yaw)
        
        rospy.loginfo("[localizer] Sending move_base goal: (%.2f, %.2f), yaw=%.2f", 
                      goal_x, goal_y, yaw)
        self.move_base_client.send_goal(goal)

    @staticmethod
    def yaw_to_quaternion(yaw):
        """
        Converts a yaw angle (in radians) to a quaternion for robot orientation.
        """
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q


if __name__ == "__main__":
    # Entry point: start the localizer node and keep it running
    node = WavingPersonLocalizer()
    rospy.spin()
