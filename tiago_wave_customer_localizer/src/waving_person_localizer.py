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
    def __init__(self):
        # --- 初始化 node ---
        rospy.init_node("waving_person_localizer")

        # 参数
        self.depth_topic = rospy.get_param("~depth_topic", "/xtion/depth_registered/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/xtion/rgb/camera_info")
        self.bbox_topic = rospy.get_param("~bbox_topic", "/wave_customer_detect/waving_person_bbox")
        self.camera_frame = rospy.get_param("~camera_frame", "xtion_rgb_optical_frame")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # 机器人希望离客人的距离（米）
        self.target_distance = rospy.get_param("~target_distance", 1.2)
        
        # 是否使用move_base导航
        self.use_move_base = rospy.get_param("~use_move_base", False)
        
        # Cooldown机制:避免频繁发送导航目标
        self.navigation_cooldown = rospy.get_param("~navigation_cooldown", 5.0)  # 秒
        self.last_navigation_time = rospy.Time(0)

        # 内部状态
        self.bridge = CvBridge()
        self.depth_img = None
        self.fx = self.fy = self.cx = self.cy = None

        # TF buffer
        self.tfb = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfb)

        # 订阅者
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.depth_cb, queue_size=1
        )
        self.caminfo_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.caminfo_cb, queue_size=1
        )
        self.bbox_sub = rospy.Subscriber(
            self.bbox_topic, RegionOfInterest, self.bbox_cb, queue_size=1
        )

        # 发布者：兼容原作者的 coordinates 话题（目标点）
        self.coord_pub = rospy.Publisher("coordinates", Point, queue_size=10)
        # 发布人位置（map 下）
        self.person_pub = rospy.Publisher("/wave_customer_detect/person_point_map", Point, queue_size=10)
        
        # move_base action client (可选)
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

    # --- 回调函数 ---

    def depth_cb(self, msg):
        """保存最近一帧深度图"""
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[localizer] Failed to convert depth image: %s", e)

    def caminfo_cb(self, msg):
        """保存相机内参"""
        K = msg.K  # row-major 3x3
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]

    def bbox_cb(self, roi):
        """收到挥手顾客的 bbox 时，计算 3D 点并发布 coordinates"""
        if self.depth_img is None or self.fx is None:
            rospy.logwarn_throttle(2.0, "[localizer] No depth image or camera info yet.")
            return

        # 1) bbox 中心像素坐标
        u_center = roi.x_offset + roi.width / 2.0
        v_center = roi.y_offset + roi.height / 2.0

        # 2) 在 bbox 中心附近取一个小窗口，做深度平均（中位数更稳）
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

        d = float(np.median(valid))  # 深度（米）

        # 3) 像素 + 深度 -> 相机坐标系下的 3D
        X_c = (u_center - self.cx) * d / self.fx
        Y_c = (v_center - self.cy) * d / self.fy
        Z_c = d

        pt_cam = PointStamped()
        pt_cam.header.stamp = rospy.Time(0)
        pt_cam.header.frame_id = self.camera_frame
        pt_cam.point.x = X_c
        pt_cam.point.y = Y_c
        pt_cam.point.z = Z_c

        # 4) 用 TF 转换到 map 坐标系（得到客人在地图中的位置）
        try:
            pt_map = self.tfb.transform(pt_cam, self.map_frame, rospy.Duration(0.2))
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[localizer] TF transform to map failed: %s", e)
            return

        person_x = pt_map.point.x
        person_y = pt_map.point.y

        # 发布 person 点（方便你在 RViz 调试）
        person_point = Point()
        person_point.x = person_x
        person_point.y = person_y
        person_point.z = 0.0
        self.person_pub.publish(person_point)

        # 5) 拿机器人当前位置（base_link 在 map 中）
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

        # 6) 计算从机器人指向人的向量
        dx = person_x - base_x
        dy = person_y - base_y
        dist = np.hypot(dx, dy)

        if dist < 1e-3:
            rospy.logwarn_throttle(1.0, "[localizer] Person pose almost identical to base, skip.")
            return

        d_target = self.target_distance

        # ---- 关键逻辑：在“距人 d_target 的圆里，选离机器人最近的点” ----
        if dist <= d_target:
            # 已经在圆内：最近点就是当前 base 位置（不再前进）
            goal_x = base_x
            goal_y = base_y
        else:
            # 在圆外：取圆周上距离机器人最近的点 = person - u * d_target
            ux = dx / dist
            uy = dy / dist
            goal_x = person_x - ux * d_target
            goal_y = person_y - uy * d_target

        # 打印调试信息：三点之间的距离关系
        dist_goal_person = np.hypot(goal_x - person_x, goal_y - person_y)
        dist_goal_base   = np.hypot(goal_x - base_x,  goal_y - base_y)

        rospy.loginfo_throttle(
            0.5,
            "[localizer] base=(%.2f, %.2f), person=(%.2f, %.2f), goal=(%.2f, %.2f), "
            "dist_base_person=%.2f, dist_goal_person=%.2f, dist_goal_base=%.2f",
            base_x, base_y, person_x, person_y, goal_x, goal_y,
            dist, dist_goal_person, dist_goal_base
        )

        # 7) 发布到 coordinates（兼容原 base_controller）
        goal_point = Point()
        goal_point.x = goal_x
        goal_point.y = goal_y
        goal_point.z = 0.0
        self.coord_pub.publish(goal_point)
        rospy.loginfo("[localizer] Published goal to 'coordinates' topic")

        # 8) 可选：如果启用了move_base，直接发送导航目标
        if self.use_move_base and self.move_base_client is not None:
            # 再次检查cooldown (防止重复发送)
            current_time = rospy.Time.now()
            time_since_last = (current_time - self.last_navigation_time).to_sec()
            if time_since_last >= self.navigation_cooldown:
                self.send_move_base_goal(goal_x, goal_y, person_x, person_y)
                # 更新最后导航时间
                self.last_navigation_time = current_time
                rospy.logwarn("[localizer] 🎯 Navigation goal sent! Next goal in %.1fs", self.navigation_cooldown)
            else:
                rospy.loginfo("[localizer] Skipping navigation goal (cooldown: %.1fs remaining)", 
                              self.navigation_cooldown - time_since_last)

    def send_move_base_goal(self, goal_x, goal_y, person_x, person_y):
        """发送MoveBase导航目标（让机器人在 goal 点上面向人）"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        
        goal.target_pose.pose.position.x = goal_x
        goal.target_pose.pose.position.y = goal_y
        goal.target_pose.pose.position.z = 0.0
        
        # 用“从 goal 指向 person”的方向作为目标朝向
        yaw = np.arctan2(person_y - goal_y, person_x - goal_x)
        goal.target_pose.pose.orientation = self.yaw_to_quaternion(yaw)
        
        rospy.loginfo("[localizer] Sending move_base goal: (%.2f, %.2f), yaw=%.2f", 
                      goal_x, goal_y, yaw)
        self.move_base_client.send_goal(goal)

    @staticmethod
    def yaw_to_quaternion(yaw):
        """将yaw角转换为四元数"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2.0)
        q.w = np.cos(yaw / 2.0)
        return q


if __name__ == "__main__":
    node = WavingPersonLocalizer()
    rospy.spin()
