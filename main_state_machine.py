"""
State Machine for TiaGo Robot Task RoboCup@Home, including tasks like reception, drink delivery, and guest interaction.
Most interactions are done using ROS topics like /startyolo, /recognized_name, /recognized_drink, etc.
Modules like YoloV5 for object detection and face recognition are computationally intensive (causing delays), 
so these modules are only started when needed.

Author: Wenrong Xue
"""
import sys
import rospy
import smach
import smach_ros
import subprocess
import actionlib
import math
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PointStamped, Twist, Point
from control_msgs.msg import PointHeadAction, PointHeadGoal, FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import time
from std_msgs.msg import String, Bool, Int32MultiArray
import message_filters
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# State for robot localization using AMCL and global localization service with own defined spin speed and threshold values.
class LocalizationState(smach.State):
    def __init__(self, max_duration=60.0, spin_speed=0.7, threshold=0.2):
        smach.State.__init__(self, outcomes=['localized', 'failed'])

        # Configuration parameters
        self.spin_speed = spin_speed  # adjustable angular velocity for spinning (rad/s)
        self.threshold = threshold  # Covariance threshold for localization accuracy
        self.max_duration = max_duration  # Maximum time to attempt localization (seconds)

        # State variables
        self.covariance_sum = float('inf')
        self.localized = False

        # ROS interfaces
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)
        self.vel_pub = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=1)
        self.localization_client = rospy.ServiceProxy("/global_localization", Empty)

    def pose_callback(self, msg):
        """Callback for AMCL pose updates to monitor localization quality."""
        # Ignore stale messages
        delta = rospy.Time.now() - msg.header.stamp
        if delta.to_sec() > 0.5:
            return

        # Check localization quality based on covariance
        self.covariance_sum = sum(msg.pose.covariance)
        if self.covariance_sum < self.threshold:
            rospy.loginfo("[Localization] Localization successful. Covariance sum: %.5f", self.covariance_sum)
            self.localized = True

    # Execute localization procedure by calling global localization service and spinning the robot.
    def execute(self, userdata):
        rospy.loginfo("[Localization] Calling /global_localization...")
        try:
            self.localization_client()
        except rospy.ServiceException as e:
            rospy.logerr("[Localization] Failed to call /global_localization: %s", e)
            return 'failed'

        rospy.loginfo("[Localization] Starting to spin robot for localization...")

        rate = rospy.Rate(10)
        twist = Twist()
        twist.angular.z = self.spin_speed
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            self.vel_pub.publish(twist)
            rate.sleep()

            if self.localized:
                rospy.loginfo("[Localization] Localization confirmed. Stopping spin.")
                break

            if (rospy.Time.now() - start_time).to_sec() > self.max_duration:
                rospy.logwarn("[Localization] Localization timed out after %.1f seconds.", self.max_duration)
                self.stop_rotation()
                return 'localized'  # Return localized even on timeout

        self.stop_rotation()
        return 'localized'

    # Stop the robot rotation by publishing zero velocity.
    def stop_rotation(self):
        stop_msg = Twist()
        self.vel_pub.publish(stop_msg)
        rospy.loginfo("[Localization] Robot rotation stopped.")

# State for all navigation task. Navigating to a pre-defined waypoint using move_base action. The waypoints are saved in waypoints.yaml
class NavigateToWaypoint(smach.State):
    def __init__(self, waypoint_param):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'])
        self.waypoint_param = waypoint_param
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)


        #✅ 新增：用于TTS的publisher（不影响原逻辑）
        self.say_pub = rospy.Publisher('/the_word_to_say', String, queue_size=10)

    #Execute navigation to the specified waypoint.
    def execute(self, userdata):
        self.client.wait_for_server()

        if not rospy.has_param(self.waypoint_param):
            rospy.logwarn(f"[Navigate] Waypoint param '{self.waypoint_param}' not found.")
            return 'failed'
        
        goal_coords = rospy.get_param(self.waypoint_param)
        if len(goal_coords) != 3:
            rospy.logwarn(f"[Navigate] Invalid coordinates for waypoint '{self.waypoint_param}': {goal_coords}")
            return 'failed'

        # The coordinates are stored in the waypoints.yaml file as (x, y, theta)
        x, y, theta = goal_coords
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(theta / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(theta / 2.0)

        rospy.loginfo(f"[Navigate] Navigating to waypoint '{self.waypoint_param}': x={x:.2f}, y={y:.2f}, theta={theta:.2f} rad")

        self.client.send_goal(goal)  # Send navigation goal
        self.client.wait_for_result() # Wait for result that the robot has reached the waypoint



        # ✅ 新增：如果到达的是 bar_table_position，发布一句话
        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[Navigate] Arrived at '{self.waypoint_param}'.")

            
            if self.waypoint_param == 'bar_table_position':
                try:
                    self.say_pub.publish(String(data="I have reached the bar"))
                    rospy.loginfo("[Navigate] Published: I have reached the bar -> /the_word_to_say")
                except Exception as e:
                    rospy.logwarn(f"[Navigate] Failed to publish bar reached sentence: {e}")

            return 'succeeded'
        else:
            rospy.logwarn(f"[Navigate] Failed to reach '{self.waypoint_param}'.")
            return 'failed'
        
class NavigateToCoordinates(smach.State):
    def __init__(self,
                 coords_topic='coordinates',
                 frame_id='map',
                 timeout=60.0,
                 wait_coordinates_timeout=30.0):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'])
        self.coords_topic = coords_topic
        self.frame_id = frame_id
        self.timeout = timeout
        self.wait_coordinates_timeout = wait_coordinates_timeout

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.last_goal_point = None  # 保存最近一次接收到的目标点
        self.last_goal_pose = None   # 保存最近一次导航成功时的机器人位姿（PoseStamped）

    def execute(self, userdata):
        rospy.loginfo("[NavigateToCoordinates] Waiting for move_base action server...")
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("[NavigateToCoordinates] move_base action server not available!")
            return 'failed'
        rospy.loginfo("[NavigateToCoordinates] Connected to move_base.")

        # 1) 等一条 /coordinates 消息
        try:
            rospy.loginfo("[NavigateToCoordinates] Waiting for coordinates on topic '%s' (timeout=%.1fs)...",
                          self.coords_topic, self.wait_coordinates_timeout)
            point = rospy.wait_for_message(self.coords_topic, Point,
                                           timeout=self.wait_coordinates_timeout)
            self.last_goal_point = point  # 保存到实例变量
        except rospy.ROSException:
            rospy.logwarn("[NavigateToCoordinates] Timeout: no coordinates received on '%s'.",
                          self.coords_topic)
            return 'failed'

        x = point.x
        y = point.y
        rospy.loginfo("[NavigateToCoordinates] Got goal coordinates: x=%.2f, y=%.2f",
                      x, y)

        # 2) 构造 MoveBaseGoal（frame_id = map）
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.frame_id
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0

        # 先用 yaw=0（面向 map 的 +x 方向），后面你可以改成朝向人
        goal.target_pose.pose.orientation.z = 0.0
        goal.target_pose.pose.orientation.w = 1.0

        rospy.loginfo("[NavigateToCoordinates] Sending MoveBase goal...")
        self.client.send_goal(goal)

        # 3) 等待 move_base 结果
        finished = self.client.wait_for_result(rospy.Duration(self.timeout))
        if not finished:
            rospy.logwarn("[NavigateToCoordinates] Navigation timed out after %.1f s, canceling goal.",
                          self.timeout)
            self.client.cancel_goal()
            return 'failed'

        state = self.client.get_state()
        rospy.loginfo("[NavigateToCoordinates] move_base state = %d", state)
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("[NavigateToCoordinates] Reached goal from /coordinates.")
            # 读取当前机器人在map下的位姿
            try:
                from geometry_msgs.msg import PoseStamped
                import tf2_ros
                tf_buffer = tf2_ros.Buffer()
                tf_listener = tf2_ros.TransformListener(tf_buffer)
                # 等待tf可用
                tf_buffer.can_transform(self.frame_id, 'base_link', rospy.Time(0), rospy.Duration(1.0))
                trans = tf_buffer.lookup_transform(self.frame_id, 'base_link', rospy.Time(0), rospy.Duration(1.0))
                pose = PoseStamped()
                pose.header = trans.header
                pose.pose.position.x = trans.transform.translation.x
                pose.pose.position.y = trans.transform.translation.y
                pose.pose.position.z = trans.transform.translation.z
                pose.pose.orientation = trans.transform.rotation
                self.last_goal_pose = pose
                rospy.loginfo("[NavigateToCoordinates] Saved robot pose: x=%.3f, y=%.3f, z=%.3f, q=(%.3f,%.3f,%.3f,%.3f)",
                              pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
                              pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w)
            except Exception as e:
                rospy.logwarn("[NavigateToCoordinates] Failed to get robot pose: %s", e)
            return 'succeeded'
        else:
            rospy.logwarn("[NavigateToCoordinates] move_base failed with state %d.", state)
            return 'failed'

class NavToSavedPose(smach.State):
    def __init__(self, nav_to_coords_state):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'])
        self.nav_to_coords_state = nav_to_coords_state
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    def execute(self, userdata):
        pose = self.nav_to_coords_state.last_goal_pose
        if pose is None:
            rospy.logwarn("[NavToSavedPose] No saved pose available.")
            return 'failed'
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("[NavToSavedPose] move_base action server not available!")
            return 'failed'
        goal = MoveBaseGoal()
        goal.target_pose = pose
        self.client.send_goal(goal)
        finished = self.client.wait_for_result(rospy.Duration(60.0))
        if not finished:
            self.client.cancel_goal()
            return 'failed'
        state = self.client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            return 'succeeded'
        else:
            return 'failed'

# 搜索顾客状态：通过topic启动检测节点
class SearchCustomersState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'])
        # 初始化publisher
        self.yolo_pub = rospy.Publisher('/startyolo', Bool, queue_size=1)
        self.customer_detect_pub = rospy.Publisher('/startcustomerdetection', Bool, queue_size=1)
        # 你可以根据需要添加更多topic

    def execute(self, userdata):
        rospy.loginfo('[SearchCustomersState] 启动yolo和customerdetection相关节点...')
        # 发布True信号
        self.yolo_pub.publish(Bool(data=True))
        self.customer_detect_pub.publish(Bool(data=True))
        rospy.loginfo('[SearchCustomersState] 已向/startyolo和/startcustomerdetection发布True')
        rospy.sleep(1.0)
        return 'succeeded'

class TakeOrderState(smach.State):
    def __init__(self,
                 trigger_topic="/start_voice_recognition",
                 recognized_topic="/recognized_order",
                 end_topic="/ask_follow_end",
                 trigger_sleep=0.2,
                 wait_result_timeout=15.0,
                 wait_end_timeout=10.0,
                 # ✅ 新增：等待连接，避免第一次丢
                 wait_subscriber_timeout=3.0):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'], output_keys=['order_ids'])

        self.trigger_pub = rospy.Publisher(trigger_topic, Bool, queue_size=1)
        self.trigger_sleep = float(trigger_sleep)

        self.wait_result_timeout = float(wait_result_timeout)
        self.wait_end_timeout = float(wait_end_timeout)

        self.wait_subscriber_timeout = float(wait_subscriber_timeout)

        self._last_ids = None
        self._done = False

        rospy.Subscriber(recognized_topic, Int32MultiArray, self._cb_ids, queue_size=1)
        rospy.Subscriber(end_topic, Bool, self._cb_end, queue_size=1)

    def _cb_ids(self, msg: Int32MultiArray):
        self._last_ids = list(msg.data) if msg.data is not None else []

    def _cb_end(self, msg: Bool):
        if msg.data:
            self._done = True

    def _wait_trigger_connection(self):
        """
        等待 publisher 和 subscriber 建链完成。
        这样只发一次 True，也能最大概率不丢。
        """
        t0 = rospy.Time.now()
        rate = rospy.Rate(50)  # 20ms
        while not rospy.is_shutdown():
            if self.trigger_pub.get_num_connections() >= 1:
                rospy.loginfo("[TakeOrderState] Trigger topic connected (subs=%d).",
                              self.trigger_pub.get_num_connections())
                return True

            if (rospy.Time.now() - t0).to_sec() > self.wait_subscriber_timeout:
                rospy.logwarn("[TakeOrderState] Wait subscriber timeout (%.1fs).",
                              self.wait_subscriber_timeout)
                return False
            rate.sleep()

    def execute(self, userdata):
        rospy.loginfo("[TakeOrderState] Trigger voice recognition...")
        self._last_ids = None
        self._done = False

        # ✅ 1) 等连接，避免第一次 publish 丢失
        self._wait_trigger_connection()

        # ✅ 2) 只发一次 True（不会触发 3 次录音）
        self.trigger_pub.publish(Bool(True))
        rospy.loginfo("[TakeOrderState] Published trigger True once.")
        rospy.sleep(self.trigger_sleep)

        rate = rospy.Rate(10)

        # ✅ 3) 必须先等到 recognized_order
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self._last_ids is not None:
                break

            if (rospy.Time.now() - start).to_sec() > self.wait_result_timeout:
                rospy.logwarn("[TakeOrderState] Timeout waiting /recognized_order (%.1fs)",
                              self.wait_result_timeout)
                return "failed"
            rate.sleep()

        ids = self._last_ids or []
        rospy.loginfo("[TakeOrderState] recognized ids: %s", ids)

        # ✅ 4) ids 必须有效
        if not ids or ids == [0]:
            rospy.logwarn("[TakeOrderState] No valid order ids (got %s).", ids)
            return "failed"

        # ✅ 5) 必须等到 /ask_follow_end=True（硬条件）
        end_start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self._done:
                rospy.loginfo("[TakeOrderState] Got /ask_follow_end=True, finishing.")
                break

            if (rospy.Time.now() - end_start).to_sec() > self.wait_end_timeout:
                rospy.logwarn("[TakeOrderState] Timeout waiting /ask_follow_end=True (%.1fs)",
                              self.wait_end_timeout)
                return "failed"
            rate.sleep()

        # ✅ 6) 三个条件都满足 -> succeeded
        # 检查 order_ids 的大小，如果少于两个元素，填充默认 [1, 2]
        if len(ids) < 2:
            ids = [1, 2]
            rospy.logwarn("[TakeOrderState] order_ids 少于两个元素，填充默认 [1, 2]")
        userdata.order_ids = ids
        return "succeeded"


class AskTakeObjectsState(smach.State):
    """
    Enter state -> publish a sentence to /the_word_to_say: "Here you are, please take the order"

    Optional:
      - wait a short time to let TTS start
      - wait for an ack topic if your TTS node provides one
      - wait for at least one subscriber connection (recommended to avoid losing the first message)
      - latch the publisher (optional)
    """

    def __init__(
        self,
        say_topic="/the_word_to_say",
        sentence="Here you are, please take the order",
        repeat=1,
        repeat_interval=1.0,
        post_wait=2.0,
        ack_topic=None,
        ack_timeout=5.0,
        wait_for_subscriber=True,
        subscriber_timeout=2.0,
        latch=False,
        pre_wait=0.0
    ):
        """
        Args:
            say_topic: topic to publish text for TTS
            sentence: what to say
            repeat: publish how many times (useful if TTS sometimes misses)
            repeat_interval: seconds between repeats
            post_wait: wait seconds after publishing (let speech play)
            ack_topic: if provided, wait for Bool(True) as acknowledgement
            ack_timeout: max seconds to wait for ack
            wait_for_subscriber: wait until at least one subscriber connects before publishing
            subscriber_timeout: max seconds to wait for subscriber connection
            latch: if True, latch the publisher (late subscribers get last msg)
            pre_wait: optional sleep before publishing (small delay to let system settle)
        """
        smach.State.__init__(self, outcomes=['succeeded', 'failed'])

        self.say_topic = say_topic
        self.sentence = sentence
        self.repeat = max(1, int(repeat))
        self.repeat_interval = float(repeat_interval)
        self.post_wait = float(post_wait)

        self.ack_topic = ack_topic
        self.ack_timeout = float(ack_timeout)

        self.wait_for_subscriber = bool(wait_for_subscriber)
        self.subscriber_timeout = float(subscriber_timeout)
        self.latch = bool(latch)
        self.pre_wait = float(pre_wait)

        # Publisher (optionally latched)
        self.say_pub = rospy.Publisher(
            self.say_topic,
            String,
            queue_size=10,
            latch=self.latch
        )

        # Ack
        self._ack_received = False
        self._ack_sub = None
        if self.ack_topic:
            self._ack_sub = rospy.Subscriber(self.ack_topic, Bool, self._cb_ack, queue_size=1)

    def _cb_ack(self, msg: Bool):
        if msg.data:
            self._ack_received = True

    def _wait_subscriber_connection(self) -> bool:
        """Wait until at least one subscriber is connected to say_topic publisher."""
        if not self.wait_for_subscriber:
            return True

        start = rospy.Time.now()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            n = self.say_pub.get_num_connections()
            if n > 0:
                rospy.loginfo("[AskTakeObjectsState] Subscriber connected: %d", n)
                return True

            if (rospy.Time.now() - start).to_sec() > self.subscriber_timeout:
                rospy.logwarn(
                    "[AskTakeObjectsState] No subscriber connected on %s within %.2fs",
                    self.say_topic, self.subscriber_timeout
                )
                # 不直接失败：仍然发布（有些系统订阅端可能稍后才连上；repeat/latch可提高成功率）
                return False

            rate.sleep()

        return False

    def execute(self, userdata):
        rospy.loginfo("[AskTakeObjectsState] Saying: %s", self.sentence)

        # Reset ack flag each time we enter the state
        self._ack_received = False

        # Optional small wait before publishing (helps startup race)
        if self.pre_wait > 0:
            rospy.sleep(self.pre_wait)

        # Wait subscriber connection (recommended)
        self._wait_subscriber_connection()

        # Publish sentence (optionally repeated)
        msg = String(data=self.sentence)
        for i in range(self.repeat):
            self.say_pub.publish(msg)
            rospy.loginfo(
                "[AskTakeObjectsState] Published (%d/%d) to %s",
                i + 1, self.repeat, self.say_topic
            )

            if i < self.repeat - 1:
                rospy.sleep(self.repeat_interval)

        # If you have an ack topic, wait for it
        if self.ack_topic:
            rospy.loginfo(
                "[AskTakeObjectsState] Waiting ack on %s (timeout=%.1fs)",
                self.ack_topic, self.ack_timeout
            )
            start = rospy.Time.now()
            rate = rospy.Rate(10)

            while not rospy.is_shutdown():
                if self._ack_received:
                    rospy.loginfo("[AskTakeObjectsState] Ack received.")
                    break

                if (rospy.Time.now() - start).to_sec() > self.ack_timeout:
                    rospy.logwarn("[AskTakeObjectsState] Ack timeout.")
                    return 'failed'

                rate.sleep()

        # Always wait a bit so the speech can be heard
        if self.post_wait > 0:
            rospy.sleep(self.post_wait)

        return 'succeeded'


import signal

        # 用于管理所有子进程，便于后续清理
launched_processes = []

class PreGraspState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
    def execute(self, userdata):
        rospy.loginfo('[PreGraspState] 启动 grasp_try.launch...')
        try:
            proc = subprocess.Popen(["roslaunch", "grasp", "grasp_try.launch"])
            launched_processes.append(proc)
            rospy.sleep(2.0)  # 可根据实际情况调整等待时间
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[PreGraspState] 启动失败: {e}")
            return 'aborted'

class OpenYoloState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.yolo_pub = rospy.Publisher('/startyolo', Bool, queue_size=1)
    def execute(self, userdata):
        rospy.loginfo('[OpenYoloState] 通过 /startyolo 发布 True 启动 yolo...')
        try:
            self.yolo_pub.publish(Bool(data=True))
            rospy.sleep(1.0)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[OpenYoloState] 发布 /startyolo 失败: {e}")
            return 'aborted'

# 用于保存第一次看到的原始订单（因为后面 RunGraspState 会 pop 掉）
ORIGINAL_ORDER_IDS = None

class PlaneSegmentationState(smach.State):
    # ✅ 类变量：跨 3 个实例共享计数
    call_count = 0

    def __init__(self):
        # ✅ 新增 input_keys，让该 state 可以读 userdata.order_ids
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=['order_ids'])

        # ✅ 新增：TTS publisher
        self.say_pub = rospy.Publisher('/the_word_to_say', String, queue_size=10)

        # id -> 名称映射
        self._id_to_name = {
            1: "cola",
            2: "sprite",
            4: "cereal",
        }

    def _format_wants(self, ids):
        """把 [1,4] -> 'cola and cereal' """
        names = [self._id_to_name[i] for i in ids if i in self._id_to_name]
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return " and ".join(names)

    def execute(self, userdata):
        # ✅ 1) 一进入 state 就更新“第几次调用”
        PlaneSegmentationState.call_count += 1
        current_call = PlaneSegmentationState.call_count

        # ✅ 2) 第一次调用时保存原始 order_ids（后续不会再被 pop 影响）
        global ORIGINAL_ORDER_IDS
        if current_call == 1:
            try:
                ORIGINAL_ORDER_IDS = list(userdata.order_ids) if hasattr(userdata, 'order_ids') else []
            except Exception:
                ORIGINAL_ORDER_IDS = []

        # ✅ 3) 仅在第一次/第二次调用时发话
        #     第一次：请放第一个物体
        #     第二次：请放第二个物体
        if current_call in (1, 2):
            wants_ids = ORIGINAL_ORDER_IDS if ORIGINAL_ORDER_IDS is not None else []
            wants_text = self._format_wants(wants_ids)

            # 当前要放的物体 = 原始列表里的第 current_call-1 个
            target_name = ""
            if len(wants_ids) >= current_call:
                target_id = wants_ids[current_call - 1]
                target_name = self._id_to_name.get(target_id, "")

            if wants_text and target_name:
                sentence = f"The customer wants {wants_text} . Now please place {target_name} on the table"
                try:
                    self.say_pub.publish(String(data=sentence))
                    rospy.loginfo("[PlaneSegmentationState] Published sentence: %s", sentence)
                except Exception as e:
                    rospy.logwarn("[PlaneSegmentationState] Failed to publish sentence: %s", e)

        # ---- 下面保持你原来的逻辑不变 ----
        rospy.loginfo('[PlaneSegmentationState] 启动 plane_segmentation_seg.launch...')
        try:
            proc = subprocess.Popen(["roslaunch", "plane_segmentation", "plane_segmentation_seg.launch"])
            launched_processes.append(proc)
            rospy.sleep(2.0)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[PlaneSegmentationState] 启动失败: {e}")
            return 'aborted'


class ObjectLabelingState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
    def execute(self, userdata):
        rospy.loginfo('[ObjectLabelingState] 启动 object_labeling_seg.launch...')
        try:
            proc = subprocess.Popen(["roslaunch", "object_labeling", "object_labeling_seg.launch"])
            launched_processes.append(proc)
            rospy.sleep(2.0)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[ObjectLabelingState] 启动失败: {e}")
            return 'aborted'

class RunGraspState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=['order_ids'])
        self.flow_sub = None
        self.received_cola = False

    def flow_callback(self, msg):
        if msg.data == 'COLA' or msg.data == 'SPRITE' or msg.data == 'CEREAL' or msg.data == 'PLATE':
            self.received_cola = True

    def execute(self, userdata):
        # 检查 order_ids 是否存在且为 list
        if 'order_ids' not in userdata or not isinstance(userdata.order_ids, list):
            rospy.logerr('[RunGraspState] order_ids 未提供或不是 list')
            return 'aborted'
        
        
        # 根据 order_ids 选择 launch_file
        if not userdata.order_ids:
            # list 为空，抓取托盘
            launch_file = 'grasp_plate.launch'
        else:
            # 取第一个并 pop
            first = userdata.order_ids.pop(0)
            if first == 1:
                launch_file = 'grasp_cola.launch'
            elif first == 2:
                launch_file = 'grasp_sprite.launch'
            elif first == 4:
                launch_file = 'grasp_cereal.launch'
            else:
                rospy.logerr(f'[RunGraspState] 未知的 order id: {first}')
                return 'aborted'
        
        rospy.loginfo(f'[RunGraspState] 启动 {launch_file}...')
        try:
            proc = subprocess.Popen(["roslaunch", "grasp", launch_file])
            launched_processes.append(proc)
            
            # 订阅 /flow_result topic
            self.flow_sub = rospy.Subscriber('/flow_result', String, self.flow_callback)
            self.received_cola = False
            
            # 等待接收到 'cola' 消息，最多等待 180 秒
            rate = rospy.Rate(10)
            start_time = rospy.Time.now()
            timeout = 180.0
            while not self.received_cola and (rospy.Time.now() - start_time).to_sec() < timeout:
                rate.sleep()
            
            # 取消订阅
            if self.flow_sub:
                self.flow_sub.unregister()
            
            if self.received_cola:
                return 'succeeded'
            else:
                rospy.logwarn('[RunGraspState] 超时未收到 /flow_result 中的 完成抓取 消息')
                return 'aborted'
        except Exception as e:
            rospy.logerr(f"[RunGraspState] 启动 {launch_file} 失败: {e}")
            return 'aborted'
        
class StopYoloState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.yolo_pub = rospy.Publisher('/stopyolo', Bool, queue_size=1)
    def execute(self, userdata):
        rospy.loginfo('[StopYoloState] 通过 /stopyolo 发布 False 停止 yolo...')
        try:
            self.yolo_pub.publish(Bool(data=True))
            rospy.sleep(1.0)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[StopYoloState] 发布 /stopyolo 失败: {e}")
            return 'aborted'

class CleanUpState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'])
    
    def execute(self, userdata):
        global launched_processes
        for proc in launched_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)  # 等待进程终止，最多5秒
            except subprocess.TimeoutExpired:
                proc.kill()  # 如果terminate没成功，强制kill
                proc.wait()
        launched_processes = []  # 清空列表
        return 'succeeded'

# 可选：后续可添加 CleanUpState 用于关闭所有子进程

    
# Main function to initialize and run the TiaGo state machine.
def main():
    rospy.init_node('tiago_party_state_machine')
    sm = smach.StateMachine(outcomes=['TASK_COMPLETED', 'TASK_FAILED'])

    with sm:
        rospy.loginfo("Adding LOCALIZE_START")
        smach.StateMachine.add('LOCALIZE_START', LocalizationState(),
                               transitions={'localized': 'NAV_TO_START',
                                            'failed': 'TASK_FAILED'})
        # rospy.loginfo("Adding NAV_TO_START")
        # # TODO: Adjust the position of TiaGo head to look front
        smach.StateMachine.add('NAV_TO_START', NavigateToWaypoint('start_position'),
                               transitions={'succeeded': 'SEARCH_CUSTOMERS',
                                            'failed': 'TASK_FAILED'})
    
        rospy.loginfo("Starting Searching Customers")
        smach.StateMachine.add('SEARCH_CUSTOMERS', SearchCustomersState(),
                               transitions={'succeeded': 'NAV_TO_PERSON',
                                            'failed': 'TASK_FAILED'})
    
        
        rospy.loginfo("Adding NAV_TO_PERSON")
        # # Start all these nodes in a new terminal each at the total beginning.
        # # roslaunch yolo_v8_detector yolo_v8_detector.launch
        # # roslaunch wave_customer_detect detect_wave.launch
        # # roslaunch tiago_wave_customer_localizer waving_person_localizer.launch
        nav_to_person_state = NavigateToCoordinates('coordinates')
        smach.StateMachine.add('NAV_TO_PERSON', nav_to_person_state,
                       transitions={'succeeded': 'TAKE_ORDER',
                            'failed': 'TASK_FAILED'})

        # TODO: Add the state of taking order from the customer using speech recognition wit
        # rospy.loginfo("Adding TAKE_ORDER")
        smach.StateMachine.add('TAKE_ORDER', TakeOrderState(),
                       transitions={'succeeded': 'NAV_TO_BAR',
                            'failed': 'TASK_FAILED'})


        rospy.loginfo("Adding NAV_TO_BAR")
        smach.StateMachine.add('NAV_TO_BAR', NavigateToWaypoint('bar_table_position'),
                       transitions={'succeeded': 'PRE_GRASP',
                            'failed': 'TASK_FAILED'})
        
        #####################################################################################
        # Grasping pipeline: 依次启动各节点，完成抓取任务 For Object 1
        rospy.loginfo("Adding PRE_GRASP")
        smach.StateMachine.add('PRE_GRASP', PreGraspState(),
                       transitions={'succeeded': 'START_YOLO',
                            'aborted': 'TASK_FAILED'})

        rospy.loginfo("Adding START_YOLO")
        smach.StateMachine.add('START_YOLO', OpenYoloState(),
                       transitions={'succeeded': 'PLANE_SEGMENTATION',
                            'aborted': 'TASK_FAILED'})

        rospy.loginfo("Adding PLANE_SEGMENTATION")
        smach.StateMachine.add('PLANE_SEGMENTATION', PlaneSegmentationState(),
                       transitions={'succeeded': 'OBJECT_LABELING',
                            'aborted': 'TASK_FAILED'})

        rospy.loginfo("Adding OBJECT_LABELING")
        smach.StateMachine.add('OBJECT_LABELING', ObjectLabelingState(),
                       transitions={'succeeded': 'RUN_GRASP',
                            'aborted': 'TASK_FAILED'})

        rospy.loginfo("Adding RUN_GRASP")
        # 假设 drink 已经存储在状态机的 userdata 中
        smach.StateMachine.add('RUN_GRASP', RunGraspState(),
                   transitions={'succeeded': 'CLEAN_UP_STATE',
                        'aborted': 'TASK_FAILED'},
                   remapping={'order_ids': 'order_ids'})
        
        rospy.loginfo("Cleaning up launched processes")
        smach.StateMachine.add('CLEAN_UP_STATE', CleanUpState(),
                       transitions={'succeeded': 'PLANE_SEGMENTATION_2'})
        #####################################################################################
        # Grasping pipeline: 依次启动各节点，完成抓取任务 For Object 2

        rospy.loginfo("Adding PLANE_SEGMENTATION")
        smach.StateMachine.add('PLANE_SEGMENTATION_2', PlaneSegmentationState(),
                       transitions={'succeeded': 'OBJECT_LABELING_2',
                            'aborted': 'TASK_FAILED'})

        rospy.loginfo("Adding OBJECT_LABELING")
        smach.StateMachine.add('OBJECT_LABELING_2', ObjectLabelingState(),
                       transitions={'succeeded': 'RUN_GRASP_2',
                            'aborted': 'TASK_FAILED'})

        rospy.loginfo("Adding RUN_GRASP")
        smach.StateMachine.add('RUN_GRASP_2', RunGraspState(),
                   transitions={'succeeded': 'CLEAN_UP_STATE_2',
                        'aborted': 'TASK_FAILED'},
                   remapping={'order_ids': 'order_ids'})
        
        rospy.loginfo("Cleaning up launched processes")
        smach.StateMachine.add('CLEAN_UP_STATE_2', CleanUpState(),
                       transitions={'succeeded': 'PLANE_SEGMENTATION_3'})
        ###################################################################################
        # Grasp Plate
        rospy.loginfo("Adding PLANE_SEGMENTATION")
        smach.StateMachine.add('PLANE_SEGMENTATION_3', PlaneSegmentationState(),
                       transitions={'succeeded': 'OBJECT_LABELING_3',
                            'aborted': 'TASK_FAILED'})

        rospy.loginfo("Adding OBJECT_LABELING")
        smach.StateMachine.add('OBJECT_LABELING_3', ObjectLabelingState(),
                       transitions={'succeeded': 'RUN_GRASP_3',
                            'aborted': 'TASK_FAILED'})

        rospy.loginfo("Adding RUN_GRASP")
        smach.StateMachine.add('RUN_GRASP_3', RunGraspState(),
                   transitions={'succeeded': 'CLEAN_UP_STATE_3',
                        'aborted': 'TASK_FAILED'},
                   remapping={'order_ids': 'order_ids'})
        
        rospy.loginfo("Cleaning up launched processes")
        smach.StateMachine.add('CLEAN_UP_STATE_3', CleanUpState(),
                       transitions={'succeeded': 'STOP_YOLO'})

        rospy.loginfo("Adding STOP_YOLO")
        smach.StateMachine.add('STOP_YOLO', StopYoloState(),
                       transitions={'succeeded': 'NAV_TO_SAVED_PERSON',
                            'aborted': 'TASK_FAILED'})
        ###################################################################################

        smach.StateMachine.add('NAV_TO_SAVED_PERSON', NavToSavedPose(nav_to_person_state),
                   transitions={'succeeded': 'ASK_TAKE_OBJECTS',
                        'failed': 'TASK_FAILED'})
        
        
        # TODO: Ask customer to take the objects from the tray.(RobotSpeaking)
        rospy.loginfo("Adding ASK_TAKE_OBJECTS")
        smach.StateMachine.add('ASK_TAKE_OBJECTS', AskTakeObjectsState(),
                       transitions={'succeeded': 'NAV_TO_START2',
                            'failed': 'TASK_FAILED'})

        ###############################################################################
        # TODO: Move back to start point
        rospy.loginfo("Adding NAV_TO_START")
        smach.StateMachine.add('NAV_TO_START2', NavigateToWaypoint('start_position'),
                               transitions={'succeeded': 'TASK_COMPLETED',
                                            'failed': 'TASK_FAILED'})
        ###############################################################################

    # Start introspection server for state machine visualization
    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()
    outcome = sm.execute()
    rospy.spin()

if __name__ == '__main__':
    main()