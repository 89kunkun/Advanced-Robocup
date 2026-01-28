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
from std_msgs.msg import String, Bool
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

        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[Navigate] Arrived at '{self.waypoint_param}'.")
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
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'], output_keys=['drink'])
        self.speech_pub = rospy.Publisher('/robot_speaking', String, queue_size=1)

    def execute(self, userdata):
        rospy.loginfo('[TakeOrderState] 通过语音与顾客交互，获取订单...')
        self.speech_pub.publish(String(data="Hello! What would you like to order?"))
        try:
            rospy.loginfo('[TakeOrderState] 等待 /drink topic 的 String 消息...')
            msg = rospy.wait_for_message('/drink', String, timeout=10.0)
            userdata.drink = msg.data.strip()
            rospy.loginfo(f'[TakeOrderState] 收到饮料订单: {userdata.drink}')
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f'[TakeOrderState] 获取饮料订单失败: {e}')
            return 'failed'

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

class PlaneSegmentationState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
    def execute(self, userdata):
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
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=['drink'])
    def execute(self, userdata):
        drink = getattr(userdata, 'drink', None)
        if drink is None:
            rospy.logerr('[RunGraspState] 未提供 drink 参数，无法选择抓取动作')
            return 'aborted'
        if drink.lower() == 'coke' or drink.lower() == 'cola':
            launch_file = 'grasp_cola.launch'
        elif drink.lower() == 'sprite':
            launch_file = 'grasp_sprite.launch'
        else:
            rospy.logerr(f"[RunGraspState] 未知饮料类型: {drink}")
            return 'aborted'
        rospy.loginfo(f'[RunGraspState] 启动 {launch_file}...')
        try:
            proc = subprocess.Popen(["roslaunch", "grasp", launch_file])
            launched_processes.append(proc)
            rospy.sleep(2.0)
            return 'succeeded'
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
        rospy.loginfo("Adding NAV_TO_START")
        # TODO: Adjust the position of TiaGo head to look front
        smach.StateMachine.add('NAV_TO_START', NavigateToWaypoint('start_position'),
                               transitions={'succeeded': 'SEARCH_CUSTOMERS',
                                            'failed': 'TASK_FAILED'})
    
        rospy.loginfo("Starting Searching Customers")
        smach.StateMachine.add('SEARCH_CUSTOMERS', SearchCustomersState(),
                               transitions={'succeeded': 'NAV_TO_PERSON',
                                            'failed': 'TASK_FAILED'})
    
        
        rospy.loginfo("Adding NAV_TO_PERSON")
        # Start all these nodes in a new terminal each at the total beginning.
        # roslaunch yolo_v8_detector yolo_v8_detector.launch
        # roslaunch wave_customer_detect detect_wave.launch
        # roslaunch tiago_wave_customer_localizer waving_person_localizer.launch
        nav_to_person_state = NavigateToCoordinates('coordinates')
        smach.StateMachine.add('NAV_TO_PERSON', nav_to_person_state,
                       transitions={'succeeded': 'TAKE_ORDER',
                            'failed': 'TASK_FAILED'})

        # TODO: Add the state of taking order from the customer using speech recognition wit
        rospy.loginfo("Adding TAKE_ORDER")
        smach.StateMachine.add('TAKE_ORDER', TakeOrderState(),
                       transitions={'succeeded': 'NAV_TO_BAR',
                            'failed': 'TASK_FAILED'})


        rospy.loginfo("Adding NAV_TO_BAR")
        smach.StateMachine.add('NAV_TO_BAR', NavigateToWaypoint('bar_table_position'),
                       transitions={'succeeded': 'PRE_GRASP',
                            'failed': 'TASK_FAILED'})
        
        #####################################################################################
        # Grasping pipeline: 依次启动各节点，完成抓取任务
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
                   transitions={'succeeded': 'STOP_YOLO',
                        'aborted': 'TASK_FAILED'},
                   remapping={'drink': 'drink'})
        
        rospy.loginfo("Adding STOP_YOLO")
        smach.StateMachine.add('STOP_YOLO', StopYoloState(),
                       transitions={'succeeded': 'NAV_TO_SAVED_PERSON',
                            'aborted': 'TASK_FAILED'})
        ####################################################################################
        


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