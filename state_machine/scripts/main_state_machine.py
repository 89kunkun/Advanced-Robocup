"""
State Machine for TiaGo Robot Task RoboCup@Home, including tasks like reception, drink delivery, and guest interaction.
Most interactions are done using ROS topics like /startyolo, /recognized_name, /recognized_drink, etc.
Modules like YoloV5 for object detection and face recognition are computationally intensive (causing delays), 
so these modules are only started when needed.

DEPENDENCIES:
- Required ROS packages: smach, actionlib, move_base, play_motion, speech recognition
- External nodes: yolo_v8_detector, wave_customer_detect, tiago_wave_customer_localizer
- Hardware: TiaGo robot with gripper, head joints, and mobile base
- Configuration: waypoints.yaml file with predefined navigation points

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
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from sensor_msgs.msg import JointState


# This state handles the initial localization of the robot. It uses AMCL and the global localization service 
# to determine the robot's pose in the map. The robot spins in place to gather information 
# from its laser scanner and localize itself with a certain confidence level.
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

# This state navigates the robot to a predefined waypoint from the `waypoints.yaml` file.
# It uses the `move_base` action server to send a navigation goal.
# The state succeeds if the robot reaches the waypoint and fails otherwise.
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

        # The coordinates are stored in the waypoints.yaml file as a list [x, y, theta]
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
        
# This state navigates the robot to a dynamic coordinate received from a ROS topic.
# It waits for a `Point` message on the specified topic and then sends a goal to `move_base`.
# After reaching the goal, it saves the robot's final pose for later use.        
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
        self.last_goal_point = None  # Store the last received goal point
        self.last_goal_pose = None   # Store the robot's pose (PoseStamped) after the last successful navigation

    def execute(self, userdata):
        rospy.loginfo("[NavigateToCoordinates] Waiting for move_base action server...")
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("[NavigateToCoordinates] move_base action server not available!")
            return 'failed'
        rospy.loginfo("[NavigateToCoordinates] Connected to move_base.")

        # 1) Wait for a /coordinates message
        try:
            rospy.loginfo("[NavigateToCoordinates] Waiting for coordinates on topic '%s' (timeout=%.1fs)...",
                          self.coords_topic, self.wait_coordinates_timeout)
            point = rospy.wait_for_message(self.coords_topic, Point,
                                           timeout=self.wait_coordinates_timeout)
            self.last_goal_point = point  # Save to instance variable
        except rospy.ROSException:
            rospy.logwarn("[NavigateToCoordinates] Timeout: no coordinates received on '%s'.",
                          self.coords_topic)
            return 'failed'

        x = point.x
        y = point.y
        rospy.loginfo("[NavigateToCoordinates] Got goal coordinates: x=%.2f, y=%.2f",
                      x, y)

        # 2) Construct MoveBaseGoal (frame_id = map)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.frame_id
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0

        # First use yaw=0 (facing the +x direction of the map), you can change it to face the person later
        goal.target_pose.pose.orientation.z = 0.0
        goal.target_pose.pose.orientation.w = 1.0

        rospy.loginfo("[NavigateToCoordinates] Sending MoveBase goal...")
        self.client.send_goal(goal)

        # 3) Wait for move_base result
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
            # Read the current robot pose in the map frame
            try:
                from geometry_msgs.msg import PoseStamped
                import tf2_ros
                tf_buffer = tf2_ros.Buffer()
                tf_listener = tf2_ros.TransformListener(tf_buffer)
                # Wait for tf to be available
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


# This state navigates the robot back to a previously saved pose.
# It retrieves the pose from the `NavigateToCoordinates` state and uses `move_base` to return to that location.
# This is useful for returning to a person's location after leaving to fetch an item.
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

# This state is responsible for initiating the customer search process.
# It activates the necessary detection nodes, such as YOLO for object detection and a custom customer detector,
# by publishing boolean messages to specific ROS topics.
class SearchCustomersState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'])
        # Initialize publishers
        self.yolo_pub = rospy.Publisher('/startyolo', Bool, queue_size=1)
        self.customer_detect_pub = rospy.Publisher('/startcustomerdetection', Bool, queue_size=1)
        # You can add more topics if needed

    def execute(self, userdata):
        rospy.loginfo('[SearchCustomersState] Starting yolo and customerdetection nodes...')
        # Publish True signal
        self.yolo_pub.publish(Bool(data=True))
        self.customer_detect_pub.publish(Bool(data=True))
        rospy.loginfo('[SearchCustomersState] Published True to /startyolo and /startcustomerdetection')
        rospy.sleep(1.0)
        return 'succeeded'
# This state manages the process of taking a customer's order using voice recognition.
# It triggers the voice recognition system and waits for the recognized order IDs.
# The state ensures that valid order IDs are received before transitioning.
class TakeOrderState(smach.State):
    def __init__(self,
                 trigger_topic="/start_voice_recognition",
                 recognized_topic="/recognized_order",
                 end_topic="/ask_follow_end",
                 trigger_sleep=0.2,
                 wait_result_timeout=15.0,
                 wait_end_timeout=10.0,
                 # New: wait for connection to avoid losing the first message
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
        Wait for the publisher and subscriber to establish a connection.
        This way, publishing True only once has the highest probability of not being missed.
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

        # 1) Wait for connection to avoid losing the first publish
        self._wait_trigger_connection()

        # 2) Publish True only once (won't trigger 3 recordings)
        self.trigger_pub.publish(Bool(True))
        rospy.loginfo("[TakeOrderState] Published trigger True once.")
        rospy.sleep(self.trigger_sleep)

        rate = rospy.Rate(10)

        # 3) Must wait for recognized_order first
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

        # 4) ids must be valid
        if not ids or ids == [0]:
            rospy.logwarn("[TakeOrderState] No valid order ids (got %s).", ids)
            return "failed"

        # 5) Must wait for /ask_follow_end=True (hard requirement)
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

        # If all three conditions are met, the state succeeds.
        # Check the size of order_ids. If it has fewer than two elements,
        # it is padded with a default value of [1, 4] (cola and cereal).
        # This is a fallback to prevent errors in the grasping pipeline if voice recognition fails.
        if len(ids) < 2:
            ids = [1, 4]
            rospy.logwarn("[TakeOrderState] order_ids has less than two elements, filling with default [1, 4]")
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
        # Publisher for gripper command (to open/close)
        self.gripper_pub = rospy.Publisher(
            "/gripper_controller/command", JointTrajectory, queue_size=1,
        )
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
                # Don't fail directly: still publish (some systems might connect subscribers later; repeat/latch can increase success rate)
                return False

            rate.sleep()

        return False
    def send_gripper_cmd(self, positions, duration):
        """
        Sends a command to the gripper controller.
        
        Args:
            positions: A list of two floats for the left and right finger joint positions.
            duration: The time in seconds the gripper should take to reach the position.
        """
        traj = JointTrajectory()
        traj.joint_names = [
            "gripper_left_finger_joint",
            "gripper_right_finger_joint",
        ]

        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = rospy.Duration(duration)
        traj.points.append(pt)
        traj.header.stamp = rospy.Time.now() + rospy.Duration(0.2)

        # Publish the command multiple times to ensure it's received
        for _ in range(3):
            self.gripper_pub.publish(traj)
            rospy.sleep(0.1)

    def open_gripper(self):
        """Opens the gripper to release an object."""
        self.send_gripper_cmd([0.04, 0.04], 3.5)

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
        # After speaking, open the gripper to allow the customer to take the items.
        for i in range(self.repeat):
            self.say_pub.publish(msg)
            rospy.loginfo(
                "[AskTakeObjectsState] Published (%d/%d) to %s",
                i + 1, self.repeat, self.say_topic
            )

            if i < self.repeat - 1:
                rospy.sleep(self.repeat_interval)
        self.open_gripper()
        rospy.sleep(4.0)
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
        #After the customer has taken the objects, tuck the arm back to a safe position.
        # Tuck arm
        rospy.loginfo("Waiting for play_motion...")
        client = actionlib.SimpleActionClient("play_motion", PlayMotionAction)
        client.wait_for_server()
        rospy.loginfo("...connected.")

        rospy.wait_for_message("joint_states", JointState)
        rospy.sleep(3.0)

        rospy.loginfo("Tuck arm...")
        goal = PlayMotionGoal()
        goal.motion_name = 'home'
        goal.skip_planning = False

        client.send_goal(goal)
        client.wait_for_result(rospy.Duration(10.0))
        rospy.loginfo("Arm tucked.")

        return 'succeeded'


import signal

# Global list used to manage all child processes for later cleanup
# This ensures proper termination of launched ROS nodes
launched_processes = []

# This state is responsible for preparing the robot for grasping an object.
# It launches the `grasp_try.launch` file, which likely sets up the necessary nodes and services for the grasping pipeline.
# This state ensures that the robot's arm and gripper are ready for the subsequent grasping operations.
class PreGraspState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
    def execute(self, userdata):
        rospy.loginfo('[PreGraspState] Starting grasp_try.launch...')
        try:
            proc = subprocess.Popen(["roslaunch", "grasp", "grasp_try.launch"])
            launched_processes.append(proc)
            rospy.sleep(2.0)  # Adjust wait time as needed
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[PreGraspState] Failed to start: {e}")
            return 'aborted'

class OpenYoloState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.yolo_pub = rospy.Publisher('/startyolo', Bool, queue_size=1)
    def execute(self, userdata):
        rospy.loginfo('[OpenYoloState] Starting yolo by publishing True to /startyolo...')
        try:
            self.yolo_pub.publish(Bool(data=True))
            rospy.sleep(1.0)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[OpenYoloState] Failed to publish to /startyolo: {e}")
            return 'aborted'

class PlaneSegmentationState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
    def execute(self, userdata):
        rospy.loginfo('[PlaneSegmentationState] Starting plane_segmentation_seg.launch...')
        try:
            proc = subprocess.Popen(["roslaunch", "plane_segmentation", "plane_segmentation_seg.launch"])
            launched_processes.append(proc)
            rospy.sleep(2.0)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[PlaneSegmentationState] Failed to start: {e}")
            return 'aborted'

class ObjectLabelingState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
    def execute(self, userdata):
        rospy.loginfo('[ObjectLabelingState] Starting object_labeling_seg.launch...')
        try:
            proc = subprocess.Popen(["roslaunch", "object_labeling", "object_labeling_seg.launch"])
            launched_processes.append(proc)
            rospy.sleep(2.0)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[ObjectLabelingState] Failed to start: {e}")
            return 'aborted'

class RunGraspState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=['order_ids'])
        self.flow_sub = None
        self.received_cola = False

    def flow_callback(self, msg):
        """Callback to monitor the grasping process completion.
        Listens for specific object names that indicate successful grasping."""
        if msg.data == 'COLA' or msg.data == 'SPRITE' or msg.data == 'CEREAL' or msg.data == 'PLATE':
            self.received_cola = True

    def execute(self, userdata):
        # Check if order_ids exists and is a list
        if 'order_ids' not in userdata or not isinstance(userdata.order_ids, list):
            rospy.logerr('[RunGraspState] order_ids not provided or not a list')
            return 'aborted'
        
        
        # Select launch_file based on order_ids
        # Order ID mapping: 1=cola, 2=sprite, 4=cereal, empty=plate
        if not userdata.order_ids:
            # list is empty, grasp the tray
            launch_file = 'grasp_plate.launch'
        else:
            # Take the first item from the order queue and process it
            first = userdata.order_ids.pop(0)
            if first == 1:
                launch_file = 'grasp_cola.launch'
            elif first == 2:
                launch_file = 'grasp_sprite.launch'
            elif first == 4:
                launch_file = 'grasp_cereal.launch'
            else:
                rospy.logerr(f'[RunGraspState] Unknown order id: {first}')
                return 'aborted'
        
        rospy.loginfo(f'[RunGraspState] Starting {launch_file}...')
        try:
            proc = subprocess.Popen(["roslaunch", "grasp", launch_file])
            launched_processes.append(proc)
            
            # Subscribe to /flow_result topic
            self.flow_sub = rospy.Subscriber('/flow_result', String, self.flow_callback)
            self.received_cola = False
            
            # Wait to receive completion message, with a timeout of 180 seconds
            # This timeout is generous to account for complex grasping operations
            rate = rospy.Rate(10)
            start_time = rospy.Time.now()
            timeout = 180.0  # 3 minutes timeout for grasping operations
            while not self.received_cola and (rospy.Time.now() - start_time).to_sec() < timeout:
                rate.sleep()
            
            # Unsubscribe
            if self.flow_sub:
                self.flow_sub.unregister()
            
            if self.received_cola:
                return 'succeeded'
            else:
                rospy.logwarn('[RunGraspState] Timeout waiting for grasp completion message on /flow_result')
                return 'aborted'
        except Exception as e:
            rospy.logerr(f"[RunGraspState] Failed to start {launch_file}: {e}")
            return 'aborted'
        
class StopYoloState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.yolo_pub = rospy.Publisher('/stopyolo', Bool, queue_size=1)
        self.head_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=1)

    def move_head(self, positions, duration):
        traj = JointTrajectory()
        traj.joint_names = ["head_1_joint", "head_2_joint"]
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = rospy.Duration(duration)
        traj.points.append(pt)
        traj.header.stamp = rospy.Time.now() + rospy.Duration(0.2)
        for _ in range(3):
            self.head_pub.publish(traj)
            rospy.sleep(0.1)

    def execute(self, userdata):
        rospy.loginfo('[StopYoloState] Stopping yolo by publishing False to /stopyolo...')
        try:
            self.yolo_pub.publish(Bool(data=True))
            rospy.sleep(1.0)
            # Reset head to neutral position for safe navigation
            self.move_head([0.0, 0.0], 1.0)
            rospy.loginfo('[StopYoloState] Head moved to [0.0, 0.0]')
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[StopYoloState] Failed to publish to /stopyolo: {e}")
            return 'aborted'

class CleanUpState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'])
    
    def execute(self, userdata):
        global launched_processes
        for proc in launched_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)  # Wait for process to terminate, max 5 seconds
            except subprocess.TimeoutExpired:
                proc.kill()  # Force kill if terminate fails
                proc.wait()
        launched_processes = []  # Clear the list
        return 'succeeded'

class AskBarMan(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'], input_keys=['order_ids'])
        self.say_pub = rospy.Publisher('/the_word_to_say', String, queue_size=10)
    def execute(self, userdata):
        ids = userdata.order_ids
        rospy.loginfo('[AskBarMan] Read order_ids: %s', ids)
        drink_names = {1: 'cola', 2: 'sprite', 4: 'cereal'}
        if len(ids)==0:
            sentence = "The order is finished. I will grasp the tray."
        elif len(ids)==2:
            object1 = drink_names.get(ids[0], 'unknown drink')
            object2 = drink_names.get(ids[1], 'unknown drink')
            sentence = f"The customer wants {object1} and {object2}. Now please put {object1} on the table." 
        elif len(ids)==1:
            object1 = drink_names.get(ids[0], 'unknown drink')
            sentence = f"Please put {object1} on the table."    
        self.say_pub.publish(String(data=sentence))
        rospy.loginfo('[AskBarMan] Published sentence: %s', sentence)
        rospy.sleep(5.0)
        return 'succeeded'
    
# Main function to initialize and run the TiaGo state machine.
# This creates a complete workflow for a service robot in a restaurant/party scenario:
# 1. Localize the robot in the environment
# 2. Navigate to start position and search for customers
# 3. Take the customer's order using voice recognition
# 4. Navigate to the bar and coordinate with bartender
# 5. Execute grasping pipeline for each ordered item
# 6. Return to customer and deliver the items
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
        # roslaunch yolo_v8_detector yolo_v8.launch
        # roslaunch wave_customer_detect detect_wave.launch
        # roslaunch tiago_wave_customer_localizer waving_person_localizer.launch
        nav_to_person_state = NavigateToCoordinates('coordinates')
        smach.StateMachine.add('NAV_TO_PERSON', nav_to_person_state,
                       transitions={'succeeded': 'TAKE_ORDER',
                            'failed': 'TASK_FAILED'})

        # The TAKE_ORDER state uses voice recognition to get the customer's order.
        # It outputs the order_ids to be used by the grasping states.
        smach.StateMachine.add('TAKE_ORDER', TakeOrderState(),
                       transitions={'succeeded': 'NAV_TO_BAR',
                            'failed': 'TASK_FAILED'})


        rospy.loginfo("Adding NAV_TO_BAR")
        smach.StateMachine.add('NAV_TO_BAR', NavigateToWaypoint('bar_table_position'),
                       transitions={'succeeded': 'ASK_BARMAN',
                            'failed': 'TASK_FAILED'})
        
        # Grasping pipeline: sequentially start nodes to complete the grasping task for Object 1
        # The pipeline consists of: ASK_BARMAN -> PRE_GRASP -> START_YOLO -> PLANE_SEGMENTATION
        # -> OBJECT_LABELING -> RUN_GRASP -> CLEAN_UP_STATE

        smach.StateMachine.add('ASK_BARMAN', AskBarMan(),
                       transitions={'succeeded': 'PRE_GRASP'},
                       remapping={'order_ids': 'order_ids'})
        
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
        # Assume the drink is already stored in the state machine's userdata
        smach.StateMachine.add('RUN_GRASP', RunGraspState(),
                   transitions={'succeeded': 'CLEAN_UP_STATE',
                        'aborted': 'TASK_FAILED'},
                   remapping={'order_ids': 'order_ids'})
        
        rospy.loginfo("Cleaning up launched processes")
        smach.StateMachine.add('CLEAN_UP_STATE', CleanUpState(),
                       transitions={'succeeded': 'ASK_BARMAN_2'})
        
        # Grasping pipeline for Object 2: Similar to Object 1 but processes the next item in order_ids
        # This allows the robot to handle multiple items in a single order sequentially

        smach.StateMachine.add('ASK_BARMAN_2', AskBarMan(),
                       transitions={'succeeded': 'PLANE_SEGMENTATION_2'},
                       remapping={'order_ids': 'order_ids'})
        
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
                       transitions={'succeeded': 'ASK_BARMAN_3'})
        
        # Final grasping sequence: Plate/Tray
        # After all drink items are processed, grasp the serving plate/tray

        smach.StateMachine.add('ASK_BARMAN_3', AskBarMan(),
                transitions={'succeeded': 'PLANE_SEGMENTATION_3'},
                remapping={'order_ids': 'order_ids'})
        
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

        smach.StateMachine.add('NAV_TO_SAVED_PERSON', NavToSavedPose(nav_to_person_state),
                   transitions={'succeeded': 'ASK_TAKE_OBJECTS',
                        'failed': 'TASK_FAILED'})
        
        
        # The robot asks the customer to take the objects from the tray.
        rospy.loginfo("Adding ASK_TAKE_OBJECTS")
        smach.StateMachine.add('ASK_TAKE_OBJECTS', AskTakeObjectsState(),
                       transitions={'succeeded': 'NAV_TO_START2',
                            'failed': 'TASK_FAILED'})

        # The robot moves back to its starting position to complete the task.
        rospy.loginfo("Adding NAV_TO_START")
        smach.StateMachine.add('NAV_TO_START2', NavigateToWaypoint('start_position'),
                               transitions={'succeeded': 'TASK_COMPLETED',
                                            'failed': 'TASK_FAILED'})


    # Start introspection server for state machine visualization
    # This allows monitoring the state machine execution via 'rosrun smach_viewer smach_viewer.py'
    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()
    outcome = sm.execute()
    rospy.spin()

if __name__ == '__main__':
    main()