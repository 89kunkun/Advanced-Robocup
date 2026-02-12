#!/usr/bin/env python3
# Use Python3 interpreter

import rospy
import numpy as np
import tf.transformations as tft
import copy
import threading

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PoseStamped, PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import moveit_commander
from moveit_commander import PlanningSceneInterface
import moveit_commander.conversions as conversions
from moveit_msgs.msg import RobotTrajectory

from std_msgs.msg import Bool, String


class GrasperPy:
    """
    This class performs a plate grasping pipeline using MoveIt.

    Workflow overview:
    1. Receive plate centroid from /plate_center (PointStamped).
    2. Build a temporary "check grasp point" (with y_offset removed).
    3. Publish it to /check_plate_grasp_point for overhang verification.
    4. Wait for /updated_grasp_point (fallback to original if timeout).
    5. Add y_offset back to construct final grasp pose.
    6. Execute grasp sequence:
       open -> pregrasp -> approach -> close -> lift
    """

    def __init__(self):
        # Initialize ROS node
        rospy.init_node("plate_grasper")

        # ============================
        # MoveIt setup
        # ============================

        # Initialize MoveIt commander
        moveit_commander.roscpp_initialize([])

        # Move group for arm only
        self.group = moveit_commander.MoveGroupCommander("arm")

        # Move group for arm + torso (for Cartesian motions)
        self.group_cartesian = moveit_commander.MoveGroupCommander("arm_torso")

        # Get planning reference frame
        self.ref_frame = self.group.get_planning_frame()
        rospy.loginfo("[GrasperPy] Using planning frame as ref_frame: '%s'", self.ref_frame)

        # End-effector link name (default: gripper_link)
        self.eef_link = rospy.get_param("~eef_link", "gripper_link")

        # Configure reference frames
        self.group.set_pose_reference_frame(self.ref_frame)
        self.group.set_end_effector_link(self.eef_link)
        self.group_cartesian.set_pose_reference_frame(self.ref_frame)
        self.group_cartesian.set_end_effector_link(self.eef_link)

        # Goal tolerances and motion scaling factors
        self.group.set_goal_tolerance(rospy.get_param("~goal_tolerance", 0.01))
        self.velocity_scaling = rospy.get_param("~velocity_scaling", 0.3)
        self.accel_scaling = rospy.get_param("~accel_scaling", 0.2)

        self.group.set_max_velocity_scaling_factor(self.velocity_scaling)
        self.group.set_max_acceleration_scaling_factor(self.accel_scaling)
        self.group_cartesian.set_max_velocity_scaling_factor(self.velocity_scaling)
        self.group_cartesian.set_max_acceleration_scaling_factor(self.accel_scaling)

        # Planner configuration
        self.group.set_planner_id(rospy.get_param("~planner_id", "RRTConnectkConfigDefault"))
        self.group.set_planning_time(rospy.get_param("~planning_time", 15.0))
        self.group.set_num_planning_attempts(rospy.get_param("~num_planning_attempts", 5))

        # ============================
        # Planning Scene (collision objects)
        # ============================

        # Interface to add/remove collision objects
        self.scene = PlanningSceneInterface(synchronous=True)

        # Table collision parameters
        self.table_size_x = float(rospy.get_param("~table_size_x", 0.2))
        self.table_size_y = float(rospy.get_param("~table_size_y", 1.0))
        self.table_thickness = float(rospy.get_param("~table_thickness", 0.4))
        self.table_z_offset = float(rospy.get_param("~table_z_offset", 0.02))
        self.table_object_id = rospy.get_param("~table_object_id", "table")

        self.table_added = False
        self.table_center_msg = None

        # ============================
        # Grasp Parameters (Plate only)
        # ============================

        # Fixed grasp orientation defined in RPY
        self.grasp_rpy = [
            rospy.get_param("~grasp_roll", np.pi / 2),
            rospy.get_param("~grasp_pitch", np.pi / 2),
            rospy.get_param("~grasp_yaw", 0.0),
        ]

        # Position offsets applied to centroid (in reference frame)
        self.x_offset = float(rospy.get_param("~x_offset", -0.13))
        self.y_offset = float(rospy.get_param("~y_offset", -0.19))
        self.z_offset = float(rospy.get_param("~z_offset", 0.0))

        # Motion distances
        self.pregrasp_dist = float(rospy.get_param("~pregrasp_dist", 0.15))
        self.approach_dist = float(rospy.get_param("~approach_dist", 0.05))
        self.lift_dist = float(rospy.get_param("~lift_dist", 0.20))

        # Approach direction (default: +Y axis)
        self.approach_direction = np.array([
            float(rospy.get_param("~approach_dir_x", 0.0)),
            float(rospy.get_param("~approach_dir_y", 1.0)),
            float(rospy.get_param("~approach_dir_z", 0.0)),
        ], dtype=float)

        # Normalize approach direction
        n = np.linalg.norm(self.approach_direction)
        if n < 1e-9:
            rospy.logwarn("[GrasperPy] approach_direction is zero, reset to [1,0,0].")
            self.approach_direction = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self.approach_direction /= n

        # ============================
        # TF Buffer
        # ============================

        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ============================
        # Gripper Publisher
        # ============================

        self.gripper_pub = rospy.Publisher(
            "/gripper_controller/command", JointTrajectory, queue_size=1
        )

        # Debug publishers
        self.pub_pregrasp_point = rospy.Publisher(
            "/pregrasp_point", PointStamped, queue_size=1, latch=True
        )

        self.done_pub = rospy.Publisher("/grasp_done", Bool, queue_size=1, latch=True)
        self.fail_pub = rospy.Publisher("/grasp_failed", String, queue_size=1, latch=True)

        # ============================
        # Overhang Handshake Topics
        # ============================

        self.check_topic = rospy.get_param("~check_topic", "/check_plate_grasp_point")
        self.updated_topic = rospy.get_param("~updated_topic", "/updated_grasp_point")
        self.updated_wait_timeout = float(rospy.get_param("~updated_wait_timeout", 4.0))

        # Publisher for overhang checking
        self.pub_check_grasp_point = rospy.Publisher(self.check_topic, PointStamped, queue_size=5)

        # Subscriber for updated grasp point
        self.sub_updated_grasp_point = rospy.Subscriber(
            self.updated_topic, PointStamped, self.updated_cb, queue_size=10
        )

        # Thread synchronization for updated grasp point
        self._updated_event = threading.Event()
        self._updated_lock = threading.Lock()
        self._updated_point_msg = None

        # One-shot execution flag
        self.executed = False

        # ============================
        # Subscribers
        # ============================

        self.sub_plate = rospy.Subscriber("/plate_center", PointStamped, self.plate_cb, queue_size=1)
        self.sub_table = rospy.Subscriber("/table_center", PointStamped, self.table_center_cb, queue_size=1)

        rospy.loginfo("[GrasperPy] Ready.")

    # ======================================================
    # Callback Functions
    # ======================================================

    def updated_cb(self, msg: PointStamped):
        """
        Store the updated grasp point received from overhang node.
        """
        with self._updated_lock:
            self._updated_point_msg = msg
        self._updated_event.set()

    def table_center_cb(self, msg: PointStamped):
        """
        Receive table center and update collision object.
        """
        self.table_center_msg = msg
        if (not self.table_added) and rospy.get_param("~use_table_collision", True):
            self.update_table_collision(msg)

    def plate_cb(self, msg: PointStamped):
        """
        Receive plate center and trigger full grasp pipeline (one-shot).
        """
        if self.executed:
            return

        self.executed = True

        # Unsubscribe to avoid repeated triggers
        try:
            if self.sub_plate is not None:
                self.sub_plate.unregister()
                self.sub_plate = None
        except Exception:
            pass

        ok = self.execute_grasp_from_plate_center(msg)

        if not ok:
            self.fail_pub.publish(String(data="execute_grasp_from_plate_center failed"))
        else:
            self.done_pub.publish(Bool(data=True))

    # ======================================================
    # Main Grasp Pipeline
    # ======================================================

    def execute_grasp_from_plate_center(self, centroid_msg: PointStamped) -> bool:
        """
        Full grasp pipeline:
        centroid -> check point -> updated point
        -> final grasp -> pregrasp -> approach -> close -> lift
        """

        # Convert PointStamped to PoseStamped (neutral orientation)
        pose_in = PoseStamped()
        pose_in.header = centroid_msg.header
        pose_in.pose.position = centroid_msg.point
        pose_in.pose.orientation.w = 1.0

        # Transform into planning reference frame
        centroid_rf = self.transform_pose(pose_in, self.ref_frame)
        if centroid_rf is None:
            return False

        # Convert RPY to quaternion
        R = tft.euler_matrix(self.grasp_rpy[0], self.grasp_rpy[1], self.grasp_rpy[2])
        q = tft.quaternion_from_matrix(R)

        # Build temporary check point (y_offset removed)
        check_pt = PointStamped()
        check_pt.header.frame_id = self.ref_frame
        check_pt.header.stamp = rospy.Time.now()
        check_pt.point.x = centroid_rf.pose.position.x + self.x_offset
        check_pt.point.y = centroid_rf.pose.position.y
        check_pt.point.z = centroid_rf.pose.position.z + self.z_offset

        # Publish check point and wait for updated point
        with self._updated_lock:
            self._updated_point_msg = None
        self._updated_event.clear()
        self.pub_check_grasp_point.publish(check_pt)

        got = self._updated_event.wait(timeout=self.updated_wait_timeout)

        if got:
            with self._updated_lock:
                updated_pt = self._updated_point_msg
        else:
            updated_pt = None

        # Fallback if no updated point received
        base_pt = updated_pt if updated_pt is not None else check_pt

        # Build final grasp pose (add y_offset back)
        grasp = PoseStamped()
        grasp.header.frame_id = self.ref_frame
        grasp.header.stamp = rospy.Time.now()
        grasp.pose.position.x = base_pt.point.x
        grasp.pose.position.y = base_pt.point.y + self.y_offset
        grasp.pose.position.z = base_pt.point.z

        grasp.pose.orientation.x = q[0]
        grasp.pose.orientation.y = q[1]
        grasp.pose.orientation.z = q[2]
        grasp.pose.orientation.w = q[3]

        # Build pregrasp pose (retreat along approach direction)
        pregrasp = copy.deepcopy(grasp)
        pregrasp.pose.position.x -= self.approach_direction[0] * self.pregrasp_dist
        pregrasp.pose.position.y -= self.approach_direction[1] * self.pregrasp_dist
        pregrasp.pose.position.z -= self.approach_direction[2] * self.pregrasp_dist

        # ============================
        # Execute grasp sequence
        # ============================

        self.open_gripper()
        rospy.sleep(0.8)

        if not self.move_to_pose(pregrasp):
            return False

        if not self.cartesian_move(self.approach_direction, self.approach_dist):
            return False

        self.close_gripper()
        rospy.sleep(2.0)

        if not self.cartesian_move([-1, 0, 1], self.lift_dist):
            return False

        return True
