#!/usr/bin/env python3
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
    grasp only the plate using /plate_center (PointStamped)

    NEW behavior:
      - Compute a "check grasp point" (centroid + x_offset, y_offset=0, z_offset)
      - Publish it to /check_plate_grasp_point for plate_overhang_node.py
      - Wait for /updated_grasp_point (timeout fallback)
      - After updated point arrives: add y_offset BACK in grasp pose, then continue
        pregrasp -> approach -> close -> lift
    """

    def __init__(self):
        rospy.init_node("plate_grasper")

        # ============================
        # MoveIt setup
        # ============================
        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander("arm")
        self.group_cartesian = moveit_commander.MoveGroupCommander("arm_torso")

        self.ref_frame = self.group.get_planning_frame()
        rospy.loginfo("[GrasperPy] Using planning frame as ref_frame: '%s'", self.ref_frame)

        self.eef_link = rospy.get_param("~eef_link", "gripper_link")
        self.group.set_pose_reference_frame(self.ref_frame)
        self.group.set_end_effector_link(self.eef_link)
        self.group_cartesian.set_pose_reference_frame(self.ref_frame)
        self.group_cartesian.set_end_effector_link(self.eef_link)

        # tolerances & speed
        self.group.set_goal_tolerance(rospy.get_param("~goal_tolerance", 0.01))
        self.velocity_scaling = rospy.get_param("~velocity_scaling", 0.3)
        self.accel_scaling = rospy.get_param("~accel_scaling", 0.2)
        self.group.set_max_velocity_scaling_factor(self.velocity_scaling)
        self.group.set_max_acceleration_scaling_factor(self.accel_scaling)
        self.group_cartesian.set_max_velocity_scaling_factor(self.velocity_scaling)
        self.group_cartesian.set_max_acceleration_scaling_factor(self.accel_scaling)

        # Planner config
        self.group.set_planner_id(rospy.get_param("~planner_id", "RRTConnectkConfigDefault"))
        self.group.set_planning_time(rospy.get_param("~planning_time", 15.0))
        self.group.set_num_planning_attempts(rospy.get_param("~num_planning_attempts", 5))

        # ============================
        # Planning Scene (collision objects)
        # ============================
        self.scene = PlanningSceneInterface(synchronous=True)

        # Table collision params
        self.table_size_x = float(rospy.get_param("~table_size_x", 0.2))
        self.table_size_y = float(rospy.get_param("~table_size_y", 1.0))
        self.table_thickness = float(rospy.get_param("~table_thickness", 0.4))
        self.table_z_offset = float(rospy.get_param("~table_z_offset", 0.02))
        self.table_object_id = rospy.get_param("~table_object_id", "table")
        self.table_added = False
        self.table_center_msg = None

        # ============================
        # Grasp Parameters (ONLY for plate)
        # ============================
        self.grasp_rpy = [
            rospy.get_param("~grasp_roll", np.pi / 2),
            rospy.get_param("~grasp_pitch", np.pi / 2),
            rospy.get_param("~grasp_yaw", 0.0),
        ]

        # Position offsets applied to centroid (in ref_frame)
        self.x_offset = float(rospy.get_param("~x_offset", -0.13))
        self.y_offset = float(rospy.get_param("~y_offset", -0.19))
        self.z_offset = float(rospy.get_param("~z_offset", 0.02))

        # motion distances
        self.pregrasp_dist = float(rospy.get_param("~pregrasp_dist", 0.15))
        self.approach_dist = float(rospy.get_param("~approach_dist", 0.05))
        self.lift_dist = float(rospy.get_param("~lift_dist", 0.20))

        # approach direction in ref_frame (default: +Y)
        self.approach_direction = np.array([
            float(rospy.get_param("~approach_dir_x", 0.0)),
            float(rospy.get_param("~approach_dir_y", 1.0)),
            float(rospy.get_param("~approach_dir_z", 0.0)),
        ], dtype=float)
        n = np.linalg.norm(self.approach_direction)
        if n < 1e-9:
            rospy.logwarn("[GrasperPy] approach_direction is zero, reset to [1,0,0].")
            self.approach_direction = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self.approach_direction /= n

        # ============================
        # TF buffer
        # ============================
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ============================
        # Gripper
        # ============================
        self.gripper_pub = rospy.Publisher(
            "/gripper_controller/command", JointTrajectory, queue_size=1
        )

        # Debug Visualization
        self.pub_pregrasp_point = rospy.Publisher(
            "/pregrasp_point", PointStamped, queue_size=1, latch=True
        )

        self.done_pub = rospy.Publisher("/grasp_done", Bool, queue_size=1, latch=True)
        self.fail_pub = rospy.Publisher("/grasp_failed", String, queue_size=1, latch=True)

        # ============================
        # NEW: Overhang handshake topics
        # ============================
        self.check_topic = rospy.get_param("~check_topic", "/check_plate_grasp_point")
        self.updated_topic = rospy.get_param("~updated_topic", "/updated_grasp_point")
        self.updated_wait_timeout = float(rospy.get_param("~updated_wait_timeout", 2.0))

        self.pub_check_grasp_point = rospy.Publisher(self.check_topic, PointStamped, queue_size=5)
        self.sub_updated_grasp_point = rospy.Subscriber(
            self.updated_topic, PointStamped, self.updated_cb, queue_size=10
        )

        self._updated_event = threading.Event()
        self._updated_lock = threading.Lock()
        self._updated_point_msg = None

        # State
        self.executed = False

        # ============================
        # Subscribers
        # ============================
        self.sub_plate = rospy.Subscriber("/plate_center", PointStamped, self.plate_cb, queue_size=1)
        self.sub_table = rospy.Subscriber("/table_center", PointStamped, self.table_center_cb, queue_size=1)

        rospy.loginfo(
            "[GrasperPy] Ready. ref_frame='%s', eef_link='%s'", self.ref_frame, self.eef_link
        )
        rospy.loginfo(
            "[GrasperPy] Overhang handshake: check_topic=%s  updated_topic=%s  timeout=%.2fs",
            self.check_topic, self.updated_topic, self.updated_wait_timeout
        )

    # ======================================================
    # Callbacks
    # ======================================================
    def updated_cb(self, msg: PointStamped):
        # Store latest updated point (no strict stamp matching; we assume one-shot pipeline)
        with self._updated_lock:
            self._updated_point_msg = msg
        self._updated_event.set()

    def table_center_cb(self, msg: PointStamped):
        self.table_center_msg = msg
        if (not self.table_added) and rospy.get_param("~use_table_collision", True):
            self.update_table_collision(msg)

    def plate_cb(self, msg: PointStamped):
        if self.executed:
            return

        self.executed = True
        rospy.loginfo(
            "[PlateGrasper] Got /plate_center, frame='%s' (%.3f, %.3f, %.3f)",
            msg.header.frame_id, msg.point.x, msg.point.y, msg.point.z
        )

        # freeze subscriptions (one-shot)
        try:
            if self.sub_plate is not None:
                self.sub_plate.unregister()
                self.sub_plate = None
        except Exception:
            pass

        ok = self.execute_grasp_from_plate_center(msg)
        if not ok:
            rospy.logerr("[PlateGrasper] Plate grasp FAILED.")
            self.fail_pub.publish(String(data="execute_grasp_from_plate_center failed"))
        else:
            rospy.loginfo("[PlateGrasper] Plate grasp DONE.")
            self.done_pub.publish(Bool(data=True))

    # ======================================================
    # Main pipeline
    # ======================================================
    def execute_grasp_from_plate_center(self, centroid_msg: PointStamped) -> bool:
        """
        Full pipeline:
          (centroid) -> build check point (y_offset=0) -> overhang updated -> add y_offset back
          -> pregrasp -> approach -> close -> lift
        """

        # 0) PointStamped -> PoseStamped (neutral orientation)
        pose_in = PoseStamped()
        pose_in.header = centroid_msg.header
        pose_in.pose.position = centroid_msg.point
        pose_in.pose.orientation.w = 1.0

        # 1) TF transform into ref_frame
        centroid_rf = self.transform_pose(pose_in, self.ref_frame)
        if centroid_rf is None:
            rospy.logerr("[GrasperPy] TF plate point -> %s failed", self.ref_frame)
            return False

        # 2) Fixed orientation from RPY
        R = tft.euler_matrix(self.grasp_rpy[0], self.grasp_rpy[1], self.grasp_rpy[2])
        q = tft.quaternion_from_matrix(R)

        # ------------------------------------------------------
        # NEW: send check point to plate_overhang_node.py
        #   check uses y_offset = 0 (as you requested)
        # ------------------------------------------------------
        check_pt = PointStamped()
        check_pt.header.frame_id = self.ref_frame
        check_pt.header.stamp = rospy.Time.now()
        check_pt.point.x = centroid_rf.pose.position.x + self.x_offset
        check_pt.point.y = centroid_rf.pose.position.y + 0.0     # IMPORTANT: y_offset removed here
        check_pt.point.z = centroid_rf.pose.position.z + self.z_offset

        rospy.loginfo(
            "[GrasperPy] Publish check grasp point (y_offset=0): (%.3f, %.3f, %.3f) -> %s",
            check_pt.point.x, check_pt.point.y, check_pt.point.z, self.check_topic
        )

        # clear and wait
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

        if updated_pt is None:
            rospy.logwarn(
                "[GrasperPy] No /updated_grasp_point within %.2fs. Fallback to check point.",
                self.updated_wait_timeout
            )
            base_pt = check_pt
        else:
            base_pt = updated_pt
            rospy.loginfo(
                "[GrasperPy] Got updated grasp point: (%.3f, %.3f, %.3f) from %s",
                base_pt.point.x, base_pt.point.y, base_pt.point.z, self.updated_topic
            )

        # ------------------------------------------------------
        # Build FINAL grasp pose:
        #   x,z from base_pt (overhang-updated)
        #   y = base_pt.y + y_offset  (add back AFTER overhang)
        # ------------------------------------------------------
        grasp = PoseStamped()
        grasp.header.frame_id = self.ref_frame
        grasp.header.stamp = rospy.Time.now()
        grasp.pose.position.x = base_pt.point.x
        grasp.pose.position.y = base_pt.point.y + self.y_offset   # IMPORTANT: add y_offset back here
        grasp.pose.position.z = base_pt.point.z

        grasp.pose.orientation.x = q[0]
        grasp.pose.orientation.y = q[1]
        grasp.pose.orientation.z = q[2]
        grasp.pose.orientation.w = q[3]

        rospy.loginfo(
            "[GrasperPy] Final grasp pose position: (%.3f, %.3f, %.3f)  (y_offset=%.3f added back)",
            grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z, self.y_offset
        )

        # 3) Build pregrasp pose (retreat from FINAL grasp pose)
        pregrasp = copy.deepcopy(grasp)
        pregrasp.pose.position.x -= self.approach_direction[0] * self.pregrasp_dist
        pregrasp.pose.position.y -= self.approach_direction[1] * self.pregrasp_dist
        pregrasp.pose.position.z -= self.approach_direction[2] * self.pregrasp_dist

        # Debug: publish pregrasp point
        ps_pre = PointStamped()
        ps_pre.header.frame_id = self.ref_frame
        ps_pre.header.stamp = rospy.Time.now()
        ps_pre.point.x = pregrasp.pose.position.x
        ps_pre.point.y = pregrasp.pose.position.y
        ps_pre.point.z = pregrasp.pose.position.z
        self.pub_pregrasp_point.publish(ps_pre)

        # ============================
        # Execute grasp
        # ============================
        rospy.loginfo("[PlateGrasper] Open gripper ...")
        self.open_gripper()
        rospy.sleep(0.8)

        rospy.loginfo("[GrasperPy] Move to pregrasp...")
        if not self.move_to_pose(pregrasp):
            return False

        rospy.loginfo("[GrasperPy] Cartesian approach...")
        if not self.cartesian_move(self.approach_direction, self.approach_dist):
            return False

        rospy.loginfo("[GrasperPy] Close gripper ...")
        self.close_gripper()
        rospy.sleep(2.0)

        rospy.loginfo("[GrasperPy] Lift up...")
        if not self.cartesian_move([-1, 0, 1], self.lift_dist):
            return False

        rospy.loginfo("[GrasperPy] Grasp plate finished")
        return True

    # ======================================================
    # Planning Scene
    # ======================================================
    def update_table_collision(self, table_center_msg: PointStamped):
        """
        Add/update a box collision object for the table in PlanningScene.
        """
        table_pose_in = PoseStamped()
        table_pose_in.header.frame_id = table_center_msg.header.frame_id
        table_pose_in.header.stamp = rospy.Time(0)
        table_pose_in.pose.position = table_center_msg.point
        table_pose_in.pose.orientation.w = 1.0

        table_pose = self.transform_pose(table_pose_in, self.ref_frame)
        if table_pose is None:
            rospy.logerr("[GrasperPy] TF table_center -> %s failed", self.ref_frame)
            return

        box_pose = PoseStamped()
        box_pose.header.frame_id = self.ref_frame
        box_pose.header.stamp = rospy.Time.now()
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = table_pose.pose.position.x
        box_pose.pose.position.y = table_pose.pose.position.y
        box_pose.pose.position.z = float(
            table_pose.pose.position.z + self.table_z_offset - self.table_thickness / 2.0
        )

        try:
            self.scene.remove_world_object(self.table_object_id)
            rospy.sleep(0.05)
        except Exception:
            pass

        self.scene.add_box(
            name=self.table_object_id,
            pose=box_pose,
            size=(self.table_size_x, self.table_size_y, self.table_thickness),
        )
        self.table_added = True
        rospy.loginfo(
            "[GrasperPy] Table collision object updated: center=(%.3f %.3f %.3f), size=(%.2f %.2f %.2f)",
            box_pose.pose.position.x, box_pose.pose.position.y, box_pose.pose.position.z,
            self.table_size_x, self.table_size_y, self.table_thickness
        )

    # ======================================================
    # TF transformer
    # ======================================================
    def transform_pose(self, pose_st: PoseStamped, target_frame: str):
        """
        Transform PoseStamped -> target_frame using tf2 (safe for Python / Noetic).
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose_st.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.3)
            )
            pose_out = tf2_geometry_msgs.do_transform_pose(pose_st, transform)
            pose_out.header.frame_id = target_frame
            pose_out.header.stamp = rospy.Time.now()
            return pose_out
        except Exception as e:
            rospy.logerr("[GrasperPy] transform_pose failed: %s", str(e))
            return None

    # ======================================================
    # MoveIt helpers
    # ======================================================
    def move_to_pose(self, pose: PoseStamped) -> bool:
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()

        self.group.set_pose_target(pose)
        success = self.group.go(wait=True)

        self.group.stop()
        self.group.clear_pose_targets()

        if not success:
            rospy.logerr("[GrasperPy] Move failed")
        return success

    def cartesian_move(self, direction, distance) -> bool:
        direction = np.asarray(direction, dtype=float)
        n = np.linalg.norm(direction)
        if n < 1e-9:
            rospy.logerr("[GrasperPy] Cartesian direction is zero!")
            return False
        direction /= n

        self.group_cartesian.stop()
        self.group_cartesian.clear_pose_targets()
        self.group_cartesian.set_start_state_to_current_state()

        start_pose = copy.deepcopy(self.group_cartesian.get_current_pose().pose)

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.ref_frame
        target_pose.pose = copy.deepcopy(start_pose)
        target_pose.pose.position.x += direction[0] * distance
        target_pose.pose.position.y += direction[1] * distance
        target_pose.pose.position.z += direction[2] * distance

        waypoints = [start_pose, target_pose.pose]

        try:
            eef_step = float(rospy.get_param("~cartesian_eef_step", 0.005))
            jump_threshold = float(rospy.get_param("~cartesian_jump_threshold", 0.0))
            plan, fraction = self._compute_cartesian_path(
                self.group_cartesian,
                waypoints,
                eef_step,
                jump_threshold,
                True,
            )
        except Exception as e:
            rospy.logerr("[GrasperPy] compute_cartesian_path exception: %s", str(e))
            return False

        if fraction < 0.95:
            rospy.logerr("[GrasperPy] Cartesian path failed (%.2f)", fraction)
            return False

        try:
            current_state = self.group_cartesian.get_current_state()
            plan = self.group_cartesian.retime_trajectory(
                current_state,
                plan,
                self.velocity_scaling,
                self.accel_scaling,
            )
        except Exception as e:
            rospy.logwarn("[GrasperPy] retime_trajectory failed: %s", str(e))

        min_dt = float(rospy.get_param("~cartesian_min_time_step", 0.01))
        if self._ensure_monotonic_time(plan, min_dt=min_dt):
            rospy.logwarn("[GrasperPy] Cartesian trajectory time fixed (min_dt=%.3f).", min_dt)

        success = self.group_cartesian.execute(plan, wait=True)
        self.group_cartesian.stop()

        if not success:
            rospy.logerr("[GrasperPy] Cartesian execute failed")
        return success

    def _compute_cartesian_path(self, group, waypoints, eef_step, jump_threshold, avoid_collisions):
        try:
            return group.compute_cartesian_path(
                waypoints,
                eef_step,
                avoid_collisions=avoid_collisions,
            )
        except Exception:
            try:
                return group.compute_cartesian_path(
                    waypoints,
                    eef_step,
                    jump_threshold,
                    avoid_collisions,
                )
            except Exception:
                poses = [conversions.pose_to_list(p) for p in waypoints]
                ser_path, fraction = group._g.compute_cartesian_path(
                    poses,
                    eef_step,
                    jump_threshold,
                    avoid_collisions,
                )
                path = RobotTrajectory()
                path.deserialize(ser_path)
                return (path, fraction)

    def _ensure_monotonic_time(self, traj, min_dt=0.01) -> bool:
        if traj is None or not hasattr(traj, "joint_trajectory"):
            return False
        points = traj.joint_trajectory.points
        if len(points) < 2:
            return False

        changed = False
        last = points[0].time_from_start.to_sec()
        if last <= 0.0:
            last = min_dt
            points[0].time_from_start = rospy.Duration(last)
            changed = True

        for i in range(1, len(points)):
            t = points[i].time_from_start.to_sec()
            if t <= last:
                last = last + min_dt
                points[i].time_from_start = rospy.Duration(last)
                changed = True
            else:
                last = t
        return changed

    # ======================================================
    # Gripper
    # ======================================================
    def close_gripper(self):
        self.send_gripper_cmd([0.0, 0.0], 2.5)

    def open_gripper(self):
        self.send_gripper_cmd([0.04, 0.04], 2.5)

    def send_gripper_cmd(self, positions, duration):
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

        for _ in range(3):
            self.gripper_pub.publish(traj)
            rospy.sleep(0.1)


if __name__ == "__main__":
    GrasperPy()
    rospy.spin()

