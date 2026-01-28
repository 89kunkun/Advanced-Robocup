#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf.transformations as tft
import copy

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PoseStamped, PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

import moveit_commander
from moveit_commander import PlanningSceneInterface
import moveit_commander.conversions as conversions
from moveit_msgs.msg import RobotTrajectory


class GraspPlatePy:
    def __init__(self):
        rospy.init_node("grasp_plate")

        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander("arm")

        self.ref_frame = self.group.get_planning_frame()
        rospy.loginfo("[GraspPlatePy] Using planning frame as ref_frame: '%s'", self.ref_frame)

        self.eef_link = rospy.get_param("~eef_link", "gripper_link")
        self.group.set_pose_reference_frame(self.ref_frame)
        self.group.set_end_effector_link(self.eef_link)

        self.group.set_goal_tolerance(float(rospy.get_param("~goal_tolerance", 0.05)))
        self.velocity_scaling = float(rospy.get_param("~velocity_scaling", 0.3))
        self.accel_scaling = float(rospy.get_param("~accel_scaling", 0.2))
        self.group.set_max_velocity_scaling_factor(self.velocity_scaling)
        self.group.set_max_acceleration_scaling_factor(self.accel_scaling)

        self.group.set_planner_id(rospy.get_param("~planner_id", "RRTConnectkConfigDefault"))
        self.group.set_planning_time(float(rospy.get_param("~planning_time", 15.0)))
        self.group.set_num_planning_attempts(int(rospy.get_param("~num_planning_attempts", 5)))

        self.scene = PlanningSceneInterface(synchronous=True)

        self.grasp_rpy = [
            float(rospy.get_param("~grasp_roll", -np.pi / 2)),
            float(rospy.get_param("~grasp_pitch", 0.0)),
            float(rospy.get_param("~grasp_yaw", np.pi / 2)),
        ]

        self.x_offset = float(rospy.get_param("~x_offset", 0.0))
        self.y_offset = float(rospy.get_param("~y_offset", 0.0))
        self.z_offset = float(rospy.get_param("~z_offset", 0.0))

        self.pregrasp_offset = np.array([
            float(rospy.get_param("~pregrasp_offset_x", -0.25)),
            float(rospy.get_param("~pregrasp_offset_y", 0.0)),
            float(rospy.get_param("~pregrasp_offset_z", 0.0)),
        ], dtype=float)

        self.approach_dist = float(rospy.get_param("~approach_dist", 0.25))
        self.lift_dist = float(rospy.get_param("~lift_dist", 0.1))

        self.open_before_grasp = bool(rospy.get_param("~open_before_grasp", True))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 1.0))

        # TF2
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Gripper
        self.gripper_pub = rospy.Publisher("/gripper_controller/command", JointTrajectory, queue_size=1)

        # Debug visualization
        self.pub_pregrasp_pose = rospy.Publisher("/pregrasp_pose", PoseStamped, queue_size=1, latch=True)
        self.pub_pregrasp_marker = rospy.Publisher("/pregrasp_marker", Marker, queue_size=1, latch=True)

        self.executed = False

        # ✅ relative name so remap works
        self.sub_plate = rospy.Subscriber("plate_center", PointStamped, self.plate_center_cb, queue_size=1)

        rospy.loginfo("[GraspPlatePy] Ready. Waiting plate_center. ref_frame='%s', eef_link='%s'",
                      self.ref_frame, self.eef_link)

    def plate_center_cb(self, msg: PointStamped):
        if self.executed:
            return
        self.executed = True

        rospy.loginfo("[GraspPlatePy] Got plate_center in frame '%s' at (%.3f %.3f %.3f)",
                      msg.header.frame_id, msg.point.x, msg.point.y, msg.point.z)

        try:
            if self.sub_plate is not None:
                self.sub_plate.unregister()
                self.sub_plate = None
        except Exception as e:
            rospy.logwarn("[GraspPlatePy] unregister subscriber exception: %s", str(e))

        ok = self.execute_grasp_from_plate_center(msg)
        if not ok:
            rospy.logerr("[GraspPlatePy] Grasp failed (one-shot).")
        else:
            rospy.loginfo("[GraspPlatePy] Grasp + lift finished (DONE).")

    def execute_grasp_from_plate_center(self, center_msg: PointStamped) -> bool:
        center_pose = PoseStamped()
        center_pose.header = center_msg.header
        center_pose.pose.position = center_msg.point
        center_pose.pose.orientation.w = 1.0

        if center_pose.header.frame_id == self.ref_frame:
            center_ref = center_pose
        else:
            rospy.loginfo("[GraspPlatePy] TF %s -> %s", center_pose.header.frame_id, self.ref_frame)
            center_ref = self.transform_pose(center_pose, self.ref_frame)
            if center_ref is None:
                rospy.logerr("[GraspPlatePy] TF plate_center -> %s failed", self.ref_frame)
                return False

        q = tft.quaternion_from_euler(*self.grasp_rpy)

        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = self.ref_frame
        grasp_pose.header.stamp = rospy.Time.now()
        grasp_pose.pose.position.x = center_ref.pose.position.x + self.x_offset
        grasp_pose.pose.position.y = center_ref.pose.position.y + self.y_offset
        grasp_pose.pose.position.z = center_ref.pose.position.z + self.z_offset
        grasp_pose.pose.orientation.x = q[0]
        grasp_pose.pose.orientation.y = q[1]
        grasp_pose.pose.orientation.z = q[2]
        grasp_pose.pose.orientation.w = q[3]

        pregrasp = PoseStamped()
        pregrasp.header = grasp_pose.header
        pregrasp.pose = copy.deepcopy(grasp_pose.pose)
        pregrasp.pose.position.x += float(self.pregrasp_offset[0])
        pregrasp.pose.position.y += float(self.pregrasp_offset[1])
        pregrasp.pose.position.z += float(self.pregrasp_offset[2])

        self.publish_pregrasp_visual(pregrasp)

        if self.open_before_grasp:
            rospy.loginfo("[GraspPlatePy] Open gripper (pre) ...")
            self.open_gripper()
            rospy.sleep(1.0)

        rospy.loginfo("[GraspPlatePy] Move to pregrasp ...")
        if not self.move_to_pose(pregrasp):
            return False

        rospy.loginfo("[GraspPlatePy] Cartesian approach (+X) dist=%.3f ...", self.approach_dist)
        if not self.cartesian_move([1, 0, 0], self.approach_dist):
            return False

        rospy.loginfo("[GraspPlatePy] Close gripper ...")
        self.close_gripper()
        rospy.sleep(1.5)

        rospy.loginfo("[GraspPlatePy] Lift up (+Z) dist=%.3f ...", self.lift_dist)
        if not self.cartesian_move([0, 0, 1], self.lift_dist):
            return False

        return True

    def transform_pose(self, pose_st: PoseStamped, target_frame: str):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose_st.header.frame_id,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout)
            )
            pose_out = tf2_geometry_msgs.do_transform_pose(pose_st, transform)
            pose_out.header.frame_id = target_frame
            pose_out.header.stamp = rospy.Time.now()
            return pose_out
        except Exception as e:
            rospy.logerr("[GraspPlatePy] transform_pose failed: %s", str(e))
            return None

    def publish_pregrasp_visual(self, pregrasp: PoseStamped):
        try:
            self.pub_pregrasp_pose.publish(pregrasp)

            mk = Marker()
            mk.header = pregrasp.header
            mk.ns = "pregrasp"
            mk.id = 0
            mk.type = Marker.ARROW
            mk.action = Marker.ADD
            mk.pose = pregrasp.pose
            mk.scale.x = 0.15
            mk.scale.y = 0.02
            mk.scale.z = 0.02
            mk.color.r = 0.0
            mk.color.g = 1.0
            mk.color.b = 0.0
            mk.color.a = 1.0
            self.pub_pregrasp_marker.publish(mk)
        except Exception as e:
            rospy.logwarn("[GraspPlatePy] publish_pregrasp_visual exception: %s", str(e))

    def move_to_pose(self, pose: PoseStamped) -> bool:
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()

        self.group.set_pose_target(pose)
        success = self.group.go(wait=True)

        self.group.stop()
        self.group.clear_pose_targets()

        if not success:
            rospy.logerr("[GraspPlatePy] Move failed")
        return success

    # ✅ FIX: enforce strictly increasing time_from_start before execute
    def _fix_trajectory_timestamps(self, traj: RobotTrajectory, dt: float = 0.05) -> RobotTrajectory:
        try:
            jt = traj.joint_trajectory
            if not jt.points or len(jt.points) < 2:
                return traj

            # set header stamp to 0 (let controller interpret relative times)
            jt.header.stamp = rospy.Time(0)

            t = 0.0
            for i, p in enumerate(jt.points):
                # make strictly increasing
                t += dt
                p.time_from_start = rospy.Duration.from_sec(t)

            traj.joint_trajectory = jt
            return traj
        except Exception as e:
            rospy.logwarn("[GraspPlatePy] _fix_trajectory_timestamps exception: %s", str(e))
            return traj

    def cartesian_move(self, direction, distance) -> bool:
        direction = np.asarray(direction, dtype=float)
        n = np.linalg.norm(direction)
        if n < 1e-9:
            rospy.logerr("[GraspPlatePy] Cartesian direction is zero!")
            return False
        direction /= n

        start_pose = copy.deepcopy(self.group.get_current_pose().pose)

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.ref_frame
        target_pose.pose = copy.deepcopy(start_pose)
        target_pose.pose.position.x += direction[0] * distance
        target_pose.pose.position.y += direction[1] * distance
        target_pose.pose.position.z += direction[2] * distance

        waypoints = [start_pose, target_pose.pose]

        try:
            eef_step = float(rospy.get_param("~cartesian_eef_step", 0.01))
            jump_threshold = float(rospy.get_param("~cartesian_jump_threshold", 0.0))
            plan, fraction = self._compute_cartesian_path(
                waypoints, eef_step, jump_threshold, True
            )
        except Exception as e:
            rospy.logerr("[GraspPlatePy] compute_cartesian_path exception: %s", str(e))
            return False

        if fraction < 0.99:
            rospy.logerr("[GraspPlatePy] Cartesian path failed (%.2f)", fraction)
            return False

        # ✅ critical fix
        plan = self._fix_trajectory_timestamps(plan, dt=0.05)

        success = self.group.execute(plan, wait=True)
        self.group.stop()

        if not success:
            rospy.logerr("[GraspPlatePy] Cartesian execute failed")
        return success

    def _compute_cartesian_path(self, waypoints, eef_step, jump_threshold, avoid_collisions):
        try:
            return self.group.compute_cartesian_path(
                waypoints, eef_step, avoid_collisions=avoid_collisions
            )
        except Exception:
            try:
                return self.group.compute_cartesian_path(
                    waypoints, eef_step, jump_threshold, avoid_collisions
                )
            except Exception:
                poses = [conversions.pose_to_list(p) for p in waypoints]
                ser_path, fraction = self.group._g.compute_cartesian_path(
                    poses, eef_step, jump_threshold, avoid_collisions
                )
                path = RobotTrajectory()
                path.deserialize(ser_path)
                return (path, fraction)

    def close_gripper(self):
        self.send_gripper_cmd([0.0, 0.0], 3.0)

    def open_gripper(self):
        self.send_gripper_cmd([0.04, 0.04], 3.5)

    def send_gripper_cmd(self, positions, duration):
        traj = JointTrajectory()
        traj.joint_names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = rospy.Duration(duration)
        traj.points.append(pt)
        traj.header.stamp = rospy.Time.now() + rospy.Duration(0.2)

        for _ in range(3):
            self.gripper_pub.publish(traj)
            rospy.sleep(0.1)


if __name__ == "__main__":
    GraspPlatePy()
    rospy.spin()

