import rospy
import numpy as np
import tf.transformations as tft

from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import MarkerArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import moveit_commander

class GrasperPy:
    def __init__(self):
        rospy.init_node("grasper")

        # ============================
        # MoveIt setup
        # ============================
        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander("arm")

        self.group.set_pose_reference_frame("base_link")
        self.group.set_end_effector_link("gripper_link")
        self.group.set_goal_tolerance(0.02)
        self.group.set_max_velocity_scaling_factor(0.3)
        self.group.set_planning_time(15.0)

        # ============================
        # Parameters
        # ============================
        self.target_label = rospy.get_param("~target_label", "bottle")

        self.x_offset = rospy.get_param("~x_offset", 0.0)
        self.y_offset = rospy.get_param("~y_offset", 0.0)
        self.z_offset = rospy.get_param("~z_offset", 0.0)

        self.grasp_rpy = [
            rospy.get_param("~grasp_roll", -np.pi / 2),
            rospy.get_param("~grasp_pitch", 0.0),
            rospy.get_param("~grasp_yaw", -np.pi / 2),
        ]

        self.pregrasp_offset = np.array([
            rospy.get_param("~pregrasp_offset_x", 0.0),
            rospy.get_param("~pregrasp_offset_y", 0.0),
            rospy.get_param("~pregrasp_offset_z", 0.1),
        ])

        # ============================
        # Gripper
        # ============================
        self.gripper_pub = rospy.Publisher(
            "/gripper_controller/command", JointTrajectory, queue_size=1,
        )

        # ============================
        # State
        # ============================
        self.get_cloud = False
        self.table_center = None

        # ============================
        # Subscribers
        # ============================
        rospy.Subscriber("/text_markers", MarkerArray, self.marker_cb, queue_size=1)
        rospy.Subscriber("/table_center", PointStamped, self.table_center_cb, queue_size=1)

        rospy.loginfo("[GrasperPy] Ready.")
    
    # ======================================================
    # Callbacks
    # ======================================================
    def table_center_cb(self, msg):
        self.table_center = msg.point

    def marker_cb(self, msg: MarkerArray):
        if self.executed:
            return
        
        for mk in msg.markers:
            if mk.text != self.target_label:
                continue

            rospy.loginfo(
                "[GrasperPy] Found target '%s' at (%.3f %.3f %.3f)",
                mk.text,
                mk.pose.position.x,
                mk.pose.position.y,
                mk.pose.position.z,
            )

            self.execute_grasp(mk)
            self.executed = True
            break

    # ======================================================
    # Core grasp logic
    # ======================================================
    def execute_grasp(self, mk):

        c = mk.pose.position
        # -------- orientation --------
        q = tft.quaternion_from_euler(*self.grasp_rpy)

        # -------- grasp pose --------
        grasp_pose = PointStamped()
        grasp_pose.header = mk.header
        grasp_pose.header.stamp = rospy.Time.now()

        grasp_pose.pose.position.x = c.x + self.x_offset
        grasp_pose.pose.position.y = c.y + self.y_offset
        grasp_pose.pose.position.z = c.z + self.z_offset

        grasp_pose.pose.orientation.x = q[0]
        grasp_pose.pose.orientation.y = q[1]
        grasp_pose.pose.orientation.z = q[2]
        grasp_pose.pose.orientation.w = q[3]

        if not self.move_to_pose(grasp_pose):
            return
        
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep(1.0)

        self.lift_up(0.05)

        # -------- pregrasp --------
        pregrasp = PointStamped()
        pregrasp.header = grasp_pose.header
        pregrasp.pose = grasp_pose.pose

        pregrasp.pose.position.x += self.pregrasp_offset[0]
        pregrasp.pose.position.y += self.pregrasp_offset[1]
        pregrasp.pose.position.z += self.pregrasp_offset[2]

        # ======================================================
        # MoveIt execution
        # ======================================================
        if not self.move_to_pose(pregrasp):
            return
        
        rospy.sleep(1.0)

        if not self.cartesion_move([1, 0, -1], 0.16):
            return
        
        self.close_gripper()
        rospy.sleep(1.5)

        self.cartesion_move([0, 0, 1], 0.01)

        # ======================================================
        # Place
        # ======================================================
        if self.table_center is None:
            rospy.logerr("No table center")
            return
        
        current_z = self.group.get_current_pose().pose.position.z

        place_pose = PointStamped()
        place_pose.header.frame_id = "base_link"
        place_pose.pose.position.x = self.table_center.x - 0.1
        place_pose.pose.position.y = self.table_center.y
        place_pose.pose.position.z = current_z - 0.16
        place_pose.pose.orientation = grasp_pose.pose.orientation

        self.move_to_pose(place_pose)
        rospy.sleep(0.5)

        self.open_gripper()
        rospy.sleep(1.0)

        self.cartesion_move([0, 0, 1], 0.02)

        rospy.loginfo("Grasp & place finished")

    # ======================================================
    # Motion helpers
    # ======================================================
    def move_to_pose(self, pose: PointStamped) -> bool:
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()

        self.group.set_pose_target(pose)
        success = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        if not success:
            rospy.logerr("Move failed")
        return success
    
    def cartesion_move(self, direction, distance):
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)

        start = self.group.get_current_pose().pose
        target = PointStamped()
        target.header.frame_id = "base_link"
        target.pose = start

        target.pose.position.x += direction[0] * distance
        target.pose.position.y += direction[1] * distance
        target.pose.position.z += direction[2] * distance

        waypoints = [start, target.pose]

        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints,
            eef_step=0.005,
            jump_threshold=0.0,
        )

        if fraction < 0.99:
            rospy.logerr("Cartesian path failed")
            return False
        
        self.group.execute(plan, wait=True)
        return True
    
    # ======================================================
    # Gripper
    # ======================================================
    def close_gripper(self):
        self.send_gripper_cmd([0.0, 0.0], 3.0)

    def open_gripper(self):
        self.send_gripper_cmd([0.04, 0.04], 3.5)

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