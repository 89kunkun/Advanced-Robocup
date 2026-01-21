import rospy
import math
import tf2_ros
import tf.transformations as tft

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class TiagoJointCommander:
    def __init__(self):
        # =========================
        # Publishers
        # =========================
        self.arm_pub = rospy.Publisher("/arm_controller/command", JointTrajectory, queue_size=1)
        self.head_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=1)
        self.gripper_pub = rospy.Publisher("/gripper_controller/command", JointTrajectory, queue_size=1)
        self.torso_pub = rospy.Publisher("/torso_controller/command", JointTrajectory, queue_size=1)    

        # =========================
        # TF
        # =========================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.sleep(1.0)  # Allow publishers to set up
        rospy.loginfo("[TiagoJointCommander] Publishers ready.")

    # ==================================================
    # Arm
    # ==================================================
    def move_arm(self):
        arm_positions = rospy.get_param(
            "arm_positions",
            [0.4, -1.17, -1.9, 2.3, -1.3, -0.45, 0.0]
        )

        if len(arm_positions) != 7:
            rospy.logwarn("default arm_positions")
            arm_positions = [0.4, -1.17, -1.9, 2.3, -1.3, -0.45, 0.0]

        traj = JointTrajectory()
        traj.joint_names = [
            "arm_1_joint",
            "arm_2_joint",
            "arm_3_joint",
            "arm_4_joint",
            "arm_5_joint",
            "arm_6_joint",
            "arm_7_joint",
        ]

        pt = JointTrajectoryPoint()
        pt.positions = arm_positions
        pt.velocities = [0.2] * 7
        pt.time_from_start = rospy.Duration(3.0)

        traj.points.append(pt)
        traj.header.stamp = rospy.Time.now()

        self.arm_pub.publish(traj)
        rospy.loginfo("Arm command sent.")

    # ==================================================
    # Head
    # ==================================================
    def move_head(self):
        head_positions = rospy.get_param(
            "head_positions",
            [0.0, 0.0]
        )

        traj = JointTrajectory()
        traj.joint_names = [
            "head_1_joint",
            "head_2_joint",
        ]

        pt = JointTrajectoryPoint()
        pt.positions = head_positions
        pt.time_from_start = rospy.Duration(1.0)

        traj.points.append(pt)
        traj.header.stamp = rospy.Time.now()

        self.head_pub.publish(traj)
        rospy.loginfo("Head command sent.")

    # ==================================================
    # Gripper
    # ==================================================
    def move_gripper(self):
        gripper_positions = rospy.get_param(
            "gripper_positions",
            [0.0, 0.0]
        )

        traj = JointTrajectory()
        traj.joint_names = [
            "gripper_left_finger_joint",
            "gripper_right_finger_joint",
        ]

        pt = JointTrajectoryPoint()
        pt.positions = gripper_positions
        pt.time_from_start = rospy.Duration(1.0)

        traj.points.append(pt)
        traj.header.stamp = rospy.Time.now()

        self.gripper_pub.publish(traj)
        rospy.loginfo("Gripper command sent.")

    # ==================================================
    # Torso
    # ==================================================
    def move_torso(self):
        torso_position = rospy.get_param("torso_position", 0.2)

        traj = JointTrajectory()
        traj.joint_names = ["torso_lift_joint"]

        pt = JointTrajectoryPoint()
        pt.positions = [torso_position]
        pt.time_from_start = rospy.Duration(1.0)

        traj.points.append(pt)
        traj.header.stamp = rospy.Time.now()

        self.torso_pub.publish(traj)
        rospy.loginfo(f"Torso command sent: {torso_position}")

    # ==================================================
    # End-effector RPY via TF
    # ==================================================
    def print_end_effector_rpy(self):
        """
        Query TF: base_link -> gripper_link
        and print roll / pitch / yaw
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                "base_link",
                "gripper_link",
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            q = trans.transform.rotation
            roll, pitch, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])

            rospy.loginfo(
                "End-effector RPY (rad): roll=%.3f pitch=%.3f yaw=%.3f",
                roll, pitch, yaw
            )
            rospy.loginfo(
                "End-effector RPY (deg): roll=%.1f pitch=%.1f yaw=%.1f",
                math.degrees(roll),
                math.degrees(pitch),
                math.degrees(yaw)
            )
        
        except Exception as e:
            rospy.logerr(f"Failed to get end-effector TF: {e}")

# ==================================================
# main
# ==================================================
if __name__ == "__main__":
    rospy.init_node("grasp_try")

    commander = TiagoJointCommander()

    commander.move_head()
    rospy.sleep(0.8)

    commander.move_arm()
    rospy.sleep(3.0)

    commander.print_end_effector_rpy()

    commander.move_gripper()
    rospy.sleep(0.8)

    commander.move_torso()
    rospy.sleep(0.8)

    rospy.loginfo("All joint commands sent.")