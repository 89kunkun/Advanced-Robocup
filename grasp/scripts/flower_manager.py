#!/usr/bin/env python3
import rospy
import subprocess
from std_msgs.msg import Bool, String

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class FlowManager:
    def __init__(self):
        rospy.init_node("flow_manager")

        self.finished = False

        # configurable flow messages
        self.flow_success_msg = rospy.get_param("~flow_success_msg", "SUCCESS")
        self.flow_failed_msg = rospy.get_param("~flow_failed_msg", "FAILED")    
        rospy.loginfo(
            "[FlowManager] flow_success_msg='%s', flow_failed_msg='%s'",
            self.flow_success_msg,
            self.flow_failed_msg
        )

        rospy.Subscriber("/grasp_done", Bool, self.done_cb, queue_size=1)
        rospy.Subscriber("/grasp_failed", String, self.fail_cb, queue_size=1)

        self.flow_pub = rospy.Publisher(
            "/flow_result", String, queue_size=1, latch=True
        )

        self.gripper_pub = rospy.Publisher(
            "/gripper_controller/command", JointTrajectory, queue_size=1
        )

        rospy.loginfo("[FlowManager] Waiting for grasp result ...")
        rospy.spin()

    def publish_flow_result(self, result: str):
        rospy.loginfo("[FlowManager] Publishing flow_result=%s", result)
        self.flow_pub.publish(String(data=result))

    def done_cb(self, msg):
        if self.finished or not msg.data:
            return

        self.finished = True
        rospy.loginfo("[FlowManager] Grasp SUCCESS → tuck arm")

        self._tuck_arm()

        rospy.loginfo("[FlowManager] Publishing flow_result=SUCCESS")
        self.publish_flow_result(self.flow_success_msg)

        rospy.loginfo("[FlowManager] Task finished → shutdown launch")
        rospy.signal_shutdown("Grasp success")

    def fail_cb(self, msg):
        if self.finished:
            return

        self.finished = True
        rospy.logerr("[FlowManager] Grasp FAILED: %s", msg.data)

        rospy.loginfo("[FlowManager] Open gripper ...")
        self._open_gripper()
        rospy.sleep(0.8)

        self._tuck_arm()

        rospy.loginfo("[FlowManager] Publishing flow_result=FAILED")
        self.publish_flow_result(self.flow_failed_msg)

        rospy.signal_shutdown("Grasp failed")

    def _tuck_arm(self):
        try:
            subprocess.call(["rosrun", "grasp", "tuck_arm.py"])
            rospy.loginfo("[FlowManager] Tuck arm finished")
        except Exception as e:
            rospy.logerr("[FlowManager] Tuck arm failed: %s", str(e))

    def _open_gripper(self):
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
    FlowManager()

