import rospy
import smach
import smach_ros
import actionlib
import actionlib_msgs
import time
import math
import os
import subprocess
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Bool
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from pal_interaction_msgs.msg import TtsActionGoal
from teleop_tools_msgs.msg import IncrementActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf.transformations as tf
from final_msg_srv.srv import DesiredObject

launched_processes = []

class InitState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'])

    def execute(self, userdata):
        rospy.loginfo("INIT: Preparing system...")
        rospy.sleep(2)
        return 'succeeded'
    
class RunLocalizationTestLaunch(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo("[Localization] Running test.launch...")
        try:
            subprocess.Popen(['roslaunch', 'localization', 'test.launch'])
            rospy.sleep(3)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[Localization] Failed to run launch: {e}")
            return 'aborted'

class WaitForAMCL(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo("Waiting for /amcl_pose to publish...")
        try:
            rospy.wait_for_message("/amcl_pose", PoseWithCovarianceStamped, timeout=10)
            rospy.loginfo("AMCL is publishing. Localization in progress...")
        except rospy.ROSException:
            rospy.logwarn("Timeout: AMCL did not publish, but continuing...")
        return 'succeeded'

class WaitForAMCLConverge(smach.State):
    def __init__(self, xy_thresh=0.05, yaw_thresh=0.1, stable_times=10):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.xy_thresh = xy_thresh
        self.yaw_thresh = yaw_thresh
        self.stable_times = stable_times
        self.current_cov = None
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.acml_cb)

    def acml_cb(self, msg):
        self.current_cov = msg.pose.covariance

    def execute(self, userdata):
        rospy.loginfo("Waiting for AMCL to converge...")
        stable_count = 0
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.current_cov is not None:
                xy_cov = self.current_cov[0] + self.current_cov[7]
                yaw_cov = self.current_cov[35]
                rospy.loginfo("acml cov: xy=%.4f, yaw=%.4f, stable_count=%d", xy_cov, yaw_cov, stable_count)
                if xy_cov < self.xy_thresh and yaw_cov < self.yaw_thresh:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= self.stable_times:
                    rospy.loginfo("AMCL has converged.")
                    return 'succeeded'
            rate.sleep()
        return 'aborted'

class WaitState(smach.State):
    def __init__(self, wait_time):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.wait_time = wait_time

    def execute(self, userdata):
        rospy.loginfo(f"Waiting for {self.wait_time} seconds...")
        rospy.sleep(self.wait_time)
        return 'succeeded'

class WaitForMessage(smach.State): #看发到话题里面的message是不是期待的消息
    def __init__(self, topic, expected_msg):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.topic = topic
        self.expected_msg = expected_msg
        self.received_message = False

    def message_callback(self, msg):
        """ 订阅回调函数，检查是否收到期望的消息 """
        rospy.loginfo(f"Received message: {msg.data}")
        if msg.data == self.expected_msg:
            self.received_message = True

    def execute(self, userdata):
        """ 订阅指定话题，并等待特定消息 """
        rospy.loginfo(f"Waiting for message on topic: {self.topic} with expected content: '{self.expected_msg}'")

        self.received_message = False
        rospy.Subscriber(self.topic, String, self.message_callback)

        rate = rospy.Rate(10)  # 10Hz
        while not self.received_message and not rospy.is_shutdown():
            rate.sleep()

        rospy.loginfo(f"Received expected message '{self.expected_msg}', transitioning to the next state.")
        return 'succeeded'

class PublishNavGoal(smach.State): #基座导航到指定位置
    def __init__(self, x, y, yaw):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.x = x
        self.y = y
        self.yaw = yaw
        # self.publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        # rospy.sleep(1)

    def execute(self, userdata):
        rospy.loginfo(f"Publishing 2D Nav Goal to ({self.x}, {self.y}) with yaw {self.yaw} degrees...")
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.x
        goal.target_pose.pose.position.y = self.y
        # goal.pose.position.z = 0.0

        quaternion = tf.quaternion_from_euler(0, 0, self.yaw)  # 确保yaw是弧度
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        client.send_goal(goal)
        client.wait_for_result()
        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("2D Nav Goal published successfully!")
            return 'succeeded'
        else:
            rospy.logwarn("Failed to publish navigation goal")
            return 'aborted'
        
class RunLocalizationTestLaunch(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo("[Localization] Running test.launch...")
        try:
            subprocess.Popen(['roslaunch', 'localization', 'test.launch'])
            rospy.sleep(3)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[Localization] Failed to run launch: {e}")
            return 'aborted'
        
class NavigationToTable(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo("[Navigation] Running navigation.launch...")
        try:
            subprocess.Popen(['roslaunch', 'navigation', 'navigation.launch'])
            rospy.sleep(3)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"[Navigation] Failed to run launch: {e}")
            return 'aborted'

class SpeakByService(smach.State):
    def __init__(self, text):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.text = text
        self.process = None

    def execute(self, userdata):
        rospy.loginfo(f"Requesting say: {self.text}")
        try:
            script_path = '/home/ikun/ros/workspaces/project/src/robot_speaking/scripts/say.py'
            cmd = ['python3', script_path]
            #cmd = ['rosrun', 'say_something', 'say.py']
            env = os.environ.copy()
            # 非阻塞启动（进程会在后台一直活着）
            self.process = subprocess.Popen(cmd, env=env)
            rospy.loginfo("say.py started as service process.")
            rospy.sleep(2)  # 给 say.py 时间启动

            rospy.wait_for_service('/say_something')
            say = rospy.ServiceProxy('/say_something', DesiredObject)
            say(self.text)
            rospy.loginfo("Say service called successfully.")

            rospy.sleep(5)  # 根据你的TTS说话时长调整
            self.process.terminate()  # 强制回收
            rospy.loginfo("say.py terminated.")

            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"Calling service say_something failed: {e}")
            return 'aborted'
        
    # def start_say_server():
    #     script_path = '/home/ikun/ros/workspaces/project/src/robot_speaking/scripts/say.py'
    #     cmd = ['python3', script_path]
    #     env = os.environ.copy()
    #     process = subprocess.Popen(cmd, env=env)
    #     time.sleep(2)  # 等待服务端ready
    #     return process

class WaitForGesture(smach.State):
    def __init__(self,timeout=20):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'],output_keys=['gesture_result'])
        self.hand_sign = None
        self.process = None
        self.timeout = timeout

    def execute(self, userdata):
        rospy.loginfo("Launching hand_sign.launch...")

        # self.process = subprocess.Popen(["roslaunch", "my_cam", "my_cam.launch"])
        # 启动手势识别launch文件
        self.process = subprocess.Popen(["roslaunch", "ros_hand_gesture_recognition", "hand_sign.launch"])

        self.hand_sign = None
        sub = rospy.Subscriber('/gesture/hand_sign', String, self.hand_sign_callback)

        # 等待手势识别
        # rate = rospy.Rate(10)
        # timeout = rospy.Time.now() + rospy.Duration(20)

        # while not rospy.is_shutdown() and self.hand_sign is None and rospy.Time.now() < timeout:
        #     rate.sleep()

        # if self.process and self.process.poll() is None:
        #     self.process.terminate()
        #     self.process.wait()
        #     rospy.loginfo("Gesture recognition stopped.")
         # 等待目标消息或超时
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.hand_sign is None:
            if (rospy.Time.now() - start_time).to_sec() > self.timeout:
                rospy.logwarn("WaitForGesture timeout.")
                break
            rate.sleep()

        # 收到目标消息或超时后，kill外部进程
        if self.process.poll() is None:  # 检查进程是否还活着
            self.process.terminate()
            try:
                self.process.wait(timeout=2)  # 等待进程优雅退出
            except subprocess.TimeoutExpired:
                self.process.kill()          # 若还不退出，强杀
            rospy.loginfo("hand_sign_recognition.py terminated.")

        sub.unregister()  # 取消订阅，防止callback泄漏

        if self.hand_sign in ["Left one", "Right one"]:
            rospy.loginfo(f"Gesture recognized: {self.hand_sign}")
            userdata.gesture_result = self.hand_sign  # 写入 userdata
            return 'succeeded'
        else:
            return 'aborted'

    def hand_sign_callback(self, msg):
        if msg.data in ["Left one", "Right one"]:
            self.hand_sign = msg.data

class FOLLOW(smach.State):
    def __init__(self, launch_file, timeout=30, process_list=None):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.launch_file = launch_file
        self.stop_signal_received = False
        self.no_movement_count = 0
        self.threshold = 0.1
        self.last_position = None # 初始化
        self.timeout = timeout  # 超时时间，单位为秒
        self.process = None
        self.process_list = process_list

    def execute(self, userdata):
        rospy.loginfo(f"Launching: {self.launch_file}")
        try:
            self.process = subprocess.Popen(["roslaunch", *self.launch_file.split()])
            rospy.loginfo("Launch process started.")
            if self.process_list is not None:
                self.process_list.append(self.process)
        except Exception as e:
            rospy.logerr(f"Failed to launch {self.launch_file}: {e}")
            return 'aborted'

        # 订阅目标位置
        rospy.Subscriber('/person_pose', PoseStamped, self.position_callback)

        # 监听外部信号
        rospy.Subscriber('/test_topic', String, self.stop_callback)

        start_time = time.time()
        rate = rospy.Rate(10)  # 10 Hz

        while not self.stop_signal_received and not rospy.is_shutdown():
            # 超时终止
            if time.time() - start_time > self.timeout:
                rospy.logwarn("FOLLOW state timed out, stopping follow.")
                self.stop_signal_received = True
                break
            rate.sleep()

        # 结束子进程
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                rospy.loginfo("Launch process terminated.")
            except Exception as e:
                rospy.logerr(f"Failed to terminate launch process: {e}")    

        return 'succeeded' if self.stop_signal_received else 'aborted'

    def position_callback(self, msg):
        current_position = (msg.pose.position.x, msg.pose.position.y)
        if self.last_position is not None:
            distance = ((current_position[0] - self.last_position[0])**2 + (current_position[1] - self.last_position[1])**2)**0.5
            if distance < self.threshold:
                self.no_movement_count += 1
            else:
                self.no_movement_count = 0
        else:
            self.no_movement_count = 0  # 首次收到消息，重置

        self.last_position = current_position  # ★★ 每次都赋值
        
        if self.no_movement_count >= 10:
            rospy.loginfo("Target stopped. Publishing stop_follow")
            stop_pub = rospy.Publisher('/stop_topic', String, queue_size=1)
            stop_pub.publish("stop_follow")
            self.stop_signal_received = True

    def stop_callback(self, msg):
        if msg.data == "stop_follow":
                rospy.loginfo("Received stop signal, stopping follow.")
                self.stop_signal_received = True
        else:
                rospy.loginfo("Unknown message received")
            

class ExecuteCommand(smach.State): 
    def __init__(self, cmd, wait_sec=0, process_list=None):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.cmd = cmd
        self.wait_sec = wait_sec  # 等待时间，单位为秒
        self.process_list = process_list

    def execute(self, userdata):
        rospy.loginfo(f"Executing command: {self.cmd}")
        try:
            env = os.environ.copy()
            process = subprocess.Popen(
                self.cmd,
                shell=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if self.process_list is not None:
                self.process_list.append(process)
            if self.wait_sec > 0:
                rospy.loginfo(f"Waiting for {self.wait_sec} seconds after executing command...")
                rospy.sleep(self.wait_sec)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"Command failed: {e}")
            return 'aborted'

class GraspTry(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo('Executing object try grasp...')
        result = subprocess.call(["roslaunch", "grasp", "grasp_try.launch"])
        if result == 0:
            rospy.loginfo("Launch succeeded")
        else:
            rospy.logerr("Launch failed")
        return 'succeeded' if result == 0 else 'aborted'
    
class ObjectDetection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo('Executing object detection...')
        try:
            process = subprocess.Popen(["roslaunch", "object_detection_world", "object_detection.launch"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            launched_processes.append(process)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"Failed to launch object_detection: {e}")
            return 'aborted'

class PlaneSegmentation(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo('Executing plane_segmentation...')
        try:
            process = subprocess.Popen(["roslaunch", "plane_segmentation", "plane_segmentation_tiago.launch"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            launched_processes.append(process)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"Failed to launch object_detection: {e}")
            return 'aborted'
    
class ObjectLabeling(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo('Executing object labeling...')
        try:
            process = subprocess.Popen(["roslaunch", "object_labeling", "object_labeling.launch"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            launched_processes.append(process)
            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"Failed to launch object_detection: {e}")
            return 'aborted'
    
class Grasp(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=['gesture_result'])

    def execute(self, userdata):
        hand_sign = userdata.gesture_result
        rospy.loginfo(f"Executing grasp based on hand sign: {hand_sign}")

        if hand_sign == "Left one":
            result = subprocess.call(["roslaunch", "grasp", "grasp_left.launch"])
        elif hand_sign == "Right one":
            result = subprocess.call(["roslaunch", "grasp", "grasp_right.launch"])
        else:
            rospy.logerr("Unknown hand sign, cannot execute grasp.")
            return 'aborted'
        
        rospy.loginfo("Shutting down previous processes...")
        for proc in launched_processes:
            proc.terminate()
            proc.wait()
        rospy.loginfo("All background nodes shut down.")

        return 'succeeded' if result == 0 else 'aborted'
    
class Rotate(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo('Executing rotation...')
        result = subprocess.call(["roslaunch", "move", "rotate.launch"])
        if result == 0:
            rospy.loginfo("Launch succeeded")
        else:
            rospy.logerr("Launch failed")
        return 'succeeded' if result == 0 else 'aborted'
    
class TuckArm(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

    def execute(self, userdata):
        rospy.loginfo('Executing tuck arm...')
        result = subprocess.call(["rosrun", "grasp", "tuck_arm.py"])
        if result == 0:
            rospy.loginfo("Launch succeeded")
        else:
            rospy.logerr("Launch failed")
        return 'succeeded' if result == 0 else 'aborted'
    
class CleanUpState(smach.State):
    def __init__(self, process_list):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.process_list = process_list

    def execute(self, userdata):
        rospy.loginfo("Cleaning up all launched processes...")
        for proc in self.process_list:
            if proc.poll() is None:
                rospy.loginfo(f"Terminating process {proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception as e:
                    rospy.logwarn(f"Failed to terminate process: {e}")
        rospy.loginfo("All processes cleaned up.")
        return 'succeeded'

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

def main():
    rospy.init_node('carry_task_manager_test', log_level=rospy.INFO)
    rospy.loginfo("Task manager initialized.")


    sm = smach.StateMachine(outcomes=['succeeded', 'aborted'], output_keys=['gesture_result'])

    with sm:

        # 1. part
        
        ############################### go to the table ###################################
        smach.StateMachine.add('INIT', InitState(), transitions={'succeeded': 'RUN_LOCALIZATION'})

        smach.StateMachine.add('RUN_LOCALIZATION', RunLocalizationTestLaunch(),
                               transitions={'succeeded': 'WAIT_FOR_AMCL', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_FOR_AMCL', WaitForAMCLConverge(xy_thresh=0.08, yaw_thresh=0.08, stable_times=10), 
                               transitions={'succeeded': 'Navigation','aborted': 'aborted'}) 
        
        smach.StateMachine.add("GO_TO_TABLE", 
                               NavigationToTable(),
                               transitions={'succeeded': 'SPEAK_ASK', 'aborted': 'aborted'})
        
        smach.StateMachine.add("Navigation", 
                               PublishNavGoal(x=-0.155 ,y=-0.003 , yaw=-0.15),
                               transitions={'succeeded': 'SPEAK_ASK', 'aborted': 'aborted'})    

   
        # 2. part

        #######################################################################################
        
        smach.StateMachine.add('SPEAK_ASK', 
                               SpeakByService(text="Hello, nice to meet you. Please let me know which bottle I need to carry."),
                               transitions={'succeeded': 'WAIT_FOR_GESTURE', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_FOR_GESTURE',
                               WaitForGesture(),
                               transitions={'succeeded': 'START_TO_GRASP'})
        ############################### go to the table ###################################


        # 3. part

        ############################### grasp ###################################
        smach.StateMachine.add('START_TO_GRASP', 
                               GraspTry(),
                               transitions={'succeeded': 'WAIT_0', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_0', 
                               WaitState(5),
                               transitions={'succeeded': 'OBJECT_DETECTION'})
    
        smach.StateMachine.add('OBJECT_DETECTION', 
                               ObjectDetection(),
                               transitions={'succeeded': 'WAIT_1', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_1', 
                               WaitState(5),
                               transitions={'succeeded': 'PLANE_SEGMENTATION'})
        
        smach.StateMachine.add('PLANE_SEGMENTATION', 
                               PlaneSegmentation(),
                               transitions={'succeeded': 'WAIT_2', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_2', 
                               WaitState(10),
                               transitions={'succeeded': 'OBJECT_LABELING'})
    
        smach.StateMachine.add('OBJECT_LABELING', 
                               ObjectLabeling(),
                               transitions={'succeeded': 'WAIT_3', 'aborted': 'aborted'})
            
        smach.StateMachine.add('WAIT_3', 
                               WaitState(5),
                               transitions={'succeeded': 'GRASP'})
        
        smach.StateMachine.add('GRASP',
                               Grasp(),
                               transitions={'succeeded': 'TUCKARM', 'aborted': 'aborted'})
        
        smach.StateMachine.add('TUCKARM',
                               TuckArm(),
                               transitions={'succeeded': 'TURN', 'aborted': 'aborted'})
        
        smach.StateMachine.add('TURN', 
                               Rotate(),
                               transitions={'succeeded': 'SPEAK_READY', 'aborted': 'aborted'}) 
        
        smach.StateMachine.add('SPEAK_READY', 
                               SpeakByService(text="Grasp completed, I an ready to follow you"),
                               transitions={'succeeded': 'YOLO', 'aborted': 'aborted'})
        
        # ################################ grasp ###################################
        
        
        # 4. part

        # ################################ person following ###################################

        smach.StateMachine.add('YOLO',
                                ExecuteCommand(cmd="roslaunch object_detection_world object_detection.launch", wait_sec=5, process_list=launched_processes),
                                transitions={'succeeded': 'FOLLOW', 'aborted': 'aborted'})

        smach.StateMachine.add('FOLLOW',
                                ExecuteCommand(cmd="roslaunch preprocessing_pointcloud preprocessing_pointcloud.launch", wait_sec=10, process_list=launched_processes),
                                transitions={'succeeded': 'START_LABELING_PERSON', 'aborted': 'aborted'})

        smach.StateMachine.add('START_LABELING_PERSON',
                                ExecuteCommand(cmd="roslaunch object_labeling_following object_labeling_following.launch", wait_sec=10, process_list=launched_processes),
                                transitions={'succeeded': 'PERSON_FOLLOW', 'aborted': 'aborted'})

        smach.StateMachine.add('PERSON_FOLLOW',
                                FOLLOW('navigation_to_person navigation_to_person.launch', process_list=launched_processes),
                                transitions={'succeeded': 'CLEANUP', 'aborted': 'aborted'})
        
        smach.StateMachine.add('CLEANUP',
                                CleanUpState(launched_processes),
                                transitions={'succeeded': 'SPEAK_FINAL'})

        # ################################ person following ###################################
        
        
        
        # ################################ give the cup ###################################

        smach.StateMachine.add('SPEAK_FINAL', 
                               SpeakByService(text="Following completed, I am ready to give the bottle."),
                               transitions={'succeeded': 'GIVE_THE_CUP', 'aborted': 'aborted'})

        smach.StateMachine.add('GIVE_THE_CUP', 
                               ExecuteCommand(cmd="roslaunch move give.launch"),
                               transitions={'succeeded': 'WAIT', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT', 
                               WaitState(10),
                               transitions={'succeeded': 'TUCKARM_1'})
        
        smach.StateMachine.add('TUCKARM_1',
                               TuckArm(),
                               transitions={'succeeded': 'FINISH', 'aborted': 'aborted'})
        
        smach.StateMachine.add('FINISH', 
                               SpeakByService(text="Task completed, I have given the bottle."),
                               transitions={'succeeded': 'succeeded', 'aborted': 'aborted'})
        ############################### give the cup ###################################



    sis = smach_ros.IntrospectionServer('carry_task_manager', sm, '/SM_ROOT')
    sis.start()

    outcome = sm.execute()
    rospy.loginfo('Carry Task Manager Outcome: %s', outcome)

    sis.stop()


if __name__ == '__main__':
    main()

    
    
    
    