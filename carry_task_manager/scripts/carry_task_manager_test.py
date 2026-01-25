import rospy
import rospkg
import smach
import smach_ros
import actionlib
import actionlib_msgs
import time
import math
import os
import subprocess
import roslaunch
import roslaunch.rlutil
import roslaunch.parent
import tf.transformations as tf

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Bool
from std_msgs.msg import String
from actionlib_msgs.msg import GoalStatusArray
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from pal_interaction_msgs.msg import TtsActionGoal
from teleop_tools_msgs.msg import IncrementActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from final_msg_srv.srv import DesiredObject

launched_processes = []

class InitState(smach.State):
    def __init__(self):
        # Initialize the SMACH state
        smach.State.__init__(self, outcomes=['succeeded'])

    def execute(self, userdata):
        # Log state entry for debugging and visualization
        rospy.loginfo("INIT: Preparing system...")
        # Simulate initialization delay 
        rospy.sleep(2)
        return 'succeeded'  # Indicate successful completion of initialization
    
class Runlocalizationlaunch(smach.State):
    def __init__(self):
        # Initialize the SMACH state with two outcomes: 'successed' and 'aborted'
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.launch = None # Will hold the launch process instance

    def execute(self, userdata):
        # Log the start of this state
        rospy.loginfo("[Localization] Running localization.launch...")
        # Define the absolute path to the launch file
        launch_file = "/home/ikun/ros/workspaces/project/src/localization/launch/localization.launch"
     
        # Check if the launch file exists before attempting to launch
        if not os.path.isfile(launch_file):
            rospy.logerr(f"[Localization] Launch file {launch_file} does not exist.")
            return 'aborted'
        
        try:
            # Generate UUID and configure logging for roslaunch
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            
            # Create a launch parent for the specified launch file
            self.launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
            
            # Start the launch process
            self.launch.start()
            rospy.loginfo("[Localization] localization.launch started.")
            
            # Optional sleep to give time for nodes to initialize
            rospy.sleep(3)  
            return 'succeeded'
        except Exception as e:
            # Log any errors that occur during the launch attempt
            rospy.logerr(f"[Localization] Failed to run launch: {e}")
            return 'aborted'
        
    def request_preempt(self):
        # Gracefully shut down the launch process if the state is preempted
        if self.launch:
            self.launch.shutdown()  

class WaitForMessage(smach.State): 
    def __init__(self, topic, expected_msg,msg_type):
        # Initialize the SMACH state with outcomes: 'succeeded' and 'aborted'
        smach.State.__init__(self, outcomes=['succeeded','aborted'])
        self.topic = topic  # Topic to subscribe to
        self.expected_msg = expected_msg  # The exact message content to wait for
        self.msg_type = msg_type  # Type of the expected ROS message
        self.received_message = False  # Flag indicating if the expected message has been received

    def message_callback(self, msg):
        """Callback function for incoming messages; checks if message matches the expected content"""
        rospy.loginfo(f"Received message: {msg.data}")
        if msg.data == self.expected_msg:
            self.received_message = True

    def execute(self, userdata):
        """Subscribe to the given topic and wait for the expected message"""
        rospy.loginfo(f"Waiting for message on topic: {self.topic} with expected content: '{self.expected_msg}'")
        self.received_message = False
        
        # Subscribe to the topic
        sub = rospy.Subscriber(self.topic, self.msg_type, self.message_callback)
        
        timeout_sec = 120  # Set timeout duration (in seconds)
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  # Loop at 10 Hz

        while not self.received_message and not rospy.is_shutdown():
            elapsed =(rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout_sec:
                rospy.logwarn(f"[WaitforMessage]Timeout after {timeout_sec} seconds, no expected message received.")
                sub.unregister()
                return 'aborted'    # Timeout occurred before receiving the expected message
            rate.sleep()

        sub.unregister()

        rospy.loginfo(f"[WaitForMessage]Received expected message, transitioning to the next state.")
        return 'succeeded'

class WaitState(smach.State):
    def __init__(self, wait_time):
        # Initialize the SMACH state with a single outcome: 'succeeded'
        smach.State.__init__(self, outcomes=['succeeded'])
        self.wait_time = wait_time  # Duration to wait in seconds

    def execute(self, userdata):
        # Duration to wait in seconds
        rospy.loginfo(f"Waiting for {self.wait_time} seconds...")
        # Sleep for the specified duration
        rospy.sleep(self.wait_time)
        # Return 'succeeded' once the wait is over
        return 'succeeded'    

class Runnavigationlaunch(smach.State):
    def __init__(self):
        # Initialize SMACH state with outcomes 'succeeded' and 'aborted'
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.launch = None # Will hold the launch process object

    def execute(self, userdata):
        # Inform that navigation launch is about to be executed
        rospy.loginfo("[Navigation] Running navigation.launch...")
        
        # Define the absolute path to the launch file
        launch_file = "/home/ikun/ros/workspaces/project/src/navigation/launch/navigation.launch"
        
        # Check if the launch file exists before launching
        if not os.path.isfile(launch_file): 
            rospy.logerr(f"[Navigation] Launch file {launch_file} does not exist.")
            return 'aborted'
        
        try:
            # Generate a unique identifier for roslaunch session
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid) 
            
            # Initialize the launch process using the launch file   
            self.launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
            self.launch.start()
            rospy.loginfo("[Navigation] navigation.launch started.")
            
            try:
                 # Wait for the move_base status topic to be published to ensure it's running
                rospy.wait_for_message("move_base/status", GoalStatusArray, timeout=10)
                rospy.loginfo("Move base is up and running.")
                return 'succeeded'
            except rospy.ROSException:
                # Timeout while waiting for move_base to be ready
                rospy.logwarn("Move base is not ready, aborting navigation.")
                return 'aborted'
            
        except Exception as e:
            # Catch any exception related to the launch process and log the error
            rospy.logerr(f"[Navigation] Failed to run launch: {e}")
            return 'aborted'
        
    def request_preempt(self):
        # Gracefully shut down the launch process when preempted
        if self.launch:
            self.launch.shutdown()

class SpeakByService(smach.State):
    def __init__(self, text):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.text = text
        self.process = None

    def execute(self, userdata):
        rospy.loginfo(f"Requesting say: {self.text}")
        try:
            # get the path to say.py
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('robot_speaking') 
            script_path = os.path.join(pkg_path, 'scripts', 'say.py')
            cmd = ['python3', script_path]
            env = os.environ.copy()
            # non-blocking launch (the process will persist running in the background)
            self.process = subprocess.Popen(cmd, env=env)
            rospy.loginfo("say.py started as service process.")
            rospy.sleep(2) # Give say.py time to start

            rospy.wait_for_service('/say_something')
            say = rospy.ServiceProxy('/say_something', DesiredObject)
            say(self.text)
            rospy.loginfo("Say service called successfully.")

            rospy.sleep(5) 
            self.process.send_signal(subprocess.signal.SIGINT)
            self.process.wait()
            rospy.loginfo("say.py terminated.")

            return 'succeeded'
        except Exception as e:
            rospy.logerr(f"Calling service say_something failed: {e}")
            return 'aborted'

class WaitForGesture(smach.State):
    def __init__(self,timeout=20):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'],output_keys=['gesture_result'])
        self.hand_sign = None
        self.process = None
        self.timeout = timeout

    def execute(self, userdata):
        rospy.loginfo("Launching hand_sign.launch...")

        # start the hand gesture recognition launch file
        self.process = subprocess.Popen(["roslaunch", "ros_hand_gesture_recognition", "hand_sign.launch"])

        self.hand_sign = None
        sub = rospy.Subscriber('/gesture/hand_sign', String, self.hand_sign_callback)

         # wait for hand_sign message or timeout
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.hand_sign is None:
            if (rospy.Time.now() - start_time).to_sec() > self.timeout:
                rospy.logwarn("WaitForGesture timeout.")
                break
            rate.sleep()

        # terminate the process if it's still running or timeout
        if self.process.poll() is None: 
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()          # force kill if it doesn't terminate
            rospy.loginfo("hand_sign_recognition.py terminated.")

        sub.unregister()  # unregister the subscriber

        if self.hand_sign in ["Left one", "Right one"]:
            rospy.loginfo(f"Gesture recognized: {self.hand_sign}")
            userdata.gesture_result = self.hand_sign  # write into userdata for grasp
            return 'succeeded'
        else:
            return 'aborted'

    def hand_sign_callback(self, msg):
        if msg.data in ["Left one", "Right one"]:
            self.hand_sign = msg.data

class FOLLOW(smach.State):
    def __init__(self, launch_file):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.launch_file = launch_file
        self.stop_signal_received = False

    def execute(self, userdata):
        rospy.loginfo(f"Launching: {self.launch_file}")

        self.process = subprocess.Popen(['roslaunch', *self.launch_file.split()])
        rospy.loginfo(f"Launch process started with PID {self.process.pid}")


        rospy.Subscriber('/test_topic', String, self.stop_callback)


        rate = rospy.Rate(10)  # 10 Hz
        while not self.stop_signal_received and not rospy.is_shutdown():
            rate.sleep()

        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            rospy.loginfo("Launch process terminated.")

        if self.stop_signal_received:
            return 'succeeded'
        else:
            return 'aborted'

    def stop_callback(self, msg):
        if msg.data == 'stop_follow':
            rospy.loginfo("Stop signal received.")
            self.stop_signal_received = True

class ExecuteCommand(smach.State): 
    def __init__(self, cmd):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.cmd = cmd

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
            rospy.sleep(3)
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
def main():
    rospy.init_node('carry_task_manager_test', log_level=rospy.INFO)
    rospy.loginfo("Task manager initialized.")


    sm = smach.StateMachine(outcomes=['succeeded', 'aborted'], output_keys=['gesture_result'])

    with sm:


        # ############################### go to the table ###################################
        smach.StateMachine.add('INIT', InitState(), transitions={'succeeded': 'SPEAK_INIT'})
       
        smach.StateMachine.add('SPEAK_INIT',
                               SpeakByService("System initialization complete. Ready to start localization."),
                               transitions={'succeeded': 'RUN_LOCALIZATION', 'aborted': 'aborted'})
        
        smach.StateMachine.add('RUN_LOCALIZATION', Runlocalizationlaunch(),
                               transitions={'succeeded': 'WAIT_FOR_LOCALIZATION_DONE', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_FOR_LOCALIZATION_DONE', 
                               WaitForMessage('/localization_done', True, Bool),
                               transitions={'succeeded': 'WAIT_Localizationlaunchfinish','aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_Localizationlaunchfinish',
                               Runnavigationlaunch(),
                               transitions={'succeeded': 'WAIT_FOR_NAVIGATION_DONE', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_FOR_NAVIGATION_DONE', 
                               WaitForMessage('/navigation_done', True, Bool),
                               transitions={'succeeded': 'WAIT_Navigationlaunchfinish', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_Navigationlaunchfinish', 
                               WaitState(10),
                               transitions={'succeeded': 'SPEAK_ASK'})
        ############################### go to the table ######################################

        ############################# Hand Gesture Recongnition ##############################
        smach.StateMachine.add('SPEAK_ASK', 
                               SpeakByService(text="Hello, nice to meet you. Please let me know which bottle I need to carry."),
                               transitions={'succeeded': 'WAIT_FOR_GESTURE', 'aborted': 'aborted'})
        
        smach.StateMachine.add('WAIT_FOR_GESTURE',
                               WaitForGesture(),
                               transitions={'succeeded': 'START_TO_GRASP', 'aborted': 'aborted'},)

        ############################# Hand Gesture Recongnition ##############################


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
                               transitions={'succeeded': 'FOLLOW', 'aborted': 'aborted'})

        ################################ grasp ###################################
        
        
        
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
        
        
        
        # ################################ give the bottle ###################################

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
        
        ############################### give the bottle ###################################



    sis = smach_ros.IntrospectionServer('carry_task_manager', sm, '/SM_ROOT')
    sis.start()

    outcome = sm.execute()
    rospy.loginfo('Carry Task Manager Outcome: %s', outcome)

    sis.stop()


if __name__ == '__main__':
    main()

    