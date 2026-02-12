#!/usr/bin/env python
# Use the system default Python interpreter to run this script

import rospy
# ROS Python client library

from actionlib import SimpleActionClient
# Used to create an action client to communicate with an action server

from pal_interaction_msgs.msg import TtsAction, TtsGoal
# Import the TTS (Text-to-Speech) action definition and goal message

from std_msgs.msg import String
# Standard ROS string message type


def callback_say(data):
    """
    Callback function for the subscriber.
    This function is automatically triggered whenever a message
    is received on the /the_word_to_say topic.
    """

    # Extract the string content from the received message
    text = data.data  

    # Print a log message in the terminal
    rospy.loginfo("I'll say: " + text)

    # Create a client that connects to the TTS action server (/tts)
    # First argument: action server name
    # Second argument: action type
    client = SimpleActionClient('/tts', TtsAction)

    # Wait until the TTS action server is available
    # This blocks execution until the server is ready
    client.wait_for_server()  

    # Create a goal object for the TTS action
    goal = TtsGoal()

    # Set the text that should be spoken
    goal.rawtext.text = text

    # Set the language ID (British English in this case)
    # You can change it to other supported languages if needed
    goal.rawtext.lang_id = "en_GB"  

    # Send the goal to the action server and wait for completion
    # This is a synchronous call (blocking)
    client.send_goal_and_wait(goal)


def main():
    """
    Main function.
    Initializes the ROS node and starts the subscriber.
    """

    # Initialize the ROS node with a custom name
    rospy.init_node('say_something_subscriber')  

    # Subscribe to the topic /the_word_to_say
    # Message type: String
    # Callback function: callback_say
    rospy.Subscriber('/the_word_to_say', String, callback_say)

    # Keep the node running and waiting for messages
    # Without this, the program would exit immediately
    rospy.spin()  


# Entry point of the script
# main() runs only if this file is executed directly
if __name__ == '__main__':
    main()
