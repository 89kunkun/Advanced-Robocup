#!/usr/bin/env python

import rospy
from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from std_msgs.msg import String

def callback_say(data):
    text = data.data  # 从话题消息中提取字符串内容
    rospy.loginfo("I'll say: " + text)

    # 创建到 TTS action server (/tts) 的客户端
    client = SimpleActionClient('/tts', TtsAction)
    client.wait_for_server()  # 等待 TTS server 上线

    # 创建一个说话的 goal
    goal = TtsGoal()
    goal.rawtext.text = text
    goal.rawtext.lang_id = "en_GB"  # 英国英语

    # 发送 goal 并等待完成
    client.send_goal_and_wait(goal)

def main():
    rospy.init_node('say_something_subscriber')  # 新节点名，可自定义

    # 订阅话题 /the_word_to_say，消息类型 String，回调函数 callback_say
    rospy.Subscriber('/the_word_to_say', String, callback_say)

    rospy.spin()  # 循环等待消息

if __name__ == '__main__':
    main()

