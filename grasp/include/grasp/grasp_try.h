#pragma once

#include <ros/ros.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

class TiagoJointCommander
{
public:
    TiagoJointCommander(ros::NodeHandle& nh);

    void moveArm();
    void moveHead();
    void moveGripper();
    void moveTorso();

private:
    ros::NodeHandle& nh_;
    ros::Publisher arm_pub_;
    ros::Publisher head_pub_;
    ros::Publisher gripper_pub_;
    ros::Publisher torso_pub_;
};