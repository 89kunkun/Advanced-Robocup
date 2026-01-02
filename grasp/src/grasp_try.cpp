# include <grasp/grasp_try.h>

TiagoJointCommander::TiagoJointCommander(ros::NodeHandle& nh)
    : nh_(nh)
{
    arm_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>("/arm_controller/command", 1);
    head_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>("/head_controller/command", 1);
    gripper_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>("/gripper_controller/command", 1);
    torso_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>("/torso_controller/command", 1);
    ros::Duration(1.0).sleep();
}

void TiagoJointCommander::moveArm()
{
    std::vector<double> arm_positions;
    if (!nh_.getParam("arm_positions", arm_positions) || arm_positions.size() != 7) {
        ROS_WARN("使用默认arm_positions");
        arm_positions = {0.4, -1.17, -1.9, 2.3, -1.3, -0.45, 0.0};
    }

    trajectory_msgs::JointTrajectory traj;
    traj.joint_names = {"arm_1_joint", "arm_2_joint", "arm_3_joint",
                        "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"};

    trajectory_msgs::JointTrajectoryPoint pt;
    pt.positions = arm_positions;
    pt.velocities = std::vector<double>(7, 0.2);
    pt.time_from_start = ros::Duration(3.0);
    traj.points.push_back(pt);

    arm_pub_.publish(traj);
    ROS_INFO("Arm command sent.");
}

void TiagoJointCommander::moveHead()
{
    std::vector<double> head_positions;
    if (!nh_.getParam("head_positions", head_positions) || head_positions.size() != 2) {
        head_positions = {0.0, 0.0};
    }

    trajectory_msgs::JointTrajectory traj;
    traj.joint_names = {"head_1_joint", "head_2_joint"};
    trajectory_msgs::JointTrajectoryPoint pt;
    pt.positions = head_positions;
    pt.time_from_start = ros::Duration(1.0);
    traj.points.push_back(pt);

    head_pub_.publish(traj);
    ROS_INFO("Head command sent.");
}

void TiagoJointCommander::moveGripper()
{
    std::vector<double> gripper_positions;
    if (!nh_.getParam("gripper_positions", gripper_positions) || gripper_positions.size() != 2) {
        gripper_positions = {0.0, 0.0};
    }

    trajectory_msgs::JointTrajectory traj;
    traj.joint_names = {"gripper_left_finger_joint", "gripper_right_finger_joint"};
    trajectory_msgs::JointTrajectoryPoint pt;
    pt.positions = gripper_positions;
    pt.time_from_start = ros::Duration(1.0);
    traj.points.push_back(pt);

    gripper_pub_.publish(traj);
    ROS_INFO("Gripper command sent.");
}

void TiagoJointCommander::moveTorso()
{
    double torso_position;
    if (!nh_.getParam("torso_position", torso_position)) {
        torso_position = 0.2;
    }

    trajectory_msgs::JointTrajectory traj;
    traj.joint_names = {"torso_lift_joint"};
    trajectory_msgs::JointTrajectoryPoint pt;
    pt.positions = {torso_position};
    pt.time_from_start = ros::Duration(1.0);
    traj.points.push_back(pt);

    torso_pub_.publish(traj);
    ROS_INFO_STREAM("Torso move command sent to position: " << pt.positions[0]);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "grasp_try");
    ros::NodeHandle nh;
    TiagoJointCommander commander(nh);

    commander.moveHead();
    ros::Duration(0.8).sleep();

    commander.moveArm();
    ros::Duration(2.0).sleep();

    commander.moveGripper();
    ros::Duration(0.8).sleep();

    commander.moveTorso();
    ros::Duration(0.8).sleep();

    ros::spinOnce();
    ros::Duration(1.0).sleep();
    return 0;
}