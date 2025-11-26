#pragma once

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/transforms.h>
#include <tf/transform_datatypes.h>

typedef pcl::PointXYZL PointT1;

class Grasper
{
public:
    Grasper(ros::NodeHandle& nh);
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

private:
    bool grasp(const sensor_msgs::PointCloud2ConstPtr& msg, int target_label);
    geometry_msgs::PoseStamped computeGraspPose(const std::vector<PointT1>& cluster_points);
    bool moveArmToTarget(const geometry_msgs::PoseStamped& target);
    void closeGripper();

    moveit::planning_interface::MoveGroupInterface move_group_;
    ros::Publisher gripper_pub_;
    int target_label_;
    double z_offset_;
    double pregrasp_offset_;
    double lift_offset_;
    double grasp_rpy_[3];
};
