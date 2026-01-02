#ifndef GRASP_H
#define GRASP_H

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/transforms.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

// MoveIt
#include <moveit/move_group_interface/move_group_interface.h>

#include <tf/transform_datatypes.h>

// 避障
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/CollisionObject.h>
#include <shape_msgs/SolidPrimitive.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/common.h>


// PCL point type withh label
typedef pcl::PointXYZL PointT1;
typedef pcl::PointCloud<PointT1> PointCloudL;

class Grasper
{
public:
        ros::Subscriber table_cloud_sub_once_;
        Grasper(ros::NodeHandle& nh);
        void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

        void tableCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
// Grasping function interface

// MoveIt-based arm control (assumed to exist elsewhere)
private:
        
        bool has_target_pose_ = false;
        geometry_msgs::PoseStamped cached_grasp_pose_;
        geometry_msgs::PoseStamped cached_pregrasp_pose_;

        bool moveArmToTarget(const geometry_msgs::PoseStamped& target);
        void closeGripper();
        void openGripper();
        bool grasp(const sensor_msgs::PointCloud2ConstPtr& msg, int traget_label);
        // bool moveArmToTargetCartesian(const geometry_msgs::PoseStamped& target);
        // bool moveEndEffectorStraightX(double distance);
        bool moveEndEffectorStraightDirection(double dx, double dy, double dz, double distance);

        ros::Publisher gripper_pub_;
        moveit::planning_interface::MoveGroupInterface move_group_;

        int target_label_;
        double x_offset_;
        double y_offset_; 
        double z_offset_; 
        double grasp_rpy_[3];
        double pregrasp_offset_x;
        double pregrasp_offset_y;
        double pregrasp_offset_z;

        sensor_msgs::PointCloud2ConstPtr cached_cloud_;
        bool got_cloud_ = false;

        geometry_msgs::Point table_center_;
        bool has_table_center_ = false;

        //
        moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
        ros::Subscriber table_cloud_sub_;

        
        bool table_obstacle_initialized_ = false;
};

#endif // GRASP_H
