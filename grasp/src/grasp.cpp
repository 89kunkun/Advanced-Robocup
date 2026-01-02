#include <grasp/grasp.h>

Grasper::Grasper(ros::NodeHandle& nh)
    : move_group_("arm") 
{
    gripper_pub_ = nh.advertise<trajectory_msgs::JointTrajectory>("/gripper_controller/command", 1);

    move_group_.setPoseReferenceFrame("base_link");
    move_group_.setEndEffectorLink("gripper_link");
    move_group_.setGoalTolerance(0.02);
    move_group_.setMaxVelocityScalingFactor(0.3);
    move_group_.setPlanningTime(15.0);
    move_group_.setStartStateToCurrentState();
    move_group_.setPlannerId("RRTConnectkConfigDefault");

    // load parameter
    nh.param("target_label", target_label_, 1);
    
    nh.param("x_offset", x_offset_, 0.0);
    nh.param("y_offset", y_offset_, 0.0);
    nh.param("z_offset", z_offset_, 0.1); // default 10cm

    nh.param("grasp_roll", grasp_rpy_[0], -M_PI/2);
    nh.param("grasp_pitch", grasp_rpy_[1], 0.0);
    nh.param("grasp_yaw", grasp_rpy_[2], -M_PI/2);

    nh.param("pregrasp_offset_x", pregrasp_offset_x, 0.0);
    nh.param("pregrasp_offset_y", pregrasp_offset_y, 0.0);
    nh.param("pregrasp_offset_z", pregrasp_offset_z, 0.0);
}

void Grasper::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    if (!got_cloud_)
    {
        cached_cloud_ = msg;
        got_cloud_ = true;
        grasp(cached_cloud_, target_label_);
    }
}

void Grasper::tableCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    // 1. pointcloud2 -> pcl pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);
    if(cloud->empty()) return;

    // 2. Fit plane
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.015);
    seg.setInputCloud(cloud);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.segment(*inliers, *coefficients);
    if(inliers->indices.empty()) return;

    // 3. Extract inliers min/max
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    pcl::PointCloud<pcl::PointXYZ>::Ptr table_inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*table_inlier_cloud);

    // Compute bounding box
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*table_inlier_cloud, min_pt, max_pt);

    double box_x = max_pt.x - min_pt.x;
    double box_y = max_pt.y - min_pt.y;
    double box_z = 0.4;

    table_center_.x = (min_pt.x + max_pt.x) / 2.0;
    table_center_.y = (min_pt.y + max_pt.y) / 2.0;
    table_center_.z = max_pt.z - box_z + 0.08;    // slightly above surface

    has_table_center_ = true;

    ROS_INFO("Table center at (%.3f, %.3f, %.3f)",
         table_center_.x,
         table_center_.y,
         table_center_.z);

    // 4.CollisionObject
    moveit_msgs::CollisionObject table_obj;
    table_obj.header.frame_id = "base_link";
    table_obj.id = "table";

    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = {box_x, box_y, box_z};

    geometry_msgs::Pose table_pose;
    table_pose.position.x = table_center_.x;
    table_pose.position.y = table_center_.y;
    table_pose.position.z = table_center_.z;
    table_pose.orientation.w = 1.0;

    table_obj.primitives.push_back(primitive);
    table_obj.primitive_poses.push_back(table_pose);
    table_obj.operation = table_obj.ADD;

    // 5. publish to planning scene
    planning_scene_interface_.applyCollisionObject(table_obj);

    table_obstacle_initialized_ = true;
    if (table_cloud_sub_once_) table_cloud_sub_once_.shutdown();
    
    ROS_INFO("[TABLE] Collision object added: size(%.3f, %.3f, %.3f)",
             box_x, box_y, box_z);

}

bool Grasper::grasp(const sensor_msgs::PointCloud2ConstPtr& msg, int target_label)
{
    // Step 1: Convert PointCloud2 to PCL PointCloud<PointXYZL>
    pcl::PointCloud<PointT1>::Ptr cloud(new pcl::PointCloud<PointT1>);
    pcl::fromROSMsg(*msg, *cloud);

    // Step 2: Extract all points with the target label
    std::vector<PointT1> target_points;
    for(const auto& pt : cloud->points)
    {
        if (pt.label == target_label)
            target_points.push_back(pt);
    }

    if (target_points.empty())
    {
        ROS_WARN("No points found with label %d", target_label);
        return false;
    }

    // Step 3: Compute centroid of the target cluster
    Eigen::Vector3d centroid(0, 0, 0);
    for (const auto& pt : target_points)
        centroid += Eigen::Vector3d(pt.x, pt.y, pt.z);
    centroid /= static_cast<double>(target_points.size());

    ROS_INFO("Centroid of target label %d: [%.3f, %.3f, %.3f]",
            target_label, centroid.x(), centroid.y(), centroid.z());

    // Step 4: Construct grasp pose
    // Orientation: PRY -> Quaternion
    tf::Quaternion q;
    q.setRPY(grasp_rpy_[0], grasp_rpy_[1], grasp_rpy_[2]);
    geometry_msgs::Quaternion quat;
    tf::quaternionTFToMsg(q, quat);

    // --- Grasp Pose ---
    geometry_msgs::PoseStamped grasp_pose;
    grasp_pose.header.frame_id = msg->header.frame_id;
    grasp_pose.header.stamp = ros::Time::now();
    grasp_pose.pose.position.x = centroid.x() + x_offset_; 
    grasp_pose.pose.position.y = centroid.y() + y_offset_;
    grasp_pose.pose.position.z = centroid.z() + z_offset_;
    grasp_pose.pose.orientation = quat;
    
    // --- Pregrasp Pose ---
    tf::Matrix3x3 rot_mat(q);
    geometry_msgs::PoseStamped pregrasp_pose = grasp_pose;
    pregrasp_pose.pose.position.x += pregrasp_offset_x;
    pregrasp_pose.pose.position.y += pregrasp_offset_y;
    pregrasp_pose.pose.position.z += pregrasp_offset_z; 
 
    // Step 5: Move the arm
    // Move to pregrasp
    if (!moveArmToTarget(pregrasp_pose)) {
        ROS_ERROR("Failed to move to pregrasp pose.");
        ros::shutdown();
        return false;
    }
    ROS_INFO("Pregrasp done");
    ros::Duration(1.0).sleep();

    // Move to grasp
    if (!moveEndEffectorStraightDirection(1, 0, -1, 0.16))
    {
        ROS_ERROR("Failed to move to grasp pose.");
        ros::shutdown();
        return false;
    }
    ROS_INFO("Grasp pose done");
    ros::Duration(1.0).sleep();

    closeGripper();
    ros::Duration(1.5).sleep();

    if (!moveEndEffectorStraightDirection(0, 0, 1, 0.001))
    {
        ROS_ERROR("Failed to lift.");
        ros::shutdown();
        return false;
    }
    ROS_INFO("Lift done");
    ros::Duration(1.0).sleep();

    // ===================================================
    // =========== MOVE TO TABLE CENTER & PLACE ===========
    // ===================================================

    if (!has_table_center_)
    {
        ROS_ERROR("No table center known, cannot place.");
        ros::shutdown();
        return false;
    }

    ROS_INFO("Moving to table center at (%.3f, %.3f)",
             table_center_.x,
             table_center_.y);

    // -------- Move horizontally above table center ----------
    double current_z = move_group_.getCurrentPose().pose.position.z;

    geometry_msgs::PoseStamped place_above;
    place_above.header.frame_id = "base_link";
    place_above.pose.position.x = table_center_.x - 0.1;
    place_above.pose.position.y = table_center_.y;
    place_above.pose.position.z = current_z - 0.16; // keep current height
    place_above.pose.orientation = quat; // keep same orientation

    if (!moveArmToTarget(place_above)) {
    ROS_ERROR("[PLACE] Failed to move horizontally to table center!");
    return false;
    }
    ROS_INFO("Moved above table center.");
    ros::Duration(0.5).sleep();

    // ---------- Open gripper to place position ----------
    openGripper();
    ros::Duration(4.0).sleep();
    ROS_INFO("Object placed.");

    // ---------- Move up to retreat position ----------
    if (!moveEndEffectorStraightDirection(0, 0, 1, 0.02)) {
        ROS_ERROR("[PLACE] Failed straight descend!");
        return false;
    }
    ROS_INFO("Descend to place position done.");

    ros::shutdown();
    return true;
}

// // test
// bool Grasper::moveArmToTarget(const geometry_msgs::PoseStamped& target)
// {
//     move_group_.setPoseTarget(target);

//     // plan
//     moveit::planning_interface::MoveGroupInterface::Plan plan;
//     bool plan_success = (move_group_.plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
//     ROS_INFO("Planning to target result: %d", plan_success);

//     if (!plan_success) return false;

//     // move
//     bool exec_success = (move_group_.execute(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
//     ROS_INFO("Execution to target result: %d", exec_success);

//     return exec_success;
// }

// move 
bool Grasper::moveArmToTarget(const geometry_msgs::PoseStamped& target)
{
    move_group_.stop();
    move_group_.clearPoseTargets();
    move_group_.setStartStateToCurrentState();

    ROS_INFO("MoveArmToTarget: setPoseTarget");
    move_group_.setPoseTarget(target);
    move_group_.setGoalTolerance(0.01);
    move_group_.setMaxVelocityScalingFactor(0.2);  // default 0.3
    move_group_.setMaxAccelerationScalingFactor(0.1);

    ROS_INFO("MoveArmToTarget: start move");
    auto result = move_group_.move();
    ROS_INFO("MoveArmToTarget: move finished");

    if (result != moveit::planning_interface::MoveItErrorCode::SUCCESS)
    {
        ROS_ERROR("Motion execution failed.");
        return false;
    }

    return true;
}

bool Grasper::moveEndEffectorStraightDirection(double dx, double dy, double dz, double distance)
{
    // Reset state so the Cartesian path starts from the real current pose
    move_group_.stop();
    move_group_.clearPoseTargets();
    move_group_.setStartStateToCurrentState();
    move_group_.setGoalTolerance(0.01);

    // normalized direction vector
    Eigen::Vector3d dir(dx, dy, dz);
    if (dir.norm() < 1e-6) {
        ROS_ERROR("Direction vector is zero!");
        return false;
    }
    dir.normalize();

    // get the current end-effector pose
    // geometry_msgs::PoseStamped current_pose = move_group_.getCurrentPose(move_group_.getEndEffectorLink());
    // geometry_msgs::Pose start = current_pose.pose;
    // geometry_msgs::Pose target = start;

    geometry_msgs::Pose start = move_group_.getCurrentPose().pose;
    geometry_msgs::Pose target = start;

    // calculate the target point
    target.position.x += dir.x() * distance;
    target.position.y += dir.y() * distance;
    target.position.z += dir.z() * distance;

    target.orientation = start.orientation; // keep same orientation

    std::vector<geometry_msgs::Pose> waypoints;
    waypoints.push_back(start);
    waypoints.push_back(target);

    moveit_msgs::RobotTrajectory trajectory;
    double fraction = move_group_.computeCartesianPath(waypoints, 0.005, 0.0, trajectory);

    ROS_INFO("Path fraction: %.2f", fraction);
    if (fraction < 0.99) {
        ROS_ERROR("Cartesian path planning failed");
        return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = trajectory;
    if (move_group_.execute(plan) != moveit::planning_interface::MoveItErrorCode::SUCCESS) {
        ROS_ERROR("Cartesian execution failed");
        return false;
    }
    ROS_INFO("Move along direction finished");
    return true;
}

void Grasper::closeGripper()
{
    trajectory_msgs::JointTrajectory traj;
    traj.joint_names = {"gripper_left_finger_joint", "gripper_right_finger_joint"};

    trajectory_msgs::JointTrajectoryPoint pt;
    pt.positions = {0.0, 0.0}; // close
    pt.time_from_start = ros::Duration(3.0);
    traj.points.push_back(pt);

    traj.header.stamp = ros::Time::now() + ros::Duration(0.2);
    for (int i = 0; i < 3; ++i)
    {
        gripper_pub_.publish(traj);
        ros::Duration(0.1).sleep();
    }
}

void Grasper::openGripper()
{
    trajectory_msgs::JointTrajectory traj;
    traj.joint_names = {"gripper_left_finger_joint", "gripper_right_finger_joint"};

    trajectory_msgs::JointTrajectoryPoint pt;
    pt.positions = {0.04, 0.04}; // open
    pt.time_from_start = ros::Duration(3.5);
    traj.points.push_back(pt);

    traj.header.stamp = ros::Time::now() + ros::Duration(0.2);
    for(int i = 0; i < 3; ++i)
    {
        gripper_pub_.publish(traj);
        ros::Duration(0.1).sleep();
    }
}
