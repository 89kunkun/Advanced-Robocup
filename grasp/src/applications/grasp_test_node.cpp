#include <ros/ros.h>
#include <grasp/grasp_test.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "grasp_node");
    ros::NodeHandle nh;

    Grasper grasper(nh);
    ros::Subscriber sub = nh.subscribe("/labeled_object_point_cloud", 1, &Grasper::pointCloudCallback, &grasper);

    ros::spin();
    return 0;
}
