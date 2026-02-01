cmake_minimum_required(VERSION 3.0.2)
project(state_machine)

find_package(catkin REQUIRED COMPONENTS
  rospy
  smach
  smach_ros
  actionlib
  geometry_msgs
  move_base_msgs
  std_msgs
  std_srvs
  control_msgs
  sensor_msgs
  cv_bridge
  message_filters
  trajectory_msgs
  control_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/main_state_machine.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY param
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
