cmake_minimum_required(VERSION 3.0.2)
project(teleop)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  geometry_msgs
)

catkin_install_python(PROGRAMS
  scripts/teleop_key.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
