cmake_minimum_required(VERSION 3.0.2)
project(sample_code)

# 1. 尋找必要的 ROS 套件
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  roslib
  tf
  nav_msgs
  visualization_msgs
)

catkin_package()

# 2. 告訴編譯器去哪裡找標頭檔 (.h)
include_directories(
  include/
  ${catkin_INCLUDE_DIRS}
)

# 3a. perception_node：感知 + RViz
#     需要：隨機森林、特徵提取
add_executable(perception_node
  src/Perception_node.cpp
  src/Random_Forest.cpp
  src/FeatureExtractor.cpp
)
target_link_libraries(perception_node
  ${catkin_LIBRARIES}
)

# 3b. localization_node：定位 + TF 廣播
#     需要：粒子濾波
add_executable(localization_node
  src/Localization_node.cpp
  src/Particle_Filter.cpp
)
target_link_libraries(localization_node
  ${catkin_LIBRARIES}
)
