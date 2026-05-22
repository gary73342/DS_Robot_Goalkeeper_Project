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

# 3. 將所有 .cpp 檔案編譯成一個名為 football_robot_fsm 的執行檔
add_executable(robot_goalkeeper_fsm
  src/main.cpp
  src/RandomForest.cpp
  src/Localization.cpp
  src/FeatureExtractor.cpp
)

# 4. 將 ROS 函式庫連結到你的執行檔上
target_link_libraries(robot_goalkeeper_fsm
  ${catkin_LIBRARIES}
)
