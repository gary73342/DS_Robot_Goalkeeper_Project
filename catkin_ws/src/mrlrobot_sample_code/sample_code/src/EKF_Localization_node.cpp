// -*- coding: utf-8 -*-
//
// EKF_Localization_node.cpp
//
// 功能與 Localization_node.cpp 相同，但定位後端改用 EKF（代替粒子濾波）。
// 發布 topic 格式與 Localization_node 完全相同，fusion_node 無需修改。
//
// 訂閱：
//   /odom             (nav_msgs/Odometry)       odom 速度，驅動 predict
//   /posts_local      (Float32MultiArray)        門柱局部座標 [x1,y1,x2,y2,...]
//   /ball_lidar_local (Float32MultiArray)        球局部座標 [detected, x, y]
//
// 發布：
//   /robot_pose       (Float32MultiArray)        [x, y, theta, var_x]
//   /ball_lidar_world (Float32MultiArray)        [detected, X, Y]（從 EKF 位姿轉換）

#include <iostream>
#include <vector>
#include <cmath>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_broadcaster.h>

#include "EKF_Localization.h"

using namespace std;

// ==============================================================================
// 全域資料（callback 寫入，main loop 讀取）
// ==============================================================================

nav_msgs::Odometry::ConstPtr latest_odom = nullptr;
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    latest_odom = msg;
}

bool  ball_local_detected = false;
float ball_local_x        = 0.0f;
float ball_local_y        = 0.0f;
void ballLocalCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    ball_local_detected = (msg->data[0] > 0.5f);
    ball_local_x        = msg->data[1];
    ball_local_y        = msg->data[2];
}

vector<Observation> latest_posts;
void postsCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    latest_posts.clear();
    for (size_t i = 0; i + 1 < msg->data.size(); i += 2)
        latest_posts.push_back({msg->data[i], msg->data[i + 1]});
}

// ==============================================================================
// main
// ==============================================================================

int main(int argc, char** argv) {

    ros::init(argc, argv, "ekf_localization_node");
    ros::NodeHandle nh;

    // --- 訂閱 ---
    ros::Subscriber odom_sub  = nh.subscribe<nav_msgs::Odometry>(
        "/odom", 1, odomCallback);
    ros::Subscriber ball_sub  = nh.subscribe<std_msgs::Float32MultiArray>(
        "/ball_lidar_local", 1, ballLocalCallback);
    ros::Subscriber posts_sub = nh.subscribe<std_msgs::Float32MultiArray>(
        "/posts_local", 1, postsCallback);

    // --- 發布（與 localization_node 完全相同，可直接替換）---
    ros::Publisher robot_pose_pub = nh.advertise<std_msgs::Float32MultiArray>(
        "/robot_pose", 1);
    ros::Publisher ball_world_pub = nh.advertise<std_msgs::Float32MultiArray>(
        "/ball_lidar_world", 1);

    // --- EKF 初始化 ---
    EKFLocalizer ekf;

    // --- TF 廣播器 ---
    static tf::TransformBroadcaster tf_broadcaster;

    // --- 迴圈頻率（與 Localization_node 相同）---
    const float loop_hz = 20.0f;
    const float dt      = 1.0f / loop_hz;
    ros::Rate rate(loop_hz);

    ROS_INFO("=== [EKF Localization] Started %.0f Hz ===", loop_hz);
    ROS_INFO("[EKF Localization] 等待兩根門柱同時出現進行幾何初始化...");

    // ==========================================================================
    // 主迴圈
    // ==========================================================================
    while (ros::ok()) {
        ros::spinOnce();

        if (latest_odom == nullptr) {
            ROS_WARN_THROTTLE(2.0, "[EKF] Waiting for odom...");
            rate.sleep();
            continue;
        }

        // ------------------------------------------------------------------
        // 階段 A：Predict（線速度與角速度均來自 odom）
        // ------------------------------------------------------------------
        float v = -(latest_odom->twist.twist.linear.x);  // 方向修正同 Localization_node
        float w = latest_odom->twist.twist.angular.z;

        if (ekf.isInitialized()) {
            ekf.predict(v, w, dt);
        }

        // ------------------------------------------------------------------
        // 階段 B：初始化 or Update
        // ------------------------------------------------------------------
        if (!ekf.isInitialized()) {
            if (latest_posts.size() >= 2) {
                if (ekf.initFromPosts(latest_posts)) {
                    ROS_INFO("[EKF] 初始化完成，開始 EKF 定位");
                } else {
                    ROS_WARN_THROTTLE(1.0, "[EKF] 幾何解算失敗，重試...");
                }
            } else {
                ROS_INFO_THROTTLE(2.0, "[EKF] 等待門柱...目前可見 %zu 根", latest_posts.size());
            }
            rate.sleep();
            continue;
        }

        // 已初始化：EKF update
        ekf.update(latest_posts);

        // ------------------------------------------------------------------
        // 階段 C：取得估計結果
        // ------------------------------------------------------------------
        float ex, ey, et;
        ekf.getEstimate(ex, ey, et);
        float var_x = ekf.getVarianceX();

        ROS_INFO_THROTTLE(0.3,
            "[EKF] Pose X=%.2f Y=%.2f Theta=%.2f var_x=%.4f posts=%zu",
            ex, ey, et, var_x, latest_posts.size());

        // ------------------------------------------------------------------
        // 階段 D：發布機器人全域姿態
        // ------------------------------------------------------------------
        {
            std_msgs::Float32MultiArray pose_msg;
            pose_msg.data = {ex, ey, et, var_x};
            robot_pose_pub.publish(pose_msg);
        }

        // ------------------------------------------------------------------
        // 階段 E：球局部座標 → 全域座標
        // ------------------------------------------------------------------
        {
            std_msgs::Float32MultiArray ball_msg;
            if (ball_local_detected) {
                float ball_global_x = ex + ball_local_x * std::cos(et)
                                         - ball_local_y * std::sin(et);
                float ball_global_y = ey + ball_local_x * std::sin(et)
                                         + ball_local_y * std::cos(et);
                ball_msg.data = {1.0f, ball_global_x, ball_global_y};
                ROS_INFO_THROTTLE(0.3, "[EKF] Ball world X=%.2f Y=%.2f",
                                  ball_global_x, ball_global_y);
            } else {
                ball_msg.data = {0.0f, 0.0f, 0.0f};
            }
            ball_world_pub.publish(ball_msg);
        }

        // ------------------------------------------------------------------
        // 階段 F：TF 廣播（map → odom → base_link）
        // ------------------------------------------------------------------
        {
            float odom_x = latest_odom->pose.pose.position.x;
            float odom_y = latest_odom->pose.pose.position.y;

            tf::Quaternion odom_q(
                latest_odom->pose.pose.orientation.x,
                latest_odom->pose.pose.orientation.y,
                latest_odom->pose.pose.orientation.z,
                latest_odom->pose.pose.orientation.w);
            double roll, pitch, odom_yaw;
            tf::Matrix3x3(odom_q).getRPY(roll, pitch, odom_yaw);

            float cx = ex - odom_x;
            float cy = ey - odom_y;
            float ct = et - static_cast<float>(odom_yaw);

            tf::Transform map_to_odom;
            map_to_odom.setOrigin(tf::Vector3(cx, cy, 0.0));
            tf::Quaternion q_corr;
            q_corr.setRPY(0, 0, ct);
            map_to_odom.setRotation(q_corr);
            tf_broadcaster.sendTransform(
                tf::StampedTransform(map_to_odom, ros::Time::now(), "map", "odom"));

            tf::Transform odom_to_base;
            odom_to_base.setOrigin(tf::Vector3(odom_x, odom_y, 0.0));
            tf::Quaternion q_odom;
            q_odom.setRPY(0, 0, odom_yaw);
            odom_to_base.setRotation(q_odom);
            tf_broadcaster.sendTransform(
                tf::StampedTransform(odom_to_base, ros::Time::now(), "odom", "base_link"));
        }

        rate.sleep();
    }

    return 0;
}
