// -*- coding: utf-8 -*-
#include <iostream>
#include <vector>
#include <cmath>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_broadcaster.h>

#include "Particle_Filter.h"

using namespace std;

// ==============================================================================
// 全域變數
// ==============================================================================

// 最新的 odom 資料
nav_msgs::Odometry::ConstPtr latest_odom = nullptr;
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    latest_odom = msg;
}

// 球的局部座標（來自 perception_node）
bool  ball_local_detected = false;
float ball_local_x        = 0.0f;
float ball_local_y        = 0.0f;
void ballLocalCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    ball_local_detected = (msg->data[0] > 0.5f);
    ball_local_x        = msg->data[1];
    ball_local_y        = msg->data[2];
}

// 門柱的局部座標（來自 perception_node）
// 格式：[x1, y1, x2, y2, ...]，打平的一維陣列
vector<Observation> latest_posts;
void postsCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    latest_posts.clear();
    // 每兩個數字是一個門柱的 (x, y)
    for (size_t i = 0; i + 1 < msg->data.size(); i += 2) {
        latest_posts.push_back({msg->data[i], msg->data[i + 1]});
    }
}

// ==============================================================================
// main
// ==============================================================================

int main(int argc, char** argv) {

    ros::init(argc, argv, "localization_node");
    ros::NodeHandle nh;

    // --- 訂閱 ---
    ros::Subscriber odom_sub = nh.subscribe<nav_msgs::Odometry>(
        "/odom", 1, odomCallback);
    ros::Subscriber ball_sub = nh.subscribe<std_msgs::Float32MultiArray>(
        "/ball_lidar_local", 1, ballLocalCallback);
    ros::Subscriber posts_sub = nh.subscribe<std_msgs::Float32MultiArray>(
        "/posts_local", 1, postsCallback);

    // --- 發布 ---
    // /robot_pose       : [x, y, theta]         機器人全域姿態，給 fusion_node
    // /ball_lidar_world : [detected, X, Y]       球的全域座標，給 fusion_node
    ros::Publisher robot_pose_pub  = nh.advertise<std_msgs::Float32MultiArray>(
        "/robot_pose", 1);
    ros::Publisher ball_world_pub  = nh.advertise<std_msgs::Float32MultiArray>(
        "/ball_lidar_world", 1);

    // --- 粒子濾波初始化 ---
    Localizer pf(300);  // 300 個粒子，精度與速度的平衡點

    // --- TF 廣播器 ---
    static tf::TransformBroadcaster tf_broadcaster;

    // --- 迴圈頻率 ---
    // 與 perception_node 保持一致（20Hz），確保每幀都能拿到最新感知資料
    float loop_hz = 20.0f;
    float dt      = 1.0f / loop_hz;
    ros::Rate rate(loop_hz);

    ROS_INFO("=== [定位節點] 啟動，%.0f Hz，%d 個粒子 ===", loop_hz, 300);

    // ==========================================================================
    // 主迴圈
    // ==========================================================================
    while (ros::ok()) {
        ros::spinOnce();

        // 等 odom 資料就緒
        if (latest_odom == nullptr) {
            ROS_WARN_THROTTLE(2.0, "[定位] 等待 odom 資料...");
            rate.sleep();
            continue;
        }

        // ----------------------------------------------------------------------
        // 階段 A：從 odom 取得實際速度，餵給粒子濾波 predict
        // ----------------------------------------------------------------------
        // 用 odom 實測速度而非 cmd_vel 指令速度，更準確反映機器人真實運動
        float actual_v = static_cast<float>(latest_odom->twist.twist.linear.x);
        float actual_w = static_cast<float>(latest_odom->twist.twist.angular.z);

        pf.predict(actual_v, actual_w, dt);

        // ----------------------------------------------------------------------
        // 階段 B：用門柱觀測更新粒子權重
        // ----------------------------------------------------------------------
        pf.update(latest_posts);

        // ----------------------------------------------------------------------
        // 階段 C：取得定位估計結果
        // ----------------------------------------------------------------------
        float ex, ey, et;
        pf.getEstimate(ex, ey, et);

        ROS_INFO_THROTTLE(0.5,
            "[定位] 全域位置 X=%.2f m  Y=%.2f m  Theta=%.2f rad  "
            "門柱數=%zu",
            ex, ey, et, latest_posts.size());

        // ----------------------------------------------------------------------
        // 階段 D：發布機器人全域姿態
        // ----------------------------------------------------------------------
        {
            std_msgs::Float32MultiArray pose_msg;
            pose_msg.data = {ex, ey, et};
            robot_pose_pub.publish(pose_msg);
        }

        // ----------------------------------------------------------------------
        // 階段 E：球的局部座標 → 全域座標，發布給 fusion_node
        // ----------------------------------------------------------------------
        {
            std_msgs::Float32MultiArray ball_msg;
            if (ball_local_detected) {
                // 旋轉矩陣：局部座標轉全域座標
                float ball_global_x = ex + ball_local_x * cos(et)
                                         - ball_local_y * sin(et);
                float ball_global_y = ey + ball_local_x * sin(et)
                                         + ball_local_y * cos(et);
                ball_msg.data = {1.0f, ball_global_x, ball_global_y};
                ROS_INFO_THROTTLE(0.5,
                    "[定位] 球全域座標 X=%.2f m  Y=%.2f m",
                    ball_global_x, ball_global_y);
            } else {
                ball_msg.data = {0.0f, 0.0f, 0.0f};
            }
            ball_world_pub.publish(ball_msg);
        }

        // ----------------------------------------------------------------------
        // 階段 F：TF 廣播（map → odom → base_link）
        // ----------------------------------------------------------------------
        float odom_x = latest_odom->pose.pose.position.x;
        float odom_y = latest_odom->pose.pose.position.y;

        tf::Quaternion odom_q(
            latest_odom->pose.pose.orientation.x,
            latest_odom->pose.pose.orientation.y,
            latest_odom->pose.pose.orientation.z,
            latest_odom->pose.pose.orientation.w);
        double odom_roll, odom_pitch, odom_yaw;
        tf::Matrix3x3(odom_q).getRPY(odom_roll, odom_pitch, odom_yaw);

        // map → odom：粒子濾波修正偏移
        // 概念：真實位置 = odom位置 + 修正偏移
        //       修正偏移 = 真實位置 - odom位置
        float correction_x     = ex - odom_x;
        float correction_y     = ey - odom_y;
        float correction_theta = et - static_cast<float>(odom_yaw);

        tf::Transform map_to_odom;
        map_to_odom.setOrigin(tf::Vector3(correction_x, correction_y, 0.0));
        tf::Quaternion q_correction;
        q_correction.setRPY(0, 0, correction_theta);
        map_to_odom.setRotation(q_correction);
        tf_broadcaster.sendTransform(
            tf::StampedTransform(map_to_odom, ros::Time::now(), "map", "odom"));

        // odom → base_link
        tf::Transform odom_to_base;
        odom_to_base.setOrigin(tf::Vector3(odom_x, odom_y, 0.0));
        tf::Quaternion q_odom;
        q_odom.setRPY(0, 0, odom_yaw);
        odom_to_base.setRotation(q_odom);
        tf_broadcaster.sendTransform(
            tf::StampedTransform(odom_to_base, ros::Time::now(), "odom", "base_link"));

        rate.sleep();
    }

    return 0;
}
