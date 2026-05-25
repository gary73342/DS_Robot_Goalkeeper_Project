// -*- coding: utf-8 -*-
//
// EKF_Localization_node.cpp
//
// 正常情況下使用 EKF 定位（EKF_PRIMARY）。
// 當 EKF innovation 連續異常高時，判定發生綁架，切換至 PF_RECOVERY：
//   PF 全域重撒粒子 → 收斂後用幾何解算重置 EKF → 切回 EKF_PRIMARY。
//
// 訂閱：
//   /odom             (nav_msgs/Odometry)
//   /posts_local      (Float32MultiArray)   [x1,y1,x2,y2,...]
//   /ball_lidar_local (Float32MultiArray)   [detected, x, y]
//
// 發布：
//   /robot_pose       (Float32MultiArray)   [x, y, theta, var_x]
//   /ball_lidar_world (Float32MultiArray)   [detected, X, Y]
//   /particle_cloud   (geometry_msgs/PoseArray)   PF 粒子點雲（RViz）

#include <iostream>
#include <vector>
#include <cmath>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>

#include "EKF_Localization.h"
#include "Particle_Filter.h"

using namespace std;

// ==============================================================================
// 狀態機定義
// ==============================================================================
enum class LocState { WAITING, EKF_PRIMARY, PF_RECOVERY };

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

    // --- 發布 ---
    ros::Publisher robot_pose_pub     = nh.advertise<std_msgs::Float32MultiArray>("/robot_pose", 1);
    ros::Publisher ball_world_pub     = nh.advertise<std_msgs::Float32MultiArray>("/ball_lidar_world", 1);
    ros::Publisher particle_cloud_pub = nh.advertise<visualization_msgs::Marker>("/particle_cloud", 1);

    // --- 定位器初始化 ---
    EKFLocalizer ekf;
    Localizer    pf(300);

    // --- 狀態機 ---
    LocState state = LocState::WAITING;

    // ★ 調教區：綁架偵測門限
    const float INNOV_THRESHOLD = 0.30f;  // 單幀 innovation L2 norm > 此值視為異常
    const int   KIDNAP_COUNT    = 5;      // 連續 N 幀異常 → 觸發綁架恢復
    const float PF_CONVERGE_VAR = 0.05f; // PF var_x 低於此值視為重定位成功
    int kidnap_frames = 0;

    // --- TF 廣播器 ---
    static tf::TransformBroadcaster tf_broadcaster;

    // --- 迴圈頻率 ---
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

        float v = -(latest_odom->twist.twist.linear.x);
        float w = latest_odom->twist.twist.angular.z;

        // ------------------------------------------------------------------
        // WAITING：等待 EKF 幾何初始化
        // ------------------------------------------------------------------
        if (state == LocState::WAITING) {
            if (latest_posts.size() >= 2) {
                if (ekf.initFromPosts(latest_posts)) {
                    float ex, ey, et;
                    ekf.getEstimate(ex, ey, et);
                    pf.reinitNearPose(ex, ey, et);
                    state = LocState::EKF_PRIMARY;
                    kidnap_frames = 0;
                    ROS_INFO("[EKF] 初始化完成，切換至 EKF_PRIMARY");
                } else {
                    ROS_WARN_THROTTLE(1.0, "[EKF] 幾何解算失敗，重試...");
                }
            } else {
                ROS_INFO_THROTTLE(2.0, "[EKF] 等待門柱...目前可見 %zu 根", latest_posts.size());
            }
            rate.sleep();
            continue;
        }

        // ------------------------------------------------------------------
        // Predict
        // ------------------------------------------------------------------
        ekf.predict(v, w, dt);
        pf.predict(v, w, dt);

        // ------------------------------------------------------------------
        // Update + 綁架偵測（只在 EKF_PRIMARY 偵測）
        // ------------------------------------------------------------------
        if (state == LocState::EKF_PRIMARY) {
            ekf.update(latest_posts);
            float innov = ekf.getLastInnovation();

            if (innov > INNOV_THRESHOLD) {
                kidnap_frames++;
                ROS_WARN_THROTTLE(0.5, "[EKF] 高 innovation=%.3f (%d/%d)",
                    innov, kidnap_frames, KIDNAP_COUNT);
            } else {
                kidnap_frames = 0;
            }

            if (kidnap_frames >= KIDNAP_COUNT) {
                pf.reinitForKidnapping(1000);
                state = LocState::PF_RECOVERY;
                kidnap_frames = 0;
                ROS_WARN("[EKF] ★ 偵測到綁架！切換至 PF_RECOVERY");
            }
        }

        pf.update(latest_posts);

        // PF 收斂後用幾何解算重置 EKF，切回 EKF_PRIMARY
        if (state == LocState::PF_RECOVERY) {
            float pf_var = pf.getVarianceX();
            if (pf_var < PF_CONVERGE_VAR && latest_posts.size() >= 2) {
                if (ekf.initFromPosts(latest_posts)) {
                    float rx, ry, rt;
                    ekf.getEstimate(rx, ry, rt);
                    pf.reinitNearPose(rx, ry, rt, 300);
                    state = LocState::EKF_PRIMARY;
                    kidnap_frames = 0;
                    ROS_INFO("[EKF] PF 收斂（var_x=%.4f），重定位成功，切回 EKF_PRIMARY", pf_var);
                }
            }
        }

        // ------------------------------------------------------------------
        // 取得最佳估計（EKF_PRIMARY 用 EKF，PF_RECOVERY 用 PF）
        // ------------------------------------------------------------------
        float ex, ey, et, var_x;
        if (state == LocState::EKF_PRIMARY) {
            ekf.getEstimate(ex, ey, et);
            var_x = ekf.getVarianceX();
        } else {
            pf.getEstimate(ex, ey, et);
            var_x = pf.getVarianceX();
        }

        const char* state_str = (state == LocState::EKF_PRIMARY) ? "EKF" : "PF";

        // ------------------------------------------------------------------
        // 發布機器人全域姿態
        // ------------------------------------------------------------------
        {
            std_msgs::Float32MultiArray pose_msg;
            pose_msg.data = {ex, ey, et, var_x};
            robot_pose_pub.publish(pose_msg);
        }

        // ------------------------------------------------------------------
        // 球局部座標 → 全域座標
        // ------------------------------------------------------------------
        {
            std_msgs::Float32MultiArray ball_msg;
            if (ball_local_detected) {
                float ball_global_x = ex + ball_local_x * std::cos(et)
                                         - ball_local_y * std::sin(et);
                float ball_global_y = ey + ball_local_x * std::sin(et)
                                         + ball_local_y * std::cos(et);
                ball_msg.data = {1.0f, ball_global_x, ball_global_y};
                ROS_INFO_THROTTLE(0.5,
                    "[EKF] [%s] X=%+.2f Y=%.2f θ=%+.3f var=%.4f posts=%zu | 球(%+.2f,%.2f)",
                    state_str, ex, ey, et, var_x, latest_posts.size(),
                    ball_global_x, ball_global_y);
            } else {
                ball_msg.data = {0.0f, 0.0f, 0.0f};
                ROS_INFO_THROTTLE(0.5,
                    "[EKF] [%s] X=%+.2f Y=%.2f θ=%+.3f var=%.4f posts=%zu | 球:無",
                    state_str, ex, ey, et, var_x, latest_posts.size());
            }
            ball_world_pub.publish(ball_msg);
        }

        // ------------------------------------------------------------------
        // PF 粒子點雲（RViz 視覺化，按權重顯示顏色：藍=低權重，綠=中，紅=高權重）
        // ------------------------------------------------------------------
        {
            const auto& particles = pf.getParticles();

            float max_w = 1e-9f;
            for (const auto& p : particles)
                if (p.weight > max_w) max_w = p.weight;

            visualization_msgs::Marker marker;
            marker.header.stamp    = ros::Time::now();
            marker.header.frame_id = "map";
            marker.ns              = "particles";
            marker.id              = 0;
            marker.type            = visualization_msgs::Marker::SPHERE_LIST;
            marker.action          = visualization_msgs::Marker::ADD;
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05;
            marker.pose.orientation.w = 1.0;

            for (const auto& p : particles) {
                geometry_msgs::Point pt;
                pt.x = p.x; pt.y = p.y; pt.z = 0.0;
                marker.points.push_back(pt);

                // 藍(0)→綠(0.5)→紅(1) 彩虹漸層
                float t = p.weight / max_w;
                std_msgs::ColorRGBA c;
                if (t < 0.5f) {
                    c.r = 0.0f; c.g = 2.0f * t; c.b = 1.0f - 2.0f * t;
                } else {
                    c.r = 2.0f * (t - 0.5f); c.g = 1.0f - 2.0f * (t - 0.5f); c.b = 0.0f;
                }
                c.a = 0.7f;
                marker.colors.push_back(c);
            }

            particle_cloud_pub.publish(marker);
        }

        // ------------------------------------------------------------------
        // TF 廣播（map → odom → base_link）
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
