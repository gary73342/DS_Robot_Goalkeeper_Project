// -*- coding: utf-8 -*-
#include <iostream>
#include <vector>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Float32MultiArray.h>

#include "Random_Forest.h"
#include "FeatureExtractor.h"

using namespace std;

// ==============================================================================
// 全域變數
// ==============================================================================

sensor_msgs::LaserScan::ConstPtr latest_scan = nullptr;

void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    latest_scan = msg;
}

// ==============================================================================
// main
// ==============================================================================

int main(int argc, char** argv) {

    ros::init(argc, argv, "perception_node");
    ros::NodeHandle nh;

    // --- 訂閱 ---
    ros::Subscriber scan_sub = nh.subscribe<sensor_msgs::LaserScan>(
        "/scan", 1, scanCallback);

    // --- 發布 ---
    // /ball_lidar_local : [detected(0/1), x_local, y_local]
    //   球在光達局部座標系下的位置，給 localization_node 轉換成全域座標
    ros::Publisher ball_local_pub = nh.advertise<std_msgs::Float32MultiArray>(
        "/ball_lidar_local", 1);

    // /posts_local : [x1, y1, x2, y2, ...]
    //   所有偵測到的門柱局部座標，給 localization_node 做粒子濾波 update
    ros::Publisher posts_pub = nh.advertise<std_msgs::Float32MultiArray>(
        "/posts_local", 1);

    // RViz 視覺化
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>(
        "/visual_markers", 10);

    // --- 載入隨機森林模型 ---
    std::string model_path = ros::package::getPath("sample_code")
                             + "/rf_model_multi_7feat.txt";
    RandomForest detector(model_path);

    // --- 迴圈頻率 ---
    float loop_hz = 20.0f;
    ros::Rate rate(loop_hz);

    ROS_INFO("=== [Perception] Started, %.0f Hz ===", loop_hz);

    // ==========================================================================
    // 主迴圈
    // ==========================================================================
    while (ros::ok()) {
        ros::spinOnce();

        if (latest_scan == nullptr) {
            ROS_WARN_THROTTLE(2.0, "[Perception] Waiting for LiDAR...");
            rate.sleep();
            continue;
        }

        // ----------------------------------------------------------------------
        // 階段 A：特徵提取 + 隨機森林分類
        // ----------------------------------------------------------------------
        vector<Segment> segments =
            FeatureExtractor::extractSegmentsFromScan(latest_scan);

        vector<Observation> found_posts;
        bool     ball_detected = false;
        Observation ball_obs;
        float    min_ball_dist = 1e6f;

        for (const auto& seg : segments) {
            if (seg.features.size() != 7) continue;

            int label = detector.predict(seg.features);
            if (label <= 0) continue;

            if (label == 2 && seg.local_y > 0) {
                // 球：Y 正向，距離 0.2m ~ 1.5m
                float d = hypot(seg.local_x, seg.local_y);
                if (d < min_ball_dist && d > 0.2f && d < 1.5f) {
                    min_ball_dist  = d;
                    ball_detected  = true;
                    ball_obs       = {seg.local_x, seg.local_y};
                }
            } else if (label == 1) {
                // 門柱：不限方向（綁架恢復時機器人可能轉向，門柱會出現在任意 local_y）
                found_posts.push_back({seg.local_x, seg.local_y});
            }
        }

        // ----------------------------------------------------------------------
        // 階段 B：發布感知結果
        // ----------------------------------------------------------------------

        // 球的局部座標
        {
            std_msgs::Float32MultiArray ball_msg;
            if (ball_detected) {
                ball_msg.data = {1.0f, ball_obs.x, ball_obs.y};
                ROS_INFO_THROTTLE(0.3, "[Perception] Ball: x=%.2f y=%.2f dist=%.2f m",
                                  ball_obs.x, ball_obs.y, min_ball_dist);
            } else {
                ball_msg.data = {0.0f, 0.0f, 0.0f};
            }
            ball_local_pub.publish(ball_msg);
        }

        // 門柱過濾：從所有偵測到的門柱中選間距最接近 0.9m 的一對發布
        // 避免機器人的腳被誤判為門柱時 EKF 收到錯誤觀測
        const float GOAL_WIDTH = 0.9f;
        const float GATE_DIST  = 0.15f;

        vector<Observation> valid_posts;
        if (found_posts.size() >= 2) {
            int   best_i = -1, best_j = -1;
            float best_err = 1e6f;
            for (size_t i = 0; i < found_posts.size(); ++i) {
                for (size_t j = i + 1; j < found_posts.size(); ++j) {
                    float dx  = found_posts[i].x - found_posts[j].x;
                    float dy  = found_posts[i].y - found_posts[j].y;
                    float err = fabs(hypot(dx, dy) - GOAL_WIDTH);
                    if (err < best_err) {
                        best_err = err;
                        best_i   = static_cast<int>(i);
                        best_j   = static_cast<int>(j);
                    }
                }
            }
            if (best_err <= GATE_DIST) {
                valid_posts.push_back(found_posts[best_i]);
                valid_posts.push_back(found_posts[best_j]);
            }
        } else {
            valid_posts = found_posts;  // 0 或 1 根：維持現有行為
        }

        // 發布門柱局部座標（打平成一維陣列：x1,y1,x2,y2,...）
        {
            std_msgs::Float32MultiArray posts_msg;
            for (const auto& p : valid_posts) {
                posts_msg.data.push_back(p.x);
                posts_msg.data.push_back(p.y);
            }
            posts_pub.publish(posts_msg);

            for (size_t i = 0; i < valid_posts.size(); ++i) {
                ROS_INFO_THROTTLE(0.3, "[Perception] Post %zu: x=%.2f y=%.2f dist=%.2f m",
                    i + 1, valid_posts[i].x, valid_posts[i].y,
                    hypot(valid_posts[i].x, valid_posts[i].y));
            }
            if (found_posts.size() >= 2 && valid_posts.size() < 2) {
                ROS_WARN_THROTTLE(0.5, "[Perception] Posts filtered out: %zu detected, no pair within %.2fm of goal width",
                    found_posts.size(), GATE_DIST);
            }
        }

        // ----------------------------------------------------------------------
        // 階段 C：RViz Marker 視覺化
        // ----------------------------------------------------------------------

        // 球（綠色圓球）
        if (ball_detected) {
            visualization_msgs::Marker ball_marker;
            ball_marker.header.frame_id = latest_scan->header.frame_id;
            ball_marker.header.stamp    = ros::Time::now();
            ball_marker.ns              = "ball";
            ball_marker.id              = 0;
            ball_marker.type            = visualization_msgs::Marker::SPHERE;
            ball_marker.action          = visualization_msgs::Marker::ADD;
            ball_marker.pose.position.x = ball_obs.x;
            ball_marker.pose.position.y = ball_obs.y;
            ball_marker.pose.position.z = -0.065f;  // 雷射與球心垂直落差
            ball_marker.scale.x         = 0.22f;
            ball_marker.scale.y         = 0.22f;
            ball_marker.scale.z         = 0.22f;
            ball_marker.color.r         = 0.0f;
            ball_marker.color.g         = 1.0f;
            ball_marker.color.b         = 0.0f;
            ball_marker.color.a         = 1.0f;
            marker_pub.publish(ball_marker);
        } else {
            visualization_msgs::Marker del;
            del.header.frame_id = latest_scan->header.frame_id;
            del.ns              = "ball";
            del.id              = 0;
            del.action          = visualization_msgs::Marker::DELETE;
            marker_pub.publish(del);
        }

        // 門柱（藍色圓柱，只顯示通過過濾的門柱對）
        for (size_t i = 0; i < 2; ++i) {
            visualization_msgs::Marker post_marker;
            post_marker.header.frame_id = latest_scan->header.frame_id;
            post_marker.header.stamp    = ros::Time::now();
            post_marker.ns              = "goal_posts";
            post_marker.id              = static_cast<int>(i);
            post_marker.type            = visualization_msgs::Marker::CYLINDER;

            if (i < valid_posts.size()) {
                post_marker.action          = visualization_msgs::Marker::ADD;
                post_marker.pose.position.x = valid_posts[i].x;
                post_marker.pose.position.y = valid_posts[i].y;
                post_marker.pose.position.z = 0.0f;
                post_marker.scale.x         = 0.08f;
                post_marker.scale.y         = 0.08f;
                post_marker.scale.z         = 0.20f;
                post_marker.color.r         = 0.0f;
                post_marker.color.g         = 0.0f;
                post_marker.color.b         = 1.0f;
                post_marker.color.a         = 1.0f;
            } else {
                post_marker.action = visualization_msgs::Marker::DELETE;
            }
            marker_pub.publish(post_marker);
        }

        rate.sleep();
    }

    return 0;
}