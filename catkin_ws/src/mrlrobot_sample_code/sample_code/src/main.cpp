#include <iostream>
#include <vector>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

#include "RandomForest.h"
#include "Localization.h"
#include "FeatureExtractor.h"

using namespace std;

// 兩種模式：巡邏、攔截
enum RobotState { PATROL, INTERCEPT};

// 儲存最新收到的雷射掃描
sensor_msgs::LaserScan::ConstPtr latest_scan = nullptr;

// 回調函數，將掃完一圈的資料存進記憶體
void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    latest_scan = msg;
}

// 儲存最新的 odom 資料
nav_msgs::Odometry::ConstPtr latest_odom = nullptr;

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    latest_odom = msg;
}

int main(int argc, char** argv) {
    // 1. 初始化 ROS 節點
    ros::init(argc, argv, "football_robot_fsm");
    ros::NodeHandle nh;

    // 訂閱光達資料，發布馬達控制指令
    ros::Subscriber scan_sub = nh.subscribe<sensor_msgs::LaserScan>("/scan", 1, scanCallback);
    ros::Publisher cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("/visual_markers", 10);
    ros::Subscriber odom_sub = nh.subscribe<nav_msgs::Odometry>("/odom", 1, odomCallback);
    
    // 2. 初始化核心模組
    std::string model_path = ros::package::getPath("sample_code") + "/rf_model_multi_7feat.txt";
    RandomForest detector(model_path);
    Localizer pf(100);

    // 3. FSM 狀態與控制參數
    RobotState current_state = PATROL;

    static tf::TransformBroadcaster tf_broadcaster;

    // P-control
    float Kp_x = 0.5f;  // X 軸爆發力：決定前進/倒車衝向目標的猛烈程度
    float Kp_yaw = 2.5f;   // 車頭穩定器：決定鎖死車頭不讓它歪掉的力道

    float current_v = 0.0f;
    float current_w = 0.0f;  
    
    // ★ 控制迴圈頻率：設定為 10 Hz
    float loop_hz = 10.0;
    ros::Rate rate(loop_hz); 
    float dt = 1.0 / loop_hz;

    ROS_INFO("=== Football Robot FSM Started ===");

    // 4. ROS 主控制迴圈
    while (ros::ok()) {
        ros::spinOnce(); 

        // 如果光達資料還沒來，就乖乖等待，什麼都別做！
        if (latest_scan == nullptr) {
            cerr << "Wait for Laser Data..." << endl;
            rate.sleep();
            continue; 
        }  
        // ---------------------------------------------------------
        // 階段 A：感知與空間遮罩
        // ---------------------------------------------------------
        vector<Segment> segments = FeatureExtractor::extractSegmentsFromScan(latest_scan);
        
        vector<Observation> found_posts;
        bool ball_detected = false;
        Observation ball_obs;
        float min_ball_dist = 1e6;

        for (const auto& seg : segments) {
            if (seg.features.size() != 7) continue;

            int label = detector.predict(seg.features);
            if (label <= 0) continue; 

            // 球在 Y 軸負向
            if (label == 2 && seg.local_y > 0) {
                float d = hypot(seg.local_x, seg.local_y);
                
                // 加上嚴格的物理限制：太近的(自己輪子)或太遠的(場外雜物)都不算！
                // 假設球場寬度約 1.5 公尺，只理會距離在 0.2m ~ 1.5m 之間的球
                if (d < min_ball_dist && d > 0.2f && d < 1.5f) {
                    min_ball_dist = d;
                    ball_detected = true;
                    ball_obs = {seg.local_x, seg.local_y};
                }
            } 
            // 球柱在 Y 軸正向
            else if (label == 1 && seg.local_y < 0) {
                found_posts.push_back({seg.local_x, seg.local_y});
            }
        }
        // ---------------------------------------------------------
        // 階段 B：定位與日誌輸出
        // ---------------------------------------------------------
        pf.predict(current_v, current_w, dt); 
        pf.update(found_posts);       

        float ex, ey, et;
        pf.getEstimate(ex, ey, et);
            
        ROS_INFO_THROTTLE(0.5, "[Localization] Global: X=%.2f, Y=%.2f, Theta=%.2f rad", ex, ey, et);

        for (size_t i = 0; i < found_posts.size(); ++i) {
            ROS_INFO_THROTTLE(0.5, "[Perception] Post %zu found! Rel: x=%.2f, y=%.2f (Dist: %.2f)", 
                                i+1, found_posts[i].x, found_posts[i].y, 
                                hypot(found_posts[i].x, found_posts[i].y));
        }
        // =========================================================
        // 廣播 map → odom（修正偏移量），而非直接廣播 map → base_link
        // =========================================================
        if (latest_odom != nullptr) {
            // 1. 取出 odom 認為自己的位置
            float odom_x = latest_odom->pose.pose.position.x;
            float odom_y = latest_odom->pose.pose.position.y;
            
            // 取出 odom 的 yaw 角
            tf::Quaternion odom_q(
                latest_odom->pose.pose.orientation.x,
                latest_odom->pose.pose.orientation.y,
                latest_odom->pose.pose.orientation.z,
                latest_odom->pose.pose.orientation.w
            );
            double odom_roll, odom_pitch, odom_yaw;
            tf::Matrix3x3(odom_q).getRPY(odom_roll, odom_pitch, odom_yaw);

            // 2. 粒子濾波算出的真實位置 (ex, ey, et) 已在上面計算好
            // 計算 map → odom 的修正偏移
            // 概念：map_to_odom = 真實位置 - odom位置
            float correction_x     = ex - odom_x;
            float correction_y     = ey - odom_y;
            float correction_theta = et - odom_yaw;

            // 3. 建構並廣播 map → odom 的 TF
            tf::Transform map_to_odom;
            map_to_odom.setOrigin(tf::Vector3(correction_x, correction_y, 0.0));
            
            tf::Quaternion q_correction;
            q_correction.setRPY(0, 0, correction_theta);
            map_to_odom.setRotation(q_correction);

            tf_broadcaster.sendTransform(
                tf::StampedTransform(map_to_odom, ros::Time::now(), "map", "odom")
            );
            // 補上 odom → base_link
            tf::Transform odom_to_base;
            odom_to_base.setOrigin(tf::Vector3(odom_x, odom_y, 0.0));
            tf::Quaternion q_odom;
            q_odom.setRPY(0, 0, odom_yaw);
            odom_to_base.setRotation(q_odom);

            tf_broadcaster.sendTransform(
                tf::StampedTransform(odom_to_base, ros::Time::now(), "odom", "base_link")
            );
        } else {
            // odom 還沒來之前，先廣播一個零偏移，避免 TF 樹斷掉
            tf::Transform identity;
            identity.setIdentity();
            tf_broadcaster.sendTransform(
                tf::StampedTransform(identity, ros::Time::now(), "map", "odom")
            );
            ROS_WARN_THROTTLE(2.0, "[TF] wait for odom data...");
        }        
        // ---------------------------------------------------------  
        // 階段 C：FSM 決策與目標點生成
        // ---------------------------------------------------------
        // 定義球門物理幾何 (全域座標系)
        float limit_x_max = 0.5f;  // 上面那根可樂瓶的 X 座標
        float limit_x_min = -0.5f; // 下面那根可樂瓶的 X 座標
            
        static float current_patrol_target_x = limit_x_max; 
        float target_yaw = 0.0f;   // 守門員車頭永遠面向正前方，不准轉彎！

        if (ball_detected) {
            current_state = PATROL;
        } else {
            current_state = PATROL;
        }

        switch (current_state) {
            case PATROL: {
                float dist_to_target = std::abs(current_patrol_target_x - ex);
                
                ROS_INFO_THROTTLE(0.5, "[Patrol] Target X=%.2f, Dist left: %.2f", current_patrol_target_x, dist_to_target);

                // 🚨 升級：視覺確認換檔機制 (Vision-Confirmed Flip)
                if (dist_to_target < 0.1f) {
                    // 只有在「眼睛有看到門柱」或是「盲猜嚴重超界 (怕撞牆)」時，才允許換目標！
                    if (!found_posts.empty() || std::abs(ex) > 0.6f) {
                        current_patrol_target_x = (current_patrol_target_x == limit_x_max) ? limit_x_min : limit_x_max;
                        ROS_INFO("[FSM-Trigger] Confirmed Reach! Switch target to X=%.2f", current_patrol_target_x);
                    } else {
                        // 瞎著眼睛飄到目標區了，不准換檔，印出警告並繼續微速往前探
                        ROS_INFO_THROTTLE(0.5, "[FSM-Warning] Blind near target! Waiting for vision...");
                    }
                }
                break;
            }
            
            case INTERCEPT: {
                float ball_global_x = ex + ball_obs.x * cos(et) - ball_obs.y * sin(et);
                float raw_target_x = ball_global_x;
                current_patrol_target_x = std::max(limit_x_min, std::min(raw_target_x, limit_x_max));
                
                ROS_INFO_THROTTLE(0.5, "[Intercept] Ball locked! Move to X=%.2f", current_patrol_target_x);
                break;
            }
            
        }

        // ---------------------------------------------------------
        // 階段 D：單軸線性控制器 (進階 PI 角度鎖定 + 距離流暢平滑版)
        // ---------------------------------------------------------
        
        // 1. 職業級角度鎖定器 (Yaw PI Controller) - 徹底根治越走越偏
        float error_yaw = target_yaw - et;
        while (error_yaw > M_PI)  error_yaw -= 2.0 * M_PI;
        while (error_yaw < -M_PI) error_yaw += 2.0 * M_PI;

        // 💡 宣告靜態變數：用來累積角度誤差的「積分池」
        static float integral_yaw = 0.0f;
        float Ki_yaw = 0.5f; // 積分增益 (修正馬達不均勻的靈魂參數)
        
        // 防飽和裝甲 (Anti-Windup)：只有在角度微歪 (小於 30 度) 時才累積積分
        if (std::abs(error_yaw) < 0.5f) {
            integral_yaw += error_yaw * dt;
        } else {
            integral_yaw = 0.0f; // 歪得太離譜時重設，避免失控
        }

        // 算出力道強壯的角度修正角速度
        current_w = (Kp_yaw * error_yaw) + (Ki_yaw * integral_yaw);
        current_w = std::max(-1.2f, std::min(current_w, 1.2f)); // 稍微拉高修正上限

        // 2. 🚀 距離感應流暢控制器 (Distance-based Smooth Velocity Profile)
        float error_x = current_patrol_target_x - ex;
        float target_v = 0.0f; 

        if (std::abs(error_x) > 0.05f) {
            float min_move_v = 0.08f; // 克服實體靜摩擦力的最小保底速度
            float direction = (error_x > 0) ? 1.0f : -1.0f;
            
            // 基礎 P 控制速度 (會隨著距離接近而自動減速)
            float p_v = Kp_x * error_x;
            
            // 平滑減速與保底油門的混合控制
            if (std::abs(p_v) < min_move_v) {
                target_v = direction * min_move_v;
            } else {
                target_v = p_v;
            }

            // 🚨 角度優先權盾牌 (Yaw-Priority Shield)
            // 如果偵測到車頭歪得有點明顯 (大於 0.15 rad，約 8.5 度)
            // 強迫線性速度直接砍半 (乘以 0.4)，把能量留給 current_w 優先把車身拉直！
            if (std::abs(error_yaw) > 0.15f) {
                target_v *= 0.4f; 
            }
        } else {
            target_v = 0.0f;     // 踩入 deadband 煞車死區
            integral_yaw = 0.0f; // 煞車時清空角度積分，免得下次起步暴衝
        }


        // 盲人手杖機制 (Blind Mode Shield)
        // 計算連續幾幀沒有看到門柱
        static int blind_counter = 0;
        if (found_posts.empty()) {
            blind_counter++;
        } else {
            blind_counter = 0;
        }

        // 預設最大線速度為 0.20
        float max_allowed_v = 0.20f;
        
        // 如果超過 0.5 秒 (10Hz迴圈下的 5 幀) 都沒看到門柱，進入盲人模式
        if (blind_counter > 5) { 
            max_allowed_v = 0.10f; // 瞎了就砍半限速，邊走邊看
            ROS_INFO_THROTTLE(0.5, "[Control] Blind Mode! Slowing down to max %.2f m/s", max_allowed_v);
        }

        // 限制最大線速度（套用盲人手杖限速）
        target_v = std::max(-max_allowed_v, std::min(target_v, max_allowed_v));
        
        // 3. ✨ 加速度斜坡濾波器 (溫柔起步機制)
        static float last_v = 0.0f;          
        float MAX_ACCEL = 0.8f;              // 起步會像絲綢一樣滑順
        float max_v_change = MAX_ACCEL * dt; 

        if (target_v > last_v) {
            current_v = std::min(target_v, last_v + max_v_change);
        } else {
            current_v = std::max(target_v, last_v - max_v_change);
        }
        last_v = current_v; 

        // 4. 發布最終精確指令
        geometry_msgs::Twist cmd_msg;
        cmd_msg.linear.x = current_v; 
        cmd_msg.linear.y = 0.0f;      
        cmd_msg.angular.z = current_w;
        

        // ---------------------------------------------------------
        // 新增階段：建構 RViz Marker 視覺化
        // ---------------------------------------------------------
        // 畫球 (綠色圓球)
        if (ball_detected) {
            visualization_msgs::Marker ball_marker;
            // 關鍵：因為 Fixed Frame 被你設在 base_scan，這裡必須綁定相同的 frame_id
            ball_marker.header.frame_id = latest_scan->header.frame_id; 
            ball_marker.header.stamp = ros::Time::now();
            ball_marker.ns = "ball";
            ball_marker.id = 0;
            ball_marker.type = visualization_msgs::Marker::SPHERE; // 圓球
            ball_marker.action = visualization_msgs::Marker::ADD;

            // 設定球的位置 (相對於雷射)
            ball_marker.pose.position.x = ball_obs.x;
            ball_marker.pose.position.y = ball_obs.y;
            ball_marker.pose.position.z = -0.065; // 依據報告，雷射與球心有 -6.5cm 的垂直落差

            // 設定大小 (報告指出 5 號足球直徑為 22cm = 0.22m)
            ball_marker.scale.x = 0.22;
            ball_marker.scale.y = 0.22;
            ball_marker.scale.z = 0.22;

            // 設定顏色 (鮮綠色，不透明)
            ball_marker.color.r = 0.0f;
            ball_marker.color.g = 1.0f;
            ball_marker.color.b = 0.0f;
            ball_marker.color.a = 1.0f;

            marker_pub.publish(ball_marker);
        } else {
            // 如果這一幀沒看到球，發布一個 DELETE 指令讓球在 RViz 中消失，避免殘影
            visualization_msgs::Marker ball_marker;
            ball_marker.header.frame_id = latest_scan->header.frame_id;
            ball_marker.ns = "ball";
            ball_marker.id = 0;
            ball_marker.action = visualization_msgs::Marker::DELETE;
            marker_pub.publish(ball_marker);
        }

        // 畫門柱 (藍色圓柱)
        for (size_t i = 0; i < 2; ++i) {
            visualization_msgs::Marker post_marker;
            post_marker.header.frame_id = latest_scan->header.frame_id;
            post_marker.header.stamp = ros::Time::now();
            post_marker.ns = "goal_posts";
            post_marker.id = i; // id 0 和 id 1 代表兩個不同的門柱
            post_marker.type = visualization_msgs::Marker::CYLINDER; // 圓柱體

            if (i < found_posts.size()) {
                // 如果有偵測到門柱，顯示它
                post_marker.action = visualization_msgs::Marker::ADD;
                post_marker.pose.position.x = found_posts[i].x;
                post_marker.pose.position.y = found_posts[i].y;
                post_marker.pose.position.z = 0.0; 

                // 設定大小 (假設可樂瓶直徑 8cm，高度 20cm)
                post_marker.scale.x = 0.08;
                post_marker.scale.y = 0.08;
                post_marker.scale.z = 0.20;

                // 設定顏色 (藍色)
                post_marker.color.r = 0.0f;
                post_marker.color.g = 0.0f;
                post_marker.color.b = 1.0f;
                post_marker.color.a = 1.0f;
            } else {
                // 如果沒看到，就隱藏該編號的門柱
                post_marker.action = visualization_msgs::Marker::DELETE;
            }
            marker_pub.publish(post_marker);
        }

        cmd_vel_pub.publish(cmd_msg); 
        

    // ★ 動態休眠：精準扣除你上面計算特徵、跑 RF、跑 PF 花掉的時間
    rate.sleep();
    }
    
    return 0;
}
