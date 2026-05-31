// -*- coding: utf-8 -*-
//
// EKF_Localization_node.cpp
//
// 狀態機：WAITING → EKF_PRIMARY ⇄ KIDNAP_RECOVERY
//
// EKF_PRIMARY：EKF 主導定位，偵測到綁架跡象時切換至 KIDNAP_RECOVERY。
// KIDNAP_RECOVERY：
//   - 命令機器人原地旋轉（state_flag=1 → fusion_node 發旋轉指令）
//   - posts>=2：幾何解算，連續 N 幀成功後切回
//   - posts==1：KidnapRecovery 假設管理，假設收斂後切回
//   - 超時：強制用最佳假設重置 EKF，切回
//
// 訂閱：
//   /odom             (nav_msgs/Odometry)
//   /posts_local      (Float32MultiArray)   [x1,y1,x2,y2,...]
//   /ball_lidar_local (Float32MultiArray)   [detected, x, y]
//
// 發布：
//   /robot_pose       (Float32MultiArray)   [x, y, theta, var_x, state_flag]
//                     state_flag: 0.0=EKF_PRIMARY, 1.0=KIDNAP_RECOVERY
//   /ball_lidar_world (Float32MultiArray)   [detected, X, Y]
//   /hypothesis_cloud (visualization_msgs/Marker)  假設點雲（RViz 除錯用）

#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <clocale>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Bool.h>
#include <std_msgs/ColorRGBA.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>

#include "EKF_Localization.h"
#include "KidnapRecovery.h"

using namespace std;

// ==============================================================================
// 狀態機定義
// ==============================================================================
enum class LocState { WAITING, EKF_PRIMARY, KIDNAP_RECOVERY };

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

bool interception_done_flag = false;
void interceptionDoneCallback(const std_msgs::Bool::ConstPtr& msg) {
    if (msg->data) interception_done_flag = true;
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

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ekf_localization_node");
    setlocale(LC_ALL, "");
    ros::NodeHandle nh;

    // --- 訂閱 ---
    ros::Subscriber odom_sub  = nh.subscribe<nav_msgs::Odometry>(
        "/odom", 1, odomCallback);
    ros::Subscriber ball_sub  = nh.subscribe<std_msgs::Float32MultiArray>(
        "/ball_lidar_local", 1, ballLocalCallback);
    ros::Subscriber posts_sub = nh.subscribe<std_msgs::Float32MultiArray>(
        "/posts_local", 1, postsCallback);
    ros::Subscriber intercept_sub = nh.subscribe<std_msgs::Bool>(
        "/interception_done", 1, interceptionDoneCallback);

    // --- 發布 ---
    ros::Publisher robot_pose_pub = nh.advertise<std_msgs::Float32MultiArray>("/robot_pose", 1);
    ros::Publisher ball_world_pub = nh.advertise<std_msgs::Float32MultiArray>("/ball_lidar_world", 1);
    ros::Publisher hyp_cloud_pub  = nh.advertise<visualization_msgs::Marker>("/hypothesis_cloud", 1);

    // --- 定位器與綁架恢復器初始化 ---
    EKFLocalizer   ekf;
    KidnapRecovery kr;

    // --- 狀態機 ---
    LocState state = LocState::WAITING;

    // ★ 調教區
    const float INNOV_THRESHOLD       = 0.30f;  // innovation L2 norm > 此值視為異常
    const int   KIDNAP_COUNT          = 15;     // 條件A：連續 N 幀觸發（750ms@20Hz）
    // 條件B sliding window：最近 60 幀中有 ≥40 幀 posts<2 就觸發
    // （避免「偶爾一幀看到兩柱就清零」的脆弱性，60 ≈ 3s @ 20Hz）
    const int   POSTS_LOW_WINDOW      = 60;
    const int   POSTS_LOW_THRESH      = 40;
    const float GOODX_VAR_THRESH      = 0.05f;  // var_x 低於此值才更新 last_good_x（可信）
    const float SETTLE_DURATION       = 5.0f;   // SETTLE 落地等待（秒）
    const float INTERCEPT_P_INFLATE   = 5.0f;   // 攔截完成後 P 矩陣膨脹係數
    // 純單柱旋轉無法 prune 假設（數學上不可能，論文也驗證），所以 SPIN 上限不再用 360°。
    // 180° 已足夠覆蓋所有「前後」朝向避開機械遮蔽，~8s @ KIDNAP_SPIN_SPEED=0.4 rad/s。
    const float SPIN_MAX_ANGLE        = static_cast<float>(M_PI);
    // 切回 sliding window：最近 20 幀中有 ≥11 幀 init 成功就切回 EKF
    // （perception 偶有單幀雜訊，連續 20 幀太脆弱，超過一半已足夠）
    const int   RETURN_WINDOW         = 20;
    const int   RETURN_THRESH         = 11;
    const float RETURN_TIMEOUT        = 20.0f;  // RETURN 逾時未切回 → HALT（疑似收斂到錯位）

    int       kidnap_frames     = 0;
    std::deque<bool> posts_low_window;     // 條件B sliding window
    std::deque<bool> return_succ_window;   // RETURN/HALT 切回 sliding window
    std::deque<bool> halt_posts_window;    // HALT 單柱監控 sliding window
    float     last_good_x       = 0.0f;  // 最後一次可信的 EKF x（綁架前），決定恢復側別
    bool      last_good_x_valid = false;
    ros::Time settle_start_time;
    ros::Time return_start_time;         // 進入 RETURN 的時間（超時判斷用）

    // --- TF 廣播器 ---
    static tf::TransformBroadcaster tf_broadcaster;

    // --- 迴圈頻率 ---
    const float loop_hz = 20.0f;
    const float dt      = 1.0f / loop_hz;
    ros::Rate   rate(loop_hz);

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
        float w =   latest_odom->twist.twist.angular.z;

        // ------------------------------------------------------------------
        // WAITING：等待兩根門柱同時出現，幾何初始化
        // ------------------------------------------------------------------
        if (state == LocState::WAITING) {
            if (latest_posts.size() >= 2) {
                if (ekf.initFromPosts(latest_posts)) {
                    state         = LocState::EKF_PRIMARY;
                    kidnap_frames = 0;
                    ROS_INFO("[EKF] 初始化完成，切換至 EKF");
                } else {
                    ROS_WARN_THROTTLE(1.0, "[EKF] 幾何解算失敗，重試...");
                }
            } else {
                ROS_INFO_THROTTLE(2.0, "[EKF] 等待門柱...目前可見 %zu 根",
                                  latest_posts.size());
            }
            rate.sleep();
            continue;
        }

        // ------------------------------------------------------------------
        // 攔截完成信號：膨脹 P 矩陣，讓 EKF 重新靠地標觀測收斂
        // ------------------------------------------------------------------
        if (interception_done_flag && state == LocState::EKF_PRIMARY) {
            ekf.inflateCovariance(INTERCEPT_P_INFLATE);
            interception_done_flag = false;
            ROS_WARN("[EKF] 收到攔截完成信號，膨脹 P 矩陣 (×%.1f) 重新收斂", INTERCEPT_P_INFLATE);
        }

        // ------------------------------------------------------------------
        // Predict（EKF_PRIMARY 和 KIDNAP_RECOVERY 都需要）
        // ------------------------------------------------------------------
        ekf.predict(v, w, dt);

        // ------------------------------------------------------------------
        // EKF_PRIMARY：EKF 更新 + 綁架偵測
        // ------------------------------------------------------------------
        if (state == LocState::EKF_PRIMARY) {
            ekf.update(latest_posts);

            float ex, ey, et;
            ekf.getEstimate(ex, ey, et);
            float  innov   = ekf.getLastInnovation();
            float  var_now = ekf.getVarianceX();
            size_t n_posts = latest_posts.size();

            // 維護 last_good_x：兩柱可見且 var 低（定位可信）時持續記錄，
            // 作為綁架觸發時判斷被綁到哪一側的依據（避免用已漂移的 ex）
            if (n_posts >= 2 && var_now < GOODX_VAR_THRESH) {
                last_good_x       = ex;
                last_good_x_valid = true;
            }

            // 條件A：innovation 異常高 + 可見門柱不足（連續計）
            bool cond_A = (innov > INNOV_THRESHOLD) && (n_posts < 2);
            if (cond_A) {
                kidnap_frames++;
                ROS_WARN_THROTTLE(0.5,
                    "[EKF] 綁架跡象A：innov=%.3f posts=%zu (%d/%d)",
                    innov, n_posts, kidnap_frames, KIDNAP_COUNT);
            } else {
                kidnap_frames = 0;
            }

            // 條件B：sliding window — 最近 POSTS_LOW_WINDOW 幀有 ≥POSTS_LOW_THRESH 幀 posts<2
            posts_low_window.push_back(n_posts < 2);
            if (static_cast<int>(posts_low_window.size()) > POSTS_LOW_WINDOW)
                posts_low_window.pop_front();
            int posts_low_count = std::count(posts_low_window.begin(),
                                              posts_low_window.end(), true);
            bool cond_B = (static_cast<int>(posts_low_window.size()) >= POSTS_LOW_WINDOW)
                          && (posts_low_count >= POSTS_LOW_THRESH);

            if (kidnap_frames >= KIDNAP_COUNT || cond_B) {
                // 用最後可信 x 的符號決定側別；無可信值時退回當前 ex
                float side_x   = last_good_x_valid ? last_good_x : ex;
                int   post_side = (side_x > 0.0f) ? 0 : 1;
                kr.reset(post_side);                 // 進入 SETTLE
                posts_low_window.clear();
                return_succ_window.clear();
                kidnap_frames     = 0;
                settle_start_time = ros::Time::now();
                state             = LocState::KIDNAP_RECOVERY;
                ROS_WARN("[KIDNAP] ★觸發 last_good_x=%+.2f%s → %s柱側 "
                         "(觸發時ekf_x=%+.2f) innov=%.3f posts=%zu B=%d/%d",
                         side_x, last_good_x_valid ? "" : "(無可信值,用ekf_x)",
                         post_side == 0 ? "右" : "左", ex, innov, n_posts,
                         posts_low_count, POSTS_LOW_THRESH);
            }
        }

        // ------------------------------------------------------------------
        // KIDNAP_RECOVERY：SETTLE → SPIN → RETURN → HALT
        // 不呼叫 ekf.update()，只做 predict（已在上方完成）
        // ------------------------------------------------------------------
        if (state == LocState::KIDNAP_RECOVERY) {
            size_t        n_posts = latest_posts.size();
            RecoveryStage stg     = kr.getStage();

            if (stg == RecoveryStage::SETTLE) {
                // 停住等落地、降漂移；時間到進 SPIN
                float el = (ros::Time::now() - settle_start_time).toSec();
                if (el >= SETTLE_DURATION) {
                    kr.setStage(RecoveryStage::SPIN);
                    ROS_WARN("[KIDNAP] SETTLE 完成（%.1fs），開始旋轉搜尋（%s柱側）",
                             el, kr.getPostSide() == 0 ? "右" : "左");
                }

            } else if (stg == RecoveryStage::SPIN) {
                kr.addSpin(w * dt);

                // 嘗試真兩柱快速路徑：posts>=2 且 initFromPosts 通過 → RETURN
                bool init_ok = false;
                if (n_posts >= 2 && ekf.initFromPosts(latest_posts)) {
                    kr.setStage(RecoveryStage::RETURN);
                    return_succ_window.clear();
                    halt_posts_window.clear();
                    return_start_time = ros::Time::now();
                    ROS_WARN("[KIDNAP] SPIN 看到兩柱，直接重定位 → RETURN");
                    init_ok = true;
                }

                // fallback：未進入 RETURN 時走假設網格路徑
                //   posts==1            → 用單柱生成 / 更新 / prune
                //   posts==0 或 假兩柱   → 只 propagate，不更新似然（觀測不可信）
                if (!init_ok && kr.getStage() == RecoveryStage::SPIN) {
                    if (n_posts == 1) {
                        const Observation& obs = latest_posts[0];
                        if (!kr.hasHypotheses()) {
                            int n = kr.generateHypotheses(obs);
                            ROS_WARN("[KIDNAP] SPIN 生成 %d 個假設（單側遍歷 36 θ）", n);
                        } else {
                            kr.propagate(w * dt);
                            kr.updateLikelihood(obs);
                            kr.prune();
                        }
                        float bx, by, bt;
                        if (kr.converged(bx, by, bt)) {
                            ekf = EKFLocalizer(bx, by, bt);
                            kr.setStage(RecoveryStage::RETURN);
                            return_succ_window.clear();
                            halt_posts_window.clear();
                            return_start_time = ros::Time::now();
                            ROS_WARN("[KIDNAP] ★收斂 θ=%+.2f rx=%+.2f ry=%+.2f 存活=%d "
                                     "→ 重置EKF，進 RETURN", bt, bx, by, kr.aliveCount());
                        }
                    } else {
                        // posts==0 或 posts>=2 但 init 失敗（假兩柱）：
                        // 只推進假設 θ 保持與旋轉同步，不用不可信觀測更新似然
                        if (kr.hasHypotheses()) kr.propagate(w * dt);
                    }
                }

                // SPIN 上限到了（180°）：仍沒看到兩柱 → 拿中間假設當折衷猜測進 RETURN
                // 整段都沒生成過任何假設（從頭到尾沒看到單柱）才退回 HALT
                if (kr.getStage() == RecoveryStage::SPIN
                        && kr.getSpinAccum() >= SPIN_MAX_ANGLE) {
                    float bx, by, bt;
                    if (kr.getMedian(bx, by, bt)) {
                        ekf = EKFLocalizer(bx, by, bt);
                        kr.setStage(RecoveryStage::RETURN);
                        return_succ_window.clear();
                        halt_posts_window.clear();
                        return_start_time = ros::Time::now();
                        ROS_WARN("[KIDNAP] 轉滿 180° 未看到兩柱 → 取中位數假設 "
                                 "θ=%+.2f rx=%+.2f ry=%+.2f 存活=%d → RETURN",
                                 bt, bx, by, kr.aliveCount());
                    } else {
                        kr.setStage(RecoveryStage::HALT);
                        return_succ_window.clear();
                        halt_posts_window.clear();
                        ROS_WARN("[KIDNAP] 轉滿 180° 期間從未生成假設 → HALT");
                    }
                }

            } else {
                // RETURN（弧線導航）或 HALT（停住）：sliding window 切回
                // 最近 RETURN_WINDOW 幀中有 ≥RETURN_THRESH 幀 init 成功就切回 EKF
                bool init_succ = (n_posts >= 2 && ekf.initFromPosts(latest_posts));
                return_succ_window.push_back(init_succ);
                if (static_cast<int>(return_succ_window.size()) > RETURN_WINDOW)
                    return_succ_window.pop_front();
                int return_succ_count = std::count(return_succ_window.begin(),
                                                    return_succ_window.end(), true);

                if (static_cast<int>(return_succ_window.size()) >= RETURN_WINDOW
                        && return_succ_count >= RETURN_THRESH) {
                    float fx, fy, ft;
                    ekf.getEstimate(fx, fy, ft);
                    kidnap_frames = 0;
                    posts_low_window.clear();
                    return_succ_window.clear();
                    halt_posts_window.clear();
                    state         = LocState::EKF_PRIMARY;
                    ROS_WARN("[KIDNAP] ★恢復完成（%d/%d in last %d）切回 EKF | X=%+.2f Y=%+.2f",
                             return_succ_count, RETURN_THRESH, RETURN_WINDOW, fx, fy);
                }

                // RETURN 超時保護：收斂到錯 pose 時會朝錯目標無限繞行 → 逾時進 HALT
                if (kr.getStage() == RecoveryStage::RETURN
                        && (ros::Time::now() - return_start_time).toSec() > RETURN_TIMEOUT) {
                    kr.setStage(RecoveryStage::HALT);
                    return_succ_window.clear();
                    halt_posts_window.clear();
                    ROS_WARN("[KIDNAP] RETURN 超時 %.0fs 未切回 → HALT，持續監控球柱",
                             RETURN_TIMEOUT);
                }

                // HALT 單柱監控：n_posts > 0 達 11/20 幀 → 直接跳 SPIN 重定位
                if (kr.getStage() == RecoveryStage::HALT) {
                    halt_posts_window.push_back(n_posts > 0);
                    if (static_cast<int>(halt_posts_window.size()) > RETURN_WINDOW)
                        halt_posts_window.pop_front();
                    int halt_posts_count = std::count(halt_posts_window.begin(),
                                                      halt_posts_window.end(), true);
                    if (static_cast<int>(halt_posts_window.size()) >= RETURN_WINDOW
                            && halt_posts_count >= RETURN_THRESH) {
                        kr.reset(kr.getPostSide());
                        kr.setStage(RecoveryStage::SPIN);
                        halt_posts_window.clear();
                        return_succ_window.clear();
                        ROS_WARN("[KIDNAP] HALT 見柱（%d/%d）→ 直接跳 SPIN 重定位",
                                 halt_posts_count, RETURN_THRESH);
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // 取得最佳估計與狀態旗標
        // ------------------------------------------------------------------
        float ex, ey, et, var_x;
        ekf.getEstimate(ex, ey, et);
        var_x = ekf.getVarianceX();

        float state_flag = (state == LocState::KIDNAP_RECOVERY) ? 1.0f : 0.0f;

        // ------------------------------------------------------------------
        // 發布機器人全域姿態（5 fields：x, y, theta, var_x, state_flag）
        // ------------------------------------------------------------------
        {
            float recovery_sub = 0.0f;
            if (state == LocState::KIDNAP_RECOVERY) {
                switch (kr.getStage()) {
                    case RecoveryStage::SETTLE: recovery_sub = 0.0f; break;
                    case RecoveryStage::SPIN:   recovery_sub = 1.0f; break;
                    case RecoveryStage::RETURN: recovery_sub = 2.0f; break;
                    case RecoveryStage::HALT:   recovery_sub = 3.0f; break;
                }
            }
            std_msgs::Float32MultiArray pose_msg;
            pose_msg.data = {ex, ey, et, var_x, state_flag, recovery_sub};
            robot_pose_pub.publish(pose_msg);
        }

        // ------------------------------------------------------------------
        // 球局部座標 → 全域座標（仍發布供 fusion；綁架期間不印球，無意義）
        // ------------------------------------------------------------------
        {
            std_msgs::Float32MultiArray ball_msg;
            if (ball_local_detected) {
                float ball_global_x = ex + ball_local_x * cosf(et)
                                         - ball_local_y * sinf(et);
                float ball_global_y = ey + ball_local_x * sinf(et)
                                         + ball_local_y * cosf(et);
                ball_msg.data = {1.0f, ball_global_x, ball_global_y};
            } else {
                ball_msg.data = {0.0f, 0.0f, 0.0f};
            }
            ball_world_pub.publish(ball_msg);
        }

        // ------------------------------------------------------------------
        // 終端機輸出：EKF_PRIMARY 印定位+球；KIDNAP_RECOVERY 印 [KIDNAP] 恢復行
        // ------------------------------------------------------------------
        if (state == LocState::EKF_PRIMARY) {
            if (ball_local_detected) {
                float bgx = ex + ball_local_x * cosf(et) - ball_local_y * sinf(et);
                float bgy = ey + ball_local_x * sinf(et) + ball_local_y * cosf(et);
                ROS_INFO_THROTTLE(0.5,
                    "[EKF] X=%+.2f Y=%.2f θ=%+.3f var=%.4f posts=%zu | 球(%+.2f,%.2f)",
                    ex, ey, et, var_x, latest_posts.size(), bgx, bgy);
            } else {
                ROS_INFO_THROTTLE(0.5,
                    "[EKF] X=%+.2f Y=%.2f θ=%+.3f var=%.4f posts=%zu | 球:無",
                    ex, ey, et, var_x, latest_posts.size());
            }
        } else {  // KIDNAP_RECOVERY
            switch (kr.getStage()) {
                case RecoveryStage::SETTLE: {
                    float el = (ros::Time::now() - settle_start_time).toSec();
                    ROS_WARN_THROTTLE(0.5, "[KIDNAP] SETTLE 落地等待 %.1f/%.1fs",
                                      el, SETTLE_DURATION);
                    break;
                }
                case RecoveryStage::SPIN: {
                    float bx, by, bt;
                    if (kr.getBest(bx, by, bt)) {
                        // // 列出所有存活假設的 (θ, log_lik, rx, ry)，方便判讀 prune 是否生效
                        // std::string alive_str;
                        // for (const auto& h : kr.getHypotheses()) {
                        //     if (!h.alive) continue;
                        //     char buf[64];
                        //     snprintf(buf, sizeof(buf),
                        //         "[θ%+.2f L%+.1f (%+.2f,%+.2f)] ",
                        //         h.theta, h.log_lik, h.rx, h.ry);
                        //     alive_str += buf;
                        // }
                        // ROS_WARN_THROTTLE(0.5,
                        //     "[KIDNAP] SPIN 轉%.0f° | 存活%d posts=%zu | %s",
                        //     kr.getSpinAccum() * 180.0f / M_PI, kr.aliveCount(),
                        //     latest_posts.size(), alive_str.c_str());
                        ROS_WARN_THROTTLE(0.5,
                            "[KIDNAP] SPIN 轉%.0f° | 存活%d best θ=%+.2f posts=%zu",
                            kr.getSpinAccum() * 180.0f / M_PI, kr.aliveCount(),
                            bt, latest_posts.size());
                    } else {
                        ROS_WARN_THROTTLE(0.5,
                            "[KIDNAP] SPIN 轉%.0f° | 尚無假設（等看到單柱）posts=%zu",
                            kr.getSpinAccum() * 180.0f / M_PI, latest_posts.size());
                    }
                    break;
                }
                case RecoveryStage::RETURN: {
                    int rc = std::count(return_succ_window.begin(),
                                        return_succ_window.end(), true);
                    ROS_WARN_THROTTLE(0.5,
                        "[KIDNAP] RETURN 機(%+.2f,%+.2f,θ=%+.2f)→目標x=0 "
                        "兩柱確認(%d/%zu, 達標 %d) posts=%zu",
                        ex, ey, et, rc, return_succ_window.size(),
                        RETURN_THRESH, latest_posts.size());
                    break;
                }
                case RecoveryStage::HALT: {
                    int rc = std::count(return_succ_window.begin(),
                                        return_succ_window.end(), true);
                    ROS_WARN_THROTTLE(1.0,
                        "[KIDNAP] HALT 停止 監看兩柱中(%d/%zu, 達標 %d) posts=%zu",
                        rc, return_succ_window.size(), RETURN_THRESH, latest_posts.size());
                    break;
                }
            }
        }

        // ------------------------------------------------------------------
        // 假設點雲（新恢復策略不使用假設集，發 DELETEALL 清除舊標記）
        // ------------------------------------------------------------------
        {
            visualization_msgs::Marker marker;
            marker.header.stamp    = ros::Time::now();
            marker.header.frame_id = "map";
            marker.ns              = "hypotheses";
            marker.id              = 0;
            marker.action          = visualization_msgs::Marker::DELETEALL;
            marker.pose.orientation.w = 1.0;
            hyp_cloud_pub.publish(marker);
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

            float ct = et - static_cast<float>(odom_yaw);
            float cx = ex - cosf(ct) * odom_x + sinf(ct) * odom_y;
            float cy = ey - sinf(ct) * odom_x - cosf(ct) * odom_y;

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
