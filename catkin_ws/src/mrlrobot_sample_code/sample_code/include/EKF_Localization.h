#ifndef EKF_LOCALIZATION_H
#define EKF_LOCALIZATION_H

#include <vector>
#include <Eigen/Dense>
#include "FeatureExtractor.h"

// ==============================================================================
// EKF 定位器
//
// 狀態向量 μ = [x, y, θ]
// 使用門柱（已知全域座標）作為地標，搭配 odom 做 predict。
// 兩根門柱同時可見時，先做幾何直接解算作為初始化或重定位。
// ==============================================================================

class EKFLocalizer {
public:
    EKFLocalizer();
    EKFLocalizer(float x, float y, float theta);  // 從已知姿態初始化

    // 從兩根門柱觀測幾何直接求解初始姿態
    // 兩根柱都看到才呼叫，回傳是否成功
    bool initFromPosts(const std::vector<Observation>& obs_list);

    // Predict 步驟：odom 運動模型
    void predict(float v, float w, float dt);

    // Update 步驟：門柱觀測修正（含 data association + Mahalanobis gate）
    void update(const std::vector<Observation>& obs_list);

    // 取得當前估計姿態（介面與 Particle_Filter 相同）
    void getEstimate(float& ex, float& ey, float& et) const;

    // 取得 X 軸方差（供 fusion_node 判斷定位是否收斂）
    float getVarianceX() const;

    // 取得上一次 update 的最大 innovation L2 norm（供外部判斷綁架）
    float getLastInnovation() const { return last_innov_norm_; }

    bool isInitialized() const { return initialized_; }

    // 攔截完成後膨脹共變異數矩陣，讓 EKF 重新靠地標收斂
    void inflateCovariance(float factor);

private:
    // 狀態均值 [x, y, θ]
    Eigen::Vector3f mu_;
    // 共變異數矩陣 Σ (3×3)
    Eigen::Matrix3f sigma_;

    bool initialized_;

    // 門柱地圖（全域座標）
    std::vector<Observation> map_landmarks_;

    // --- 雜訊參數（在建構子中設定）---
    float noise_v_;          // 線速度雜訊標準差 (m/s)
    float noise_w_;          // 角速度雜訊標準差 (rad/s)
    float noise_obs_;        // 觀測雜訊標準差 (m)
    float mahal_gate_;       // Mahalanobis gate 閾值（chi-square 2DOF 95% = 5.99）
    float last_innov_norm_;  // 上次 update 的最大 innovation L2 norm

    // 幾何解算（輔助 initFromPosts）
    // obs_a / obs_b：兩觀測；land_a / land_b：對應的全域門柱
    // 回傳 true 且填入 out_x, out_y, out_theta
    bool solveGeometric(const Observation& obs_a, const Observation& land_a,
                        const Observation& obs_b, const Observation& land_b,
                        float& out_x, float& out_y, float& out_theta) const;

    // 對單一地標做 EKF 修正步驟
    void updateOneLandmark(const Observation& obs, const Observation& land);

    // 預測在當前 μ 下，地標 land 應出現的局部座標 h(μ)
    Eigen::Vector2f predictObservation(const Observation& land) const;

    // h(μ) 相對於 μ 的 Jacobian H (2×3)
    Eigen::Matrix<float, 2, 3> observationJacobian(const Observation& land) const;

    static float normalizeAngle(float angle);

    // LiDAR 感測器在機器人中心的偏移量（沿 body +X，即全域後方 8cm）
    static constexpr float LIDAR_OFFSET = 0.08f;
};

#endif // EKF_LOCALIZATION_H
