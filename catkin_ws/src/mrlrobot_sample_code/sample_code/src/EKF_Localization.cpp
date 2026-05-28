// -*- coding: utf-8 -*-
#include "EKF_Localization.h"
#include <cmath>
#include <iostream>

using namespace Eigen;

// ==============================================================================
// 建構子
// ==============================================================================

EKFLocalizer::EKFLocalizer()
    : initialized_(false)
{
    map_landmarks_ = { {-0.45f, 0.0f}, {0.45f, 0.0f} };
    noise_v_    = 0.04f;
    noise_w_    = 0.10f;
    noise_obs_  = 0.20f;
    mahal_gate_ = 5.99f;
    mu_              = Vector3f::Zero();
    sigma_           = Matrix3f::Identity() * 1000.0f;
    last_innov_norm_ = 0.0f;
}

EKFLocalizer::EKFLocalizer(float x, float y, float theta)
    : initialized_(true)
{
    map_landmarks_ = { {-0.45f, 0.0f}, {0.45f, 0.0f} };
    noise_v_    = 0.04f;
    noise_w_    = 0.10f;
    noise_obs_  = 0.20f;
    mahal_gate_ = 5.99f;
    mu_ << x, y, normalizeAngle(theta);
    sigma_           = Matrix3f::Identity() * 0.50f;
    last_innov_norm_ = 0.0f;
}

// ==============================================================================
// 從兩根門柱幾何直接解算初始姿態
// ==============================================================================

bool EKFLocalizer::initFromPosts(const std::vector<Observation>& obs_list)
{
    if (obs_list.size() < 2) return false;

    const Observation& o0 = obs_list[0];
    const Observation& o1 = obs_list[1];
    const Observation& lL = map_landmarks_[0]; // (-0.45, 0)
    const Observation& lR = map_landmarks_[1]; // (+0.45, 0)

    float rx, ry, rt;

    // 嘗試兩種 data association，取物理合法的那種
    // 合法條件：|rx| <= 0.5m，ry 在 [0, 2.0]m 之間
    auto isValid = [](float x, float y) {
        return std::fabs(x) <= 0.50f && y >= -0.10f && y <= 2.0f;
    };

    // Assignment A：obs[0] → 左柱，obs[1] → 右柱
    if (solveGeometric(o0, lL, o1, lR, rx, ry, rt) && isValid(rx, ry)) {
        mu_ << rx, ry, rt;
        sigma_ = Matrix3f::Identity() * 0.30f; // 幾何解算後仍有光達雜訊，不宜過小
        initialized_ = true;
        std::cout << "[EKF] 幾何初始化成功 (A→L,B→R): X=" << rx
                  << " Y=" << ry << " θ=" << rt << std::endl;
        return true;
    }

    // Assignment B：obs[0] → 右柱，obs[1] → 左柱
    if (solveGeometric(o0, lR, o1, lL, rx, ry, rt) && isValid(rx, ry)) {
        mu_ << rx, ry, rt;
        sigma_ = Matrix3f::Identity() * 0.30f;
        initialized_ = true;
        std::cout << "[EKF] 幾何初始化成功 (A→R,B→L): X=" << rx
                  << " Y=" << ry << " θ=" << rt << std::endl;
        return true;
    }

    return false;
}

// ==============================================================================
// 幾何直接解算（私有）
// ==============================================================================

bool EKFLocalizer::solveGeometric(const Observation& obs_a, const Observation& land_a,
                                   const Observation& obs_b, const Observation& land_b,
                                   float& out_x, float& out_y, float& out_theta) const
{
    // 兩門柱在世界座標的差值
    float dX = land_b.x - land_a.x; // 通常 = 0.9
    float dY = land_b.y - land_a.y; // 通常 = 0

    // 兩觀測在局部座標的差值
    float dx = obs_b.x - obs_a.x;
    float dy = obs_b.y - obs_a.y;

    // 防止除以零
    float denom = std::sqrt(dx * dx + dy * dy);
    if (denom < 1e-4f) return false;

    // 由聯立方程組解 θ：
    //   dX = dx*cosθ - dy*sinθ
    //   dY = dx*sinθ + dy*cosθ
    // → θ = atan2(dX*dy - dY*dx, dX*dx + dY*dy) 後取解
    out_theta = std::atan2(dX * dy - dY * dx,
                            dX * dx + dY * dy);

    // 注意：上式等價於 atan2(-dy, dx) 只在 dY=0 時成立
    // 使用更通用的旋轉矩陣反解：
    // [dX]   [dx  -dy] [cosθ]
    // [dY] = [dy   dx] [sinθ]
    // 解：cosθ = (dX*dx + dY*dy) / denom²
    //     sinθ = (dX*dy - dY*dx) / ... (已由 atan2 處理)

    float cosT = std::cos(out_theta);
    float sinT = std::sin(out_theta);

    // 由 obs_a 解算機器人位置
    float rx_a = land_a.x - obs_a.x * cosT + obs_a.y * sinT;
    float ry_a = land_a.y - obs_a.x * sinT - obs_a.y * cosT;

    // 由 obs_b 解算機器人位置（交叉驗證）
    float rx_b = land_b.x - obs_b.x * cosT + obs_b.y * sinT;
    float ry_b = land_b.y - obs_b.x * sinT - obs_b.y * cosT;

    // 殘差太大代表 data association 錯誤
    float residual = std::hypot(rx_a - rx_b, ry_a - ry_b);
    if (residual > 0.30f) return false;

    out_x = (rx_a + rx_b) * 0.5f;
    out_y = (ry_a + ry_b) * 0.5f;
    out_theta = normalizeAngle(out_theta);
    return true;
}

// ==============================================================================
// Predict 步驟
// ==============================================================================

void EKFLocalizer::predict(float v, float w, float dt)
{
    float x  = mu_(0);
    float y  = mu_(1);
    float th = mu_(2);

    // 運動模型（與 Particle_Filter 相同）
    mu_(0) = x  + v * std::cos(th) * dt;
    mu_(1) = y  + v * std::sin(th) * dt;
    mu_(2) = normalizeAngle(th + w * dt);

    // Jacobian F（3×3）
    Matrix3f F = Matrix3f::Identity();
    F(0, 2) = -v * std::sin(th) * dt;
    F(1, 2) =  v * std::cos(th) * dt;

    // 過程雜訊 Q（簡化對角形式）
    Matrix3f Q = Matrix3f::Zero();
    Q(0, 0) = noise_v_ * noise_v_ * dt;
    Q(1, 1) = noise_v_ * noise_v_ * dt;
    Q(2, 2) = noise_w_ * noise_w_ * dt;

    sigma_ = F * sigma_ * F.transpose() + Q;
}

// ==============================================================================
// Update 步驟：data association + 逐柱 EKF 修正
// ==============================================================================

void EKFLocalizer::update(const std::vector<Observation>& obs_list)
{
    if (obs_list.empty()) return;

    last_innov_norm_ = 0.0f;

    int n_land = static_cast<int>(map_landmarks_.size()); // = 2
    int n_obs  = static_cast<int>(obs_list.size());

    // 對每個觀測找最近（Mahalanobis）地標，避免重複配對
    std::vector<bool> land_used(n_land, false);

    for (int i = 0; i < n_obs; ++i) {
        float  best_dist = mahal_gate_; // 超過 gate 則丟棄
        int    best_j    = -1;

        for (int j = 0; j < n_land; ++j) {
            if (land_used[j]) continue;

            Vector2f z_hat = predictObservation(map_landmarks_[j]);
            Matrix<float, 2, 3> H = observationJacobian(map_landmarks_[j]);
            Matrix2f R = Matrix2f::Identity() * (noise_obs_ * noise_obs_);
            Matrix2f S = H * sigma_ * H.transpose() + R;

            Vector2f innov;
            innov(0) = obs_list[i].x - z_hat(0);
            innov(1) = obs_list[i].y - z_hat(1);

            float d2 = innov.transpose() * S.inverse() * innov;
            if (d2 < best_dist) {
                best_dist = d2;
                best_j    = j;
            }
        }

        if (best_j >= 0) {
            land_used[best_j] = true;
            updateOneLandmark(obs_list[i], map_landmarks_[best_j]);
        }
    }
}

// ==============================================================================
// 單一地標 EKF 修正
// ==============================================================================

void EKFLocalizer::updateOneLandmark(const Observation& obs, const Observation& land)
{
    Vector2f z_hat = predictObservation(land);
    Matrix<float, 2, 3> H = observationJacobian(land);
    Matrix2f R = Matrix2f::Identity() * (noise_obs_ * noise_obs_);

    Matrix2f S = H * sigma_ * H.transpose() + R;
    Matrix<float, 3, 2> K = sigma_ * H.transpose() * S.inverse();

    Vector2f innov;
    innov(0) = obs.x - z_hat(0);
    innov(1) = obs.y - z_hat(1);

    float norm = innov.norm();
    if (norm > last_innov_norm_) last_innov_norm_ = norm;

    mu_    = mu_ + K * innov;
    sigma_ = (Matrix3f::Identity() - K * H) * sigma_;

    mu_(2) = normalizeAngle(mu_(2));
}

// ==============================================================================
// 預測觀測值 h(μ)：地標在機器人局部座標的預測位置
// ==============================================================================

Vector2f EKFLocalizer::predictObservation(const Observation& land) const
{
    float dx = land.x - mu_(0);
    float dy = land.y - mu_(1);
    float th = mu_(2);

    Vector2f z_hat;
    z_hat(0) =  dx * std::cos(th) + dy * std::sin(th); // 局部 x（前方）
    z_hat(1) = -dx * std::sin(th) + dy * std::cos(th); // 局部 y（左方）
    return z_hat;
}

// ==============================================================================
// 觀測 Jacobian H（2×3）
// ==============================================================================

Matrix<float, 2, 3> EKFLocalizer::observationJacobian(const Observation& land) const
{
    float dx = land.x - mu_(0);
    float dy = land.y - mu_(1);
    float th = mu_(2);
    float cosT = std::cos(th);
    float sinT = std::sin(th);

    Matrix<float, 2, 3> H;
    // ∂h/∂x  ∂h/∂y  ∂h/∂θ
    H(0, 0) = -cosT;
    H(0, 1) = -sinT;
    H(0, 2) = -dx * sinT + dy * cosT;

    H(1, 0) =  sinT;
    H(1, 1) = -cosT;
    H(1, 2) = -dx * cosT - dy * sinT;

    return H;
}

// ==============================================================================
// 公開介面
// ==============================================================================

void EKFLocalizer::getEstimate(float& ex, float& ey, float& et) const
{
    ex = mu_(0);
    ey = mu_(1);
    et = mu_(2);
}

float EKFLocalizer::getVarianceX() const
{
    return sigma_(0, 0); // x 的方差
}

float EKFLocalizer::normalizeAngle(float angle)
{
    while (angle >  M_PI) angle -= 2.0f * M_PI;
    while (angle < -M_PI) angle += 2.0f * M_PI;
    return angle;
}
