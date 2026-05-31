// -*- coding: utf-8 -*-
#include "KidnapRecovery.h"
#include <cmath>
#include <algorithm>

// ==============================================================================
// 常數
// ==============================================================================
const float KidnapRecovery::MAP_POSTS[2][2] = {{0.45f, 0.0f}, {-0.45f, 0.0f}};
const float KidnapRecovery::LIDAR_OFFSET    = 0.08f;   // sensor 在中心後方 8cm
const float KidnapRecovery::MIN_DIST        = 0.05f;
const float KidnapRecovery::MAX_DIST        = 1.50f;

const int   KidnapRecovery::N_THETA    = 36;     // 10° 步距
const float KidnapRecovery::ZONE_X_MIN = 0.45f;  // |rx| 下限
const float KidnapRecovery::ZONE_X_MAX = 0.65f;  // |rx| 上限
const float KidnapRecovery::ZONE_Y_MIN = 0.0f;   // ry 下限（不容許球門線後方，砍掉不可能擺放位置）
const float KidnapRecovery::ZONE_Y_MAX = 0.70f;  // ry 上限

// bearing 雜訊遠小於 distance（Kose et al. 2006 實測結論）→ σ_φ < σ_r
const float KidnapRecovery::SIGMA_R   = 0.10f;   // 距離殘差標準差 (m)
const float KidnapRecovery::SIGMA_PHI = 0.05f;   // 方位角殘差標準差 (rad ≈ 2.9°)

const float KidnapRecovery::PRUNE_MARGIN    = 5.0f;
const int   KidnapRecovery::SURVIVOR_THRESH = 2;
const int   KidnapRecovery::CONVERGE_FRAMES = 8;

// ==============================================================================
KidnapRecovery::KidnapRecovery()
    : post_side_(0), stage_(RecoveryStage::SETTLE),
      converge_count_(0), spin_accum_(0.0f) {}

void KidnapRecovery::reset(int post_side)
{
    post_side_      = post_side;
    stage_          = RecoveryStage::SETTLE;
    hyps_.clear();
    converge_count_ = 0;
    spin_accum_     = 0.0f;
}

// ------------------------------------------------------------------------------
// 由單柱觀測 obs 與假設朝向 theta，反推機器人「中心」全域座標。
//   先由 obs 求 sensor 全域位置 S，再補回 +8cm（sensor 在中心後方）得中心。
//     dx = ox·cosθ − oy·sinθ ; dy = ox·sinθ + oy·cosθ   (dx=Lx−Sx, dy=Ly−Sy)
//     Sx = Lx − dx ; Sy = Ly − dy
//     cx = Sx + OFFSET·cosθ ; cy = Sy + OFFSET·sinθ
// ------------------------------------------------------------------------------
void KidnapRecovery::centerFromObs(const Observation& obs, float theta,
                                   float& cx, float& cy) const
{
    float Lx = MAP_POSTS[post_side_][0];
    float Ly = MAP_POSTS[post_side_][1];
    float c = std::cos(theta), s = std::sin(theta);

    float dx = obs.x * c - obs.y * s;
    float dy = obs.x * s + obs.y * c;
    float Sx = Lx - dx;
    float Sy = Ly - dy;
    cx = Sx + LIDAR_OFFSET * c;
    cy = Sy + LIDAR_OFFSET * s;
}

// ------------------------------------------------------------------------------
// 由假設中心 pose 預測單柱在 sensor 局部座標的觀測。
//   S = (cx − OFFSET·cosθ, cy − OFFSET·sinθ)（sensor 在中心後方）
//   ox =  (Lx−Sx)·cosθ + (Ly−Sy)·sinθ
//   oy = −(Lx−Sx)·sinθ + (Ly−Sy)·cosθ
// ------------------------------------------------------------------------------
void KidnapRecovery::predictObs(const Hypothesis& h, float& ox, float& oy) const
{
    float Lx = MAP_POSTS[post_side_][0];
    float Ly = MAP_POSTS[post_side_][1];
    float c = std::cos(h.theta), s = std::sin(h.theta);

    float Sx = h.rx - LIDAR_OFFSET * c;
    float Sy = h.ry - LIDAR_OFFSET * s;
    float dx = Lx - Sx;
    float dy = Ly - Sy;
    ox =  dx * c + dy * s;
    oy = -dx * s + dy * c;
}

// ------------------------------------------------------------------------------
// 生成 36 個 θ 假設（單側），zone 過濾後存入 hyps_。
// ------------------------------------------------------------------------------
int KidnapRecovery::generateHypotheses(const Observation& obs)
{
    hyps_.clear();
    converge_count_ = 0;

    float dist = std::sqrt(obs.x * obs.x + obs.y * obs.y);
    if (dist < MIN_DIST || dist > MAX_DIST) return 0;

    for (int k = 0; k < N_THETA; ++k) {
        float theta = -static_cast<float>(M_PI)
                      + k * (2.0f * static_cast<float>(M_PI) / N_THETA);

        float cx, cy;
        centerFromObs(obs, theta, cx, cy);

        // zone 過濾：側別固定，|rx| 與 ry 落在綁架窄帶內
        bool x_ok = (post_side_ == 0)
            ? (cx >= ZONE_X_MIN && cx <= ZONE_X_MAX)
            : (cx <= -ZONE_X_MIN && cx >= -ZONE_X_MAX);
        bool y_ok = (cy >= ZONE_Y_MIN && cy <= ZONE_Y_MAX);
        if (!x_ok || !y_ok) continue;

        Hypothesis h;
        h.theta   = normalizeAngle(theta);
        h.rx      = cx;
        h.ry      = cy;
        h.log_lik = 0.0f;
        h.alive   = true;
        hyps_.push_back(h);
    }
    return aliveCount();
}

void KidnapRecovery::propagate(float dtheta)
{
    for (auto& h : hyps_) {
        if (!h.alive) continue;
        h.theta = normalizeAngle(h.theta + dtheta);
    }
}

void KidnapRecovery::updateLikelihood(const Observation& obs)
{
    float r_obs   = std::sqrt(obs.x * obs.x + obs.y * obs.y);
    float phi_obs = std::atan2(obs.y, obs.x);

    for (auto& h : hyps_) {
        if (!h.alive) continue;
        float ox, oy;
        predictObs(h, ox, oy);
        float r_pred   = std::sqrt(ox * ox + oy * oy);
        float phi_pred = std::atan2(oy, ox);

        float dr   = r_obs - r_pred;
        float dphi = normalizeAngle(phi_obs - phi_pred);

        h.log_lik += -0.5f * (dr * dr / (SIGMA_R * SIGMA_R)
                            + dphi * dphi / (SIGMA_PHI * SIGMA_PHI));
    }
}

void KidnapRecovery::prune()
{
    float best = -1e30f;
    for (const auto& h : hyps_)
        if (h.alive && h.log_lik > best) best = h.log_lik;
    if (best <= -1e29f) return;

    for (auto& h : hyps_)
        if (h.alive && h.log_lik < best - PRUNE_MARGIN)
            h.alive = false;
}

bool KidnapRecovery::hasHypotheses() const
{
    return aliveCount() > 0;
}

int KidnapRecovery::aliveCount() const
{
    int n = 0;
    for (const auto& h : hyps_) if (h.alive) ++n;
    return n;
}

bool KidnapRecovery::getBest(float& out_x, float& out_y, float& out_theta) const
{
    float best = -1e30f;
    const Hypothesis* bp = nullptr;
    for (const auto& h : hyps_)
        if (h.alive && h.log_lik > best) { best = h.log_lik; bp = &h; }
    if (!bp) return false;
    out_x = bp->rx; out_y = bp->ry; out_theta = bp->theta;
    return true;
}

bool KidnapRecovery::getMedian(float& out_x, float& out_y, float& out_theta) const
{
    std::vector<const Hypothesis*> alive;
    for (const auto& h : hyps_) if (h.alive) alive.push_back(&h);
    if (alive.empty()) return false;
    const Hypothesis* h = alive[alive.size() / 2];
    out_x = h->rx; out_y = h->ry; out_theta = h->theta;
    return true;
}

// ------------------------------------------------------------------------------
// 收斂判定（R-MCL Algorithm 1）：存活數 <= SURVIVOR_THRESH 連續 CONVERGE_FRAMES 幀。
// ------------------------------------------------------------------------------
bool KidnapRecovery::converged(float& out_x, float& out_y, float& out_theta)
{
    int n = aliveCount();
    if (n >= 1 && n <= SURVIVOR_THRESH) {
        converge_count_++;
    } else {
        converge_count_ = 0;
    }

    if (converge_count_ >= CONVERGE_FRAMES)
        return getBest(out_x, out_y, out_theta);

    return false;
}

float KidnapRecovery::normalizeAngle(float a)
{
    while (a >  static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
    return a;
}
