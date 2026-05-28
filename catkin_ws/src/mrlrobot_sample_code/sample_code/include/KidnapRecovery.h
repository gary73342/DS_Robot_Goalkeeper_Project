#ifndef KIDNAP_RECOVERY_H
#define KIDNAP_RECOVERY_H

#include <vector>
#include <cmath>
#include "FeatureExtractor.h"  // Observation

// ==============================================================================
// KidnapRecovery — 綁架恢復（θ 假設網格 / R-MCL 簡化版）
//
// 靈感來源：Kose et al., "Comparison of Localization Methods for a Robot Soccer
// Team", IJARS Vol.3 No.4 (2006) 的 R-MCL（Reverse Monte Carlo Localization）。
// 本實作大幅簡化以套用守門員場地特性：
//   - 場地左右對稱（門柱 ±0.45,0），單柱觀測有不可分辨的鏡像問題。
//     故先用綁架前的 last_good_x 符號鎖定「哪一側」（由 Node 傳入 post_side），
//     只在該側遍歷 θ 假設，從源頭排除鏡像 → 才能收斂。
//   - 狀態空間 (x,y,θ) 降維：固定側別後只搜 θ（36 個，10° 步距），
//     每個 θ 由幾何唯一決定機器人中心 (rx,ry)，是確定性三元組而非粒子群。
//   - 收斂判據沿用論文 R-MCL Algorithm 1：逐幀淘汰低似然假設，
//     當「存活假設數」降到門檻並持續數幀 → 視為收斂（max_grid_array < th_ML）。
//   - 似然加權沿用論文實測結論：bearing(方位角) 雜訊遠小於 distance(距離)，
//     故 σ_bearing < σ_range，方位角權重較大。
//
// 子狀態：
//   SETTLE  觸發後停住等落地、降漂移（fusion 不動）
//   SPIN    慢速原地旋轉（繞機器人中心），逐幀累積假設似然直到收斂
//   RETURN  收斂後重置 EKF，弧線導航回場（fusion go-to-goal 朝 x=0）
//   HALT    轉滿一圈仍未收斂 → 停住（fusion 不動）
//
// LiDAR 偏移：感測器位於機器人「中心後方」8cm（本體座標 (-0.08, 0)，
//   forward = +local x = 全域 -X）。旋轉繞中心 → 中心固定、sensor 繞 8cm 圓。
//   假設追蹤「機器人中心」，觀測預測時才換算到 sensor 位置。
// ==============================================================================

enum class RecoveryStage { SETTLE, SPIN, RETURN, HALT };

struct Hypothesis {
    float theta;     // 機器人中心朝向（全域，rad）
    float rx, ry;    // 機器人中心全域座標（旋轉中固定）
    float log_lik;   // 累積對數似然
    bool  alive;     // 是否仍存活（未被淘汰）
};

class KidnapRecovery {
public:
    KidnapRecovery();

    // 進入 RECOVERY 時呼叫：設定使用哪根柱子（由 last_good_x 符號決定），
    // 清空假設、重置為 SETTLE。
    // post_side: 0=右柱(+0.45,0)，1=左柱(-0.45,0)
    void reset(int post_side);

    // SPIN 第一個有效幀：用單柱觀測生成 36 個 θ 假設（單側），
    // 經 zone 過濾後存入 hyps_。回傳生成的存活假設數。
    int generateHypotheses(const Observation& obs);

    // 逐幀推進：所有存活假設的 θ 加上 odom 旋轉增量（中心位置不動）。
    void propagate(float dtheta);

    // 逐幀更新：用單柱觀測對每個存活假設累積對數似然（含 sensor 偏移建模）。
    void updateLikelihood(const Observation& obs);

    // 淘汰：刪掉 log_lik 落後最佳超過 PRUNE_MARGIN 的假設。
    void prune();

    // 是否已生成假設（generateHypotheses 是否已被呼叫且有存活假設）
    bool hasHypotheses() const;

    // 存活假設數
    int aliveCount() const;

    // 收斂判定（R-MCL 法）：存活數 <= SURVIVOR_THRESH 連續 converge_frames_ 幀。
    // 收斂時回傳 true 並填入最佳假設的中心 pose。
    // 此函式每幀呼叫一次（內部維護連續幀計數）。
    bool converged(float& out_x, float& out_y, float& out_theta);

    // 取得目前最佳假設（最高 log_lik），供日誌輸出。回傳是否有存活假設。
    bool getBest(float& out_x, float& out_y, float& out_theta) const;

    // 累積旋轉量（rad，絕對值），供 Node 判斷是否轉滿一圈。
    void  addSpin(float dtheta) { spin_accum_ += std::abs(dtheta); }
    float getSpinAccum() const  { return spin_accum_; }

    // 子狀態存取
    RecoveryStage getStage()          const { return stage_; }
    void          setStage(RecoveryStage s) { stage_ = s; }
    int           getPostSide()       const { return post_side_; }

private:
    int                     post_side_;
    RecoveryStage           stage_;
    std::vector<Hypothesis> hyps_;
    int                     converge_count_;  // 收斂條件連續滿足的幀數
    float                   spin_accum_;       // 累積旋轉量

    // 由單柱觀測 obs 與假設朝向 theta，反推機器人「中心」全域座標。
    // 含 +8cm 偏移補回（sensor → center）。
    void centerFromObs(const Observation& obs, float theta,
                       float& cx, float& cy) const;

    // 由假設中心 pose 預測單柱在 sensor 局部座標的觀測（含 sensor 偏移）。
    void predictObs(const Hypothesis& h, float& ox, float& oy) const;

    static float normalizeAngle(float a);

    // --- 地圖與幾何常數 ---
    static const float MAP_POSTS[2][2];   // [0]=右(+0.45,0)，[1]=左(-0.45,0)
    static const float LIDAR_OFFSET;      // sensor 在中心後方距離 (m)，本體 (-OFFSET,0)
    static const float MIN_DIST;          // 觀測距離下限（過近視為雜訊，m）
    static const float MAX_DIST;          // 觀測距離上限（m）

    // --- 假設網格參數 ---
    static const int   N_THETA;           // θ 候選數（36，10° 步距）
    static const float ZONE_X_MIN;        // |rx| 下限
    static const float ZONE_X_MAX;        // |rx| 上限
    static const float ZONE_Y_MIN;        // ry 下限
    static const float ZONE_Y_MAX;        // ry 上限

    // --- 似然參數（bearing 權重大於 distance，本論文實測結論）---
    static const float SIGMA_R;           // 距離殘差標準差 (m)
    static const float SIGMA_PHI;         // 方位角殘差標準差 (rad)

    // --- 淘汰 / 收斂參數 ---
    static const float PRUNE_MARGIN;      // log_lik 落後最佳超過此值即淘汰
    static const int   SURVIVOR_THRESH;   // 存活數 <= 此值視為收斂候選
    static const int   CONVERGE_FRAMES;   // 收斂條件需連續滿足的幀數
};

#endif // KIDNAP_RECOVERY_H
