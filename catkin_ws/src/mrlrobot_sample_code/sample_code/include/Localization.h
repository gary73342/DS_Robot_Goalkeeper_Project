#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include <vector>
#include <random>

// 局部座標結構 (光達觀測到的物體座標)
struct Observation {
    float x; // 向前 (m)
    float y; // 向左 (m)
};

// 粒子結構
struct Particle {
    float x;
    float y;
    float theta;
    float weight;
};

class Localizer {
private:
    int num_particles;
    std::vector<Particle> particles;
    std::vector<Observation> map_landmarks; // 全域地圖中的門柱位置
    std::default_random_engine gen;

    // 運動模型雜訊參數 (根據你底盤的打滑程度調整)
    float noise_v; 
    float noise_w; 
    float noise_obs; // 觀測雜訊 (光達測距誤差)

    // 內部方法宣告
    void normalizeAndResample();
    float computeEffectiveParticles(); // 計算有效粒子數

public:
    // 建構子宣告
    Localizer(int n);

    // 運動模型更新 (Predict)
    void predict(float v, float w, float dt);

    // 觀測模型更新 (Update)
    void update(const std::vector<Observation>& obs_list);

    // 取得當前最高置信度的估計姿態
    void getEstimate(float& ex, float& ey, float& et);
};

#endif // LOCALIZATION_H