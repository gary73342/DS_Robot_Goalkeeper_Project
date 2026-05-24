#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include <vector>
#include <random>

#include <FeatureExtractor.h>

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

    // 取得所有粒子（供 RViz 視覺化）
    const std::vector<Particle>& getParticles() const { return particles; }
};

#endif // LOCALIZATION_H