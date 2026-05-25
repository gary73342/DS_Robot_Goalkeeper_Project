#include "Particle_Filter.h"
#include <cmath>
#include <numeric>
#include <iostream>

using namespace std;

// 建構子實作
Localizer::Localizer(int n) : num_particles(n) {
    // 1. 假設球門在Y軸正向， 寬度 1 m
    map_landmarks = {{0.45f, 0.0f}, {-0.45f, 0.0f}};


    // 2. 系統雜訊參數 (強烈建議依據實車情況調校)
    noise_v = 0.04f;   // 速度雜訊 (m/s)
    noise_w = 0.1f;    // 角速度雜訊 (rad/s)
    noise_obs = 0.15f; // 觀測距離雜訊 (m)

    // 3. 均勻散佈初始化
    random_device rd;
    gen.seed(rd());
    // 粒子初始化：機器人在 Y=0.5 附近，X 在 ±0.45 之間
    uniform_real_distribution<float> dist_x(-0.45f, 0.45f);
    uniform_real_distribution<float> dist_y(0.3f, 0.7f);    // 0.5m ± 0.2m
    uniform_real_distribution<float> dist_theta(-0.2f, 0.2f);

    float init_weight = 1.0f / num_particles;
    for(int i = 0; i < num_particles; ++i) {
        particles.push_back({dist_x(gen), dist_y(gen), dist_theta(gen), init_weight});
    }
}

// Predict 階段：依據 Odometry 推算 (升級側滑模型)
void Localizer::predict(float v, float w, float dt) {
    // 考慮速度本身的隨機誤差 (前後打滑與轉向打滑)
    normal_distribution<float> dist_v(0.0, noise_v);
    normal_distribution<float> dist_w(0.0, noise_w);
    
    // 守門員專屬升級：側向滑移雜訊 (Lateral Slip Noise)
    // 模擬機器人急停倒車時，底盤發生的微小橫向漂移
    normal_distribution<float> dist_vy(0.0, 0.02f); 

    for(auto& p : particles) {
        float noisy_v = v + dist_v(gen);  // 實際前後走的速度
        float noisy_w = w + dist_w(gen);  // 實際轉動的速度
        float noisy_vy = dist_vy(gen);    // 偷偷側滑的速度！

        // 升級版運動學模型：將 X 軸前進與 Y 軸側滑同時投影到全域座標
        p.x += (noisy_v * cos(p.theta) - noisy_vy * sin(p.theta)) * dt;
        p.y += (noisy_v * sin(p.theta) + noisy_vy * cos(p.theta)) * dt;
        p.theta += noisy_w * dt;

        // 保持角度在 -PI 到 PI 之間
        while (p.theta > M_PI) p.theta -= 2.0 * M_PI;
        while (p.theta < -M_PI) p.theta += 2.0 * M_PI;
    }
}

// Update 階段
void Localizer::update(const std::vector<Observation>& obs_list) {
    // 防呆：如果這一幀沒看到任何門柱，就只依靠預測，不更新權重
    if (obs_list.empty()) return;

    float weight_sum = 0.0f;

    for (auto& p : particles) {
        float prob = 1.0f; // 該粒子的聯合機率

        // 物理邊界：機器人不可能跑到防守底線後方（Y<0）
        // 這條約束消除鏡像歧義：單根門柱時，Y<0 的對稱粒子群會被淘汰
        if (p.y < 0.0f) {
            p.weight *= 1e-6f;
            continue;
        }

        for (const auto& obs : obs_list) {
            // 將光達測到的門柱，依據這個粒子的「假想姿態」投影到全域地圖
            float gx_pred = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
            float gy_pred = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);

            // 資料關聯 (Data Association)：最近鄰居法 (Nearest Neighbor)
            // 尋找地圖上距離這個假想門柱最近的真實門柱
            float min_dist_sq = 1e6;
            for (const auto& land : map_landmarks) {
                float dist_sq = pow(gx_pred - land.x, 2) + pow(gy_pred - land.y, 2);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                }
            }
            // 如果這個假想門柱離地圖上真實門柱太遠 (超過 0.6 公尺)，代表此粒子位置錯誤，
            // 給予重懲罰讓它在重採樣時被淘汰。不可 continue 跳過（否則壞粒子 weight 不降，
            // 永遠無法收斂）。
            if (min_dist_sq > pow(0.25f, 2)) {
                prob *= 1e-4f;
                continue;
            }

            // 使用高斯分佈計算似然度 (Likelihood)
            // 加上一個極小的常數，防止似然度降為 0 導致後續計算崩潰
            float likelihood = exp(-min_dist_sq / (2.0f * pow(noise_obs, 2))) + 1e-4f;
            prob *= likelihood;
        }
        
        p.weight *= prob;
        weight_sum += p.weight;
    }

    // 正規化權重
    if (weight_sum > 0) {
        for (auto& p : particles) {
            p.weight /= weight_sum;
        }
    } else {
        // 例外處理：如果所有粒子都離觀測結果太遠，平均分配權重 (避免崩潰)
        float uniform_w = 1.0f / num_particles;
        for (auto& p : particles) p.weight = uniform_w;
    }

    // ★ 實務關鍵：判斷是否需要重採樣
    float n_eff = computeEffectiveParticles();
    // 只有當有效粒子數低於總數的一半時，才執行重採樣
    if (n_eff < (num_particles / 2.0f)) {
        normalizeAndResample();
    }
}

// 計算有效粒子數 (N_eff = 1 / sum(weight^2))
float Localizer::computeEffectiveParticles() {
    float sum_sq_weights = 0.0f;
    for (const auto& p : particles) {
        sum_sq_weights += p.weight * p.weight;
    }
    return 1.0f / (sum_sq_weights + 1e-9f);
}

// 重採樣邏輯：真正的低方差重採樣 (Low Variance Resampling)
void Localizer::normalizeAndResample() {
    std::vector<Particle> new_particles;
    new_particles.reserve(num_particles); // 效能優化：預先配置記憶體空間

    // 1. 產生一個 0 到 (1/N) 之間的隨機起點 r
    std::uniform_real_distribution<float> dist(0.0f, 1.0f / num_particles);
    float r = dist(gen);
    
    // 2. 梳子演算法初始化
    float c = particles[0].weight;
    int i = 0;
    float uniform_w = 1.0f / num_particles;

    // 3. 用等間距的梳子 (m/N) 一次刮過所有粒子
    for (int m = 0; m < num_particles; ++m) {
        float U = r + static_cast<float>(m) / num_particles; // 梳齒的位置
        
        // 尋找對應的權重區間 (確保 i 不會越界)
        while (U > c && i < num_particles - 1) {
            i++;
            c += particles[i].weight;
        }
        
        // 抽中該粒子，複製並重置權重
        Particle p_new = particles[i];
        p_new.weight = uniform_w;
        new_particles.push_back(p_new);
    }
    
    particles = new_particles;
}

// 取得粒子雲在 X 軸的加權方差
float Localizer::getVarianceX() {
    float ex = 0.0f;
    for (const auto& p : particles) ex += p.weight * p.x;
    float var = 0.0f;
    for (const auto& p : particles) var += p.weight * (p.x - ex) * (p.x - ex);
    return var;
}

// 取得估計值
void Localizer::getEstimate(float& ex, float& ey, float& et) {
    ex = 0.0f; ey = 0.0f; 
    float s_theta = 0.0f, c_theta = 0.0f;
    
    for (const auto& p : particles) {
        // 利用權重進行加權平均，比直接平均更精準
        ex += p.weight * p.x;
        ey += p.weight * p.y;
        s_theta += p.weight * sin(p.theta);
        c_theta += p.weight * cos(p.theta);
    }
    
    et = atan2(s_theta, c_theta);
}
