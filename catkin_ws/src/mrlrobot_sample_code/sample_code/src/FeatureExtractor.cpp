#include "FeatureExtractor.h"
#include <cmath>
#include <numeric>

using namespace std;

vector<Segment> FeatureExtractor::extractSegmentsFromScan(const sensor_msgs::LaserScan::ConstPtr& scan_msg) {
    vector<Point2D> valid_points;
    vector<Segment> result_segments;

    // 1. 座標轉換與無效點過濾
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
        float r = scan_msg->ranges[i];
        if (std::isnan(r) || std::isinf(r) || r < 0.01f) {
            continue; // 過濾 NaN 與 0
        }
        
        float angle = scan_msg->angle_min + i * scan_msg->angle_increment;
        Point2D pt;
        pt.x = r * cos(angle);
        pt.y = r * sin(angle);
        pt.r = r;
        valid_points.push_back(pt);
    }

    if (valid_points.empty()) return result_segments;

    // 2. 分群 (Segmentation) - 依據歐氏距離 0.05m
    vector<vector<Point2D>> clusters;
    vector<Point2D> current_cluster;
    current_cluster.push_back(valid_points[0]);

    for (size_t i = 1; i < valid_points.size(); ++i) {
        float dx = valid_points[i].x - valid_points[i-1].x;
        float dy = valid_points[i].y - valid_points[i-1].y;
        float dist = std::hypot(dx, dy);

        if (dist < SEGMENT_THRESHOLD) {
            current_cluster.push_back(valid_points[i]);
        } else {
            clusters.push_back(current_cluster);
            current_cluster.clear();
            current_cluster.push_back(valid_points[i]);
        }
    }
    clusters.push_back(current_cluster); // 存入最後一個

    // 3. 特徵萃取 (Feature Extraction)
    for (const auto& cluster : clusters) {
        int num_points = cluster.size();
        if (num_points < 2) continue; // 點數太少無法算特徵，直接捨棄

        Segment seg;
        
        // 計算該群的幾何中心，做為 local_x, local_y 供定位與攔截使用
        float sum_x = 0, sum_y = 0;
        for (const auto& pt : cluster) {
            sum_x += pt.x; sum_y += pt.y;
        }
        seg.local_x = sum_x / num_points;
        seg.local_y = sum_y / num_points;

        // 計算 7 維特徵
        seg.features = calculate7DFeatures(cluster);
        result_segments.push_back(seg);
    }

    return result_segments;
}

std::vector<float> FeatureExtractor::calculate7DFeatures(const std::vector<Point2D>& pts) {
    int N = pts.size();
    float eps = 2.2204e-16f; // 對應 MATLAB 的 eps

    // 1. 距離 (feat_r) & 6. 點數 (feat_count)
    float sum_r = 0;
    for (const auto& p : pts) sum_r += p.r;
    float feat_r = sum_r / N;
    float feat_count = static_cast<float>(N);

    // 2. 標準差 (feat_std) - 遵守 MATLAB 除以 N-1 的規則
    float sum_sq_diff = 0;
    for (const auto& p : pts) {
        sum_sq_diff += pow(p.r - feat_r, 2);
    }
    float feat_std = sqrt(sum_sq_diff / (N - 1));

    // 7. 梯度 (feat_grad)
    float sum_grad = 0;
    for (int i = 1; i < N; ++i) {
        sum_grad += abs(pts[i].r - pts[i-1].r);
    }
    float feat_grad = sum_grad / (N - 1);

    // 5. 跨度 (feat_span)
    float span_dx = pts.back().x - pts.front().x;
    float span_dy = pts.back().y - pts.front().y;
    float feat_span = std::hypot(span_dx, span_dy);

    // 3. 線性度 (feat_lin) - 使用閉式解求特徵值，免除 Eigen 依賴
    float feat_lin = 0.0f;
    if (N >= 3) {
        float mean_x = 0, mean_y = 0;
        for (const auto& p : pts) { mean_x += p.x; mean_y += p.y; }
        mean_x /= N; mean_y /= N;

        float c_xx = 0, c_yy = 0, c_xy = 0;
        for (const auto& p : pts) {
            c_xx += pow(p.x - mean_x, 2);
            c_yy += pow(p.y - mean_y, 2);
            c_xy += (p.x - mean_x) * (p.y - mean_y);
        }
        c_xx /= (N - 1); c_yy /= (N - 1); c_xy /= (N - 1); // Covariance matrix

        // 解特徵方程: lambda^2 - Trace*lambda + Det = 0
        float trace = c_xx + c_yy;
        float det = c_xx * c_yy - c_xy * c_xy;
        float discriminant = sqrt(max(0.0f, trace * trace - 4 * det));
        
        float lambda1 = (trace + discriminant) / 2.0f;
        float lambda2 = (trace - discriminant) / 2.0f;
        
        float min_eig = min(lambda1, lambda2);
        float sum_eig = lambda1 + lambda2;
        feat_lin = min_eig / (sum_eig + eps);
    }

    // 4. 曲率 (feat_curv) - 完美鏡像 MATLAB 的幾何公式
    float feat_curv = 0.0f;
    if (N >= 3) {
        float sum_curv = 0;
        for (int i = 1; i < N - 1; ++i) {
            float x1 = pts[i-1].x, y1 = pts[i-1].y;
            float x2 = pts[i].x,   y2 = pts[i].y;
            float x3 = pts[i+1].x, y3 = pts[i+1].y;

            // 這裡完全照搬你 MATLAB 腳本的計算方式，維持特徵一致性
            float area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
            float a = std::hypot(x2 - x1, y2 - y1);
            float b = std::hypot(x3 - x2, y3 - y2);
            float c = std::hypot(x3 - x1, y3 - y1);

            float R = (a * b * c) / (4.0f * abs(area2) + eps);
            sum_curv += 1.0f / (R + eps);
        }
        feat_curv = sum_curv / (N - 2); // 共有 N-2 個內部點
    }

    // 回傳 7 維特徵，順序必須與模型訓練時嚴格一致
    return {feat_r, feat_std, feat_lin, feat_curv, feat_span, feat_count, feat_grad};
}
