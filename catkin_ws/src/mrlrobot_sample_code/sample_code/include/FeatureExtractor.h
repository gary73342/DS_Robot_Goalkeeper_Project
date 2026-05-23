#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include <sensor_msgs/LaserScan.h>

// 局部座標結構 (光達觀測到的物體座標)
struct Observation {
    float x; // 向前 (m)
    float y; // 向左 (m)
};


// 單一雷射點結構
struct Point2D {
    float x;
    float y;
    float r; // 距離 (Radius)
};

// 分段與特徵結構
struct Segment {
    float local_x; // 該 Segment 的中心點 X
    float local_y; // 該 Segment 的中心點 Y
    std::vector<float> features; // 7維特徵: [r, std, lin, curv, span, count, grad]
};

class FeatureExtractor {
public:
    // 將 ROS 的 LaserScan 轉換為帶有 7 維特徵的 Segments
    static std::vector<Segment> extractSegmentsFromScan(const sensor_msgs::LaserScan::ConstPtr& scan_msg);

private:
    // 距離閾值 (對應 MATLAB 的 threshold = 0.05)
    static constexpr float SEGMENT_THRESHOLD = 0.05f; 

    // 內部計算 7 維特徵的數學函數
    static std::vector<float> calculate7DFeatures(const std::vector<Point2D>& pts);
};

#endif // FEATURE_EXTRACTOR_H