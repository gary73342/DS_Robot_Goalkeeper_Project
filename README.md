# Autonomous Robot Goalkeeper Based on LiDAR-Camera Sensor Fusion

**Liu Wei-Ting** | National Central University, Dept. of Mathematics

---

## 專案簡介

本專案實作一套自主機器人守門員系統，部署於差動驅動式 minibot 平台，搭載 2D LiDAR 與 Intel RealSense D435i RGB-D 相機，在 ROS 1 環境下即時運作。系統整合感知、定位、決策三層架構，無需人工介入即可自動攔截來球，在 30 次實體測試中達到 **87% 整體攔截成功率**。

---

## 系統架構

```
LiDAR (/scan)
    └─→ perception_node ─→ 球與球柱的本地座標
Camera (/camera/color/image_raw)
    └─→ yolo_ros_node ─→ 球的世界座標
/odom ──────────────→ EKF_Localization_node ─→ /robot_pose, /ball_lidar_world
                                                        │
                              ┌─────────────────────────┘
                              ▼
                         fusion_node ─→ /cmd_vel（底盤控制）
```

### 感知層

- **LiDAR**：對原始點雲進行分群，提取 7 維特徵向量，以 **隨機森林** 分類球（95.3% recall）與球柱（99.4% recall），整體準確率 99.1%
- **相機**：以微調的 **YOLOv8n** 模型偵測球（mAP@0.5 = 99%、Recall = 97.5%），並透過預先標定的 Homography 矩陣將像素座標投影至世界座標

### 定位層

- **EKF 定位**：融合里程計與球柱地標觀測，持續估計機器人姿態，巡邏期間 X 軸定位誤差維持在 10 cm 以內
- **三狀態機**：WAITING → EKF_PRIMARY ⇄ RECOVERY
- **綁架恢復模組**（R-MCL 啟發）：機器人被突然移位時，在 20 秒內於指定恢復區域成功重定位率超過 90%

### 感測器融合與決策層

- **非同步 EKF 球追蹤**：以常速度模型融合 LiDAR（10 Hz）與相機（30 Hz）的非同步觀測，估計球的位置與速度
- **四狀態攔截機****：PATROL → TRACK → INTERCEPT → INTERCEPTED
- **P 控制器**：根據預測落點驅動機器人側移攔截，最大速度 0.22 m/s

---

## 硬體與環境

| 項目 | 規格 |
|------|------|
| 機器人底盤 | 差動驅動 minibot |
| 感測器 | 2D LiDAR（10 Hz）、Intel RealSense D435i（30 Hz）|
| 運算 | 筆電（透過本地 ROS 網路連線）|
| 作業系統 | ROS 1（Distrobox 環境）|
| 場地 | 2.0 × 3.0 m，球門寬 0.9 m（柱距 ±0.45 m）|

---

## 實驗結果

| 指標 | 數值 |
|------|------|
| 整體攔截成功率（30 次）| 87%（26/30）|
| 中央球攔截率 | 90%（9/10）|
| 側邊球攔截率 | 85~90% |
| EKF 定位誤差（巡邏中）| < 10 cm |
| 綁架恢復成功率 | > 90%（20 秒內）|
| YOLOv8n mAP@0.5 | 99% |
| 隨機森林整體準確率 | 99.1% |

---

## 專案結構

```
catkin_ws/src/mrlrobot_sample_code/sample_code/
├── src/          # C++ 節點原始碼（perception, EKF localization）
├── include/      # 標頭檔
└── launch/       # RViz / URDF launch 檔
python/
└── fusion_node.py  # 感測器融合與攔截決策（Python）
matlab/           # LiDAR 資料分析與隨機森林訓練
runs/             # YOLOv8 訓練模型權重
```

---

## 相關連結

| 資源 | 連結 |
|------|------|
| Demo 影片 | [Watch Demo](YOUR_VIDEO_URL_HERE) |
| ROS環境搭建 | [HackMD](https://hackmd.io/@Gary73342/B1ghqouCWe) |
| 相機設定與用 | [HackMD](https://hackmd.io/@Gary73342/rknFdwDJfg) |
| 相機資料處理 | [HackMD](https://hackmd.io/@Gary73342/BJbxLOokfe) |

---
