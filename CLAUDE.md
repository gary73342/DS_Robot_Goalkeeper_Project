# 守門員機器人專案 — CLAUDE.md

## 專案簡介

機器人守門員，完成感知（球與球柱偵測）、定位（機器人姿態估計）、攔截（移動決策）三大功能。
開發環境：**Distrobox + ROS1**，在筆電上運行。**不需要也不應該執行 catkin_make**，由使用者自行在 Distrobox 環境內編譯。

**硬體**：minibot 機器人、LiDAR、D435i 相機、筆電（主運算）

---

## 資料夾結構

| 路徑 | 說明 |
|------|------|
| `catkin_ws/` | ROS1 工作空間，主要程式碼在此 |
| `catkin_ws/src/mrlrobot_sample_code/sample_code/src/` | 主要 C++ 節點原始碼 |
| `catkin_ws/src/mrlrobot_sample_code/sample_code/include/` | 主要 C++ 節點標頭檔 |
| `catkin_ws/src/mrlrobot_sample_code/sample_code/launch/` | RViz 跟  urdf 的 launch檔 |
| `logs/` | 每次測試的日誌檔（依時間戳命名） |
| `data/` | LiDAR、相機錄影、抽幀照片等原始資料 |
| `matlab/` | MATLAB 光達資料與隨機森林程式碼 |
| `python/` | Python 程式碼（含 fusion_node.py） |
| `Robot_Goalkeeper.v1i.yolov8/` | YOLO 訓練用的 label 與資料集檔案（非必要不讀） |
| `runs/` | YOLO 訓練好的模型檔案（非必要不讀） |
| `TODO.txt` | 已完成、待解決、待優化、待測試事項的完整記錄 |
| `.sh檔案` | 使用 tmux 開啟節點跟關閉節點的檔案 |
---

## ROS 節點架構

### 資料流總覽

```
LiDAR (/scan)
    └─→ perception_node ─→ /posts_local, /ball_lidar_local
                                   │
Camera (/camera/color/image_raw)   │
    └─→ yolo_ros_node ─→ /ball_camera_world
                              │    │
                              ▼    ▼
/odom ─────────────→ localization_node (EKF_Localization_node.cpp)
                              │
                              ├─→ /robot_pose
                              └─→ /ball_lidar_world
                                        │
                              ┌─────────┴────────────┐
                              ▼                      ▼
                    /ball_lidar_world      /ball_camera_world
                    /robot_pose            /odom
                              └──── fusion_node ─────┘
                                          │
                                          ▼
                                      /cmd_vel（底盤執行）
```

### 各節點說明

| 節點 | 語言 | 訂閱 | 發布 | 功能 |
|------|------|------|------|------|
| `perception_node` | C++ | `/scan` | `/ball_lidar_local`, `/posts_local` | LiDAR 處理，輸出球與球柱的本地座標 |
| `yolo_ros_node` | Python | `/camera/color/image_raw` | `/ball_camera_world` | YOLOv8n 視覺偵測球，輸出世界座標 |
| `EKF_Localization_node` | C++ | `/odom`, `/posts_local`, `/ball_lidar_local` | `/robot_pose`, `/ball_lidar_world`, `/hypothesis_cloud` | EKF 定位（WAITING→EKF⇄RECOVERY），轉換球到世界座標；`/robot_pose` 為 5 fields：[x, y, theta, var_x, state_flag] |
| `fusion_node` | Python | `/ball_lidar_world`, `/ball_camera_world`, `/robot_pose`, `/odom` | `/cmd_vel` | 雙感測器融合決策，輸出控制指令；RECOVERY 期間改發旋轉指令 |

---

## 演算法分工

- **感知層**：隨機森林（球 + 球柱偵測）、YOLOv8n（球偵測）
- **定位層**：EKF 主導（`EKF_Localization.cpp`），`KidnapRecovery.cpp` 負責綁架恢復
- **融合/攔截**：`fusion_node.py`，含雙感測器融合邏輯與攔截決策

---

## 座標系定義（已確認）

- **原點**：防守端底線中點
- **X+**：球場右側（面對來球的右手邊）
- **Y+**：來球方向（踢球端）
- **機器人守門朝向**：全域 -X 方向（robot forward = global -X）
- `odom.linear.x > 0` = 機器人往 -X 移動（前進）
- `cmd_vel.linear.x = -speed`（正 speed 代表往 +X 走）
- **球門柱位置**：(±0.45, 0)，球門寬 0.9m

---
## 重要提醒

- 不要執行 catkin_make 或任何 ROS 編譯指令，使用者在自己的 Distrobox 環境內編譯
- 修改完程式碼後告知使用者自行編譯測試
- 用繁體中文
- 做完任務或是要結束對話前，由使用者提醒撰寫日誌到logs/log資料夾裡，內容包含討論內容、主要改動，檔名為日期加標題
- **修改任何程式碼之前，必須先討論方案並告知改動範圍**（包含：會動到哪些檔案、哪些函式、改動的核心邏輯），取得使用者確認後才能動手修改
- 主動獨立審查：任務完成或收尾時，主動找邏輯漏洞與未設想的邊界情況，給推薦與理由，不要只等指令
- 提選項時附推薦與 trade-off：不只列選項，要說明推薦哪個、原因是什麼，使用者自己判斷

---

## ★ 改動前三確認（強制流程）

**每一次動手改 code 之前，無論改動大小，必須先列出以下清單讓使用者確認，確認後才能開始：**

1. **改哪些檔案** — 列出所有會被新增、修改、刪除的檔案
2. **改哪些函式／介面** — 新增哪些、刪除哪些、修改哪些，特別標明對外介面（header）的變化
3. **核心邏輯是什麼** — 一句話說清楚這次改動在做什麼

格式範例：
```
改動範圍確認：
  檔案：EKF_Localization.h（修改）、EKF_Localization.cpp（修改）
  介面：新增 EKFLocalizer(x,y,theta) 建構子，移除 forceReset()
  邏輯：用建構子取代 forceReset，讓 EKFLocalizer 不依賴外部重置語意
請確認這是你要的再繼續。
```

**這條規則存在的原因**：過去因為沒有在改動前充分確認，導致改了又刪、刪了又改，浪費時間且容易出錯。三確認是防止這種情況的強制流程。
