#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_node.py

職責：
  1. 訂閱 /ball_camera_world  (相機，~30Hz) [detected, X, Y, conf]
  2. 訂閱 /ball_lidar_world   (光達，~10Hz) [detected, X, Y]
  3. 訂閱 /robot_pose                       [x, y, theta]
  4. Asynchronous EKF 融合兩個感測器
  5. 攔截點預測 + 延遲補償
  6. P-Control → 發布 /cmd_vel

執行：
  python3 fusion_node.py
"""

import sys
import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32MultiArray, Bool, ColorRGBA
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from log import setup_log

# ==============================================================================
# ★ 調教區：所有需要調整的參數都在這裡，不需要動其他地方
# ==============================================================================

# --- 球場幾何 ---
FIELD_X_MIN    = -0.45
FIELD_X_MAX    =  0.45
DEFENSE_LINE_Y =  0.5   # 機器人守在離底線 0.5m 處

# 攔截模式硬性邊界（略小於球門柱，預留煞車距離）
GUARD_X_MIN    = -0.42
GUARD_X_MAX    =  0.42

# --- 實際場地邊界（用於過濾場外誤偵測）---
# 注意：這跟 FIELD_X_MIN/MAX 不同，那是機器人的移動範圍；這是實體場地大小
BALL_FIELD_X_MIN = -1.0   # 場地左邊界（公尺）
BALL_FIELD_X_MAX = +1.0   # 場地右邊界（公尺）
BALL_FIELD_Y_MAX =  2.88  # 場地遠端邊界（公尺）：與相機標定遠端 (±0.45, 2.88) 一致
BALL_FIELD_Y_MIN =  0.5   # 防守線後方邊界：球低於此值（已越過防守線、在機器人身後）一律忽略

# --- 相機信心門檻 ---
CONF_MIN_CAM = 0.70   # 低於此值的偵測直接忽略，不進入 EKF

# --- EKF 雜訊參數 ---
# 過程雜訊（Q）：數字越大代表你越不相信物理模型，EKF 反應越靈敏但越抖
Q_XY  = 0.05  # 位置的過程雜訊（調小：讓 EKF 更信任平滑的軌跡）
Q_VXY = 0.5   # 速度的過程雜訊（調大：縮短平滑窗，讓速度估計快速跟上快球，代價是較抖）

# 相機基礎量測雜訊（R_base）：數字越大代表越不信任相機
R_BASE_CAM   = 0.01  # 相機很穩，給小雜訊讓 EKF 主要信任相機
# 光達基礎量測雜訊
R_BASE_LIDAR = 0.04  # 光達在 1.5m 內可信，降低雜訊以增加貢獻
# 光達遠距離懲罰（超過 1.5m 後每增加 1m 增加多少雜訊）
R_LIDAR_ALPHA = 0.5

# Mahalanobis Gate 門限（chi-square 2DOF 95% = 5.99）
MAHAL_GATE = 5.99

# --- 控制參數 ---
KP_LINEAR        = 2.0    # P-Control 增益
MAX_SPEED        = 0.22   # 最大線速度（m/s）
MIN_SPEED        = 0.08   # 最小啟動速度（m/s）
STOP_THRESHOLD   = 0.02   # 死區：誤差小於此值停止（m）

KP_ANGULAR        = 2.0   # 角度修正 P-Control 增益（加大以克服機械偏差）
MAX_ANGULAR_SPEED = 0.5   # 最大旋轉速度（rad/s）
THETA_DEAD_ZONE   = 0.03  # 角度死區（rad，約 1.7°），低於此值不修正

# Y 軸漂移修正
# 往 +X（current_speed > 0）→ theta_target 正 → case 1 → +Y
# 往 -X（current_speed < 0）→ theta_target 負 → case 4 → +Y
# 調教：穩定後 robot_y 若低於 0.45 → 調大 KP_Y_CORRECTION；高於 0.55 → 調小
Y_CORRECTION_THRESHOLD = 0.05  # Y 偏差超過此值才啟動校正 (m)
KP_Y_CORRECTION        = 1.5   # theta 目標增益
Y_CORRECTION_MAX       = 0.12  # theta 目標上限 (rad ≈ 6.9°)

# 速度 Ramp（緩慢起步 / 煞車）
PATROL_MAX_SPEED    = 0.15   # 巡邏最大速度（m/s）—— 刻意低於攔截上限(0.22)，便於目視區分兩種模式
MAX_ACCEL           = 0.8    # 巡邏加速度上限（m/s²）
INTERCEPT_MAX_ACCEL = 2.5    # 攔截加速度上限（m/s²）—— 追球需要更猛的起步
MAX_DECEL           = 1.2    # 減速度上限（m/s²）—— 煞車稍快但不急停

# LOST 狀態：連續幾秒沒有有效觀測就停止攔截
LOST_TIMEOUT      = 0.5   # 秒：進入 LOST 模式（回巡邏），EKF 繼續預測
EKF_RESET_TIMEOUT = 1.5   # 秒：球真的消失才 reset EKF，清除速度記憶

# 定位收斂門檻：EKF X 軸方差低於此值時視為收斂，切換至絕對位置控制
LOCALIZED_VAR_THRESHOLD = 0.02   # 對應標準差約 0.14m

# 綁架恢復（SETTLE→SPIN→RETURN→HALT）
KIDNAP_SPIN_SPEED   = 0.4    # SPIN 慢速原地旋轉角速度（rad/s）
KIDNAP_RETURN_SPEED = 0.15   # RETURN 弧線前進速度（m/s，cmd.linear.x 正值 = 沿車頭前進）
RETURN_GOAL_X       = 0.0    # RETURN 導航目標 X（場地中心，保證走到看得到兩柱）
RETURN_GOAL_Y       = DEFENSE_LINE_Y  # 目標 Y（不強制精確，巡邏線附近即可）
RETURN_GOAL_REACHED = 0.10   # 距目標小於此值不再前進（m）
RETURN_THETA_ALIGN  = 0.3    # |θ 誤差| > 此值時只旋轉不平移（rad，約 17°）

# 綁架恢復「位置先驗區」—— 僅供 RViz 顯示。必須與 KidnapRecovery.cpp 的
# ZONE_X_MAX / ZONE_Y_MIN / ZONE_Y_MAX 保持一致。C++ 端實際過濾是兩側窄帶
# （|x|∈[0.45,0.65]，單柱 fallback 用），這裡按需求畫成含中間的一整條帶
# （x∈[-0.65,0.65]）；中間靠兩柱幾何重定位即可恢復，故畫整條更貼近實際能力。
KIDNAP_ZONE_X_MAX = 0.65   # 對應 C++ ZONE_X_MAX
KIDNAP_ZONE_Y_MIN = 0.0    # 對應 C++ ZONE_Y_MIN
KIDNAP_ZONE_Y_MAX = 0.70   # 對應 C++ ZONE_Y_MAX

# 未收斂時的 odom 相對巡邏範圍（±公尺，從啟動點算起）
PATROL_RELATIVE_RANGE   = 0.2

# 攔截完成判斷
INTERCEPT_SPEED_THRESH  = 0.15  # 球視為靜止的速度閾值（m/s）
INTERCEPT_DIST_THRESH   = 0.30  # 球視為貼近機器人的距離閾值（m）
INTERCEPT_CONFIRM_TIME  = 0.5   # 條件需持續多久才確認攔截完成（秒）

# --- 落點預測觸發條件 ---
BALL_INCOMING_SPEED = 0.4   # 全向球速達此值才啟動落點預測（m/s）
INTERCEPT_T_MAX     = 5.0   # 預測時間上限（秒），防止球速極慢時落點飛太遠

# 系統延遲補償（從發出指令到馬達開始動的延遲，單位秒）
# SYSTEM_DELAY    = 0.08
# --- 來球判斷 ---
# 球的 Y 方向速度小於此值（負數，朝機器人）才觸發攔截
# BALL_INCOMING_VY = -0.1  # m/s

# ==============================================================================
# 工具函式
# ==============================================================================

def _in_field(x, y):
    """球是否在實體場地範圍內"""
    return (BALL_FIELD_X_MIN <= x <= BALL_FIELD_X_MAX
            and BALL_FIELD_Y_MIN <= y <= BALL_FIELD_Y_MAX)

# ==============================================================================
# Asynchronous EKF
# ==============================================================================

class AsyncEKF:
    """
    狀態向量 X = [x, y, vx, vy]^T
    任一感測器資料到達時，各自獨立觸發 predict + correction。
    兩個感測器共享同一個狀態，不需要等待對方。
    """

    def __init__(self):
        self.x = np.zeros((4, 1))          # 狀態向量
        self.P = np.eye(4) * 1000.0        # 共變異數矩陣（初始很大，代表完全不確定）
        self.H = np.array([[1, 0, 0, 0],   # 觀測矩陣：只觀測位置 (x, y)
                            [0, 1, 0, 0]], dtype=float)
        self.initialized = False
        self.last_update_time = None

    def _build_F(self, dt):
        """建立狀態轉移矩陣（Constant Velocity 模型）"""
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

    def _build_Q(self, dt):
        """建立過程雜訊矩陣"""
        return np.diag([Q_XY, Q_XY, Q_VXY, Q_VXY]) * dt

    def predict(self, dt):
        """預測步驟：任一感測器資料到達時呼叫"""
        if not self.initialized:
            return
        dt = max(dt, 1e-4)   # 防止 dt=0 造成數值問題
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _mahalanobis(self, z, R):             # ← 加入 R 參數
        z = np.array(z).reshape(2, 1)
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R   # ← 加上 R
        try:
            d2 = float(innov.T @ np.linalg.inv(S) @ innov)
        except np.linalg.LinAlgError:
            return float('inf')
        return d2

    def correction(self, z, R):
        """
        修正步驟：帶入觀測值 z=[x,y] 和對應的量測雜訊矩陣 R
        回傳 True 表示此次觀測通過 Mahalanobis gate 並完成更新
        """
        z = np.array(z).reshape(2, 1)

        # 第一筆觀測直接初始化，不做 gate 檢查
        if not self.initialized:
            self.x[0, 0] = z[0, 0]
            self.x[1, 0] = z[1, 0]
            self.x[2, 0] = 0.0
            self.x[3, 0] = 0.0
            # 位置很確定（剛量到）→ 小方差；速度全未知且球常從靜止突然加速 →
            # 大方差，讓前幾幀的速度修正增益高，加速初期速度收斂
            self.P = np.diag([1.0, 1.0, 100.0, 100.0])
            self.initialized = True
            rospy.loginfo("[EKF] 初始化完成，球的初始位置: X=%.3f Y=%.3f",
                          z[0, 0], z[1, 0])
            return True

        # Mahalanobis gate：拒絕離預測點太遠的雜訊觀測
        d2 = self._mahalanobis(z, R)
        if d2 > MAHAL_GATE:
            rospy.logdebug("[EKF] 觀測被 gate 拒絕，Mahal dist²=%.2f", d2)
            return False

        # 標準 Kalman Correction
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innov = z - self.H @ self.x
        self.x = self.x + K @ innov
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return True

    def get_state(self):
        """回傳當前狀態 [x, y, vx, vy]"""
        return self.x.flatten()

    def reset(self):
        """重置 EKF（球消失時呼叫）"""
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1000.0
        self.initialized = False
        self.last_update_time = None

# ==============================================================================
# 動態 R 矩陣計算
# ==============================================================================

def compute_R_camera(conf):
    """
    相機的動態量測雜訊矩陣。
    場地簡單、遠距離信心依然高，移除距離懲罰，只由 conf 縮放。
    """
    r_val = R_BASE_CAM / max(conf, 0.01)
    r_val = np.clip(r_val, R_BASE_CAM * 0.5, R_BASE_CAM * 3)
    return np.eye(2) * r_val

def compute_R_lidar(ball_world_x, ball_world_y, robot_x, robot_y):
    """
    光達的動態量測雜訊矩陣。
    - 球離機器人越遠，光達特徵退化，雜訊越大
    """
    d = math.hypot(ball_world_x - robot_x, ball_world_y - robot_y)
    if d <= 1.5:
        r_val = R_BASE_LIDAR
    else:
        r_val = R_BASE_LIDAR + R_LIDAR_ALPHA * (d - 1.5) ** 2
    r_val = np.clip(r_val, R_BASE_LIDAR, R_BASE_LIDAR * 30)
    return np.eye(2) * r_val

# ==============================================================================
# 主節點
# ==============================================================================

class FusionNode:

    def __init__(self):
        rospy.init_node("fusion_node")

        self.ekf = AsyncEKF()

        # 機器人姿態（由 perception_node 發布）
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0
        self.has_pose    = False

        # LOST 狀態計時
        self.last_valid_obs_time = None
        self.is_lost = True

        # 定位收斂狀態
        self.is_localized = False

        # 綁架恢復狀態（由 /robot_pose[4] 決定）
        self.is_kidnap_recovery = False
        self.recovery_sub = 0   # 0=SETTLE,1=SPIN,2=RETURN,3=HALT（由 /robot_pose[5] 決定）

        # odom 追蹤（未收斂時用相對位移巡邏）
        self.odom_x = 0.0
        self.odom_x_origin = None   # 第一筆 odom 到來時記錄

        # 巡邏模式狀態
        self.patrol_target_x = FIELD_X_MAX          # 收斂後的絕對位置目標
        self.patrol_target_relative = PATROL_RELATIVE_RANGE  # 未收斂時的相對位移目標
        self.patrol_timer = rospy.Timer(rospy.Duration(0.1),self._patrol_callback)

        # 速度 Ramp 狀態（緩慢起步 / 煞車）
        self.current_speed  = 0.0   # 上次實際發出的速度
        self.ramp_last_time = None

        # 攔截完成狀態
        self.is_intercepted              = False
        self.intercept_condition_start   = None

        # Publisher
        self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pub_interception_done = rospy.Publisher(
            "/interception_done", Bool, queue_size=1)
        self.pub_field = rospy.Publisher(
            "/field_boundary", Marker, queue_size=1, latch=True)
        rospy.Timer(rospy.Duration(1.0), self._publish_field_boundary)
        self.pub_predict_marker = rospy.Publisher(
            "/predict_marker", Marker, queue_size=1)
        self.pub_kidnap_zone = rospy.Publisher(
            "/kidnap_zone", Marker, queue_size=1, latch=True)
        rospy.Timer(rospy.Duration(1.0), self._publish_kidnap_zone)

        # Subscribers
        rospy.Subscriber("/ball_camera_world", Float32MultiArray,
                         self.camera_callback, queue_size=1)
        rospy.Subscriber("/ball_lidar_world",  Float32MultiArray,
                         self.lidar_callback,  queue_size=1)
        rospy.Subscriber("/robot_pose",        Float32MultiArray,
                         self.pose_callback,   queue_size=1)
        rospy.Subscriber("/odom",              Odometry,
                         self.odom_callback,   queue_size=1)

        rospy.loginfo("=" * 50)
        rospy.loginfo("[Fusion] 節點啟動")
        rospy.loginfo("[Fusion] 防守線 Y = %.2f m", DEFENSE_LINE_Y)
        rospy.loginfo("[Fusion] 訂閱 /ball_camera_world, /ball_lidar_world, /robot_pose")
        rospy.loginfo("[Fusion] 發布 /cmd_vel")
        rospy.loginfo("=" * 50)

    # ------------------------------------------------------------------
    # RViz 場地邊界 Marker
    # ------------------------------------------------------------------

    def _publish_field_boundary(self, _event=None):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp    = rospy.Time.now()
        marker.ns              = "field"
        marker.id              = 0
        marker.type            = Marker.LINE_LIST
        marker.action          = Marker.ADD
        marker.scale.x         = 0.02   # 線寬 2 cm
        marker.color           = ColorRGBA(0.0, 1.0, 0.0, 0.8)
        marker.pose.orientation.w = 1.0

        # 場地四條邊：直接綁定 _in_field 判定範圍（x: ±1.0，y: 0.5~2.88），
        # 讓綠框永遠等於「球算不算在場內」的實際邊界，畫面與邏輯一致
        # LINE_LIST：每兩個點構成一條線段
        xmin, xmax = BALL_FIELD_X_MIN, BALL_FIELD_X_MAX
        ymin, ymax = BALL_FIELD_Y_MIN, BALL_FIELD_Y_MAX
        corners = [
            (xmin, ymin), (xmax, ymin),  # 防守線（近端）
            (xmax, ymin), (xmax, ymax),  # 右邊線
            (xmax, ymax), (xmin, ymax),  # 遠端線
            (xmin, ymax), (xmin, ymin),  # 左邊線
        ]
        for (x, y) in corners:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        self.pub_field.publish(marker)

    def _publish_kidnap_zone(self, _event=None):
        """RViz 顯示綁架恢復的位置先驗區（含中間的一整條帶，純顯示）"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp    = rospy.Time.now()
        marker.ns              = "kidnap_zone"
        marker.id              = 0
        marker.type            = Marker.LINE_LIST
        marker.action          = Marker.ADD
        marker.scale.x         = 0.02   # 線寬 2 cm
        marker.color           = ColorRGBA(1.0, 0.5, 0.0, 0.8)  # 橙色，區別於綠色場地框
        marker.pose.orientation.w = 1.0

        # 一整條帶的矩形外框（x: ±0.68，y: -0.10~0.85）
        x      = KIDNAP_ZONE_X_MAX
        y0, y1 = KIDNAP_ZONE_Y_MIN, KIDNAP_ZONE_Y_MAX
        corners = [
            (-x, y0), (x, y0),   # 下緣
            (x, y0),  (x, y1),   # 右緣
            (x, y1),  (-x, y1),  # 上緣
            (-x, y1), (-x, y0),  # 左緣
        ]
        for (px, py) in corners:
            p = Point()
            p.x = px
            p.y = py
            p.z = 0.0
            marker.points.append(p)

        self.pub_kidnap_zone.publish(marker)

    # ------------------------------------------------------------------
    # 共用：計算 dt 並觸發 EKF predict
    # ------------------------------------------------------------------

    def _ekf_predict(self, stamp):
        """
        用感測器 header.stamp 計算精確 dt，觸發 EKF predict。
        回傳 dt（秒）。
        """
        now = stamp.to_sec()
        if self.ekf.last_update_time is None:
            self.ekf.last_update_time = now
            return 0.0
        dt = now - self.ekf.last_update_time
        dt = max(min(dt, 1.0), 1e-4)   # 限制在合理範圍，防止異常跳變
        self.ekf.last_update_time = now
        self.ekf.predict(dt)
        return dt

    # ------------------------------------------------------------------
    # 機器人姿態 callback
    # ------------------------------------------------------------------

    def odom_callback(self, msg):
        current_x = msg.pose.pose.position.x
        if self.odom_x_origin is None:
            self.odom_x_origin = current_x
            rospy.loginfo("[Fusion] odom 原點記錄：X=%.3f", current_x)
        self.odom_x = current_x

    def pose_callback(self, msg):
        self.robot_x     = msg.data[0]
        self.robot_y     = msg.data[1]
        self.robot_theta = msg.data[2]
        self.has_pose    = True

        if len(msg.data) >= 4 and not self.is_localized:
            var_x = msg.data[3]
            if var_x < LOCALIZED_VAR_THRESHOLD:
                self.is_localized = True
                rospy.logwarn("[Fusion] 定位收斂！var_x=%.4f → 切換至絕對位置控制", var_x)

        if len(msg.data) >= 5:
            self.is_kidnap_recovery = (msg.data[4] > 0.5)
        if len(msg.data) >= 6:
            self.recovery_sub = int(msg.data[5] + 0.5)

    # ------------------------------------------------------------------
    # 相機 callback（~30Hz）
    # ------------------------------------------------------------------

    def camera_callback(self, msg):
        if not self.has_pose:
            return

        detected = msg.data[0] > 0.5
        stamp    = rospy.Time.now()   # 相機 msg 沒有 header，用收到時間近似

        self._ekf_predict(stamp)

        if not detected:
            self._check_lost()
            return

        X_cam  = msg.data[1]
        Y_cam  = msg.data[2]
        conf   = msg.data[3]

        if conf < CONF_MIN_CAM:
            self._check_lost()
            return

        R = compute_R_camera(conf)
        accepted = self.ekf.correction([X_cam, Y_cam], R)

        if accepted:
            if _in_field(X_cam, Y_cam):
                self.last_valid_obs_time = rospy.Time.now().to_sec()
                self.is_lost = False
                self._publish_control()
            else:
                if not self.is_lost:
                    rospy.logwarn("[Fusion] 球在場外 (%.2f, %.2f)，立即切換至巡邏", X_cam, Y_cam)
                    self.is_lost = True
                    self._stop_robot()
                else:
                    rospy.logwarn_throttle(1.0,
                        "[Fusion] 球在場外 (%.2f, %.2f)，維持巡邏", X_cam, Y_cam)
                self._check_lost()

    # ------------------------------------------------------------------
    # 光達 callback（~10Hz）
    # ------------------------------------------------------------------

    def lidar_callback(self, msg):
        if not self.has_pose:
            return

        detected = msg.data[0] > 0.5
        stamp    = rospy.Time.now()

        self._ekf_predict(stamp)

        if not detected:
            self._check_lost()
            return

        X_lidar = msg.data[1]
        Y_lidar = msg.data[2]

        R = compute_R_lidar(X_lidar, Y_lidar,
                             self.robot_x, self.robot_y)
        accepted = self.ekf.correction([X_lidar, Y_lidar], R)

        if accepted:
            if _in_field(X_lidar, Y_lidar):
                self.last_valid_obs_time = rospy.Time.now().to_sec()
                self.is_lost = False
                self._publish_control()
            else:
                if not self.is_lost:
                    rospy.logwarn("[Fusion] 光達：球在場外 (%.2f, %.2f)，立即切換至巡邏",
                                  X_lidar, Y_lidar)
                    self.is_lost = True
                    self._stop_robot()
                self._check_lost()

    # ------------------------------------------------------------------
    # 速度 Ramp：限制每次發布的速度變化率，避免急停急衝
    # ------------------------------------------------------------------

    def _apply_ramp(self, target_speed, max_accel=MAX_ACCEL):
        now = rospy.Time.now().to_sec()
        if self.ramp_last_time is None:
            self.ramp_last_time = now
            self.current_speed = 0.0
            return 0.0
        dt = min(now - self.ramp_last_time, 0.2)   # 防止長時間暫停後一次跳太多
        self.ramp_last_time = now

        # 加速 vs 減速取決於「速度大小變化」而非 delta 正負。
        # 例：current=0, target=-0.22 時 delta=-0.22<0 看似減速，但實際是「從靜止加速到 -X 方向」，
        # 應該用 max_accel（否則往 -X 追球的加速度會被誤限到 MAX_DECEL，導致左右不對稱）。
        if abs(target_speed) >= abs(self.current_speed):
            rate = max_accel
        else:
            rate = MAX_DECEL
        delta = target_speed - self.current_speed
        delta = max(min(delta, rate * dt), -rate * dt)
        self.current_speed += delta
        return self.current_speed

    # ------------------------------------------------------------------
    # LOST 狀態檢查
    # ------------------------------------------------------------------

    def _check_lost(self):
        if self.last_valid_obs_time is None:
            return
        elapsed = rospy.Time.now().to_sec() - self.last_valid_obs_time

        if elapsed > LOST_TIMEOUT and not self.is_lost:
            if self.is_intercepted:
                rospy.logwarn("[Fusion] ★ 攔截後球消失（被拿走），發送 interception_done 信號")
                self.pub_interception_done.publish(Bool(data=True))
                self.is_intercepted = False
                self.intercept_condition_start = None
            rospy.logwarn("[Fusion] 球消失超過 %.1f 秒，進入 LOST 模式（EKF 繼續預測）", LOST_TIMEOUT)
            self.is_lost = True
            self._stop_robot()

        if elapsed > EKF_RESET_TIMEOUT and self.ekf.initialized:
            rospy.logwarn("[Fusion] 球消失超過 %.1f 秒，重置 EKF", EKF_RESET_TIMEOUT)
            self.ekf.reset()

    # ------------------------------------------------------------------
    # 攔截決策 + P-Control
    # ------------------------------------------------------------------
    '''
    def _publish_control(self):
        """
        從 EKF 後驗狀態預測攔截點，計算 P-Control 速度，發布 /cmd_vel。
        """
        state = self.ekf.get_state()
        bx, by, vx, vy = state

        cmd = Twist()

        ball_speed = math.hypot(vx, vy)

        # 球速太小（靜止或微動）→ 直接追當前 X，不做預測
        # 避免光達雜訊造成的微小速度被 t_intercept 放大
        if ball_speed < 0.05 or vy >= BALL_INCOMING_VY:
            target_x = bx
            mode_str = "TRACK"
        else:
            # 球正在靠近，預測攔截點
            if abs(vy) < 1e-3:
                t_intercept = 0.0
            else:
                t_intercept = (DEFENSE_LINE_Y - by) / vy

            t_intercept = max(t_intercept, 0.0)
            t_intercept = min(t_intercept, 2.0)
            # 攔截點 X + 延遲補償
            target_x = bx + vx * (t_intercept + SYSTEM_DELAY)
            mode_str = "INTERCEPT"

        # 限制攔截點在球場範圍內
        target_x = np.clip(target_x, FIELD_X_MIN, FIELD_X_MAX)

        # P-Control
        error = target_x - self.robot_x

        if abs(error) < STOP_THRESHOLD:
            speed = 0.0
        else:
            speed = KP_LINEAR * error
            # 最小速度補償
            if abs(speed) < MIN_SPEED:
                speed = math.copysign(MIN_SPEED, speed)
            speed = np.clip(speed, -MAX_SPEED, MAX_SPEED)

        cmd.linear.x  = -speed   # 負號：正 cmd_vel = 機器人往 -X，需反向
        cmd.angular.z = 0.0
        self.pub_vel.publish(cmd)

        # 終端機輸出（方便 debug，不需要 rostopic echo）
        rospy.loginfo_throttle(0.2,
            "[Fusion] %s | 球(%.2f,%.2f) v=(%.2f,%.2f) | "
            "目標X=%.2f 機器人X=%.2f | 速度=%+.2f",
            mode_str, bx, by, vx, vy,
            target_x, self.robot_x, speed)
    '''
    def _publish_predict_marker(self, x, y, mode):
        """在 RViz 中畫出落點（INTERCEPT=紅橙，TRACK=藍灰）"""
        marker = Marker()
        marker.header.frame_id    = "map"
        marker.header.stamp       = rospy.Time.now()
        marker.ns                 = "predict"
        marker.id                 = 0
        marker.type               = Marker.SPHERE
        marker.action             = Marker.ADD
        marker.pose.position.x    = x
        marker.pose.position.y    = y
        marker.pose.position.z    = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        marker.lifetime = rospy.Duration(0.3)
        if mode == "INTERCEPT":
            marker.color = ColorRGBA(1.0, 0.2, 0.0, 0.9)   # 紅橙色：預測落點
        else:
            marker.color = ColorRGBA(0.4, 0.4, 1.0, 0.6)   # 藍灰色：追蹤目標
        self.pub_predict_marker.publish(marker)

    def _check_intercept(self, bx, by, vx, vy):
        speed = math.hypot(vx, vy)
        dist  = math.hypot(bx - self.robot_x, by - self.robot_y)

        conditions_met = (
            speed < INTERCEPT_SPEED_THRESH and
            dist  < INTERCEPT_DIST_THRESH  and
            by    > DEFENSE_LINE_Y
        )

        now = rospy.Time.now().to_sec()
        if conditions_met:
            if self.intercept_condition_start is None:
                self.intercept_condition_start = now
            elif now - self.intercept_condition_start >= INTERCEPT_CONFIRM_TIME:
                self.is_intercepted = True
                self.intercept_condition_start = None
                rospy.logwarn("[Fusion] ★ 攔截完成！停住等球被拿走")
        else:
            self.intercept_condition_start = None

    def _publish_control(self):
        # KIDNAP_RECOVERY 期間：交給恢復控制
        if self.is_kidnap_recovery:
            self._do_recovery_control()
            return

        # 攔截完成：停住等球消失（球消失由 _check_lost 處理）
        if self.is_intercepted:
            self._stop_robot()
            return

        state = self.ekf.get_state()
        bx, by, vx, vy = state
        ball_speed = math.hypot(vx, vy)

        t_intercept = 0.0
        if ball_speed >= BALL_INCOMING_SPEED and vy < 0:
            t_intercept = (DEFENSE_LINE_Y - by) / vy
            t_intercept = max(0.0, min(t_intercept, INTERCEPT_T_MAX))
            predict_x = bx + vx * t_intercept
            # 落點在門框外（|x| > FIELD_X_MAX）代表這球不會進門，無需攔截：
            # 不去堵門柱角落，退回 TRACK 守球當前 X。
            # marker_x 一律畫真實點（不 clip），方便在 RViz 判斷預測準不準
            if abs(predict_x) > FIELD_X_MAX:
                target_x = bx
                mode_str = "TRACK"
                marker_x = predict_x      # 門外真實落點（已算出，畫出來驗證預測）
            else:
                target_x = predict_x
                mode_str = "INTERCEPT"
                marker_x = predict_x
        else:
            target_x = bx
            mode_str = "TRACK"
            marker_x = bx                 # 沒進預測分支，沒算落點，畫球當前 X

        # marker 畫真實點（不 clip）；target_x 才 clip 給機器人控制用
        self._publish_predict_marker(marker_x, DEFENSE_LINE_Y, mode_str)
        target_x = np.clip(target_x, GUARD_X_MIN, GUARD_X_MAX)

        error = target_x - self.robot_x

        if abs(error) < STOP_THRESHOLD:
            speed = 0.0
        else:
            speed = KP_LINEAR * error
            if abs(speed) < MIN_SPEED:
                speed = math.copysign(MIN_SPEED, speed)
            speed = np.clip(speed, -MAX_SPEED, MAX_SPEED)

        speed = self._apply_ramp(speed, INTERCEPT_MAX_ACCEL)

        # 硬性邊界：已超出球門範圍且仍朝外移動時強制歸零
        if self.robot_x > GUARD_X_MAX and speed > 0:
            speed = 0.0
        elif self.robot_x < GUARD_X_MIN and speed < 0:
            speed = 0.0

        cmd = Twist()
        cmd.linear.x = -speed
        cmd.angular.z = self._theta_correction()
        self.pub_vel.publish(cmd)

        rospy.loginfo_throttle(0.5,
            "[Fusion] %s | 機X=%+.2f 落點X=%+.2f 誤差=%+.2f 球速=%.2f t=%.2fs",
            mode_str, self.robot_x, target_x, error, ball_speed, t_intercept)

        self._check_intercept(bx, by, vx, vy)


    def _theta_correction(self):
        theta_target = 0.0
        if self.is_localized and abs(self.current_speed) > 0.01:
            y_error = DEFENSE_LINE_Y - self.robot_y
            if abs(y_error) > Y_CORRECTION_THRESHOLD:
                # 往 +X（current_speed > 0）→ theta 正；往 -X → theta 負
                direction_sign = 1 if self.current_speed > 0 else -1
                theta_target = float(np.clip(
                    direction_sign * KP_Y_CORRECTION * y_error,
                    -Y_CORRECTION_MAX, Y_CORRECTION_MAX))

        theta_error = self.robot_theta - theta_target
        if abs(theta_error) < THETA_DEAD_ZONE:
            return 0.0
        angular = -KP_ANGULAR * theta_error
        return float(np.clip(angular, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

    def _do_recovery_control(self):
        """
        KIDNAP_RECOVERY 恢復控制：依子狀態 recovery_sub 分派動作。
          0=SETTLE（停住等落地）  1=SPIN（慢速原地旋轉）
          2=RETURN（弧線導航回場） 3=HALT（停住）
        """
        # 不清 ramp_last_time（跟 _stop_robot 一致）：恢復切回 EKF_PRIMARY 後
        # 第一拍能用累積 dt 直接接近 max_accel × dt_cap，不會白白損失一拍
        self.current_speed = 0.0
        cmd = Twist()

        if self.recovery_sub == 1:    # SPIN：慢速原地旋轉找柱（全程 0 平移）
            cmd.linear.x  = 0.0
            cmd.angular.z = KIDNAP_SPIN_SPEED
            rospy.logwarn_throttle(1.0, "[Fusion] RECOVERY SPIN — 慢速旋轉搜尋球柱")

        elif self.recovery_sub == 2:  # RETURN：先轉回守門姿態 (θ=0) 再 1D X 控制
            # θ 控制：驅動 → 0（守門姿態 = LiDAR +X 對齊全域 +X，物理朝向 -X）
            # 這個專案的 θ_EKF 是 LiDAR +X 在全域的方向，θ=0 ⇔ 物理面 -X
            theta_error = self.robot_theta
            while theta_error >  math.pi: theta_error -= 2.0 * math.pi
            while theta_error < -math.pi: theta_error += 2.0 * math.pi
            cmd.angular.z = float(np.clip(-KP_ANGULAR * theta_error,
                                          -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

            # X 平移：只在 θ 接近 0 時啟動。底盤是差速驅動，cmd.linear.x = 前進方向；
            # θ 沒對齊就平移會沿錯誤方向衝（上次往 +Y 衝就是吃這個）
            error_x = RETURN_GOAL_X - self.robot_x
            if abs(theta_error) > RETURN_THETA_ALIGN or abs(error_x) < RETURN_GOAL_REACHED:
                cmd.linear.x = 0.0
            else:
                speed = KP_LINEAR * error_x
                if abs(speed) < MIN_SPEED:
                    speed = math.copysign(MIN_SPEED, speed)
                speed = float(np.clip(speed, -KIDNAP_RETURN_SPEED, KIDNAP_RETURN_SPEED))
                cmd.linear.x = -speed

            rospy.logwarn_throttle(1.0,
                "[Fusion] RECOVERY RETURN — 機(%.2f,%.2f,θ=%.2f) "
                "X誤差=%+.2f θ誤差=%+.2f v=%+.2f w=%+.2f",
                self.robot_x, self.robot_y, self.robot_theta,
                error_x, theta_error, cmd.linear.x, cmd.angular.z)

        else:                          # SETTLE(0) / HALT(3)：停住
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            label = "SETTLE 落地等待" if self.recovery_sub == 0 else "HALT 停止"
            rospy.logwarn_throttle(1.0, "[Fusion] RECOVERY %s — 停住", label)

        self.pub_vel.publish(cmd)

    def _stop_robot(self):
        # 只清速度、保留 ramp_last_time：下次重新啟動時 ramp 會把停滯期間
        # 累積的 dt 算進來，第一拍就能跳到接近 max_accel × dt_capped(0.2s) 的速度，
        # 避免「球短暫消失再出現」時還要從 0 重新爬升
        self.current_speed = 0.0
        cmd = Twist()
        self.pub_vel.publish(cmd)

    def _patrol_callback(self, event):
        """
        10Hz 定時器：只有在 LOST 狀態下才執行巡邏。
        攔截模式時由 camera/lidar callback 發出指令，這裡不干涉。

        定位未收斂：用 odom 相對位移巡邏（±PATROL_RELATIVE_RANGE），
                    不依賴可能錯誤的 EKF 輸出。
        定位已收斂：改用絕對位置（robot_x），在 FIELD_X_MIN ~ FIELD_X_MAX 之間巡邏。
        """
        # KIDNAP_RECOVERY 期間：交給恢復控制
        if self.is_kidnap_recovery:
            self._do_recovery_control()
            return

        if not self.is_lost:
            return

        if self.is_localized:
            # --- 絕對位置巡邏 ---
            if not self.has_pose:
                return
            error = self.patrol_target_x - self.robot_x
            if abs(error) < 0.08:
                self.patrol_target_x = (FIELD_X_MIN
                                        if self.patrol_target_x == FIELD_X_MAX
                                        else FIELD_X_MAX)
            rospy.loginfo_throttle(0.5,
                "[Fusion] PATROL | 機X=%+.2f 目標X=%+.2f",
                self.robot_x, self.patrol_target_x)
        else:
            # --- odom 相對位移巡邏（定位未收斂）---
            if self.odom_x_origin is None:
                return
            relative_x = self.odom_x - self.odom_x_origin
            error = self.patrol_target_relative - relative_x
            if abs(error) < 0.08:
                self.patrol_target_relative = -self.patrol_target_relative  # 左右切換
            rospy.loginfo_throttle(0.5,
                "[Fusion] PATROL(REL) | 機X=%+.2f 目標X=%+.2f",
                relative_x, self.patrol_target_relative)

        speed = KP_LINEAR * error
        speed = np.clip(speed, -PATROL_MAX_SPEED, PATROL_MAX_SPEED)
        if abs(speed) < MIN_SPEED and abs(error) > 0.08:
            speed = math.copysign(MIN_SPEED, speed)

        speed = self._apply_ramp(speed)

        cmd = Twist()
        cmd.linear.x = -speed
        cmd.angular.z = self._theta_correction()
        self.pub_vel.publish(cmd)
    # ------------------------------------------------------------------
    # 啟動
    # ------------------------------------------------------------------

    def _shutdown(self):
        rospy.logwarn("[Fusion] 節點關閉，發送停止指令")
        cmd = Twist()
        for _ in range(5):
            self.pub_vel.publish(cmd)

    def start(self):
        rospy.on_shutdown(self._shutdown)
        rospy.spin()


# ==============================================================================
# 入口
# ==============================================================================

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    setup_log(log_path, sample_rate=1)
    try:
        node = FusionNode()
        node.start()
    except rospy.ROSInterruptException:
        pass