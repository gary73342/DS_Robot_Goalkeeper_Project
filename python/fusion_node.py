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
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
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
BALL_FIELD_Y_MAX =  3.5   # 場地遠端邊界（公尺）

# --- 相機信心門檻 ---
CONF_MIN_CAM = 0.70   # 低於此值的偵測直接忽略，不進入 EKF

# --- EKF 雜訊參數 ---
# 過程雜訊（Q）：數字越大代表你越不相信物理模型，EKF 反應越靈敏但越抖
Q_XY  = 0.05  # 位置的過程雜訊（調小：讓 EKF 更信任平滑的軌跡）
Q_VXY = 0.1   # 速度的過程雜訊（調小：避免靜止時速度估計亂跳）

# 相機基礎量測雜訊（R_base）：數字越大代表越不信任相機
R_BASE_CAM   = 0.01  # 相機很穩，給小雜訊讓 EKF 主要信任相機
# 光達基礎量測雜訊
R_BASE_LIDAR = 0.20  # 光達跳動大，給大雜訊降低其影響
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
PATROL_MAX_SPEED = 0.20   # 巡邏最大速度（m/s）
MAX_ACCEL        = 0.8    # 加速度上限（m/s²）—— 起步緩慢加速
MAX_DECEL        = 1.2    # 減速度上限（m/s²）—— 煞車稍快但不急停

# LOST 狀態：連續幾秒沒有有效觀測就停止攔截
LOST_TIMEOUT      = 0.5   # 秒：進入 LOST 模式（回巡邏），EKF 繼續預測
EKF_RESET_TIMEOUT = 1.5   # 秒：球真的消失才 reset EKF，清除速度記憶

# 定位收斂門檻：粒子雲 X 軸方差低於此值時視為收斂，切換至絕對位置控制
LOCALIZED_VAR_THRESHOLD = 0.02   # 對應標準差約 0.14m

# 未收斂時的 odom 相對巡邏範圍（±公尺，從啟動點算起）
PATROL_RELATIVE_RANGE   = 0.2

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
    return BALL_FIELD_X_MIN <= x <= BALL_FIELD_X_MAX and y <= BALL_FIELD_Y_MAX

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
            self.P = np.eye(4) * 1.0
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

def compute_R_camera(ball_world_x, ball_world_y, robot_x, robot_y, conf):
    """
    相機的動態量測雜訊矩陣。
    - 球離相機越遠（深度越大），雜訊越大
    - YOLO conf 越低，雜訊越大
    球到相機的距離用球到機器人的距離近似（相機架在球場邊，跟機器人距離固定）
    """
    d = math.hypot(ball_world_x - robot_x, ball_world_y - robot_y)
    # d = ball_world_y
    d_ref = 1.5   # 參考距離（公尺）
    conf = max(conf, 0.01)   # 防止除以零

    scale = (d / d_ref) ** 2 / conf
    r_val = R_BASE_CAM * scale
    r_val = np.clip(r_val, R_BASE_CAM, R_BASE_CAM * 50)
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

        # Publisher
        self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

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

        R = compute_R_camera(X_cam, Y_cam,
                              self.robot_x, self.robot_y, conf)
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

    def _apply_ramp(self, target_speed):
        now = rospy.Time.now().to_sec()
        if self.ramp_last_time is None:
            self.ramp_last_time = now
            self.current_speed = 0.0
            return 0.0
        dt = min(now - self.ramp_last_time, 0.2)   # 防止長時間暫停後一次跳太多
        self.ramp_last_time = now

        delta = target_speed - self.current_speed
        if delta > 0:
            delta = min(delta, MAX_ACCEL * dt)
        else:
            delta = max(delta, -MAX_DECEL * dt)
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
    def _publish_control(self):
        state = self.ekf.get_state()
        bx, by, vx, vy = state

        target_x = np.clip(bx, FIELD_X_MIN, FIELD_X_MAX)
        error = target_x - self.robot_x

        if abs(error) < STOP_THRESHOLD:
            speed = 0.0
        else:
            speed = KP_LINEAR * error
            if abs(speed) < MIN_SPEED:
                speed = math.copysign(MIN_SPEED, speed)
            speed = np.clip(speed, -MAX_SPEED, MAX_SPEED)

        speed = self._apply_ramp(speed)

        # 硬性邊界：已超出球門範圍且仍朝外移動時強制歸零
        if self.robot_x > GUARD_X_MAX and speed > 0:
            speed = 0.0
        elif self.robot_x < GUARD_X_MIN and speed < 0:
            speed = 0.0

        cmd = Twist()
        cmd.linear.x = -speed
        cmd.angular.z = self._theta_correction()
        self.pub_vel.publish(cmd)

        ball_speed = math.hypot(vx, vy)
        rospy.loginfo_throttle(0.5,
            "[Fusion] TRACK | 機X=%+.2f 落點X=%+.2f 誤差=%+.2f 球速=%.2f",
            self.robot_x, target_x, error, ball_speed)


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

    def _stop_robot(self):
        self.current_speed  = 0.0   # 清除 ramp 狀態，下次起步重新累加
        self.ramp_last_time = None
        cmd = Twist()
        self.pub_vel.publish(cmd)

    def _patrol_callback(self, event):
        """
        10Hz 定時器：只有在 LOST 狀態下才執行巡邏。
        攔截模式時由 camera/lidar callback 發出指令，這裡不干涉。

        定位未收斂：用 odom 相對位移巡邏（±PATROL_RELATIVE_RANGE），
                    不依賴可能錯誤的粒子濾波輸出。
        定位已收斂：改用絕對位置（robot_x），在 FIELD_X_MIN ~ FIELD_X_MAX 之間巡邏。
        """
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