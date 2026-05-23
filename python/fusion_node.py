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

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

# ==============================================================================
# ★ 調教區：所有需要調整的參數都在這裡，不需要動其他地方
# ==============================================================================

# --- 球場幾何 ---
FIELD_X_MIN    = -0.45
FIELD_X_MAX    =  0.45
DEFENSE_LINE_Y =  0.5   # 機器人守在離底線 0.5m 處

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
KP_LINEAR       = 2.0    # P-Control 增益
MAX_SPEED       = 0.22   # 最大線速度（m/s）
MIN_SPEED       = 0.08   # 最小啟動速度（m/s）
STOP_THRESHOLD  = 0.02   # 死區：誤差小於此值停止（m）

# LOST 狀態：連續幾秒沒有有效觀測就停止攔截
LOST_TIMEOUT    = 0.5    # 秒

# 系統延遲補償（從發出指令到馬達開始動的延遲，單位秒）
# SYSTEM_DELAY    = 0.08
# --- 來球判斷 ---
# 球的 Y 方向速度小於此值（負數，朝機器人）才觸發攔截
# BALL_INCOMING_VY = -0.1  # m/s

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

        # 巡邏模式狀態
        self.patrol_target_x = FIELD_X_MAX   # 先往右邊走
        self.patrol_timer    = rospy.Timer(rospy.Duration(0.1),
                                           self._patrol_callback)

        # Publisher
        self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # Subscribers
        rospy.Subscriber("/ball_camera_world", Float32MultiArray,
                         self.camera_callback, queue_size=1)
        rospy.Subscriber("/ball_lidar_world",  Float32MultiArray,
                         self.lidar_callback,  queue_size=1)
        rospy.Subscriber("/robot_pose",        Float32MultiArray,
                         self.pose_callback,   queue_size=1)

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

    def pose_callback(self, msg):
        self.robot_x     = msg.data[0]
        self.robot_y     = msg.data[1]
        self.robot_theta = msg.data[2]
        self.has_pose    = True

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

        R = compute_R_camera(X_cam, Y_cam,
                              self.robot_x, self.robot_y, conf)
        accepted = self.ekf.correction([X_cam, Y_cam], R)

        if accepted:
            self.last_valid_obs_time = rospy.Time.now().to_sec()
            self.is_lost = False
            self._publish_control()

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
            self.last_valid_obs_time = rospy.Time.now().to_sec()
            self.is_lost = False
            self._publish_control()

    # ------------------------------------------------------------------
    # LOST 狀態檢查
    # ------------------------------------------------------------------

    def _check_lost(self):
        """連續 LOST_TIMEOUT 秒沒有有效觀測 → 停止攔截"""
        if self.last_valid_obs_time is None:
            return
        elapsed = rospy.Time.now().to_sec() - self.last_valid_obs_time
        if elapsed > LOST_TIMEOUT:
            if not self.is_lost:
                rospy.logwarn("[Fusion] 球消失超過 %.1f 秒，進入 LOST 模式", LOST_TIMEOUT)
                self.is_lost = True
                self.ekf.reset()
            self._stop_robot()

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

        cmd.linear.x  = speed
        cmd.angular.z = 0.0   # 守門員不需要轉彎
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

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = 0.0
        self.pub_vel.publish(cmd)

        rospy.loginfo_throttle(0.2,
            "[TRACK] 球(%+.2f, %.2f) v=(%+.2f, %+.2f) | "
            "機器人X=%+.2f 誤差=%+.2f | 速度=%+.2f",
            bx, by, vx, vy, self.robot_x, error, speed)


    def _stop_robot(self):
        cmd = Twist()
        self.pub_vel.publish(cmd)

    def _patrol_callback(self, event):
        """
        10Hz 定時器：只有在 LOST 狀態下才執行巡邏。
        攔截模式時由 camera/lidar callback 發出指令，這裡不干涉。
        """
        if not self.is_lost or not self.has_pose:
            return

        error = self.patrol_target_x - self.robot_x

        # 到達巡邏目標附近就切換方向
        if abs(error) < 0.08:
            self.patrol_target_x = (FIELD_X_MIN
                                    if self.patrol_target_x == FIELD_X_MAX
                                    else FIELD_X_MAX)
            rospy.loginfo_throttle(1.0, "[Fusion] PATROL 換向 → 目標X=%.2f",
                                   self.patrol_target_x)

        # P-Control（巡邏速度限制在 MAX_SPEED 的一半，不用跑那麼快）
        speed = KP_LINEAR * error
        speed = max(min(speed,  MAX_SPEED * 0.5), -MAX_SPEED * 0.5)
        if abs(speed) < MIN_SPEED and abs(error) > 0.08:
            speed = math.copysign(MIN_SPEED, speed)

        cmd = Twist()
        cmd.linear.x = speed
        self.pub_vel.publish(cmd)

        side = "右側(+%.2f)" % FIELD_X_MAX if self.patrol_target_x > 0 else "左側(%.2f)" % FIELD_X_MIN
        rospy.loginfo_throttle(0.5,
            "[PATROL] 前往%s | 機器人X=%+.2f 誤差=%+.2f | 速度=%+.2f",
            side, self.robot_x, error, speed)
    # ------------------------------------------------------------------
    # 啟動
    # ------------------------------------------------------------------

    def start(self):
        rospy.spin()


# ==============================================================================
# 入口
# ==============================================================================

if __name__ == "__main__":
    try:
        node = FusionNode()
        node.start()
    except rospy.ROSInterruptException:
        pass