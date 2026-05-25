#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_ros_node.py

功能：
  1. 啟動時若無 H.npy，進入互動式校準模式（點四個角點 → 計算 Homography → 存檔）
  2. 校準完成後進入偵測模式，訂閱相機影像
  3. 透過 UDP 呼叫 yolo_server.py 取得球的像素座標
  4. 將像素座標透過 H 矩陣轉換為球場全域座標
  5. 發布 /ball_camera_world (Float32MultiArray: [detected, X_world, Y_world, conf])
     同時在終端機印出偵測結果，不需要另開 rostopic echo

執行前確認：
  - yolo_server.py 已在另一個終端機執行
  - Intel D435i 相機節點已啟動 (roslaunch realsense2_camera rs_camera.launch ...)
  - H.npy 放在與本腳本相同的資料夾（或由本腳本校準產生）
"""

import rospy
import socket
import json
import os
import sys
import numpy as np
import cv2
from log import setup_log

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# ==============================================================================
# 全域設定
# ==============================================================================

# H.npy 存放位置（與本腳本同資料夾）
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
H_FILE_PATH  = os.path.join(SCRIPT_DIR, "H.npy")

# 球場全域座標（你定義的座標系，單位公尺）
# 順序必須和校準時點擊的角點順序完全一致
WORLD_POINTS = np.array([
    [-0.45, 0.00],  # 左下角（防守端左側）
    [ 0.45, 0.00],  # 右下角（防守端右側）
    [ 0.45, 2.88],  # 右上角（踢球端右側）
    [-0.45, 2.88],  # 左上角（踢球端左側）
    [-0.45, 0.96],  # 左邊 1/3 處
    [-0.45, 1.92],  # 左邊 2/3 處
    [ 0.45, 0.96],  # 右邊 1/3 處
    [ 0.45, 1.92],  # 右邊 2/3 處
], dtype=np.float32)

# YOLO server UDP 設定
YOLO_SERVER_IP   = "127.0.0.1"
YOLO_SERVER_PORT = 9999
YOLO_TIMEOUT_SEC = 0.1

# conf 低於此值視為無效偵測，不進 EKF
CONF_THRESHOLD = 0.4

# ==============================================================================
# ROS Topic 相機包裝（讓 HomographyCalibrator 可以用 cap.read() 介面）
# ==============================================================================

class RosTopicCapture:
    """
    HomographyCalibrator 可以從 ROS topic 拿畫面
    每次呼叫 cap.read() 就去訂閱一次 /camera/color/image_raw 拿一幀，校準視窗就能持續顯示即時畫面。
    """
    def __init__(self, node):
        self._node = node   # YoloRosNode，用來呼叫 _grab_frame_from_topic

    def read(self):
        frame = self._node._grab_frame_from_topic(timeout=3.0)
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        pass   # 不需要做任何事


# ==============================================================================
# 校準模組
# ==============================================================================

class HomographyCalibrator:
    """
    互動式 Homography 校準工具。
    開啟 OpenCV 視窗，讓使用者依序點擊八個標定點，
    計算 H 矩陣後存成 H.npy。
    """

    POINT_LABELS = [
        "1: Left-Bottom     (-0.45, 0.00)",
        "2: Right-Bottom    ( 0.45, 0.00)",
        "3: Right-Top       ( 0.45, 2.88)",
        "4: Left-Top        (-0.45, 2.88)",
        "5: Left-1/3        (-0.45, 0.96)",
        "6: Left-2/3        (-0.45, 1.92)",
        "7: Right-1/3       ( 0.45, 0.96)",
        "8: Right-2/3       ( 0.45, 1.92)",
    ]

    NUM_POINTS = len(WORLD_POINTS)   # 8

    def __init__(self):
        self.clicked_points = []   # 收集像素座標
        self.display_frame  = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.clicked_points) >= self.NUM_POINTS:
            return

        self.clicked_points.append([x, y])
        idx = len(self.clicked_points) - 1
        print(f"  [校準] 點 {idx+1} 已記錄：像素 ({x}, {y})  ←→  全域 {WORLD_POINTS[idx].tolist()}")

        # 在畫面上畫圓點和標籤
        cv2.circle(self.display_frame, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(self.display_frame, f"P{idx+1}", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self, cap):
        """
        cap: cv2.VideoCapture 物件（或從 ROS topic 取一幀）
        回傳：計算好的 H 矩陣 (3x3 numpy array)，失敗回傳 None
        """
        print("\n" + "="*55)
        print("  Homography 校準模式")
        print("="*55)
        print(f"  請依序點擊球場 {self.NUM_POINTS} 個標定點：")
        for label in self.POINT_LABELS:
            print(f"    {label}")
        print(f"\n  點完 {self.NUM_POINTS} 個點後按 [Enter] 計算，按 [ESC] 取消")
        print("="*55 + "\n")

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", 960, 540)
        cv2.setMouseCallback("Calibration", self._mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[錯誤] 無法從相機讀取影像")
                return None

            self.display_frame = frame.copy()

            # 在畫面上顯示目前狀態
            n = len(self.clicked_points)
            remaining = self.NUM_POINTS - n
            if remaining > 0:
                msg = f"請點第 {n+1} 個點：{self.POINT_LABELS[n]}"
            else:
                msg = f"{self.NUM_POINTS} 個點已完成！按 [Enter] 確認，[R] 重新來過"

            cv2.putText(self.display_frame, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            # 重新繪製已點的圓
            for i, pt in enumerate(self.clicked_points):
                cv2.circle(self.display_frame, tuple(pt), 8, (0, 255, 0), -1)
                cv2.putText(self.display_frame, f"P{i+1}", (pt[0]+10, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Calibration", self.display_frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:   # ESC
                print("[校準] 已取消")
                cv2.destroyAllWindows()
                return None

            elif key == ord('r') or key == ord('R'):
                self.clicked_points = []
                print(f"[校準] 重新來過，請重新點擊 {self.NUM_POINTS} 個標定點")

            elif key == 13 and len(self.clicked_points) == self.NUM_POINTS:  # Enter
                break

        cv2.destroyAllWindows()

        # 計算 Homography
        pixel_pts = np.array(self.clicked_points, dtype=np.float32)
        world_pts = WORLD_POINTS.copy()

        H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("[校準] Homography 計算失敗！請確認四個點不共線")
            return None

        # 驗證：把每個像素點反算回全域座標，看誤差多大
        print("\n  [校準] 驗證誤差：")
        for i, (px, py) in enumerate(self.clicked_points):
            pt = np.array([[[px, py]]], dtype=np.float32)
            world = cv2.perspectiveTransform(pt, H)[0][0]
            expected = WORLD_POINTS[i]
            err = np.linalg.norm(world - expected)
            print(f"    P{i+1}: 算出 ({world[0]:.3f}, {world[1]:.3f})  "
                  f"期望 ({expected[0]:.3f}, {expected[1]:.3f})  誤差 {err*100:.1f} cm")

        return H

    def show_verification_grid(self, cap, H):
        """
        在影像上疊加球場格線，讓你目視確認 H 矩陣是否正確。
        按任意鍵關閉。
        """
        print("\n[校準] 顯示驗證格線，確認格線是否對齊球場邊界...")
        print("  按任意鍵繼續")

        # 定義球場格線的全域座標（邊界 + 中線）
        grid_world = [
            # 邊界
            [(-0.5, 0.0), ( 0.5, 0.0)],
            [( 0.5, 0.0), ( 0.5, 3.0)],
            [( 0.5, 3.0), (-0.5, 3.0)],
            [(-0.5, 3.0), (-0.5, 0.0)],
            # 中線
            [(-0.5, 1.5), ( 0.5, 1.5)],
        ]

        H_inv = np.linalg.inv(H)

        def world_to_pixel(wx, wy):
            pt = np.array([[[wx, wy]]], dtype=np.float32)
            px = cv2.perspectiveTransform(pt, H_inv)[0][0]
            return (int(px[0]), int(px[1]))

        cv2.namedWindow("Verification", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verification", 960, 540)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis = frame.copy()

            for (wx1, wy1), (wx2, wy2) in grid_world:
                p1 = world_to_pixel(wx1, wy1)
                p2 = world_to_pixel(wx2, wy2)
                cv2.line(vis, p1, p2, (0, 255, 255), 2)

            cv2.putText(vis, "Verification: press any key to continue",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("Verification", vis)

            if cv2.waitKey(30) & 0xFF != 255:
                break

        cv2.destroyAllWindows()


# ==============================================================================
# 主節點
# ==============================================================================

class YoloRosNode:

    def __init__(self):
        rospy.init_node("yolo_detector")

        # UDP socket 連接 yolo_server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(YOLO_TIMEOUT_SEC)

        self.bridge = CvBridge()
        self.H      = None  # Homography 矩陣，校準後載入

        # Publisher：輸出全域座標給 fusion_node
        # 格式：[detected(0/1), X_world, Y_world, conf]
        self.pub = rospy.Publisher("/ball_camera_world",
                                   Float32MultiArray, queue_size=1)

        # Publisher：標註影像，供 RViz Image display 使用
        self.vis_pub = rospy.Publisher("/ball_detection_image",
                                       Image, queue_size=1)

        # 統計用
        self._det_count  = 0
        self._miss_count = 0

    def _grab_frame_from_topic(self, timeout=10.0):
        """
        從 ROS topic 抓一幀影像回來。
        校準和驗證都用這個方式持續取得畫面，不需要 VideoCapture。
        回傳 frame (BGR numpy array)，失敗回傳 None。
        """
        try:
            msg = rospy.wait_for_message("/camera/color/image_raw",
                                         Image, timeout=timeout)
            return self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except rospy.ROSException:
            rospy.logerr("[YOLO] 等待相機影像逾時，請確認相機節點已啟動")
            return None

    def _load_or_calibrate(self):
        """
        嘗試載入 H.npy，若不存在則進入校準流程。
        回傳 True 表示成功取得 H 矩陣。
        """
        if os.path.exists(H_FILE_PATH):
            self.H = np.load(H_FILE_PATH)
            rospy.loginfo(f"[YOLO] 已載入 Homography 矩陣：{H_FILE_PATH}")
            print(f"\n[YOLO] H 矩陣內容：\n{self.H}\n")
            return True

        # 沒有 H.npy，進入校準
        rospy.logwarn("[YOLO] 找不到 H.npy，啟動校準模式")
        print("[YOLO] 從 ROS topic 取得相機影像，請確認相機節點已啟動...")

        # 先抓一幀確認相機活著
        test_frame = self._grab_frame_from_topic(timeout=10.0)
        if test_frame is None:
            return False

        # 建立一個會持續從 topic 拉畫面的假 cap 物件
        # HomographyCalibrator.run() 需要 cap.read() 介面，這裡用 RosCapture 包裝
        ros_cap = RosTopicCapture(self)

        calibrator = HomographyCalibrator()
        H = calibrator.run(ros_cap)

        if H is None:
            return False

        # 顯示驗證格線
        calibrator.show_verification_grid(ros_cap, H)

        # 儲存
        np.save(H_FILE_PATH, H)
        rospy.loginfo(f"[YOLO] H 矩陣已儲存至 {H_FILE_PATH}")
        print(f"\n[YOLO] H 矩陣：\n{H}\n")

        self.H = H
        return True

    def _pixel_to_world(self, cx, cy):
        """
        將 YOLO 偵測到的 bounding box 底部中心點（像素）
        轉換為球場全域座標（公尺）。
        回傳 (X_world, Y_world)
        """
        # 取 bounding box 底部中心，球接觸地面的點比中心更準確
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(world[0]), float(world[1])

    def _call_yolo(self, frame):
        """
        把影像存成暫存檔，透過 UDP 送給 yolo_server，取回偵測結果。
        回傳 dict: {"detected": 0/1, "cx": px, "cy": py, "w": w, "h": h, "conf": c}
        """
        tmp_path       = "/dev/shm/yolo_frame_tmp.jpg"
        final_path     = "/dev/shm/yolo_frame.jpg"
        cv2.imwrite(tmp_path, frame)
        os.replace(tmp_path, final_path)   # 原子操作，避免 server 讀到一半的檔案

        self.sock.sendto(final_path.encode(), (YOLO_SERVER_IP, YOLO_SERVER_PORT))

        try:
            data, _ = self.sock.recvfrom(1024)
            return json.loads(data.decode())
        except socket.timeout:
            return {"detected": 0, "cx": 0, "cy": 0, "w": 0, "h": 0, "conf": 0.0}

    def image_callback(self, msg):
        """
        相機影像 callback：
          1. 呼叫 YOLO server
          2. 像素座標 → 全域座標
          3. 發布 /ball_camera_world
          4. 終端機印出結果
        """
        if self.H is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # --- YOLO 推論 ---
        r = self._call_yolo(frame)

        result_msg = Float32MultiArray()
        vis_frame = frame.copy()

        if r["detected"] == 1 and r.get("conf", 1.0) >= CONF_THRESHOLD:
            # bounding box 底部中心點（球接觸地面）
            cx_bottom = r["cx"]
            cy_bottom = r["cy"] + r["h"] / 2.0

            X_world, Y_world = self._pixel_to_world(cx_bottom, cy_bottom)
            conf = float(r.get("conf", 1.0))

            result_msg.data = [1.0, X_world, Y_world, conf]

            # 畫紅色框與信心度
            x1 = int(r["cx"] - r["w"] / 2.0)
            y1 = int(r["cy"] - r["h"] / 2.0)
            x2 = int(r["cx"] + r["w"] / 2.0)
            y2 = int(r["cy"] + r["h"] / 2.0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_frame, f"ball {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            self._det_count += 1
            if self._det_count % 10 == 1:
                print(f"[YOLO] ✓ X={X_world:+.3f}m  Y={Y_world:.3f}m  conf={conf:.2f}")

        else:
            result_msg.data = [0.0, 0.0, 0.0, 0.0]
            self._miss_count += 1
            if self._miss_count % 10 == 1:
                print(f"[YOLO] ✗ 未偵測到球  (miss:{self._miss_count})")

        self.pub.publish(result_msg)
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_frame, "bgr8"))

    def start(self):
        """
        節點主入口：先校準，成功後才開始訂閱相機 topic。
        """
        if not self._load_or_calibrate():
            rospy.logerr("[YOLO] 校準失敗，節點終止")
            sys.exit(1)

        rospy.Subscriber("/camera/color/image_raw", Image,
                         self.image_callback, queue_size=1,
                         buff_size=2**24)

        rospy.loginfo("[YOLO] 節點就緒，開始偵測...")
        print("\n[YOLO] 訂閱 /camera/color/image_raw")
        print("[YOLO] 發布 /ball_camera_world  格式: [detected, X_world, Y_world, conf]")
        print("[YOLO] 偵測結果會直接印在這個終端機，不需要 rostopic echo\n")

        rospy.spin()


# ==============================================================================
# 入口
# ==============================================================================

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    setup_log(log_path, sample_rate=5)
    try:
        node = YoloRosNode()
        node.start()
    except rospy.ROSInterruptException:
        pass