#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 收集laser data (限時自動關閉版)

import rospy
import math
import datetime
from sensor_msgs.msg import LaserScan

counter = 0
fp = None
record_time = 30.0  # 預設錄製時間 (秒)

def RAD2DEG(r):
    return r * 180.0 / math.pi

def callback(event):
    global counter
    counter += 1
    # 改為進度提示，讓終端機看起來更專業
    rospy.loginfo("[採集進度] 已運行 %.1f 秒...", counter * 0.1)

def auto_stop_callback(event):
    """
    時間到達時觸發的 Callback。負責優雅地關閉節點。
    """
    rospy.loginfo("========================================")
    rospy.loginfo("⏱️ 達到設定的採集時間 (%.1f 秒)！", record_time)
    rospy.loginfo("準備停止採集並安全保存檔案...")
    rospy.loginfo("========================================")
    # 發送關閉訊號，這會打破 rospy.spin() 的迴圈
    rospy.signal_shutdown("Data collection time limit reached.")

def scanCallback(scan):
    global fp
    # 確保在檔案關閉或尚未開啟時不寫入資料
    if fp is None or fp.closed:
        return

    scan_num = len(scan.ranges)

    # 將每個點寫入檔案 (angle: radians; range: meters)
    for i in range(scan_num):
        angle = scan.angle_min + i * scan.angle_increment
        distance = scan.ranges[i]
        
        # 把非數值處理為 NaN（MATLAB 會識別）
        if not math.isfinite(distance):
            distance = float('nan')
            
        fp.write("{:0.6f} {:0.6f}\n".format(angle, distance))

    # 空行分隔不同掃描 (frame)[cite: 13]
    fp.write("\n")
    fp.flush()

def listener():
    global fp, record_time
    rospy.init_node('py_sample_save', anonymous=True)

    # 【工程師的最佳實踐】：將錄製時間設為 ROS Parameter
    # 這樣你可以用 rosrun package_name get_laser_data.py _time:=30 來動態改變時間
    record_time = rospy.get_param('~time', 60.0)

    # 建一個以時間命名的檔案，方便管理[cite: 13]
    timestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "laser_goal_{}s_{}.dat".format(int(record_time), timestr)
    rospy.loginfo("將雷射數據保存至: %s", filename)
    fp = open(filename, "w")

    # 10Hz 進度計時器 (每 0.1 秒印一次)[cite: 13]
    timer1 = rospy.Timer(rospy.Duration(0.1), callback)
    
    # 【新增】：一次性計時器，倒數 60 秒後觸發 auto_stop_callback
    stop_timer = rospy.Timer(rospy.Duration(record_time), auto_stop_callback, oneshot=True)

    sub = rospy.Subscriber("/scan", LaserScan, scanCallback)

    try:
        rospy.loginfo(">>> 開始錄製！請開始讓機器人與目標物移動 <<<")
        rospy.spin() # 程式會停在這裡，直到被 signal_shutdown 喚醒[cite: 13]
    except KeyboardInterrupt:
        rospy.loginfo("偵測到強制中斷 (Ctrl+C)...")
    finally:
        # 無論是時間到還是手動中斷，都會執行這區塊，確保檔案完整關閉[cite: 13]
        rospy.loginfo("關閉檔案流，確保最後一幀資料完整...")
        if fp and not fp.closed:
            fp.close()
        rospy.loginfo("資料採集節點已完全退出。")

if __name__ == '__main__':
    listener()
