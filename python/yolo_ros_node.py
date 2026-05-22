#!/usr/bin/env python3
import rospy, socket, json, tempfile, os
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

bridge = CvBridge()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.1)
pub = rospy.Publisher("/ball_detection", Float32MultiArray, queue_size=1)

def image_callback(msg):
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")

    tmp_path = "/tmp/yolo_frame.jpg"
    
    # 先寫到暫存檔，確認寫完再改名，確保 server 讀到的是完整檔案
    tmp_write_path = "/tmp/yolo_frame_tmp.jpg"
    cv2.imwrite(tmp_write_path, frame)
    os.replace(tmp_write_path, tmp_path)  # os.replace 是原子操作，確保不會讀到一半
    
    sock.sendto(tmp_path.encode(), ("127.0.0.1", 9999))
    
    result_msg = Float32MultiArray()
    try:
        data, _ = sock.recvfrom(1024)
        r = json.loads(data.decode())
        result_msg.data = [float(r["detected"]), r["cx"], r["cy"], r["w"], r["h"]]
    except socket.timeout:
        result_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    pub.publish(result_msg)

rospy.init_node("yolo_detector")
rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
rospy.loginfo("YOLO ROS node started")
rospy.spin()
