#!/usr/bin/env python3
import socket, json
import cv2
from ultralytics import YOLO

model = YOLO("/home/weiting/DS_Robot_Goalkeeper_Project/runs/detect/train/weights/best.pt")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 9999))

print("YOLO server ready...")
while True:
    data, addr = sock.recvfrom(65535)
    img_path = data.decode()
    
    frame = cv2.imread(img_path)
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        response = {"detected": 0, "cx": 0.0, "cy": 0.0,
                    "w": 0.0, "h": 0.0, "conf": 0.0}
    else:
        best = max(boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        conf = float(best.conf[0])          
        response = {
            "detected": 1,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "w":  x2 - x1,
            "h":  y2 - y1,
            "conf": conf,       # 信心度           
        }

    sock.sendto(json.dumps(response).encode(), addr)