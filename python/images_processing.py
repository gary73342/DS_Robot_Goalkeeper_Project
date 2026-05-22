import os
import cv2
import rosbag
from cv_bridge import CvBridge

def extract_images_from_bag(bag_path, output_dir, topic_name, frame_step=5):
    """
    從 rosbag 中抽取影像並存成 jpg
    :param bag_path: bag 檔案路徑
    :param output_dir: 圖片輸出的資料夾
    :param topic_name: 影像的 topic 名稱
    :param frame_step: 降頻抽樣的間隔 (預設每 5 幀抽一張)
    """
    # 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bag = rosbag.Bag(bag_path, "r")
    bridge = CvBridge()
    count = 0
    saved_count = 0

    print(f"開始讀取 {bag_path} ...")
    
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if count % frame_step == 0:
            try:
                # 將 ROS Image message 轉換為 OpenCV 影像
                # 若是彩色影像，bgr8 是最安全的格式
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                
                # 組合檔名 (包含時間戳記，這對未來的 Sensor Fusion 很有幫助)
                img_name = f"frame_{t.to_nsec()}.jpg"
                img_path = os.path.join(output_dir, img_name)
                
                cv2.imwrite(img_path, cv_img)
                saved_count += 1
                
            except Exception as e:
                print(f"轉換影像時發生錯誤: {e}")
                
        count += 1

    bag.close()
    print(f"抽取完成！共處理 {count} 幀，實際儲存 {saved_count} 張圖片於 {output_dir}")

if __name__ == '__main__':
    # 執行設定
    BAG_FILE = os.path.expanduser('~/DS_Robot_Goalkeeper_Project/data/camera_data/camera0518_0320.bag')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FOLDER = os.path.join(script_dir, '..', 'data', 'images_dataset')
    IMAGE_TOPIC = '/camera/color/image_raw'
    
    # 這裡的 frame_step 建議設為 5 或 10，根據你球移動的速度決定
    extract_images_from_bag(BAG_FILE, OUTPUT_FOLDER, IMAGE_TOPIC, frame_step=5)
