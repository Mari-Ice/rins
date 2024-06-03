import numpy as np

from qreader import QReader

import cv2
from cv_bridge import CvBridge
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

if __name__ == '__main__':

    qreader = QReader()
    
    qr_code_path = "/home/domen/colcon_ws/rins/src/dis_tutorial3/worlds/task3_meshes/qr_url.png"
    cv_image = cv2.imread(qr_code_path)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
    (decoded,) = qreader.detect_and_decode(cv_image)
    
    if decoded:
        print(f"Decoded: {decoded}")
    else:
        print("No QR code found")
    
    if(decoded.startswith("http")):
        with urllib.request.urlopen(decoded) as response:
            arr = np.asarray(bytearray(response.read()), dtype="uint8")
            img = cv2.imdecode(arr, -1)
            cv2.imshow("Image", img)
            cv2.waitKey(0)
        
        