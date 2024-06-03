import numpy as np

from qreader import QReader
import urllib.request

import cv2
from cv_bridge import CvBridge

from rclpy.node import Node

from sensor_msgs.msg import Image
import message_filters

class qr_reader(Node):
	
	def __init__(self):
		self.qreader = QReader()
		self.bridge = CvBridge()
		
		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_image_sub], 10, 0.05, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_callback)
	
	def rgb_callback(self, rgb_data):
		cv_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
		cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
		
		(decoded,) = self.qreader.detect_and_decode(cv_image)
		
		if decoded:
			print(f"Decoded: {decoded}")
		
		if(decoded.startswith("http")):
			with urllib.request.urlopen(decoded) as response:
				arr = np.asarray(bytearray(response.read()), dtype="uint8")
				img = cv2.imdecode(arr, -1)
				cv2.imshow("QR code image", img)
				cv2.waitKey(0)
		
		