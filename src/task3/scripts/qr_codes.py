#! /usr/bin/env python3
import numpy as np

from qreader import QReader
import urllib.request

import cv2
from cv_bridge import CvBridge

from rclpy.node import Node

from sensor_msgs.msg import Image
import message_filters
from std_srvs.srv import Trigger
import rclpy


class Qr_reader(Node):
	
	def __init__(self):
		super().__init__('qr_reader')
		self.qreader = QReader()
		self.bridge = CvBridge()
		self.last_img = None

		self.srv_read_qr = self.create_service(Trigger, 'read_qr', self.read_qr_callback)
		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_image_sub], 10, 0.05, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_callback)
	
	def rgb_callback(self, rgb_data):
		self.last_img = rgb_data
		return
	
	def read_qr_callback(self, request, response):
		if(not self.last_img):
			return
		cv_image = self.bridge.imgmsg_to_cv2(self.last_img, "bgr8")
		cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
		
		(decoded,) = self.qreader.detect_and_decode(cv_image)
		
		if decoded:
			print(f"Decoded: {decoded}")
			response.success = True
		else:
			response.message = 'Nothing detected in QR code!'
			response.success = False
		if("http" in decoded):
			words = decoded.split()
			url = [w for w in words if "http" in w]
			url = url[0]
			with urllib.request.urlopen(url) as response:
				arr = np.asarray(bytearray(response.read()), dtype="uint8")
				img = cv2.imdecode(arr, -1)
				cv2.imshow("QR code image", img)
				cv2.waitKey(1)
				print('sending response')
		return response

def main():
	print('QR_reader node starting.')

	rclpy.init(args=None)
	node = Qr_reader()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
