#!/usr/bin/env python3

import statistics
import random
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from geometry_msgs.msg import Point
import message_filters

from ultralytics import YOLO
import sys
import os

class face_extractor(Node):
	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_image_sub], 10, 0.05, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_callback)

		self.model = YOLO("yolov8n.pt")
		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

		self.face_count = 0
		self.prev_img_stats = [0,0,0,0]

		pwd = os.getcwd()
		self.gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]


	def rgb_callback(self, rgb_data):
		cv_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
		res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)
		for x in res:
			bbox = x.boxes.xyxy
			if bbox.nelement() == 0: #No element
				continue
			bbox = bbox[0]
			if(bbox[2] < 5 or bbox[3] < 5): #Too small
				continue

			img_stats = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
			pis = self.prev_img_stats
			diff = abs(img_stats[0] - pis[0]) + abs(img_stats[1] - pis[1]) + abs(img_stats[2] - pis[2]) + abs(img_stats[3] - pis[3])
			self.prev_img_stats = img_stats

			if(diff > 1):
				self.face_count += 1
				cv2.imwrite(f"{self.gpath}/mona_images/mona_{self.face_count:04}.jpg", cv_image)

			print(self.face_count)

			# draw rectangle
			cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 3)

		cv2.imshow("Image", cv_image)
		key = cv2.waitKey(1)
		if key==27: #Esc
			print("exiting")
			exit()

def main():
	rclpy.init(args=None)
	node = face_extractor()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
