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

		edge_img = cv_image.copy()
		hsv_image = cv2.cvtColor(edge_img, cv2.COLOR_BGR2HSV)
		edge_img[hsv_image[:,:,1] < 200] = (255,255,255)
		edge_img[(hsv_image[:,:,0] > 30) & (hsv_image[:,:,0] < 220)] = (255,255,255)
		gray = 255 - cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
		contours_tupled, hierarchy = cv2.findContours(image=gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

		contours = []
		for c in contours_tupled:
			c = cv2.convexHull(c)

			x_coords = c[:,:,0]
			y_coords = c[:,:,1]
			min_x_index = np.argmin(x_coords)
			max_x_index = np.argmax(x_coords)

			#CE y pri mx > kot y pri Mx potem dodas minx
			#sicer dodas max

			minx = int(x_coords[min_x_index])
			maxx = int(x_coords[max_x_index])
			center = 0.5*(minx+maxx)	
		
			left_block = []
			right_block = []

			for i,p in enumerate(c):
				p = p[0]
				if(p[0] <= center):
					left_block.append(p[1])	
				if(p[0] >= center):
					right_block.append(p[1])

			minx_y = max(left_block)
			maxx_y = max(right_block)
			
			append_pts = []
			for p in c:
				p = p[0]
				if(p[1] == 0): #Touches top
					if(minx_y < maxx_y):
						append_pts.append( [maxx,0] )
					else:
						append_pts.append( [minx,0] )
						
			for p in append_pts:
				c = np.vstack((c, np.array([[p]])))

			c = cv2.convexHull(c)
			contours.append(c)

		mask = np.zeros_like(gray)
		for i in range(len(contours)):
			cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
			
		masked = cv_image.copy()
		masked[mask == 0] = (255,255,255)
		#cv2.imshow("Masked", masked)

		# for c in contours:
		# 	c_box = cv2.boundingRect(c)	
		# 	cv_image = cv2.rectangle(cv_image, (c_box[0], c_box[1]), (c_box[0] + c_box[2], c_box[1] + c_box[3]), (0,255,0), 2)

		res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)
		for x in res:
			bbox = x.boxes.xyxy
			if bbox.nelement() == 0: #No element
				continue
			bbox = bbox[0]
			if(abs(bbox[2] - bbox[0]) < 50 or abs(bbox[3] - bbox[1]) < 80): #Too small
				continue
		   
			img_stats = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
			pis = self.prev_img_stats
			diff = abs(img_stats[0] - pis[0]) + abs(img_stats[1] - pis[1]) + abs(img_stats[2] - pis[2]) + abs(img_stats[3] - pis[3])
			self.prev_img_stats = img_stats
			print(self.face_count)

			#cv_image = cv2.rectangle(cv_image, (img_stats[0], img_stats[1]), (img_stats[2], img_stats[3]), (0,0,255), 3)
			
			img_surface = abs(img_stats[2] - img_stats[0]) * abs(img_stats[3] - img_stats[1])
	
			min_err = float("inf")
			min_index = 0
			cbox = img_surface
			
			for i,c in enumerate(contours):
				c_box = cv2.boundingRect(c)	
				c_box = [c_box[0], c_box[1], c_box[0] + c_box[2], c_box[1] + c_box[3]]
				c_surface = abs(c_box[2] - c_box[0]) * abs(c_box[3] - c_box[1])
				error = abs(c_surface - img_surface)
				if(error < min_err):
					min_err = error
					min_index = i
					cbox = [int(c_box[0]), int(c_box[1]), int(c_box[2]), int(c_box[3])]
				#cv_image = cv2.rectangle(cv_image, (c_box[0], c_box[1]), (c_box[2], c_box[3]), (0,255,0), 2)
		
			print(min_err)
			if(min_err < 10000):
			 	#cv_image = cv2.rectangle(cv_image, (cbox[0], cbox[1]), (cbox[2], cbox[3]), (255,0,0), 2)
				mona_img = masked[int(cbox[1]):int(cbox[3]), int(cbox[0]):int(cbox[2])]

				if(diff > 1):
					self.face_count += 1
					cv2.imwrite(f"{self.gpath}/mona_images/mona_{self.face_count:04}.jpg", mona_img)
					cv2.imshow("Mona", mona_img)

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
