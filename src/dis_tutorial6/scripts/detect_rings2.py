#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
import tf2_geometry_msgs as tfg
import message_filters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

class sky_mask(Node):
	def __init__(self):
		super().__init__('sky_mask')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/oakd/rgb/preview/depth/points")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub], 10, 0.3, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_pc_callback)

		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
		cv2.namedWindow("FatMask", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Circle", cv2.WINDOW_NORMAL)
		cv2.waitKey(1)
		
		cv2.moveWindow('Image',  1   ,1)
		cv2.moveWindow('Mask',   415 ,1)
		cv2.moveWindow('FatMask',830 ,1)
		cv2.moveWindow('Circle', 1245,1)

		# cv2.createTrackbar('A', "Image", 0, 100, self.nothing)
		# cv2.createTrackbar('B', "Image", 0, 100, self.nothing)

	def nothing(self, data):
		pass

	def rgb_pc_callback(self, rgb, pc):
		img = self.bridge.imgmsg_to_cv2(rgb, "bgr8")

		A = max(1,cv2.getTrackbarPos('A', 'Image'))
		B = max(1,cv2.getTrackbarPos('B', 'Image'))

		if((A % 2) == 0):
			A+=1
		if((B % 2) == 0):
			B+=1

		height     = pc.height
		width      = pc.width
		point_step = pc.point_step
		row_step   = pc.row_step		

		a = pc2.read_points_numpy(pc, field_names= ("x", "y", "z"))
		a = a.reshape((height,width,3))
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# Tole se zna pridet prav, ...
		# qube_mask = np.full((height, width), 0, dtype=np.uint8)
		# qube_mask[(hsv_img[:,:,1] > 35) & (hsv_img[:,:,2] > 35)] = 255
		# qube_mask[a[:,:,2] < 0.4] = 0
		# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (A,B))
		# qube_mask = cv2.morphologyEx(qube_mask, cv2.MORPH_OPEN, kernel1)
		# cv2.imshow("QubeMask", qube_mask)
		# img[qube_mask==255] = (179,179,179)
		# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		mask = np.full((height, width), 0, dtype=np.uint8)
		mask[(hsv_img[:,:,1] > 35) & (hsv_img[:,:,2] > 35)] = 255
		mask[a[:,:,2] < 0.4] = 0

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
		fat_mask = cv2.dilate(mask, kernel)
		
		contours, hierarchy = cv2.findContours(image=fat_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			x+=2
			y+=2
			w-=4
			h-=4

			cx = int(x + w/2)
			cy = int(y + h/2)

			if(w < 8 or h < 8):
				continue

			# pet zarkov, za vecjo stabilnost
			r0 = sum(a[cy,cx])
			r1 = sum(a[cy+1,cx])
			r2 = sum(a[cy,cx+1])
			r3 = sum(a[cy-1,cx])
			r4 = sum(a[cy,cx-1])

			if(min(r0,r1,r2,r3,r4) < 10000):
				continue

			circle_img = img[y:y+h,x:x+w]
			cv2.imshow("Circle", circle_img)

			color = circle_img.sum(axis=(0,1))
			color -= min(color)

			color_max = max(color)
			if(color_max == 0):
				continue

			color = (color / color_max)
			hue = (int(360 * rgb_to_hsv([color[2], color[1], color[0]])[0]) + 0) % 360
			color = (color*255)
			color = [int(color[0]), int(color[1]), int(color[2])]

			# print(f"hue: {hue}, color: {color}")
			
			color_index = int(((hue + 60)%360) / 120)
			color_names = ["red", "green", "blue"]
			color_name = color_names[color_index]
			# print(f"color: {color_name}")

			cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)
			cv2.putText(img, color_name, (cx ,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

		cv2.imshow("Image", img)
		cv2.imshow("Mask", mask)
		cv2.imshow("FatMask", fat_mask)
		cv2.waitKey(1)

def main():
	print("OK")
	rclpy.init(args=None)
	node = sky_mask()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
