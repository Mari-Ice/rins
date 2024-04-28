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
		cv2.namedWindow("Circle", cv2.WINDOW_NORMAL)

		# cv2.namedWindow("c0", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("c1", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("c2", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("c3", cv2.WINDOW_NORMAL)

		cv2.waitKey(1)
		
		cv2.moveWindow('Image',  1   ,1)
		cv2.moveWindow('Mask',   415 ,1)
		cv2.moveWindow('Circle', 830 ,1)
		# cv2.moveWindow('Circle', 1245,1)
		
		cv2.moveWindow("c0", 1   ,450)
		cv2.moveWindow("c1", 415 ,450)
		cv2.moveWindow("c2", 830 ,450)
		cv2.moveWindow("c3", 1245,450)


	def cont_pos(self, cont, a, mask1):
		x,y,w,h = cv2.boundingRect(cont)
		masked_ring_a = a[y:y+h,x:x+w].copy()
		masked_ring_a[mask1[y:y+h,x:x+w]!=255] = float('NaN')
		masked_ring_a[(a[y:y+h,x:x+w]>10000)]  = float('NaN')

		result = np.array([0,0,0],dtype=np.float64)
		cnt = 1

		for y in masked_ring_a:
			for x in y:
				if(not np.isnan(np.min(x))):
					result[0] += x[0]
					result[1] += x[1]
					result[2] += x[2]
					cnt += 1
		result *= 1/cnt
		return result
		
	def nothing(self, data):
		pass

	def rgb_pc_callback(self, rgb, pc):
		cv2.waitKey(1)
		img = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
		img_display = img.copy()
		height	 = pc.height
		width	  = pc.width
		point_step = pc.point_step
		row_step   = pc.row_step		

		a = pc2.read_points_numpy(pc, field_names= ("x", "y", "z"))
		a = a.reshape((height,width,3))
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# Ring mask
		mask1 = np.full((height, width), 0, dtype=np.uint8)
		mask1[(hsv_img[:,:,1] > 25) & (hsv_img[:,:,2] > 25)] = 255
		mask1[a[:,:,2] < 0.4] = 0
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
		mask1 = cv2.dilate(mask1, kernel)

		# Get ring contours
		contours, hierarchy = cv2.findContours(image=mask1, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

		
		#cv2.imshow("Image", img)


		LENGTH = len(contours)
		status = np.zeros((LENGTH,1))
		
		for i,cnt1 in enumerate(contours):
			mean1 = self.cont_pos(cnt1, a, mask1)
			if(np.linalg.norm(mean1) < 0.01):
				continue
			
			x = i
			if i != LENGTH-1:
				for j,cnt2 in enumerate(contours[i+1:]):
					mean2 = self.cont_pos(cnt2, a, mask1)
					if(np.linalg.norm(mean2) < 0.01):
						continue

					x = x+1
					dist = (np.linalg.norm(mean2-mean1) < 0.25)
					if dist == True:
						val = min(status[i],status[x])
						status[x] = status[i] = val
					else:
						if status[x]==status[i]:
							status[x] = i+1
		
		unified = []
		if(len(status) != 0):
			maximum = int(status.max())+1
			for i in range(maximum):
				pos = np.where(status==i)[0]
				if pos.size != 0:
					cont = np.vstack(tuple(contours[i] for i in pos))
					hull = cv2.convexHull(cont)
					unified.append(hull)

		# cv2.drawContours(img, unified, -1, (0,0,255), 2)
		# cv2.drawContours(img, contours, -1, (0,255,0), 1)
		# for i,c in enumerate(contours):
		# 	dist = self.cont_pos(c, a, mask1)
		# 	x,y,w,h = cv2.boundingRect(c)
		# 	cv2.putText(img, f"{dist[0]:.2f}", (x ,y+ 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (x,y), 1)
		# 	cv2.putText(img, f"{dist[1]:.2f}", (x ,y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (x,y), 1)
		# 	cv2.putText(img, f"{dist[2]:.2f}", (x ,y+19), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (x,y), 1)

		# cv2.imshow("Image", img)
		# return	

		for i,c in enumerate(unified):
			touches_top = any(0 in y for y in c[:,:,1])
			touches_left = any(0 in x for x in c[:,:,0])
			touches_right = any(width-1 in x for x in c[:,:,0])
			if(touches_top and touches_left):
				c = np.vstack((c, np.array([[[0, 0]]])))
			if(touches_top and touches_right):
				c = np.vstack((c, np.array([[[width-1, 0]]])))
			hull = cv2.convexHull(c)
			cv2.fillPoly(mask1, pts=[hull], color=255)


		# cv2.imshow("Mask1", mask1)

		# Floor mask
		mask = np.full((height, width), 255, dtype=np.uint8)
		mask[(a[:,:,2] < 1000) | (mask1==255)] = 0

		# Get sky contours
		contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
	
		# Fill sky and ground
		mask = np.full((height, width), 0, dtype=np.uint8)
		mask[(a[:,:,2] < 1000)] = 255
		for i,c in enumerate(contours):
			cv2.floodFill(mask, None, c[0][0], 255) 

		cv2.imshow("Mask", mask)
		mask = 255-mask
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
		mask = cv2.dilate(mask, kernel)
		mask[a[:,:,2] > 1000] = 0
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
		mask = cv2.dilate(mask, kernel)

		contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

		for i,c in enumerate(contours):
			x,y,w,h = cv2.boundingRect(c)
			if(w < 7 or h < 7):
				continue

			x1 = max(0,x-0)
			y1 = max(0,y-0)
				
			x2 = min(width , x+w+0)
			y2 = min(height, y+h+0)

			cx = int((x1 + x2)/2)
			cy = int((y1 + y2)/2)

			circle_img  = img[y1:y2,x1:x2].copy()
			circle_mask = mask[y1:y2,x1:x2]
			circle_img[circle_mask==0] = (0,0,0)
			cv2.imshow(f"Circle", circle_img)

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

			cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 1)
			cv2.putText(img_display, color_name, (cx ,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

		cv2.imshow("Image", img_display)

def main():
	print("OK")
	rclpy.init(args=None)
	node = sky_mask()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
