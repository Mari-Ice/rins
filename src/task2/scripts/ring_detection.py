#!/usr/bin/env python3

import random
import time
import math
import rclpy
import cv2
import numpy as np
import tf2_geometry_msgs as tfg
import message_filters
from enum import Enum
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
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
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from lifecycle_msgs.srv import GetState
from task2.msg import RingInfo

amcl_pose_qos = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)


#	Kamera naj bo v looking_for_rings polozaju.
#	Zaznat je treba camera clipping, ker takrat zadeve ne delajo prav. 

def lin_map(x, from_min, from_max, to_min, to_max):
	normalized_x = (x - from_min) / (from_max - from_min)
	mapped_value = normalized_x * (to_max - to_min) + to_min
	return mapped_value
def clamp(x, min_x, max_x):
	return min(max(x,min_x),max_x)
		
def separate(depth, dep=0):
	gut = np.sort(depth[depth!=0])
	if(len(gut) < 10):
		return []

	if(dep > 10): #max depth
		mask = np.zeros_like(depth, dtype=np.uint8)
		mask[depth>0] = 255
		return [mask]

	max_diff = 0
	max_diff_index = 0
	for i in range(1,len(gut)):
		b = gut[i]
		a = gut[i-1]	
		diff = b-a
		if(diff > max_diff):
			max_diff = diff
			max_diff_index = i

	m = gut[max_diff_index]
	m_min = (gut[0] + gut[1] + gut[2])/3
	m_max = (gut[-1] + gut[-2] + gut[-3])/3
	m_diff = m_max - m_min

	thresh = 0.025
	if(max_diff < thresh):
		mask = np.zeros_like(depth, dtype=np.uint8)
		mask[depth>0] = 255
		return [mask]

	g1 = depth.copy()
	g1[depth<m] = 0

	g2 = depth.copy()
	g2[depth>=m] = 0

	return separate(g1, dep+1) + separate(g2, dep+1)

class RingDetection(Node):
	def __init__(self):
		super().__init__('ring_detection')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.bridge = CvBridge()

		# For listening and loading the TF
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.rgb_sub = message_filters.Subscriber(self, Image,		 "/top_camera/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/top_camera/rgb/preview/depth/points")
		self.depth_sub = message_filters.Subscriber(self, Image,	 "/top_camera/rgb/preview/depth")

		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub, self.depth_sub], 10, 0.3, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_pc_callback)

		#msg publisher
		self.rings_info_pub = self.create_publisher(RingInfo, "/rings_info", QoSReliabilityPolicy.BEST_EFFORT)

		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

		# cv2.namedWindow("Depth0", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("Depth1", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("Depth2", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("Depth3", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("Depth4", cv2.WINDOW_NORMAL)

		cv2.waitKey(1)
		cv2.moveWindow('Image',  1   ,1)
		cv2.moveWindow('Mask',   415 ,1)
		cv2.moveWindow('Depth',   830 ,1)

		cv2.moveWindow('Depth1',   1 ,500)
		cv2.moveWindow('Depth2',   415 ,500)
		cv2.moveWindow('Depth3',   830 ,500)
		cv2.moveWindow('Depth4',   1245 ,500)

		cv2.createTrackbar('A', "Image", 0, 1000, self.nothing)
		cv2.createTrackbar('B', "Image", 0, 1000, self.nothing)
		
		cv2.setTrackbarPos("A", "Image", 100)
		cv2.setTrackbarPos("B", "Image", 200)

	def nothing(self, data):
		pass

	def rgb_pc_callback(self, rgb, pc, depth_raw):
		cv2.waitKey(1)
		img = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
		height	 = pc.height
		width	  = pc.width
		point_step = pc.point_step
		row_step   = pc.row_step		

		img_display = img.copy()

		xyz = pc2.read_points_numpy(pc, field_names= ("y", "z", "x"))
		xyz = xyz.reshape((height,width,3))

		mask = np.full((height, width), 0, dtype=np.uint8)
		mask[(xyz[:,:,1] > -0.162) & (xyz[:,:,1] < 1000)] = 255
		#mask[(xyz[:,:,1] > 0.4) & (xyz[:,:,1] < 1000)] = 255


		A = 2*(cv2.getTrackbarPos("A", 'Image') / 1000) - 1
		# B = 2*(cv2.getTrackbarPos("B", 'Image') / 1000) - 1
		img_display[(xyz[:,:,1] <= A)] = (0,0,0) ##TEGA DEJ

		depth_raw = self.bridge.imgmsg_to_cv2(depth_raw, "32FC1")
		depth_raw[depth_raw==np.inf] = 0
		
		depth = depth_raw.copy()
		depth[mask!=255] = 0
		# depth = (depth / np.max(depth)) * 255
		# depth = np.array(depth, dtype=np.uint8)

		#nastavi vse slike na belo
		mpty = np.full((height, width), 255, dtype=np.uint8)
		# for i in range(10):
		#	 cv2.imshow(f"Depth{i}", mpty)

		masks = separate(depth) #okej tole torej dela kjut razdeli glede na globino. To bi torej lahko blo plug'n play v detect rings2?
		for j,m in enumerate(masks):
			#cv2.imshow(f"Depth{j}", m)
			mask1 = m
			#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
			#mask = cv2.dilate(mask1, kernel)
			mask = mask1.copy()
			cv2.floodFill(mask, None, (int(width/2),int(height-1)), 255) 
			mask = 255-mask
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
			#mask = cv2.dilate(mask, kernel) #za spodnjo kamero
			#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
			mask = cv2.dilate(mask, kernel)
			mask[xyz[:,:,1] > 1000] = 0

			#cv2.imshow(f"Mask{j}", mask)

			contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
			for i,c in enumerate(contours):
				x,y,w,h = cv2.boundingRect(c)
				if(w < 5 or h < 5):
					continue

				x1 = max(0,x-0)
				y1 = max(0,y-0)
					
				x2 = min(width , x+w+0)
				y2 = min(height, y+h+0)

				cx = int((x1 + x2)/2)
				cy = int((y1 + y2)/2)

				circle_xyz = xyz[y1:y2,x1:x2]
				circle_img  = img[y1:y2,x1:x2].copy()
				circle_mask = mask1[y1:y2,x1:x2]
				circle_img[circle_mask==0] = (0,0,0)
				#cv2.imshow(f"Circle{j}_{i}", circle_img)

				color = circle_img.sum(axis=(0,1))
				color -= min(color)

				color_max = max(color)
				if(color_max == 0):
					continue

				color = (color / color_max)
				hue = (int(360 * rgb_to_hsv([color[2], color[1], color[0]])[0]) + 0) % 360
				color_uint = (color*255)
				color_uint = [int(color_uint[0]), int(color_uint[1]), int(color_uint[2])]
				
				color_index = int(((hue + 60)%360) / 120)
				color_names = ["red", "green", "blue"]
				color_name = color_names[color_index]
				# print(f"color: {color_name}")

				cv2.rectangle(img_display, (x1, y1), (x2, y2), color_uint, 1)
				#cv2.putText(img_display, color_name, (cx ,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

				
				#Tu dobis krog torej velikost kroga / luknje
				ring_points = circle_xyz[circle_mask != 0]
				colors_set = circle_img[circle_mask != 0]
				ring_position = np.sum(ring_points, axis=0) / len(ring_points) #TODO: transform ring point to base_link tf

				avg_color = np.sum(colors_set, axis=0) / len(colors_set)
				color_dist_sq = (colors_set - avg_color)**2
				avg_color_dist = np.mean(color_dist_sq)

				q_colors	= math.exp(-0.0004*avg_color_dist)
				q_size	  = 1.0-math.exp(-10.0*(min(w,h) / min(width,height)))
				q_okroglost = math.exp(-3.27*(1-min(w,h)/max(w,h))**2)
				q_distance  = math.exp(-0.08*(0.15 - np.linalg.norm(ring_position))**2)

				q_geom = (q_size + q_okroglost + q_distance)/3
				q = min(q_geom, q_colors)

				# #TO posljemo
				# cv2.putText(img_display, f"{q1:.3f}", (x2 ,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) #Preveri ali je to sploh smiselno
				# cv2.putText(img_display, f"{q2:.3f}", (x2 ,cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) #Preveri ali je to sploh smiselno
				# cv2.putText(img_display, f"{q3:.3f}", (x2 ,cy+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) #Preveri ali je to sploh smiselno
				# cv2.putText(img_display, f"{q4:.3f}", (x2 ,cy+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) #Preveri ali je to sploh smiselno
				cv2.putText(img_display, f"{q:.2f}", (x2 ,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_uint, 2) #Preveri ali je to sploh smiselno
				
				msg = RingInfo()
				msg.q = q
				msg.color = [color[2], color[1], color[0]]
				msg.position = [float(ring_position[2]), float(ring_position[0]), float(ring_position[1]) ]
				self.rings_info_pub.publish(msg)

		cv2.imshow("Image", img_display)

def main():
	print("OK")
	rclpy.init(args=None)
	node = RingDetection()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
