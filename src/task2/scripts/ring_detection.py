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

def separate(depth, dep=0):
	if(dep > 10): #max depth
		return []

	gut = np.sort(depth[depth!=0])
	if(len(gut) < 10):
		return []

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
	m_min = gut[0]
	m_max = gut[-1]

	if(max_diff < 0.01): #Todo: ugotovit ali je kaksna veza te ocene glede na razdaljo, ipd...
		mask = np.zeros_like(depth, dtype=np.uint8)
		mask[depth>0] = 255
		return [mask]

	#print(max_diff, m_max - m_min)

	g1 = depth.copy()
	g1[depth<m] = 0

	g2 = depth.copy()
	g2[depth>=m] = 0

	return separate(g1, dep+1) + separate(g2, dep+1)

def join_contours(contours, data, contour_norm, max_dist):
	LENGTH = len(contours)
	status = np.zeros((LENGTH,1))
	
	for i,cnt1 in enumerate(contours):
		mean1 = contour_norm(cnt1, data)
		
		if(np.linalg.norm(mean1) < 0.01): #Todo not ikzakli rajt
			continue
		
		x = i
		if i != LENGTH-1:
			for j,cnt2 in enumerate(contours[i+1:]):
				mean2 = contour_norm(cnt2, data)
				
				if(np.linalg.norm(mean2) < 0.01): #Todo not ikzakli rajt
					continue

				x = x+1
				dist = (np.linalg.norm(mean2-mean1) < max_dist)
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
	return unified

# def cont_pos(cont, data):
# 	a, mask1 = data
# 


class RingDetection(Node):
	def __init__(self):
		super().__init__('ring_detection')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.bridge = CvBridge()
		self.scan = None

		# For listening and loading the TF
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.rgb_sub = message_filters.Subscriber(self, Image,		 "/top_camera/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/top_camera/rgb/preview/depth/points")
		self.depth_sub = message_filters.Subscriber(self, Image,	 "/top_camera/rgb/preview/depth")

		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub, self.depth_sub], 10, 0.3, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_pc_callback)

		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

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
		mask[(xyz[:,:,1] > -0.168) & (xyz[:,:,1] < 1000)] = 255
		#mask[(xyz[:,:,1] > 0.4) & (xyz[:,:,1] < 1000)] = 255


		# A = 2*(cv2.getTrackbarPos("A", 'Image') / 1000) - 1
		# B = 2*(cv2.getTrackbarPos("B", 'Image') / 1000) - 1
		# img_display[(xyz[:,:,1] > A) & (xyz[:,:,1] < 1000)] = (0,0,0) ##TEGA DEJ

		depth_raw = self.bridge.imgmsg_to_cv2(depth_raw, "32FC1")
		depth_raw[depth_raw==np.inf] = 0
		
		depth = depth_raw.copy()
		depth[mask!=255] = 0
		# depth = (depth / np.max(depth)) * 255
		# depth = np.array(depth, dtype=np.uint8)

		#nastavi vse slike na belo
		mpty = np.full((height, width), 255, dtype=np.uint8)
		# for i in range(10):
		# 	cv2.imshow(f"Depth{i}", mpty)

		masks = separate(depth) #okej tole torej dela kjut razdeli glede na globino. To bi torej lahko blo plug'n play v detect rings2?
		for j,m in enumerate(masks):
			#cv2.imshow(f"Depth{i}", m)
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
				color = (color*255)
				color = [int(color[0]), int(color[1]), int(color[2])]
				
				color_index = int(((hue + 60)%360) / 120)
				color_names = ["red", "green", "blue"]
				color_name = color_names[color_index]
				# print(f"color: {color_name}")

				cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 1)
				cv2.putText(img_display, color_name, (cx ,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

				
				#Tu dobis krog torej velikost kroga / luknje
                ring_points = xyz[circle_img != (0,0,0)]
				colors_set = circle_img[circle_img != (0,0,0)]
                ring_position = sum(ring_points) / len(ring_points) #TODO: transform ring point to base_link tf

				q_size = min(w,h) / min(width,height)
				q_colors = math.exp(-0.1*np.std(colors_set[:]))
				q_okroglost = math.exp(-0.1*(1-max(w,h)/min(w,h)**2))
                q_distance = math.exp(-0.1*(0.15 - np.linalg.norm(ring_position))**2)
                
                q = min(q_size, q_okroglost, q_colors, q_distance)
                #TO posljemo
				cv2.putText(img_display, f"{std}", (x2 ,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) #Preveri ali je to sploh smiselno
                
                msg = RingInfo()
                msg.q = q
                msg.color = color
                msg.position = ring_position
                ##TODO pusblish msg



		#ocena:
		# velikost luknje
		# okroglost luknje
		# enotnost barve
		# oddaljuenost od obroca


		# edges = cv2.Canny(image=depth, threshold1=A, threshold2=B) # Canny Edge Detection
		# cv2.imshow('Canny Edge Detection', edges)
			
		cv2.imshow("Mask", mask)
		cv2.imshow("Image", img_display)

		depth_img = depth_raw.copy()
		depth_img = (depth_img / np.max(depth_img)) * 255
		depth_img = np.array(depth_img, dtype=np.uint8)
		cv2.imshow("Depth", depth_img)

def main():
	print("OK")
	rclpy.init(args=None)
	node = RingDetection()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
