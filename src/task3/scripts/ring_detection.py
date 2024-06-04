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
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from lifecycle_msgs.srv import GetState
from task3.msg import RingInfo

#	Kamera naj bo v looking_for_rings polozaju.
#	Zaznat je treba camera clipping, ker takrat zadeve ne delajo prav. 

def nothing(data):
	pass

def lin_map(x, from_min, from_max, to_min, to_max):
	normalized_x = (x - from_min) / (from_max - from_min)
	mapped_value = normalized_x * (to_max - to_min) + to_min
	return mapped_value
def clamp(x, min_x, max_x):
	return min(max(x,min_x),max_x)

def array2point(arr):
	p = Point()
	p.x = float(arr[0])
	p.y = float(arr[1])
	p.z = float(arr[2])
	return p

def create_point_marker(position, header):
	marker = Marker()
	marker.header = header
	
	marker.type = 2
	
	scale = 0.1
	marker.scale.x = scale
	marker.scale.y = scale
	marker.scale.z = scale
	
	marker.color.r = 1.0
	marker.color.g = 1.0
	marker.color.b = 1.0
	marker.color.a = 1.0
	marker.pose.position = array2point(position)
	
	return marker
		
def separate(depth, dep=0):
	gut = np.sort(depth[depth!=0])
	if(len(gut) < 10):
		return []

	if(dep > 7): #max depth
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
		self.tf_buffer = Buffer()
		self.rgb_sub = message_filters.Subscriber(self, Image,		 "/top_camera/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/top_camera/rgb/preview/depth/points")
		self.depth_sub = message_filters.Subscriber(self, Image,	 "/top_camera/rgb/preview/depth")

		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub, self.depth_sub], 10, 0.03, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_pc_callback)

		#msg publisher
		self.ring_info_pub = self.create_publisher(RingInfo, "/ring_info", QoSReliabilityPolicy.BEST_EFFORT)
		self.ring_marker_pub = self.create_publisher(Marker, "/ring_marker", QoSReliabilityPolicy.BEST_EFFORT)

		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		self.received_any_data = False
		print("Init")

	def rgb_pc_callback(self, rgb, pc, depth_raw):
		if(not self.received_any_data):
			self.received_any_data = True
			print("Data\nOK")

		cv2.waitKey(1)
		img = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
		height	 = pc.height
		width	  = pc.width
		point_step = pc.point_step
		row_step   = pc.row_step		
		xyz = pc2.read_points_numpy(pc, field_names= ("y", "z", "x"))
		xyz = xyz.reshape((height,width,3))

		#obrezemo vse slike, da je lazje za racunat.
		depth_raw = self.bridge.imgmsg_to_cv2(depth_raw, "32FC1")
		depth_raw[depth_raw==np.inf] = 0

		cut_y1 = 30
		cut_y2 = height - 60
		img = img[cut_y1:cut_y2,:,:]
		xyz = xyz[cut_y1:cut_y2,:,:]
		depth_raw = depth_raw[cut_y1:cut_y2,:]
		height = height - 90

		img_display = img.copy()
		mask = np.full((height, width), 0, dtype=np.uint8)
		mask[(xyz[:,:,1] > -0.162) & (xyz[:,:,1] < 1000)] = 255 #Height masking and depth image masking
		
		depth = depth_raw.copy()
		depth[mask!=255] = 0

		masks = separate(depth) 
		for j,m in enumerate(masks):
			#cv2.imshow(f"Depth{j}", m)
			mask1 = m
			mask = mask1.copy()
			cv2.floodFill(mask, None, (int(width/2),int(height-1)), 255) 
			mask = 255-mask
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
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
				circle_img[circle_mask==0] = (255,0,255)
				#cv2.imshow(f"Circle{j}_{i}", circle_img)

				ring_points = circle_xyz[circle_mask != 0]
				colors_set = circle_img[circle_mask != 0]
				ring_position = np.sum(ring_points, axis=0) / len(ring_points)
				avg_color = np.mean(colors_set, axis=0)

				if(avg_color[0] < 40 and avg_color[1] < 40 and avg_color[2] < 40):
					color_index = 3 #black
					color = (0.,0.,0.)
					color_uint = (0,0,0)
				else:
					color = avg_color - min(avg_color)
					color_max = max(color)
					if(color_max == 0):
						continue

					color = (color / color_max)
					hue = (int(360 * rgb_to_hsv([color[2], color[1], color[0]])[0]) + 0) % 360
					color_uint = (color*255)
					color_uint = [int(color_uint[0]), int(color_uint[1]), int(color_uint[2])]
					color_index = int(((hue + 60)%360) / 120)

				color_names = ["red", "green", "blue", "black"]
				color_name = color_names[color_index]

				color_dist_sq = (colors_set - avg_color)**2
				avg_color_dist = np.mean(color_dist_sq)

				if(color_index == 3):#black
					avg_color_dist *= 0.1

				q_colors	= math.exp(-0.0004*avg_color_dist)
				q_size	    = 1.0-math.exp(-10.0*(min(w,h) / min(width,height)))
				q_okroglost = math.exp(-3.27*(1-min(w,h)/max(w,h))**2)
				q_distance  = math.exp(-0.08*(0.15 - np.linalg.norm(ring_position))**2)

				q = q_colors * q_size * q_okroglost * q_distance

				cv2.rectangle(img_display, (x1, y1), (x2, y2), color_uint, 1)
				cv2.putText(img_display, color_name, (x2 ,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_uint, 2)
				cv2.putText(img_display, f"{q:.2f}", (x2 ,cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_uint, 2)
			
				#Calc ring norma
				left_point_index = np.argmin(ring_points, axis=0)[0]
				left_point = ring_points[left_point_index]
				normal = np.cross(np.array([0,1,0]), (ring_position - left_point))
				normal[1] = 0
				nlen = np.linalg.norm(normal)
				if(nlen > 0.0001):
					normal /= nlen

				#Ustvarimo marker
				marker_header = Header()
				marker_header.frame_id = "/top_camera_link"
				marker_header.stamp = rgb.header.stamp
				marker = create_point_marker([float(ring_position[2]), float(ring_position[0]), float(ring_position[1])], marker_header)
				marker.id = random.randint(1,100000)
				marker.lifetime = Duration(seconds=.1).to_msg()
				self.ring_marker_pub.publish(marker)

				# Calculate global position
				time_now = rclpy.time.Time()
				timeout = Duration(seconds=0.5)
				transform = self.tf_buffer.lookup_transform("map", "top_camera_link", time_now, timeout)	

				position_point = PointStamped() #robot global pos
				position_point.header.frame_id = "/map"
				position_point.header.stamp = rgb.header.stamp
				position_point.point.x = float(ring_position[2])
				position_point.point.y = float(ring_position[0])
				position_point.point.z = float(ring_position[1])
				pp_global = tfg.do_transform_point(position_point, transform)
				
				msg = RingInfo()
				msg.color_index = color_index
				msg.position_relative = [float(ring_position[2]), float(ring_position[0]), float(ring_position[1])]
				msg.normal_relative = [float(normal[2]), float(normal[0]), float(normal[1])]
				msg.q = q
				msg.position = [float(pp_global.point.x), float(pp_global.point.y), float(pp_global.point.z)]

				self.ring_info_pub.publish(msg)

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
