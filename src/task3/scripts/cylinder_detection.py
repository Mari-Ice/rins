#!/usr/bin/env python3

import os
import random
import time
import math
import rclpy
import cv2
import numpy as np
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
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from lifecycle_msgs.srv import GetState
from task3.msg import CylinderInfo
from std_msgs.msg import Header

amcl_pose_qos = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

#	Kamera naj bo v looking_for_cylinders polozaju.

def array2point(arr):
	p = Point()
	p.x = float(arr[0])
	p.y = float(arr[1])
	p.z = float(arr[2])
	return p

def create_point_marker(point, header):
	marker = Marker()
	marker.header = header

	marker.type = 2

	marker.scale.x = 0.1
	marker.scale.y = 0.1
	marker.scale.z = 0.1

	marker.color.r = 1.0
	marker.color.g = 1.0
	marker.color.b = 1.0
	marker.color.a = 1.0
	marker.pose.position = array2point(point)
	
	return marker

class CylinderDetection(Node):
	def __init__(self):
		super().__init__('cylinder_detection')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		pwd = os.getcwd()
		self.gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]

		self.bridge = CvBridge()
		self.rgb_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/oakd/rgb/preview/depth/points")
		self.depth_sub = message_filters.Subscriber(self, Image,	 "/oakd/rgb/preview/depth")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub, self.depth_sub], 20, 0.03, allow_headerless=False) 
		self.ts.registerCallback(self.sensors_callback)

		self.marker_pub = self.create_publisher(Marker, "/cylinder_marker", QoSReliabilityPolicy.BEST_EFFORT)
		self.data_pub = self.create_publisher(CylinderInfo, "/cylinder_info", QoSReliabilityPolicy.BEST_EFFORT)

		self.received_any_data = False
		print("Init")
		return

	def sensors_callback(self, rgb_data, pc, depth_data):
		if(not self.received_any_data):
			self.received_any_data = True
			print("Data\nOK")

		# Prep data formats
		img 	= self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
		height	= pc.height
		width	= pc.width
		xyz = pc2.read_points_numpy(pc, field_names= ("x", "y", "z"))
		xyz = xyz.reshape((height,width,3))
		depth_raw = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
		depth_raw[depth_raw==np.inf] = 0
		hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		img_display = img.copy()

		cylinder_mask = np.zeros((height, width), dtype=np.uint8)
		cylinder_mask[(hsv_image[:,:,1] > 100) | (hsv_image[:,:,2] < 40)] = 255					#Color mask
		cylinder_mask[(xyz[:,:,2] < -0.19) | (xyz[:,:,2] > 0.15)] = 0 #Height mask

		#cv2.imshow(f"Mask", cylinder_mask)

		cv2.waitKey(1)

		contours_tupled, hierarchy = cv2.findContours(image=cylinder_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
		for i,c in enumerate(contours_tupled):
			c = cv2.convexHull(c)
			x1, y1, width, height = cv2.boundingRect(c)	
			if(width < 20 or height < 50):
				continue
			x2 = x1 + width
			y2 = y1 + height	

			# TODO, Meh, zis okaj
			x1 += 5
			y1 += 5
			x2 -= 5
			y2 -= 5
			width -= 10
			height -= 10

			per_cylinder_depth = depth_raw[y1:y2,x1:x2].copy()

			line_gradient = np.mean(per_cylinder_depth, axis=0)
			line_gradient -= np.min(line_gradient)
			line_gradient /= 0.3

			der = line_gradient[1:] - line_gradient[0:-1]

			if(der[0] > 0 or der[-1] < 0):
				continue

			k, n = np.polyfit(np.arange(0,len(der)),der,1)
			
			if(k < 0.00001 or k > 0.01):
				continue


			per_cylinder_pc = xyz[y1:y2,x1:x2]
			closest = xyz[int(y1+height/2),int(x1+width/2)]
			closest_dir = closest.copy() / np.linalg.norm(closest)
			center = closest + closest_dir * 0.125

			cheader = Header()
			cheader.stamp = rgb_data.header.stamp
			cheader.frame_id = "/oakd_link"
			cmarker = create_point_marker(center, cheader)
			self.marker_pub.publish(cmarker)

			# Ekstra podatki ki jih racunamo (barva, polozaj, razdalja, kvaliteta, kot)
			per_cylinder_img = img[y1:y2,x1:x2]
			img_pixels = np.reshape(per_cylinder_img, (-1,3))

			avg_color = np.mean(img_pixels, axis=0)	

			color_dist_sq = (img_pixels - avg_color)**2
			avg_color_dist = np.mean(color_dist_sq)
			
			per_cylinder_img_hsv = hsv_image[y1:y2,x1:x2]
			per_cylinder_hsv_values = np.reshape(per_cylinder_img_hsv, (-1,3))

			avg_hsv = np.mean(per_cylinder_hsv_values, axis=0)	
			hue = int(avg_hsv[0] * 360/255)
			color_index = int(((hue + 60)%360) / 120)

			if(avg_color[0] < 40 and avg_color[1] < 40 and avg_color[2] < 40):
				color_index = 3 #black

			q_color	= math.exp(-0.0004*avg_color_dist)
			q_size = 1
			q_distance  = math.exp(-0.08*(0.15 - np.linalg.norm(center))**2)

			cinfo = CylinderInfo()
			cinfo.color_index = color_index
			cinfo.color = avg_color.tolist()
			cinfo.position_relative = center.tolist()
			cinfo.yaw_relative = math.atan2(center[1], center[0])
			cinfo.quality = q_color * q_size * q_distance
			self.data_pub.publish(cinfo)

			img_display = cv2.rectangle(img_display, (x1, y1), (x2,y2), (0,0,255), 2)
			
		# cv2.imshow(f"Image", img_display)
		return

def main():
	rclpy.init(args=None)
	node = CylinderDetection()
	
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
