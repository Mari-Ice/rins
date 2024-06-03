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
		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/oakd/rgb/preview/depth/points")
		self.depth_sub = message_filters.Subscriber(self, Image,	 "/top_camera/rgb/preview/depth")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub, self.depth_sub], 20, 0.1, allow_headerless=False) 
		self.ts.registerCallback(self.sensors_callback)

		self.marker_pub = self.create_publisher(Marker, "/cylinder_marker", QoSReliabilityPolicy.BEST_EFFORT)

		print("Init")
		return

	def sensors_callback(self, rgb_data, pc, depth_data):
		# Prep data formats
		img 	= self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
		height	= pc.height
		width	= pc.width
		xyz = pc2.read_points_numpy(pc, field_names= ("x", "y", "z"))
		xyz = xyz.reshape((height,width,3))
		depth_raw = self.bridge.imgmsg_to_cv2(depth_raw, "32FC1")
		depth_raw[depth_raw==np.inf] = 0
		hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		cylinder_mask = np.zeros((height, width), dtype=np.uint8)
		cylinder_mask[hsv_image[:,:,1] > 180] = 255				    #Color mask
		cylinder_mask[(xyz[:,:,2] < -0.2) | (xyz[:,:,2] > 0.4)] = 0 #Height mask
		#TODO: popravi za crno barvo

		contours_tupled, hierarchy = cv2.findContours(image=cylinder_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
		for c in contours_tupled:
			c = cv2.convexHull(c)
			x1, y1, width, height = cv2.boundingRect(c)	
			if(width < 10 or height < 10): #TODO: Nastavi prave meje...
				continue
			x2 = x1 + width
			y2 = y1 + height	

			per_cylinder_depth = depth_raw[y1:y2,x1:x2]
			line_gradient = np.mean(per_cylinder_depth, axis=0)
			line_gradient -= np.min(line_gradient)

			print(f"Line gradient: {line_gradient}")

			# Okej, to more bit zdej bit vedno neki zlo podobnega za cilindre.
			# TODO: preveri kaj dobis, fitaj mogoce na nek znan gradient...
			
			# Recimo da zgornje velja. Tedaj je treba najdit polozaj, zamaknjenost, barvo, kvaliteto
			# Podobobno kot pri ring_detection...
			# Pa poslat marker v rviz...

			# Vec opcij.
			#	1. Povprecje (vseh) vidnih tock
			#	2. Najdes najblizjo tocko in v pristejes radij cilindra. (Mogoce bolj ziher idk)

			# Probamo prvo opcijo 1:
			per_cylinder_pc = xyz[y1:y2,x1:x2]
			center = np.mean(per_cylinder_pc, axis=0)

			cheader = Header()
			cheader.stamp = rgb_data.header.stamp
			cheader.frame_id = "/oakd"
			cmarker = create_point_marker(center, cheader)
			self.marker_pub.publish(cmarker)

			# Ekstra podatki ki jih racunamo
			per_cylinder_img = img[y1:y2,x1:x2]
			avg_color = np.mean(per_cylinder_img, axis=0)	
			color_dist_sq = (per_cylinder_img[:] - avg_color)**2
			avg_color_dist = np.mean(color_dist_sq)
			
			per_cylinder_img_hsv = hsv_image[y1:y2,x1:x2]
			avg_hsv = np.mean(per_cylinder_img_hsv, axis=0)	
			color_index = int(((avg_hsv[0] + 45)%256) / 85) #Mogoce niso prave vresnoti tuke

			q_color	= math.exp(-0.0004*avg_color_dist)
			q_size = 1
			q_distance  = math.exp(-0.08*(0.15 - np.linalg.norm(center))**2)

			cinfo = CylinderInfo()
			cinfo.color_index = color_index
			cinfo.color = avg_color.tolist()
			cinfo.position = center.tolist()
			cinfo.quality = q_color * q_size * q_distance
			
		return

def main():
	rclpy.init(args=None)
	node = CylinderDetection()
	
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
