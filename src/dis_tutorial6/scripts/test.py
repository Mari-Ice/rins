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
from sklearn.cluster import DBSCAN

amcl_pose_qos = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

def sign(x):
	if(x<0):
		return -1.
	return 1.	
def positive_angle(angle):
	while(angle < 0):
		angle += 2*math.pi
	return math.fmod(angle, 2*math.pi)

class test(Node):
	def __init__(self):
		super().__init__('floor_mask')

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

		self.rgb_sub = message_filters.Subscriber(self, Image,	     "/top_camera/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/top_camera/rgb/preview/depth/points")
		self.depth_sub = message_filters.Subscriber(self, Image,	 "/top_camera/rgb/preview/depth")

		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub, self.depth_sub], 10, 0.3, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_pc_callback)

		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

		cv2.namedWindow("Depth1", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Depth2", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Depth3", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Depth4", cv2.WINDOW_NORMAL)

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

		xyz = pc2.read_points_numpy(pc, field_names= ("y", "z", "x"))
		xyz = xyz.reshape((height,width,3))

		mask = np.full((height, width), 0, dtype=np.uint8)
		mask[(xyz[:,:,1] > -0.15) & (xyz[:,:,1] < 1000)] = 255

		depth_raw = self.bridge.imgmsg_to_cv2(depth_raw, "32FC1")
		depth = depth_raw.copy()
		depth[depth==np.inf] = 0
		depth[mask!=255] = 0
		# depth = (depth / np.max(depth)) * 255
		# depth = np.array(depth, dtype=np.uint8)

		gut = np.sort(depth[depth!=0])
		if(len(gut) > 0):
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

			g1 = depth.copy()
			print(max_diff, m_max - m_min)
			#if(m_max - m_min < 100):
			if(max_diff < 0.01):
				cv2.imshow("Depth1", g1)
			else:
				g1[depth<m] = 0
				g2 = depth.copy()
				g2[depth>=m] = 0
				cv2.imshow("Depth1", g1)
				cv2.imshow("Depth2", g2)


		#print(f"gutl: {len(gut)} uniq: {len(np.unique(gut))}, gut: {gut}")

		A = 1000*(cv2.getTrackbarPos("A", 'Image') / 1000)
		B = 1000*(cv2.getTrackbarPos("B", 'Image') / 1000)

		# edges = cv2.Canny(image=depth, threshold1=A, threshold2=B) # Canny Edge Detection
		# cv2.imshow('Canny Edge Detection', edges)
			
		cv2.imshow("Mask", mask)
		cv2.imshow("Image", img)
		#cv2.imshow("Depth", depth)

def main():
	print("OK")
	rclpy.init(args=None)
	node = test()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
