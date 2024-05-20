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
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from lifecycle_msgs.srv import GetState
from task2.msg import CylinderInfo

amcl_pose_qos = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)


#	Kamera naj bo v looking_for_cylinders polozaju.
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

class CylinderDetection(Node):
	def __init__(self):
		super().__init__('cylinder_detection')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.bridge = CvBridge()

		# For listening and loading the TF
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		
		self.cylinder_sub = message_filters.Subscriber(self, PointCloud2, "/cylinder")

		self.ts = message_filters.ApproximateTimeSynchronizer( [self.cylinder_sub], 10, 0.3, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_pc_callback)

		#msg publisher
		self.cylinder_info_pub = self.create_publisher(CylinderInfo, "/cylinder_info", QoSReliabilityPolicy.BEST_EFFORT)

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

		# cv2.createTrackbar('A', "Image", 0, 1000, self.nothing)
		# cv2.createTrackbar('B', "Image", 0, 1000, self.nothing)
		
		cv2.setTrackbarPos("A", "Image", 100)
		cv2.setTrackbarPos("B", "Image", 200)


		self.start_time = time.time()

	def nothing(self, data):
		pass

	def rgb_pc_callback(self, pc):

		if((time.time() - self.start_time) < 3):
			return

		cv2.waitKey(1)		
		xyzrgb = pc2.read_points_numpy(pc, field_names= ("y", "z", "x", "rgb"))
		xyz = xyzrgb[:, 0:3]
		rgb = xyzrgb[:, 3]

		rgb = rgb.astype(int)
		f = lambda x: [(x >> 16) & 0x0000ff, (x >> 8)  & 0x0000ff, x & 0x0000ff]
		rgb = np.array(f(rgb))

		print("xyzrgb")
		print(xyzrgb)
		print("xyz")
		print(xyz)
		print("rgb")
		print(rgb)

		hsv = rgb_to_hsv(rgb)
		avg_saturation = np.mean(hsv[:, 1], axis=0)
		avg_color = np.mean(rgb, axis=0)

		color_dist_sq = (rgb - avg_color)**2
		avg_color_dist = np.mean(color_dist_sq)

		q_color_dist = math.exp(-0.0004*avg_color_dist)
		q_color_saturation = avg_saturation
		q = q_color_dist * q_color_saturation

		print(f"q_color_dist: {q_color_dist:.2f}, q_color_saturation: {q_color_saturation:.2f}, q: {q:.2f}")



def main():
	print("OK")
	rclpy.init(args=None)
	node = CylinderDetection()
	
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
