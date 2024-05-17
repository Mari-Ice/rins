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
from sensor_msgs.msg import Image, PointCloud2, LaserScan
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
from task2.msg import RingInfo

amcl_pose_qos = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)


#	Kamera naj bo v looking_for_rings polozaju.
#	Zaznat je treba camera clipping, ker takrat zadeve ne delajo prav. 


def clamp(x, minx, maxx):
	return min(max(x, minx), maxx)
def deg2rad(deg):
	return (deg/180.0) * math.pi
def pos_angle(rad):
	while(rad > 2*math.pi):
		rad -= 2*math.pi
	while(rad < 0):
		rad += 2*math.pi
	return rad
def millis():
	return round(time.time() * 1000)	

class Test(Node):
	def __init__(self):
		super().__init__('test')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.bridge = CvBridge()

		# For listening and loading the TF

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		#self.rgb_sub = message_filters.Subscriber(self, Image,		 "/top_camera/rgb/preview/image_raw")
		self.laser_sub  = message_filters.Subscriber(self, LaserScan, "/scan")

		self.ts = message_filters.ApproximateTimeSynchronizer( [self.laser_sub], 10, 0.3, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_laser_callback)

		self.marker_pub = self.create_publisher(Marker, "/test", QoSReliabilityPolicy.BEST_EFFORT)

		self.start_time = time.time()
		return

	def nothing(self, data):
		return

	def get_point(self, laser, x):
		width = 250 #TODO, to bi lahko vzel is slike
		angle_increment = laser.angle_increment #TODO, to je zdej za pravega robota...
		fov = deg2rad(55) 
		fi = (fov * (0.5 - x/width)) 

		n = (int(fi/angle_increment) + 270)
		while(n > len(laser.ranges)):
			n -= len(laser.ranges)
		while(n < 0):
			n+=len(laser.ranges)

		rn = laser.ranges[n]
		fi1 = fi - deg2rad(90)
		return np.array([ rn * math.cos(fi1), rn* math.sin(fi1), 0 ])

	def rgb_laser_callback(self, laser):

		#print(laser.ranges)
		point = self.get_point(laser, 250/2)

		marker = Marker()

		marker.header.frame_id = "/rplidar_link"
		marker.header.stamp = laser.header.stamp

		marker.type = 2
		marker.id = 1

		# set the scale of the marker
		scale = 0.1
		marker.scale.x = scale
		marker.scale.y = scale
		marker.scale.z = scale

		# set the color
		marker.color.r = 1.0
		marker.color.g = 0.0
		marker.color.b = 0.0
		marker.color.a = 1.0

		# set the pose of the marker
		marker.pose.position.x = float(point[0])
		marker.pose.position.y = float(point[1])
		marker.pose.position.z = float(0.0)
		self.marker_pub.publish(marker)

		point = self.get_point(laser, 0)
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		marker.color.a = 1.0
		marker.id = 2	
		marker.pose.position.x = float(point[0])
		marker.pose.position.y = float(point[1])
		self.marker_pub.publish(marker)
		return

def main():
	print("OK")
	rclpy.init(args=None)
	node = Test()
	
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
	return

if __name__ == '__main__':
	main()
