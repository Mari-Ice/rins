#!/usr/bin/env python3

import random
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, Point
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.duration import Duration
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import message_filters
from ultralytics import YOLO
from task3.msg import ParkingSpotInfo

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
def vec_normalize(vec):
	norm = np.linalg.norm(vec)
	if(norm < 0.01):
		return vec
	return vec / norm

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

def fit_circle(points): 
	if(len(points) < 10):
		return None

	mat_A = np.zeros((points.shape[0], 3), dtype=np.float64)
	mat_A[:,0:2] = points
	mat_A[:,2] = 1

	vec_B = mat_A[:,0]**2 + mat_A[:,1]**2
	
	#Resimo linearen sistem	 A*[a,b,c]^T=b
	a,b,c = np.linalg.lstsq(mat_A, vec_B, rcond=None)[0]

	cx = a/2
	cy = b/2
	r = math.sqrt(4*c+a*a+b*b)/2

	return [cx,cy,r]

class face_detection(Node):
	def __init__(self):
		super().__init__('face_detection')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])
		
		self.device = self.get_parameter('device').get_parameter_value().string_value
		self.bridge = CvBridge()

		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.point_cloud_sub = message_filters.Subscriber(self, PointCloud2, "/oakd/rgb/preview/depth/points")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_image_sub, self.point_cloud_sub], 20, 0.03, allow_headerless=False) 
		self.ts.registerCallback(self.sensors_callback)

		self.marker_pub = self.create_publisher(Marker, "/parking_spot_marker", QoSReliabilityPolicy.BEST_EFFORT)
		self.data_pub   = self.create_publisher(ParkingSpotInfo, "/parking_spot_info", QoSReliabilityPolicy.BEST_EFFORT)

		self.received_any_data = False
		print("Init")
		return

	def sensors_callback(self, rgb_data, pc_data):
		if(not self.received_any_data):
			self.received_any_data = True
			print("Data\nOK")

		img = None
		try:
			img = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
		except CvBridgeError as e:
			print(e)
			return

		img_display = img.copy()

		height	   = pc_data.height
		width	   = pc_data.width
		pc_xyz = pc2.read_points_numpy(pc_data, field_names= ("x", "y", "z"))
		pc_xyz = pc_xyz.reshape((height,width,3))

		height_mask = np.zeros((height, width), dtype=np.uint8)
		height_mask[pc_xyz[:,:,2] < -0.24] = 255

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
		height_mask = cv2.erode(height_mask, kernel)

		img[height_mask != 255] = (255,255,255)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_,img = cv2.threshold(img,60,255,cv2.THRESH_BINARY)

		edge_points = pc_xyz[:,:,0:2][img != 255]
		if(len(edge_points) < 15):
			return
		
		avg_pt = np.mean(edge_points, axis=0)
		diff = (edge_points - avg_pt)
		dist_sq = diff[:,0]**2 + diff[:,1]**2
		# print(min(dist_sq), max(dist_sq), np.mean(dist_sq))

		edge_points = edge_points[dist_sq < 0.2]
		if(len(edge_points) < 10):
			return

		#Zdej pa fitas krog gor.
		cx,cy,r = fit_circle(edge_points)

		if(r < 0.15 or r > 0.30):
			return

		q_radius = circle_quality = math.pow(math.e, -0.1*(abs(0.24 - r)))
		q_distance  = math.exp(-0.08*(0.15 - np.linalg.norm(np.array([cx,cy])))**2)

		#Send marker
		header = Header()
		header.stamp = rgb_data.header.stamp
		header.frame_id = "/oakd_link"
		marker = create_point_marker([cx,cy,-0.24], header)
		marker.id = random.randint(1,100000)
		marker.lifetime = Duration(seconds=.1).to_msg()
		self.marker_pub.publish(marker)

		fi = math.atan2(cy, cx)
		msg = ParkingSpotInfo()
		msg.position_relative = [cx,cy,-0.24]
		msg.yaw_relative = fi
		msg.quality = q_radius * q_distance
		self.data_pub.publish(msg)

		# cv2.imshow("HeightMask", height_mask)
		# cv2.imshow("Img", img)
		cv2.waitKey(1)
		return	

def main():
	rclpy.init(args=None)
	node = face_detection()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
