#!/usr/bin/env python3

import statistics
import random
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from geometry_msgs.msg import Point
import message_filters

from ultralytics import YOLO
import sys

#from task1.msg import FaceNormal
# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

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
def nothing():
	return

class detect_faces(Node):
	face_id = 0
		
	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.img_height = 0
		self.img_width = 0
		#self.cam_fov_y = deg2rad(55) #TODO, odvisno od naprave.
		#self.cam_fov_x = deg2rad(55)
		self.cam_fov_y = deg2rad(90) 
		self.cam_fov_x = deg2rad(90)

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.laser_sub  = message_filters.Subscriber(self, LaserScan, "/scan")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_image_sub, self.laser_sub], 10, 0.05, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_laser_callback)

		marker_topic = "/people_marker"
		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)

		marker_topic1 = "/img1"
		self.marker_pub1 = self.create_publisher(Marker, marker_topic1, QoSReliabilityPolicy.BEST_EFFORT)
		marker_topic2 = "/img2"
		self.marker_pub2 = self.create_publisher(Marker, marker_topic2, QoSReliabilityPolicy.BEST_EFFORT)

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.model = YOLO("yolov8n.pt")

		self.faces = []
		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")
		self.t1 = millis()

		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

		cv2.createTrackbar('FovX', "Image", 55, 130, self.change_fovX)
		cv2.createTrackbar('FovY', "Image", 55, 130, self.change_fovY)
		cv2.createTrackbar('Height', "Image", 15, 200, self.change_height)
		self.t_height = 0.15

	def change_fovX(self, val):
		self.cam_fov_x = deg2rad(max(val, 1))
	def change_fovY(self, val):
		self.cam_fov_y = deg2rad(max(val, 1))
	def change_height(self, val):
		self.t_height = val*0.01

	def remove_noise(self, meje_arr):
		filter_size = 15
		new_meje = meje_arr.copy()
		for x in range(len(meje_arr)-filter_size):
			nset = []
			for i in range(filter_size):
				nset.append(meje_arr[x + i])
			new_meje[x+int(filter_size/2)] = int(statistics.median(nset))
		return new_meje

	def generate_mask(self, laser):
		if(self.img_height == 0 or self.img_width == 0):
			return

		visina = self.t_height
		fov = self.cam_fov_y
		meje_arr = []
		mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

		for x in range(self.img_width):
			rn = np.linalg.norm(self.get_point(laser, x))
			y = rn * math.tan(fov/2)
			meja = int(self.img_height/2 * (1 - visina/y))
			
			# if(random.uniform(0,1) < 0.01): #Dodamo sum v simulciji, da lahko testiramo odpravljanje suma.. TODO odstrani potem tole
			# 	meja = random.randrange(0,self.img_height)
			
			meje_arr.append(meja)

		meje_arr = self.remove_noise(meje_arr)
		for x in range(self.img_width):
			mask[meje_arr[x]:,x] = 255
		return (mask)

	def rgb_laser_callback(self, rgb_data, laser_data):
		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")

			self.img_height = cv_image.shape[0]
			self.img_width = cv_image.shape[1]

			mask = self.generate_mask(laser_data)
			cv_image[mask == 0] = 255

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over results 
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue

				# self.get_logger().info(f"Person has been detected!")

				bbox = bbox[0]

				# draw rectangle
				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)
				self.faces.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))

			cv2.imshow("Image", cv_image)
			key = cv2.waitKey(1)
			if key==27: #Esc
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

		self.find_faces(laser_data)

	def get_point(self, laser, x):
		fi_screen = (self.cam_fov_x * (0.5 - x/self.img_width))
		fi = fi_screen - math.pi/2
		n = int(pos_angle(fi - laser.angle_min)/laser.angle_increment) 
		rn = laser.ranges[n]
		#print(f"x: {x}, n: {n}, fi: {fi}")
		return np.array([ rn*math.cos(fi), rn*math.sin(fi), 0 ])

	def find_faces(self, data):
		if(self.img_height == 0):
			return

		if(millis() - self.t1 < 4000): #pocakaj, da se zadeve inicializirajo.
			return

		# Gremo cez vse face, ki smo jih zaznali v zadnjem frame-u
		for x,y,z,w in self.faces:
			if(abs(x-y) < 15 or abs(z-w) < 10):
				continue

			d1 = self.get_point(data, x)
			d2 = self.get_point(data, z)
			
			if(len(d1) == 0 or len(d2) == 0):
				return

			#ce so robovi slike prevec dalec narazen ali pa preblizu skupaj
			if(np.linalg.norm(d2-d1) > 0.5):
				continue
			if(np.linalg.norm(d2-d1) < 0.05):
				continue
			#Ce je obraz dalec od robota, ga bomo zazanli rajsi kdaj ko bomo blizje ...
			if(np.linalg.norm(d1) > 2.6): 
				continue

			p1 = PointStamped()
			p1.header.frame_id = "/rplidar_link"
			p1.header.stamp = self.get_clock().now().to_msg()
			p1.point.x = float(d1[0])
			p1.point.y = float(d1[1])
			p1.point.z = float(d1[2])

			p2 = PointStamped()
			p2.header.frame_id = "/rplidar_link"
			p2.header.stamp = self.get_clock().now().to_msg()
			p2.point.x = float(d2[0])
			p2.point.y = float(d2[1])
			p2.point.z = float(d2[2])

			#zdej pa te tocke transformiramo v globalne (map) koordinate
			time_now = rclpy.time.Time()
			timeout = Duration(seconds=10.0)
			trans = self.tf_buffer.lookup_transform("map", "rplidar_link", time_now, timeout)	

			p1 = tfg.do_transform_point(p1, trans)
			p2 = tfg.do_transform_point(p2, trans)

			d1 = np.array([
				p1.point.x,
				p1.point.y,
				p1.point.z
			]);
			d2 = np.array([
				p2.point.x,
				p2.point.y,
				p2.point.z
			]);

			marker = Marker()

			marker.header.frame_id = "/map"
			marker.header.stamp = data.header.stamp

			marker.type = 2

			# Set the scale of the marker
			scale = 0.1
			marker.scale.x = scale
			marker.scale.y = scale
			marker.scale.z = scale

			# Set the color
			marker.color.r = 1.0
			marker.color.g = 1.0
			marker.color.b = 1.0
			marker.color.a = 1.0

			# Set the pose of the marker
			marker.pose.position.x = float(d1[0])
			marker.pose.position.y = float(d1[1])
			marker.pose.position.z = float(d1[2])

			self.marker_pub1.publish(marker)
			
			# Set the pose of the marker
			marker.pose.position.x = float(d2[0])
			marker.pose.position.y = float(d2[1])
			marker.pose.position.z = float(d2[2])

			# Set the color
			marker.color.r = 1.0
			marker.color.g = 0.0
			marker.color.b = 1.0
			marker.color.a = 1.0

			self.marker_pub2.publish(marker)

			origin = 0.5 * (np.array(d1).astype(float) +  np.array(d2).astype(float))
			vector_up = np.array([0,0,1])
			vector_right = np.array(d1).astype(float) -  np.array(d2).astype(float)
			vector_fwd = np.cross(vector_up, vector_right)
	  
			fwd_len = np.linalg.norm(vector_fwd)
			if fwd_len <= 0.02:
				return
			vector_fwd = vector_fwd / fwd_len

			# create marker
			marker = Marker()

			marker.header.frame_id = "/map"
			marker.header.stamp = data.header.stamp
			
			marker.type = 0
			marker.id = detect_faces.face_id
			detect_faces.face_id+=1

			# Set the scale of the marker
			scale = 0.1
			marker.scale.x = scale
			marker.scale.y = scale
			marker.scale.z = scale

			# Set the color
			marker.color.r = 1.0
			marker.color.g = 0.0
			marker.color.b = 0.0
			marker.color.a = 1.0

			startpoint = Point()
			startpoint.x = origin[0]
			startpoint.y = origin[1]
			startpoint.z = origin[2]

			endpoint = Point()

			endpoint.x = origin[0] + vector_fwd[0]
			endpoint.y = origin[1] + vector_fwd[1]
			endpoint.z = origin[2] + vector_fwd[2]

			marker.points.append(startpoint)
			marker.points.append(endpoint)
			
			self.marker_pub.publish(marker)

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
