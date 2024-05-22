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

from rclpy.duration import Duration
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from geometry_msgs.msg import Point
import message_filters
from task1.msg import PersonInfo

from ultralytics import YOLO
import sys

#from task1.msg import FaceNormal
# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

def vec_normalize(np_vec):
	norm = np.linalg.norm(np_vec)
	if(norm < 0.01):
		return np.array([0,0,0])
	return np_vec / norm

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

def array2point(arr):
	p = Point()
	p.x = float(arr[0])
	p.y = float(arr[1])
	p.z = float(arr[2])
	return p


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
		self.simulation = False

		if(self.simulation):
			self.cam_fov_y = deg2rad(90)
			self.cam_fov_x = deg2rad(90)
		else:	
			self.cam_fov_y = deg2rad(55)
			self.cam_fov_x = deg2rad(55)

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.laser_sub  = message_filters.Subscriber(self, LaserScan, "/scan")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_image_sub, self.laser_sub], 10, 0.2, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_laser_callback)

		self.marker_pub = self.create_publisher(Marker, "/people_marker", QoSReliabilityPolicy.BEST_EFFORT)
		self.person_pub = self.create_publisher(PersonInfo, "/people_info", QoSReliabilityPolicy.BEST_EFFORT)

		self.marker_pub1 = self.create_publisher(Marker, "/img1", QoSReliabilityPolicy.BEST_EFFORT)
		self.marker_pub2 = self.create_publisher(Marker, "/img2", QoSReliabilityPolicy.BEST_EFFORT)

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.model = YOLO("yolov8n.pt")

		self.faces = []
		self.t1 = millis()

		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

		if(self.simulation):
			cv2.createTrackbar('FovX', "Image", 90, 130, self.change_fovX)
			cv2.createTrackbar('FovY', "Image", 90, 130, self.change_fovY)
			cv2.createTrackbar('Height', "Image", 15, 200, self.change_height)
		else:
			cv2.createTrackbar('FovX', "Image", 55, 130, self.change_fovX)
			cv2.createTrackbar('FovY', "Image", 55, 130, self.change_fovY)
			cv2.createTrackbar('Height', "Image", 15, 200, self.change_height)

		self.t_height = 0.15
		print(f"OK, simulation: {self.simulation}")
		return

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
		
			if(self.simulation):
				if(random.uniform(0,1) < 0.01): #Dodamo sum v simulciji
					meja = random.randrange(0,self.img_height)
			
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

	def create_point_marker(self, stamp,  point):
		marker = Marker()

		marker.header.frame_id = "/map"
		marker.header.stamp = stamp

		marker.type = 2

		scale = 0.1
		marker.scale.x = scale
		marker.scale.y = scale
		marker.scale.z = scale

		marker.color.r = 1.0
		marker.color.g = 1.0
		marker.color.b = 1.0
		marker.color.a = 1.0
		marker.pose.position = array2point(point)
		
		return marker
	def create_arrow_marker(self, stamp, m_id, origin, endpoint):
		marker = Marker()
		marker.header.frame_id = "/map"
		marker.header.stamp = stamp
		marker.lifetime = Duration(seconds=4.2).to_msg()
		
		marker.type = 0
		#marker.id = detect_faces.face_id
		marker.id = m_id
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

		startpoint = array2point(origin)
		endpoint = array2point(endpoint)

		marker.points.append(startpoint)
		marker.points.append(endpoint)

		return marker	
	
	def find_faces(self, data):
		if(self.img_height == 0):
			return

		if(millis() - self.t1 < 4000): #pocakaj, da se zadeve inicializirajo.
			return

		# Gremo cez vse face, ki smo jih zaznali v zadnjem frame-u
		for x1,y1,x2,y2 in self.faces:
			if(abs(x1-x2) < 15 or abs(y1-y2) < 15):
				continue

			r1 = self.get_point(data, x1)
			r2 = self.get_point(data, x2)
			r3 = self.get_point(data, (x1+x2)/2)

			lin_error = np.linalg.norm(r3 - (r1+r2)/2)
			
			if(len(r1) == 0 or len(r2) == 0):
				return

			if(lin_error > 0.1):
				return

			#ce so robovi slike prevec dalec narazen ali pa preblizu skupaj
			if(np.linalg.norm(r2-r1) > 0.5):
				continue
			if(np.linalg.norm(r2-r1) < 0.1):
				continue
			
			#Ce je obraz dalec od robota, ga bomo zazanli rajsi kdaj ko bomo blizje ...
			distance = np.linalg.norm((r1+r2)/2)
			if(distance > 2.6): 
				continue

			p1 = PointStamped()
			p1.header.frame_id = "/rplidar_link"
			p1.header.stamp = self.get_clock().now().to_msg()
			p1.point = array2point(r1)

			p2 = PointStamped()
			p2.header.frame_id = "/rplidar_link"
			p2.header.stamp = self.get_clock().now().to_msg()
			p2.point = array2point(r2)

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

			self.marker_pub1.publish(self.create_point_marker(data.header.stamp,  d1))
			self.marker_pub2.publish(self.create_point_marker(data.header.stamp,  d2))

			origin = 0.5 * (np.array(d1).astype(float) +  np.array(d2).astype(float))
			vector_up = np.array([0,0,1])
			vector_right = np.array(d1).astype(float) -  np.array(d2).astype(float)
			vector_fwd = np.cross(vector_up, vector_right)
	  
			fwd_len = np.linalg.norm(vector_fwd)
			if fwd_len <= 0.02:
				return
			vector_fwd = vector_fwd / fwd_len

			self.marker_pub.publish(self.create_arrow_marker(data.header.stamp, detect_faces.face_id, origin, origin + vector_fwd))

			vec = vec_normalize(r2-r1)
			fi = math.acos(np.array([-1,0,0]).dot(vec))

			q_dist = math.exp(-distance)
			q_lin  = math.exp(-60*lin_error)
			q_angle = 1.0 - clamp(abs(fi)/(math.pi), 0, 1)
			q_edges = 1.0 if (x1 > 1 and x2 < self.img_width-2) else 0.8
			
			pi = PersonInfo()
			pi.origin = [float(origin[0]), float(origin[1]), float(origin[2])]
			pi.normal = [float(vector_fwd[0]), float(vector_fwd[1]), float(vector_fwd[2])]
			pi.quality = q_dist * q_lin * q_angle * q_edges

			self.person_pub.publish(pi)

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
