#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from geometry_msgs.msg import Point

from ultralytics import YOLO
import sys

#from task1.msg import FaceNormal
# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

def clamp(x, minx, maxx):
	return min(max(x, minx), maxx)

class detect_faces(Node):
	face_id = 0
	min_y = 1000
	max_y = 0
		
	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		marker_topic = "/people_marker"
		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)

		marker_topic1 = "/img1"
		self.marker_pub1 = self.create_publisher(Marker, marker_topic1, QoSReliabilityPolicy.BEST_EFFORT)
		marker_topic2 = "/img2"
		self.marker_pub2 = self.create_publisher(Marker, marker_topic2, QoSReliabilityPolicy.BEST_EFFORT)


		# For listening and loading the TF
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.model = YOLO("yolov8n.pt")

		self.faces = []

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")
		self.start_time = self.get_clock().now()

	def rgb_callback(self, data):

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			# self.get_logger().info(f"Running inference on image...")

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

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27: #Esc
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step		

		# print(f"pointcloud size: {width} x {height}")

		# Gremo cez vse face, ki smo jih zaznali v zadnjem frame-u
		for x,y,z,w in self.faces:
			x = clamp(x, 0, width-1)
			z = clamp(z, 0, width-1)
			y = clamp(y, 0, height-1)
			w = clamp(y, 0, height-1)

			if(abs(x-y) < 15 or abs(z-w) < 10):
				continue

			# get 3-channel representation of the poitn cloud in numpy format
			a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
			a = a.reshape((height,width,3))

			# Ker so slike lahko postavljene razlicno, enkrat gledajo ces steno, drugic ne...,
			# Zberemo y koordianto tako, da bi naj naceloma vedno delalo.
			# Druga resistev bi bla, da dobis y koordinato tako, da mapiras laser na sliko in vzames tisti y.

			#print(f"pixels: ({x},{y}), ({z},{w})")


			# Dokaj dirty trick, z bisekcijo, najdemo koordinato z ustrezno z visino tj. na sredini stene
			# Stena je na visini 0.09m, visina stene pa je 0.25 m
			# torej zelimo najti visino 0.09 + 0.125 = 0.215 (+- 0.05)
		
			center_x = int((x+z)*0.5)
			y = None

			for i in range(110, 170):
				value = a[i, center_x, 2]
				#print(f"value: {value}")
				if(abs(value - (-0.1)) < 0.05):
					y = i
					break
			
			if(y is None): 
				print("Could not found proper y.")
				continue

			# TODO
			# Tu bi zdej lahko se preveril ali je slika na ravni povrsini.
			# Tj ali so tocke med d1 in d2 na daljici

			d1 = a[y,x]
			d2 = a[y,z]
		
			detect_faces.min_y = min(detect_faces.min_y, y)
			detect_faces.max_y = max(detect_faces.max_y, y)

			print(f"y: {y}, min_y: {detect_faces.min_y}, max_y: {detect_faces.max_y}, d1: {d1}, d2: {d2}")

			p1 = PointStamped()
			p1.header.frame_id = "/oakd_link"
			p1.header.stamp = self.get_clock().now().to_msg()
			p1.point.x = float(d1[0])
			p1.point.y = float(d1[1])
			p1.point.z = float(d1[2])

			p2 = PointStamped()
			p2.header.frame_id = "/oakd_link"
			p2.header.stamp = self.get_clock().now().to_msg()
			p2.point.x = float(d2[0])
			p2.point.y = float(d2[1])
			p2.point.z = float(d2[2])

			p3 = PointStamped() #robot global pos
			p3.header.frame_id = "/oakd_link"
			p3.header.stamp = self.get_clock().now().to_msg()
			p3.point.x = 0.0
			p3.point.y = 0.0
			p3.point.z = 0.0

			if(np.linalg.norm(d2-d1) > 0.5):
				continue
			if(np.linalg.norm(d2-d1) < 0.05):
				continue


		
			#print(data.header.stamp, rclpy.time.Time())
			#zdej pa te tocke transformiramo v globalne (map) koordinate
			time_now = rclpy.time.Time()
			timeout = Duration(seconds=10.0)
			trans = self.tf_buffer.lookup_transform("map", "oakd_link", time_now, timeout)	

			p1 = tfg.do_transform_point(p1, trans)
			p2 = tfg.do_transform_point(p2, trans)
			p3 = tfg.do_transform_point(p3, trans)

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
			d3 = np.array([
				p3.point.x,
				p3.point.y,
				p3.point.z
			]);

			if(np.linalg.norm(d3-d1) > 2.6): #Ce je obraz dalec od robota, ga bomo zazanli rajsi kdaj ko bomo blizje ...
				continue

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
			
			#TODO: ugotovi kako dobit pravi cajt, ker ce das clock().now() in uporabis karkoli razen /map zadeva ne dela...	
			#marker.header.frame_id = "/oakd_link"
			#marker.header.stamp = self.get_clock().now().to_msg()

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
