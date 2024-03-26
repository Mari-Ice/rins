#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

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


		self.model = YOLO("yolov8n.pt")

		self.faces = []

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def rgb_callback(self, data):

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over results
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue

				self.get_logger().info(f"Person has been detected!")

				bbox = bbox[0]

				# draw rectangle
				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)
				self.faces.append((int(bbox[0]), int(bbox[3]), int(bbox[2])))

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

		print(f"pointcloud size: {width} x {height}")

		# iterate over face coordinates
		for x,y,z in self.faces:
			x = clamp(x, 0, width-1)
			z = clamp(z, 0, width-1)
			y = clamp(y, 0, height-1)

			if(abs(x-y) < 5 or abs(x-z) < 5):
				continue

			print(f"pixels: ({x},{y}), ({z},{y})")

			# get 3-channel representation of the poitn cloud in numpy format
			a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
			a = a.reshape((height,width,3))

			#Tole zna bit tudi ok
			# x = int(x * 240/256)
			# z = int(x * 240/256)

			d1 = a[y,x]
			d2 = a[y,z]
		
			if(np.linalg.norm(d2-d1) > 1):
				continue

			marker = Marker()

			marker.header.frame_id = "/oakd_link"
			marker.header.stamp = data.header.stamp

			marker.type = 2
			marker.id = 0

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
			if fwd_len == 0:
				return
			vector_fwd = vector_fwd / fwd_len


			# create marker
			marker = Marker()

			marker.header.frame_id = "/oakd_link"
			marker.header.stamp = data.header.stamp

			marker.type = 0
			marker.id = 0

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
