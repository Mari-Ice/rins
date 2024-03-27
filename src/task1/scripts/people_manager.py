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
import math
from geometry_msgs.msg import Point
from collections import deque
import time

import tf_transformations

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

def millis():
    return round(time.time() * 1000)
def mag(vec):
	return np.sqrt(vec.dot(vec))

class Face():
	face_id = 0
	tresh_xy = 0.3
	tresh_cos = math.cos(20 * 3.14/180)

	def __init__(self, marker):
		self.origin = np.array([
			marker.points[0].x,
			marker.points[0].y,
			marker.points[0].z
		])
		self.endpoint = np.array([
			marker.points[1].x,
			marker.points[1].y,
			marker.points[1].z
		])
		self.normal = self.endpoint - self.origin

		self.id = Face.face_id
		Face.face_id += 1

		self.num = 1
		self.num_tresh = 2
		self.visited = False

	def compare(self, face): #mogoce bi blo lazje primerjat ze izracunane keypointe
		kp1 = self.origin + 0.3 * self.normal
		kp2 = face.origin + 0.3 * face.normal
		return mag(kp1-kp2) < 0.5


class detect_faces(Node):
	last_n_faces = []
	last_marker_time = 0

	rolling_origin = np.array([0, 0, 0])

	def __init__(self):
		super().__init__('people_manager')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])
		
		self.faces = []

		marker_topic = "/people_marker"

		self.marker = self.create_subscription(Marker, marker_topic, self.marker_callback, 10)
		self.publisher = self.create_publisher(Marker, '/detected_faces', QoSReliabilityPolicy.BEST_EFFORT)
		
		self.get_logger().info(f"Node has been initialized! Reading from {marker_topic}.")

	def marker_callback(self, marker):
		new_face = Face(marker)
		if not (np.isfinite(new_face.origin).all() and np.isfinite(new_face.normal).all()):
			return

		# detect_faces.rolling_origin   = 0.9 * detect_faces.rolling_origin   + 0.1 * new_face.origin	
		# if(np.linalg.norm(detect_faces.rolling_origin - new_face.origin) > 0.05):
		# 	return

		notFound = True
		for face in self.faces:
			if(face.compare(new_face)): #naceloma bi blo boljse, ce bi sli cez vse in poiskai tistega, ki najbolj ustreza, 
										#ce so meje nastavljene prevec nenatancno, se zgodi, da ustreza vecim obrazom ...
				#face.origin = 0.9 * face.origin + 0.1 * new_face.origin
                #face.normal = 0.8 * face.normal + 0.2 * new_face.normal Tu je treba se normirat, ker taksan vsota ne ohrani razdalje...
				face.num += 1
				notFound = False
				if(not face.visited):
					#if(face.num > face.num_tresh):
					point = Marker()
					point.type = 2
					point.id = face.id
					point.header.frame_id = "/map"
					point.header.stamp = marker.header.stamp
					
					point.scale.x = 0.15
					point.scale.y = 0.15
					point.scale.z = 0.15

					point.color.r = 0.0
					point.color.g = 1.0
					point.color.b = 0.0
					point.color.a = 1.0
					point.pose.position.x = face.origin[0] + face.normal[0] * 0.3
					point.pose.position.y = face.origin[1] + face.normal[1] * 0.3
					point.pose.position.z = face.origin[2] + face.normal[2] * 0.3

					# marker should be turned towards the face (opposite from the normal)
					marker_normal = -face.normal
					q = tf_transformations.quaternion_from_euler(0, 0, math.atan2(marker_normal[1], marker_normal[0]))
					point.pose.orientation.x = q[0]
					point.pose.orientation.y = q[1]
					point.pose.orientation.z = q[2]
					point.pose.orientation.w = q[3]

					self.publisher.publish(point)
					face.visited = True
				break 
		if(notFound):
			self.faces.append(new_face)
	
		self.get_logger().info(f"Got a marker {marker.points[0]} {marker.points[1]}")
		self.get_logger().info(f"FACES: {len(self.faces)}")
		print()

def main():
	print('People manager node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
