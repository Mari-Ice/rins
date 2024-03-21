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



# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

class Face():
    def __init__(self, marker):
        self.x = marker.pose.position.x
        self.y = marker.pose.position.y
        self.z = marker.pose.position.z
        self.num = 1

    def compare(face):
        if(math.abs(self.x - face.x) < 1 and math.abs(self.y - face.y) < 1 and math.abs(self.z - face.z) < 1):
            return True
        return False


class detect_faces(Node):

	def __init__(self):
		super().__init__('people_manager')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])
        
        self.faces = []

		marker_topic = "/people_marker"

		self.marker = self.create_subscription(Marker, marker_topic, self.marker_callback)
		self.get_logger().info(f"Node has been initialized! Reading from {marker_topic}.")

    def marker_callback(marker):
        new_face = Face(marker)
        notFound = True
        for face in self.faces:
            if(face.compare(new_face)):
                face.x = (face.x + new_face.x) / 2
                face.y = (face.y + new_face.y) / 2
                face.z = (face.z + new_face.z) / 2 
                face.num += 1
                notFound = False
                break
        if(notFound):
            self.faces.append(new_face)
        
        self.get_logger().info(f"Got a marker {marker.pose.position.x} {marker.pose.posititon.y} {marker.pose.position.z}")
        self.get_logger().info(f"FACES: {len(self.faces)}")

def main():
	print('People manager node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()