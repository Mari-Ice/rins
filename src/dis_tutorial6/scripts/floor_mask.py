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

class floor_mask(Node):
	mask_img = None

	def __init__(self):
		super().__init__('floor_mask')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		# For listening and loading the TF
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

	def rgb_callback(self, data):
		cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		cv2.imshow("Image", cv_image)

		if(not self.mask_img is None):
			cv2.imshow("Mask", self.mask_img)

			masked_image = cv2.bitwise_and(cv_image,self.mask_img)
			cv2.imshow("Masked", masked_image)
		
		cv2.waitKey(1)

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step		

		a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
		a = a.reshape((height,width,3))

		img = np.zeros((height, width, 3), np.uint8)
		img[a[:,:,2] < -0.235] = (255,255,255)
		self.mask_img = img

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = floor_mask()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
