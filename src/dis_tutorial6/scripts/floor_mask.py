#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
import tf2_geometry_msgs as tfg
import message_filters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

class sky_mask(Node):
	def __init__(self):
		super().__init__('sky_mask')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_sub = message_filters.Subscriber(self, Image, "/top_camera/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/top_camera/rgb/preview/depth/points")
		#self.depth_sub = message_filters.Subscriber(self, Image, "/top_camera/rgb/preview/depth")

		#self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub, self.depth_sub], 10, 0.3, allow_headerless=False) 
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub], 10, 0.3, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_pc_callback)

		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
		cv2.waitKey(1)
		cv2.moveWindow('Image',  1   ,1)
		cv2.moveWindow('Mask',   415 ,1)

		cv2.createTrackbar('A', "Image", 0, 1000, self.nothing)
		
	def nothing(self, data):
		pass

	def draw_depth(self, data):
		try:
			depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
		except CvBridgeError as e:
			print(e)

		depth_image[depth_image==np.inf] = 0
		
		# Do the necessairy conversion so we can visuzalize it in OpenCV
		image_1 = depth_image / 65536.0 * 255
		image_1 = image_1/np.max(image_1)*255

		image_viz = np.array(image_1, dtype= np.uint8)

		cv2.imshow("Depth", image_viz)

	#def rgb_pc_callback(self, rgb, pc, depth_raw):
	def rgb_pc_callback(self, rgb, pc):
		cv2.waitKey(1)
		img = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
		height	 = pc.height
		width	  = pc.width
		point_step = pc.point_step
		row_step   = pc.row_step		

		a = pc2.read_points_numpy(pc, field_names= ("x", "y", "z"))
		a = a.reshape((height,width,3))

		thresh = 5*cv2.getTrackbarPos("A", 'Image') / 1000

		mask = np.full((height, width), 0, dtype=np.uint8)
		mask[(img[:,:,:] < 10).all(axis=2)] = 255
		mask[a[:,:,0] < thresh] = (0)

		cv2.imshow("Image", img)
		cv2.imshow("Mask", mask)

def main():
	print("OK")
	rclpy.init(args=None)
	node = sky_mask()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
