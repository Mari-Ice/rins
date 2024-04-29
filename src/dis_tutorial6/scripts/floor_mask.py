#!/usr/bin/env python3

import math
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

		# For listening and loading the TF
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

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
		cv2.setTrackbarPos("A", "Image", 133)
		
	def nothing(self, data):
		pass

	def fit_circle(self, points): 
		#btw mi sicer vemo, da more met krog radij 0.25
		#amapak je dosti tezje resit tak sistem heh
		if(len(points) < 3):
			return None
		
		mat_A = points.copy()
		mat_A[:,2] = 1
		vec_B = mat_A[:,0]**2 + mat_A[:,1]**2
		
		#Resimo linearen sistem	 A*[a,b,c]^T=b
		a,b,c = np.linalg.lstsq(mat_A, vec_B)[0]

		cx = a/2
		cy = b/2
		r = math.sqrt(4*c+a*a+b*b)/2

		return [cx,cy,r]
		
	#def rgb_pc_callback(self, rgb, pc, depth_raw):
	def rgb_pc_callback(self, rgb, pc):
		cv2.waitKey(1)
		img = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
		height	 = pc.height
		width	  = pc.width
		point_step = pc.point_step
		row_step   = pc.row_step		

		## Tocke bi lahko transformiral vsaj v koordiante na oakd, zato, da ko se roka premakne, stvari se zmeraj isto delajo ...
		## Amapak ker zgleda da v pythonovem api-ji ni fukncije ki bi tranformirala cel point cloud, tega potem ne bom delal.

		# #tocke transformiramo v globalne (map) koordinate
		# time_now = rclpy.time.Time()
		# timeout = Duration(seconds=10.0)
		# trans = self.tf_buffer.lookup_transform("map", "top_camera", time_now, timeout)	
		# pc = do_transform_cloud(pc, trans)

		xyz = pc2.read_points_numpy(pc, field_names= ("y", "z", "x"))
		xyz = xyz.reshape((height,width,3))

		thresh = 5*cv2.getTrackbarPos("A", 'Image') / 1000
		mask = np.full((height, width), 0, dtype=np.uint8)
		mask[(img[:,:,:] < 10).all(axis=2)] = 255
		mask[xyz[:,:,2] < thresh] = (0)

		circle = self.fit_circle(xyz[mask==255])
		if(circle != None):	
			circle[1] += 0.20
			print(circle)

		#naceloma ko enkrat mas krog, se obrnes, tako, da je y koordianta pozitivna. Tj. Najprej dolocis kot in se med obracanjem ne oziras ali se krog vidis ali ne.
		#razen, ce ga vidis boljse od prej, potem lahko malo popravis kot, do katerega se mores obrant ...

		#Ko se enkrat obrnes tako, zracunas zracunas razdaljo koliko dales se mores premaknati naprej
		#Podobno kot prej, se tu premaknes naprej za to razdaljo. Tu naceloma ne mores vec zgubit kroga, ...

		#print(xyz[mask==255])
		# #print(len(xyz[mask==255]))
		# #print((xyz[mask==255]))

		cv2.imshow("Mask", mask)
		cv2.imshow("Image", img)

def main():
	print("OK")
	rclpy.init(args=None)
	node = sky_mask()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
