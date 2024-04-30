#!/usr/bin/env python3

import time
import math
import rclpy
import cv2
import numpy as np
import tf2_geometry_msgs as tfg
import message_filters
from enum import Enum
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
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
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from lifecycle_msgs.srv import GetState

class ParkState(Enum):
	IDLE = 0
	ROTATING = 1
	DRIVING = 2
	PARKED = 3

amcl_pose_qos = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

class floor_mask(Node):
	def __init__(self):
		super().__init__('floor_mask')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.bridge = CvBridge()
		self.scan = None


		self.initial_pose_received = False
		self.position = None
		self.rotation = None
		self.yaw = 0
		self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
															  'amcl_pose',
															  self._amclPoseCallback,
															  amcl_pose_qos)
		self.waitUntilNav2Active()

		# For listening and loading the TF
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.rgb_sub = message_filters.Subscriber(self, Image, "/top_camera/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/top_camera/rgb/preview/depth/points")
		#self.depth_sub = message_filters.Subscriber(self, Image, "/top_camera/rgb/preview/depth")

		#self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub, self.depth_sub], 10, 0.3, allow_headerless=False) 
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub], 10, 0.3, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_pc_callback)

		self.teleop_pub = self.create_publisher(Twist, "cmd_vel", 10)

		self.park_state = ParkState.IDLE
		self.circle_quality = 0
		self.start_yaw = 0
		self.target_yaw = 0

		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
		cv2.waitKey(1)
		cv2.moveWindow('Image',  1   ,1)
		cv2.moveWindow('Mask',   415 ,1)

		cv2.createTrackbar('A', "Image", 0, 1000, self.nothing)
		cv2.setTrackbarPos("A", "Image", 133)

	def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
		"""Block until the full navigation system is up and running."""
		self._waitForNodeToActivate(localizer)
		if not self.initial_pose_received:
			time.sleep(1)
		self._waitForNodeToActivate(navigator)
		print('Nav2 is ready for use!')
		return

	def _waitForNodeToActivate(self, node_name):
		# Waits for the node within the tester namespace to become active
		print(f'Waiting for {node_name} to become active..')
		node_service = f'{node_name}/get_state'
		state_client = self.create_client(GetState, node_service)
		while not state_client.wait_for_service(timeout_sec=1.0):
			print(f'{node_service} service not available, waiting...')

		req = GetState.Request()
		state = 'unknown'
		while state != 'active':
			print(f'Getting {node_name} state...')
			future = state_client.call_async(req)
			rclpy.spin_until_future_complete(self, future)
			if future.result() is not None:
				state = future.result().current_state.label
				print(f'Result of get_state: {state}')
			time.sleep(2)
		return

	def _amclPoseCallback(self, msg):
		self.initial_pose_received = True
		#self.current_pose = msg.pose
		self.position = msg.pose.pose.position
		self.rotation = msg.pose.pose.orientation

		q = self.rotation
		#yaw = math.atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
		#yaw = math.asin(-2.0*(q.x*q.z - q.w*q.y))
		yaw = math.atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
		self.yaw = yaw

		return

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
		a,b,c = np.linalg.lstsq(mat_A, vec_B, rcond=None)[0]

		cx = -a/2
		cy = b/2
		r = math.sqrt(4*c+a*a+b*b)/2

		return [cx,cy,r]
	
	def positive_angle(self,angle):
		while(angle < 0):
			angle += 2*math.pi
		return math.fmod(angle, 2*math.pi)
		#return math.fmod(angle + math.pi, 2*math.pi)

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

		##TODO, namesto, da se uporablja samo zelo majhne hitrosti, bi se lahko kak PID regulator vrgu gor...

		circle = self.fit_circle(xyz[mask==255])
		if(circle != None):	
			circle_quality = circle_quality = math.pow(math.e, -(abs(0.24 - circle[2])))
			circle[1] += 0.20
			if(self.park_state == ParkState.IDLE or circle_quality >= self.circle_quality):
				self.circle_quality = circle_quality
				self.start_yaw = self.yaw
				self.target_yaw = -math.pi/2 + math.atan2(circle[1], circle[0]) 
			if(self.park_state == ParkState.IDLE):
				self.park_state = ParkState.ROTATING
			if(self.park_state == ParkState.DRIVING):
				fwd_error = circle[1] #naceloma bi lahko vzel tudi normo, amapak hej, tisto lahko povzroci pa druge probleme
				print(f"Fwd_error: {fwd_error}")
			
				cmd_msg = Twist()
				cmd_msg.angular.z = 0.
				if(abs(fwd_error) < 0.02): #stop rotating, next_state
					cmd_msg.linear.x = 0.
					self.park_state = ParkState.PARKED
					print("Parked")
				else:
					cmd_msg.linear.x = 0.1 * (fwd_error) / abs(fwd_error)
				self.teleop_pub.publish(cmd_msg)

		if(self.park_state == ParkState.ROTATING):
			yaw_error = self.positive_angle(-math.pi + self.positive_angle(self.target_yaw) - self.positive_angle(self.yaw) + self.positive_angle(self.start_yaw)) - math.pi
			print(f"Yaw error: {yaw_error}, circle_quality: {self.circle_quality}")
			
			cmd_msg = Twist()
			cmd_msg.linear.x = 0.
			if(abs(yaw_error) < 0.02): #stop rotating, next_state
				cmd_msg.angular.z = 0.
				self.park_state = ParkState.DRIVING
			else:
				cmd_msg.angular.z = 0.5 * (yaw_error) / abs(yaw_error)
			self.teleop_pub.publish(cmd_msg)
					
			
		cv2.imshow("Mask", mask)
		cv2.imshow("Image", img)

def main():
	print("OK")
	rclpy.init(args=None)
	node = floor_mask()
	#node.waitUntilNav2Active()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
