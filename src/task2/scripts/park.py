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
from std_msgs.msg import String

##Note: Roka more bit v parked2 poziciji, da kaj od tega dela.

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

def sign(x):
	if(x<0):
		return -1.
	return 1.	
def positive_angle(angle):
	while(angle < 0):
		angle += 2*math.pi
	return math.fmod(angle, 2*math.pi)

class park(Node):
	def __init__(self):
		super().__init__('park')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.bridge = CvBridge()
		self.scan = None

		self.initial_pose_received = False
		# self.position = None
		# self.rotation = None
		# self.yaw = 0
		# self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
		# 													  'amcl_pose',
		# 													  self._amclPoseCallback,
		# 													  amcl_pose_qos)
		# self.waitUntilNav2Active()

		# For listening and loading the TF
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.rgb_sub = message_filters.Subscriber(self, Image, "/top_camera/rgb/preview/image_raw")
		self.pc_sub  = message_filters.Subscriber(self, PointCloud2, "/top_camera/rgb/preview/depth/points")

		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_sub, self.pc_sub], 10, 0.3, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_pc_callback)

		self.teleop_pub = self.create_publisher(Twist, "cmd_vel", 10)
		self.cmd_sub = self.create_subscription(String, "/park_cmd", self.park_cmd_callback, qos_profile_sensor_data)

		self.pixel_locations = None
		self.pixel_locations_set = False
		self.park_state = ParkState.PARKED
		self.circle_quality = 0

		#cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
		cv2.waitKey(1)
		cv2.moveWindow('Image',  1   ,1)
		cv2.moveWindow('Mask',   415 ,1)

		# cv2.createTrackbar('A', "Image", 0, 1000, self.nothing)
		# cv2.setTrackbarPos("A", "Image", 133)

	def park_cmd_callback(self, data):
		print(f"park_cmd_callback: {data}")
		self.park_state = ParkState.IDLE
		return

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
		p = msg.pose.pose.position
		self.position = np.array([p.x, p.y, p.z])
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

		if(not self.pixel_locations_set):
			self.pixel_locations = np.full((height, width,3), 0, dtype=np.float64)
			self.pixel_locations_set = True
			for y in range(0, height):
				for x in range(0, width):
					self.pixel_locations[y][x] = [x,y,0]
		
		#print(self.pixel_locations)

		## Tocke bi lahko transformiral vsaj v koordiante na oakd, zato, da ko se roka premakne, stvari se zmeraj isto delajo ...
		## Amapak ker zgleda da v pythonovem api-ji ni fukncije ki bi tranformirala cel point cloud, tega potem ne bom delal.

		# #tocke transformiramo v globalne (map) koordinate
		# time_now = rclpy.time.Time()
		# timeout = Duration(seconds=10.0)
		# trans = self.tf_buffer.lookup_transform("map", "top_camera", time_now, timeout)	
		# pc = do_transform_cloud(pc, trans)

		xyz = pc2.read_points_numpy(pc, field_names= ("y", "z", "x"))
		xyz = xyz.reshape((height,width,3))

		thresh = 0.665
		mask = np.full((height, width), 0, dtype=np.uint8)
		mask[(img[:,:,:] < 10).all(axis=2)] = 255
		mask[xyz[:,:,2] < thresh] = (0)

		#TODO
		#Zdej sem spremenil, da se poravnala izkljucno samo na sliko.
		#Kaj tocno naredit v primeru, ko slabo vidimo krog?

		#TODO
		#V primeru ko nic ne vidimo, se ravnamo po nav2 stacku.
		#Kadar krog vidimo 

		circle = self.fit_circle(self.pixel_locations[mask==255])
		if(circle != None):	
			circle_quality = circle_quality = math.pow(math.e, -0.1*(abs(78.59 - circle[2])))

			if(circle_quality > 0.2):
				img = cv2.arrowedLine(img, (160,180), (int(circle[0]), int(circle[1])), (0,0,255), 5)  

				if(self.park_state != ParkState.PARKED and (self.park_state == ParkState.IDLE or circle_quality > self.circle_quality)):
					print("Rotating")
					self.park_state = ParkState.ROTATING
					self.circle_quality = circle_quality

				if(self.park_state == ParkState.DRIVING):
					error_raw = circle[1] - 180
					error = abs(error_raw)
					error_dir = -sign(error_raw)
					kp = 0.01
					min_vel = 0.1

					out_strength = max(min_vel, error * kp)
					output = error_dir * out_strength
						
					cmd_msg = Twist()
					cmd_msg.angular.z = 0.
					if(abs(error) < 3): #stop driving, parked.
						cmd_msg.linear.x = 0.
						self.park_state = ParkState.PARKED
						print("PARKED")
					else:
						cmd_msg.linear.x = output
					self.teleop_pub.publish(cmd_msg)

				if(self.park_state == ParkState.ROTATING):
					error_raw = positive_angle(-math.pi/2 + math.atan2(circle[1]-height, circle[0]-160)) - math.pi

					error = abs(error_raw)
					error_dir = -sign(error_raw)
					kp = 2.0
					min_vel = 0.2

					out_strength = max(min_vel, error * kp)
					output = error_dir * out_strength
					
					cmd_msg = Twist()
					cmd_msg.linear.x = 0.
					if(abs(error) < 0.01): #stop rotating, next_state
						cmd_msg.angular.z = 0.
						self.park_state = ParkState.DRIVING
						print("Rotated")
					else:
						cmd_msg.angular.z = output
					self.teleop_pub.publish(cmd_msg)
			
		cv2.imshow("Mask", mask)
		#cv2.imshow("Image", img)

def main():
	print("OK")
	rclpy.init(args=None)
	node = park()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
