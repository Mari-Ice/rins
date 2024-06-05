#!/usr/bin/python3

import face_extractor
import math
import random
import time
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import tf2_geometry_msgs as tfg
from cv_bridge import CvBridge, CvBridgeError

from enum import Enum
from flags import Flags
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import ColorRGBA
from std_msgs.msg import String

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from lifecycle_msgs.srv import GetState

import message_filters
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, PoseStamped, Quaternion, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from task3.msg import RingInfo, Waypoint, FaceInfo
from task3.srv import Color
from std_srvs.srv import Trigger
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

qos_profile = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

def array2point(arr):
	p = Point()
	p.x = float(arr[0])
	p.y = float(arr[1])
	p.z = float(arr[2])
	return p

def create_point_marker(point, header):
	marker = Marker()
	marker.header = header

	marker.type = 2

	marker.scale.x = 0.1
	marker.scale.y = 0.1
	marker.scale.z = 0.1

	marker.color.r = 1.0
	marker.color.g = 1.0
	marker.color.b = 1.0
	marker.color.a = 1.0
	marker.pose.position = array2point(point)
	
	return marker

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
def sharp_angle(alpha, beta):
	alpha = pos_angle(alpha)
	beta = pos_angle(beta)
	direction = 1

	if(alpha < beta):
		direction = alpha
		alpha = beta
		beta = direction
		direction = -1
	
	if(beta+math.pi >= alpha):
		return direction * (alpha - beta)
	return direction * (alpha + beta - 2*math.pi)

class Clusters():
	class ClusterResult(Flags):
		FOUND_POTENTIAL	= 1
		NEW_VALID 		= 2
		NEW_BEST 		= 4
		MERGED   		= 8

	def __init__(self, data_position_fcn, data_quality_fcn, quality_threshold, min_cluster_dist):
		self.data_position_fcn = data_position_fcn
		self.data_quality_fcn  = data_quality_fcn
		self.quality_threshold = quality_threshold
		self.min_cluster_dist  = min_cluster_dist
		self.data_arr = [] # Oblike [ [data1_best, data1_last], [data2_best, data2_last], ... ]

		return

	@staticmethod
	def _mag_sq(vec):
		return np.dot(vec,vec)
	@staticmethod
	def _mag(vec):
		return math.sqrt(Clusters._mag_sq(vec))

	def _closest_cluster_to_data(self, data):
		result_data_index = -1
		result_distance = float("Inf")

		ndp = self.data_position_fcn(data)
		for i, d in enumerate(self.data_arr):
			dp1 = self.data_position_fcn(d[0])		
			dp2 = self.data_position_fcn(d[1])		
			dist1 = Clusters._mag_sq(ndp - dp1)
			dist2 = Clusters._mag_sq(ndp - dp2)
			dist = min(dist1, dist2)
			if(dist < result_distance):
				result_distance = dist
				result_data_index = i

		result_distance = math.sqrt(result_distance)
		return [result_distance, result_data_index]	
	
	def _cleanup_potential_clusters(self):
		for j, d1 in enumerate(self.data_arr):
			q1 = self.data_quality_fcn(d1[0])
			if(q1 < self.quality_threshold):
				continue
			dp1 = self.data_position_fcn(d1[0])		

			for i, d2 in enumerate(self.data_arr):
				q2 = self.data_quality_fcn(d2[0])
				if(q1 >= self.quality_threshold):
					continue
				
				dp2 = self.data_position_fcn(d2[0])		
				dist = Clusters._mag(dp2-dp1)
				if(dist < 1.0):
					self.data_arr.remove(d2)
		return
	
	def _merge_data(self, target_index, data):
		result = Clusters.ClusterResult.MERGED
		new_data = [self.data_arr[target_index][0], data]

		new_data_q = self.data_quality_fcn(data)
		old_data_q = self.data_quality_fcn(new_data[0])

		if(new_data_q > self.quality_threshold and old_data_q <= self.quality_threshold):
			result |= (Clusters.ClusterResult.NEW_VALID | Clusters.ClusterResult.NEW_BEST)
		if(new_data_q > old_data_q):
			new_data = [data, data]
			result |= Clusters.ClusterResult.NEW_BEST
				
		self.data_arr[target_index] = new_data
		return result

	def add(self, data):
		status = Clusters.ClusterResult.FOUND_POTENTIAL
		min_dist, cluster_index = self._closest_cluster_to_data(data)

		if(min_dist > self.min_cluster_dist): 
			cluster_index = len(self.data_arr)
			self.data_arr.append([data,data])
			new_data_q = self.data_quality_fcn(data)
			if(new_data_q > self.quality_threshold):
				status |= (Clusters.ClusterResult.NEW_VALID | Clusters.ClusterResult.NEW_BEST)
		elif(cluster_index >= 0):
			status |= self._merge_data(cluster_index, data)

		if(bool(status & Clusters.ClusterResult.NEW_BEST)):
			self._cleanup_potential_clusters()

		return [status, cluster_index]
	def get_best(self, index):
		if(index >= len(self.data_arr)):
			return None
		return self.data_arr[index][0]
	def get_last(self, index):
		if(index >= len(self.data_arr)):
			return None
		return self.data_arr[index][1]


class Nav2State(Enum):
	IDLE = 0
	LONG_IDLE = 1
	NAVIGATING = 2
	WAITING_FOR_CANCELLATION = 3

def fdata_position_fcn(fdata):
	return fdata[1]
def fdata_quality_fcn(fdata):
	return float(fdata[0].quality)

class MasterNode(Node):
	def __init__(self):
		super().__init__('master_node')
		
		self.map_np = None
		self.map_data = {"map_load_time":None, "resolution":None, "width":None, "height":None, "origin":None}
		self.occupancy_grid_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_profile)
		self.park_search_circle_size = 10

		pwd = os.getcwd()
		self.gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]
		self.costmap = cv2.cvtColor(cv2.imread(f"{self.gpath}/src/dis_tutorial3/maps/costmap.pgm"), cv2.COLOR_BGR2GRAY)

		self.teleop_pub = self.create_publisher(Twist, "cmd_vel", 20)

		self.tf_buffer   = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		
		self.goal_handle_inited = False
		self.goal_handle = None

		self.nav2_has_goal_pending = False
		self.nav2_current_goal = None
		self.nav2state = Nav2State.IDLE
		self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
		self.initial_pose_received = False
		self.position = None
		self.rotation = None
		self.yaw = 0
		self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amclPoseCallback, qos_profile)

		t0 = millis()

		self.waitUntilNav2Active()
		self.nav2timer = self.create_timer(1/10, self.nav2_timer_loop)
		self.nav2_t1 = millis()

		while(millis() - t0 < 7000):
			time.sleep(0.1)

		# Face sub
		self.update_timer = self.create_timer(1/10, self.update)
		self.face_sub = self.create_subscription(FaceInfo, "/face_info", self.face_callback, qos_profile_sensor_data)
		self.face_clusters = Clusters(fdata_position_fcn, fdata_quality_fcn, 0.1, 0.5)
		self.visited = []

		# Rotate to target
		self.rotate2target_t2 = millis()
		
		#To zamenjas potem z state machinom...
		self.face_index = -1
		self.navigating_to_face = False
		self.state = 0 
		self.rotate_error_fcn_list = [self.yaw_img_error, self.yaw_acml_error]

		self.bridge = CvBridge()
		self.img = None
		self.mona_imgs = []

		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_image_sub], 10, 0.05, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_callback)

		# PCA stuff
		self.fe = face_extractor.face_extractor()
		cv2.namedWindow("ErrorMona", cv2.WINDOW_NORMAL)

		print("OK")
		return

	def rgb_callback(self, rgb_data):
		self.img = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")

	def yaw_acml_error(self):
		return -sharp_angle(self.yaw + math.pi/2, self.nav2_current_goal[1])
	def yaw_img_error(self):
		fdata = self.face_clusters.get_last(self.face_index)	

		if(millis() - fdata[3] > 1000):
			return None
		return fdata[0].yaw_relative

	def update(self): # To tudi potem postane del state machina
		if(self.nav2state == Nav2State.LONG_IDLE and self.navigating_to_face):
			self.visited.append(self.face_index)
			self.navigating_to_face = False
			self.state = 1
		if(self.state == 1):
			if(self.rotate_to_target_loop(self.rotate_error_fcn_list)):
				self.state = 2
		if(self.state == 2):
			if(self.move_closer_to_img()):
				self.state = 3
		if(self.state == 3):
			self.mona_imgs = []
			self.mona_imgs.append(self.img)
			self.state = 4
		if(self.state == 4):
			fdata = self.face_clusters.get_last(self.face_index)	
			result = self.fe.find_anomalies(self.mona_imgs[0], np.array(fdata[0].img_bounds_xyxy))
			if(not result[0]): #Fuck
				pass
			else:
				cv2.imshow("ErrorMona", result[1])
				cv2.waitKey(1)
				err = np.sum(result[1])
				print("PCA err: ", err)
				if(err > 1000):
					print("FAKE")
				else:
					print("REAL")
				pass		
			self.state = 0

		return 

	def goal_accepted_callback(self, future):
		if self.goal_handle_inited and not self.goal_handle.accepted:
			self.nav2state = Nav2State.IDLE
			self.nav2_t1 = millis()
			self.get_logger().error('Goal was rejected!')
			return	

		self.goal_handle = future.result()
		self.goal_handle_inited = True

		self.result_future = self.goal_handle.get_result_async()
		self.result_future.add_done_callback(self.get_result_callback)
		return

	def get_result_callback(self, future):
		status = future.result().status
		self.nav2state = Nav2State.IDLE
		self.nav2_t1 = millis()
		
		# if(status == GoalStatus.STATUS_CANCELED):
		# 	pass	
		# elif(status != GoalStatus.STATUS_SUCCEEDED):
		# 	# self.get_logger().info(f'Goal failed with status code: {status}')
		return

	def stop_nav2(self):
		if(self.nav2state != Nav2State.NAVIGATING):
			return

		while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
			self.get_logger().info("'NavigateToPose' action server not available, waiting...")

		if(self.goal_handle_inited):
			self.nav_to_pose_client._cancel_goal_async(self.goal_handle)
			self.nav2state = Nav2State.WAITING_FOR_CANCELLATION
		return

	def nav2_timer_loop(self):
		# print("nav2state: ", self.nav2state)
		if(self.nav2state == Nav2State.IDLE and millis() - self.nav2_t1 > 1400):
			self.nav2state = Nav2State.LONG_IDLE

		if(self.nav2state == Nav2State.WAITING_FOR_CANCELLATION):
			return
		if(self.nav2_has_goal_pending):	
			self.nav2_has_goal_pending = False
			self.nav2state = Nav2State.NAVIGATING
			
			pos_yaw = self.nav2_current_goal
			pose = PoseStamped()
			pose.header.frame_id = 'map'
			pose.header.stamp = self.get_clock().now().to_msg()
			pose.pose.position.x = float(pos_yaw[0][0])
			pose.pose.position.y = float(pos_yaw[0][1])
			quat_tf = quaternion_from_euler(0, 0, pos_yaw[1])
			quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3]) # for tf_turtle
			pose.pose.orientation = quat_msg

			while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
				self.get_logger().info("'NavigateToPose' action server not available, waiting...")
			goal_msg = NavigateToPose.Goal()
			goal_msg.pose = pose
			goal_msg.behavior_tree = ""
			send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
			send_goal_future.add_done_callback(self.goal_accepted_callback)
		return	

	def nav2_goto(self, position, yaw):
		fx, fy = self.get_valid_close_position(position[0], position[1])
		position = np.array([fx,fy])

		if(self.nav2state == Nav2State.NAVIGATING):
			#najprej preverimo ali je goal sploh razlicen lel.
			if(self.nav2_current_goal != None):
				if(np.linalg.norm(position - self.nav2_current_goal[0]) < 0.05 and abs(sharp_angle(yaw, self.nav2_current_goal[1])) < 0.035):
					return #prakticno isti goal
			self.stop_nav2()
		else:
			self.nav2state == Nav2State.NAVIGATING
		
		self.nav2_current_goal = [position, yaw]
		self.nav2_has_goal_pending = True

	def transform_point(self, point, from_link, to_link):
		time_now = rclpy.time.Time()
		timeout = Duration(seconds=0.5)
		trans = self.tf_buffer.lookup_transform(to_link, from_link, time_now, timeout)	
		
		p = PointStamped() 
		p.header.frame_id = "/map"
		p.header.stamp = time_now.to_msg()
		p.point.x = float(point[0])
		p.point.y = float(point[1])
		p.point.z = float(point[2])
		p_global = tfg.do_transform_point(p, trans)
	
		return np.array([p_global.point.x, p_global.point.y, p_global.point.z])

	def face_callback(self, finfo):

		global_position = self.transform_point(np.array(finfo.position_relative), "oakd_link", "map")
		global_normal  = self.transform_point((np.array(finfo.position_relative) + np.array(finfo.normal_relative)), "oakd_link", "map") - global_position

		fdata = [finfo, global_position, global_normal, millis()]	 
		status, index = self.face_clusters.add(fdata)

		if(self.state != 0 or self.navigating_to_face):
			return

		if(not index in self.visited):
			# print(f"Face cluster status: {status}")
			if(status & Clusters.ClusterResult.NEW_BEST):
				self.nav2_goto(global_position + 0.3 * global_normal, -math.atan2(global_normal[0], global_normal[1]))
				self.navigating_to_face = True
				self.face_index = index

		return

	def map_pixel_to_world(self, x, y, theta=0):
		assert not self.map_data["resolution"] is None
		world_x = x*self.map_data["resolution"] + self.map_data["origin"][0]
		world_y = (self.map_data["height"]-y)*self.map_data["resolution"] + self.map_data["origin"][1]
		return [world_x, world_y]

	def world_to_map_pixel(self, world_x, world_y, world_theta=0.2):
		assert self.map_data["resolution"] is not None
		x = int((world_x - self.map_data["origin"][0])/self.map_data["resolution"])
		y = int(self.map_data["height"] - (world_y - self.map_data["origin"][1])/self.map_data["resolution"] )
		return [x, y]

	def map_callback(self, msg):
		self.get_logger().info(f"Read a new Map (Occupancy grid) from the topic.")
		self.map_np = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
		self.map_np = np.flipud(self.map_np)
		self.map_np[self.map_np==0] = 127
		self.map_np[self.map_np==100] = 0
		self.map_data["map_load_time"]=msg.info.map_load_time
		self.map_data["resolution"]=msg.info.resolution
		self.map_data["width"]=msg.info.width
		self.map_data["height"]=msg.info.height
		quat_list = [msg.info.origin.orientation.x,
					 msg.info.origin.orientation.y,
					 msg.info.origin.orientation.z,
					 msg.info.origin.orientation.w]
		self.map_data["origin"]=[msg.info.origin.position.x,
								 msg.info.origin.position.y,
								 euler_from_quaternion(quat_list)[-1]]
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

	def get_valid_close_position(self, ox,oy):
		try:
			mx,my = self.world_to_map_pixel(ox,oy)
		except:	
			return [ox,oy]

		height, width = self.costmap.shape[:2]
		pixel_locations = np.full((height, width, 2), 0, dtype=np.uint16)
		for y in range(0, height):
			for x in range(0, width):
				pixel_locations[y][x] = [x,y]

		mask1 = np.full((height, width), 0, dtype=np.uint8)
		cv2.circle(mask1,(mx,my),self.park_search_circle_size,255,-1)
		mask1[self.costmap < 200] = 0
		pts = np.array(pixel_locations[mask1==255])
		if(len(pts) > 0):
			#Find closest to ox,oy
			norms = np.linalg.norm((pts - np.array([mx,my])), axis=1)
			index_of_smallest_norm = np.argmin(norms)
			center = pts[index_of_smallest_norm]
			return self.map_pixel_to_world(center[0], center[1])
		else:
			return [ox,oy]

	def rotate_to_target_loop(self, error_fcn_list):
		error = None
		for err_fcn in error_fcn_list:
			e = err_fcn()
			if(e != None):
				error = e
				break
		if(error == None): #V tem primeru se lahko npr vrtis 
			cmd_msg = Twist()
			cmd_msg.angular.z = 1.8
			cmd_msg.linear.x = 0.
			self.teleop_pub.publish(cmd_msg)
			return False
		
		# print(f"Rotate error: {error}")

		if(abs(error) > deg2rad(4)):
			self.rotate2target_t2 = millis()

		if(millis() - self.rotate2target_t2 > 2000):
			return True

		kp = 1.0
		
		vel = error * kp
		cmd_msg = Twist()
		cmd_msg.angular.z = vel
		cmd_msg.linear.x = 0.
		self.teleop_pub.publish(cmd_msg)
		return False	

	def move_closer_to_img(self):
		print("Moving closer")
		fdata = self.face_clusters.get_last(self.face_index)	
		error = min(465 - fdata[0].img_bounds_xyxy[3], fdata[0].img_bounds_xyxy[1] - 85)
		if(error < 0):
			return True
	
		print("Move err: ", error)
		cmd_msg = Twist()
		cmd_msg.angular.z = 0.
		cmd_msg.linear.x = error * 0.001 + 0.01
		self.teleop_pub.publish(cmd_msg)
		return False


def main():
	rclpy.init(args=None)
	rd_node = MasterNode()
	rclpy.spin(rd_node)
	cv2.destroyAllWindows()
	return

if __name__ == '__main__':
	main()
