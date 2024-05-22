#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState

from visualization_msgs.msg import Marker

from tf_transformations import quaternion_from_euler, euler_from_quaternion

from std_srvs.srv import Trigger
from task1.msg import PersonInfo, GoalKeypoint

#import tf_transformations

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

import math
import numpy as np
from geometry_msgs.msg import Twist
from enum import Enum

STOP_AFTER_THREE = True

qos_profile = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

amcl_pose_qos = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

class LookaroundState(Enum):
	IDLE = 0
	LOOKING_AROUND = 1
	LOOKING_FOR_FACE = 2
class GreetState(Enum):
	NOT_GREETING = 0,
	DRIVING_TO_KEYPOINT = 1,
	TALKING = 2

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

def nothing():
	return

class MapGoals(Node):
	def __init__(self):
		super().__init__('map_goals')

		#Dodaj nazaj stare not za simulacijo

		self.simulation = False
		if(self.simulation):
			self.keypoints = [
				[-0.6600, -0.6000, 0], [-1.7100, -0.4000, 0], [-1.7600, -0.8000, 0], [-0.5100, -0.3000, 0], [-0.1600, -0.9500, 0], [-0.5100, -1.3000, 0],
				[-0.3600, -2.0000, 0], [ 0.0400, -1.7500, 0], [ 3.3400, -1.6500, 0], [ 3.2900, -1.8000, 0], [ 3.2400, -0.9000, 0], [ 1.6400,  0.0000, 0],
				[ 0.9400, -0.3500, 0], [ 0.8900,  1.8500, 0], [ 0.0900,  2.2500, 0], [ 0.0400,  1.9500, 0], [-1.2100,  1.3000, 0], [-0.9100,  0.9500, 0],
				[-1.6100,  1.1500, 0], [-1.6600,  3.1500, 0], [-1.8100,  4.4500, 0], [-1.3100,  4.4500, 0], [-1.6100,  3.3500, 0], [-1.1600,  2.9500, 0],
				[-0.7100,  3.3500, 0], [ 1.3400,  3.4000, 0], [ 2.2900,  2.6000, 0], [ 1.8900,  2.0000, 0], [ 2.4400,  1.3500, 0], [ 2.0900, -1.9000, 0]
			]
		else:
			self.keypoints = [
        	    [-2.6000,  2.3790, 0], [-1.2000,  2.3790, 0], [-0.6500,  1.9290, 0],
        	    [-0.0500,  2.3790, 0], [-0.8500,  1.5290, 0], [ 0.0000,  0.2790, 0],
        	    [-1.8000,  0.4790, 0], [-2.0500,  1.2790, 0], [-0.7500,  1.7290, 0]
			]
		self.last_keypoint = [0,0,0]
		self.face_keypoints = []
		self.keypoint_index = 0
		self.forward = True

		self.future = None
		
		self.goal_handle_inited = False
		self.goal_handle = None

		# Basic ROS stuff
		timer_frequency = 10
		map_topic = "/map"
		timer_period = 1/timer_frequency

		# Functional variables
		self.result_future = None
		self.currently_navigating = False
		self.greeting_state = GreetState.NOT_GREETING
		self.looking_around_state = LookaroundState.IDLE
		self.lookaround_timer = 0
		self.looking_around_preinit = False
		self.start_yaw = 0
		self.prev_yaw = 0
		self.angle_traveled = 0
		self.face_yaw = 0
		self.yaw_error_integral = 0
		self.t1 = 0
		
		self.clicked_x = None
		self.clicked_y = None
		self.ros_occupancy_grid = None
		self.map_np = None
		self.map_data = {"map_load_time":None,
						 "resolution":None,
						 "width":None,
						 "height":None,
						 "origin":None} # origin will be in the format [x,y,theta]
		
		self.greeted_faces = set()
		self.current_greet_face_id = -1

		# Subscribe to map, and create an action client for sending goals
		self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
		self.faces = self.create_subscription(GoalKeypoint, '/face_keypoints', self.add_face, 10)
		
		self.teleop_pub = self.create_publisher(Twist, "cmd_vel", 10)

		self.initial_pose_received = False
		self.position = None
		self.rotation = None
		self.yaw = 0
		self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
															  'amcl_pose',
															  self._amclPoseCallback,
															  amcl_pose_qos)
		self.waitUntilNav2Active()

		# Create a timer, to do the main work.
		self.timer = self.create_timer(timer_period, self.timer_callback)

		self.client = self.create_client(Trigger, '/say_hello')
		self.waiting_for_cancellation = False

		print(f"OK, simulation: {self.simulation}")
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

	def get_prev_keypoint(self):
		new_index = self.keypoint_index-1
		if(not self.forward):
			new_index = len(self.keypoints)-1-self.keypoint_index-1
		while(new_index < 0):
			new_index += len(self.keypoints)
	
		self.keypoint_index -= 1
		if self.keypoint_index < 0:
			self.keypoint_index = len(self.keypoints)-1

		new_kp = self.keypoints[new_index]	
		direction = [ new_kp[0]-self.last_keypoint[0], new_kp[1]-self.last_keypoint[1] ] 
		q = math.atan2(direction[1], direction[0])
		new_kp[2] = q
		#print(f"YAW: {q}, direction: {direction}")
		
		self.last_keypoint = new_kp
		return new_kp

	def get_next_keypoint(self):
		if(len(self.face_keypoints) > 0):
			self.greeting_state = GreetState.DRIVING_TO_KEYPOINT
			result, face_id = self.face_keypoints.pop(0)
			self.face_yaw = result[2]
			self.current_greet_face_id = face_id
			return result
	 
		new_index = self.keypoint_index
		if(not self.forward):
			new_index = len(self.keypoints)-1-self.keypoint_index
		
		self.keypoint_index += 1
		if self.keypoint_index >= len(self.keypoints):
			self.keypoint_index = 0
			self.forward = not self.forward

		new_kp = self.keypoints[new_index]	
		direction = [ new_kp[0]-self.last_keypoint[0], new_kp[1]-self.last_keypoint[1] ] 
		q = math.atan2(direction[1], direction[0])
		new_kp[2] = q
		#print(f"YAW: {q}, direction: {direction}")
		
		self.last_keypoint = new_kp
		return new_kp

	def timer_callback(self): 
		if STOP_AFTER_THREE and len(self.greeted_faces) >= 3:
			self.get_logger().info(f"\n\nDone, i found and greeted with 3 faces.\n\n")
			self.stop_following_keypoints()
			rclpy.shutdown()
			exit(0)

		if self.future and self.future.done():
			result = self.future.result()
			self.get_logger().info(f"greeting result: {result}")
			self.greeting_state = GreetState.NOT_GREETING
			self.future = None
			self.greeted_faces.add(self.current_greet_face_id)
	
		if(self.waiting_for_cancellation):
			#print("Waiting for cancellation")
			return

		if not self.currently_navigating:
			if(self.looking_around_state != LookaroundState.IDLE):
				self.lookaround_update()
			elif(self.greeting_state == GreetState.NOT_GREETING):
				world_x, world_y, orientation = self.get_next_keypoint()
				goal_pose = self.generate_goal_message(world_x, world_y, orientation)
				self.go_to_pose(goal_pose)
		return

	def generate_goal_message(self, x, y, theta=0.2):
		goal_pose = PoseStamped()
		goal_pose.header.frame_id = 'map'
		goal_pose.header.stamp = self.get_clock().now().to_msg()

		goal_pose.pose.position.x = x
		goal_pose.pose.position.y = y
		goal_pose.pose.orientation = self.yaw_to_quaternion(theta)
		
		return goal_pose

	def map_pixel_to_world(self, x, y, theta=0):
		### Convert a pixel in an numpy image, to a real world location
		### Works only for theta=0
		assert not self.map_data["resolution"] is None

		# Apply resolution, change of origin, and translation
		# 
		world_x = x*self.map_data["resolution"] + self.map_data["origin"][0]
		world_y = (self.map_data["height"]-y)*self.map_data["resolution"] + self.map_data["origin"][1]

		# Apply rotation
		return world_x, world_y

	def world_to_map_pixel(self, world_x, world_y, world_theta=0.2):
		### Convert a real world location to a pixel in a numpy image
		### Works only for theta=0
		assert self.map_data["resolution"] is not None

		# Apply resolution, change of origin, and translation
		# x is the first coordinate, which in opencv and numpy that is the matrix row - vertical
		# vertical
		x = int((world_x - self.map_data["origin"][0])/self.map_data["resolution"])
		y = int(self.map_data["height"] - (world_y - self.map_data["origin"][1])/self.map_data["resolution"] )
		
		# Apply rotation
		return x, y

	def go_to_pose(self, pose):
		"""Send a `NavToPose` action request."""
		self.currently_navigating = True

		while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
			self.get_logger().info("'NavigateToPose' action server not available, waiting...")

		goal_msg = NavigateToPose.Goal()
		goal_msg.pose = pose
		goal_msg.behavior_tree = ""

		#self.get_logger().info('Attempting to navigate to goal: ' + str(pose.pose.position.x) + ' ' + str(pose.pose.position.y) + '...')
		self.send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
		
		# Call this function when the Action Server accepts or rejects a goal
		self.send_goal_future.add_done_callback(self.goal_accepted_callback)
		return

	def goal_accepted_callback(self, future):
		self.goal_handle = future.result()
		self.goal_handle_inited = True

		if not self.goal_handle.accepted:
			self.get_logger().error('Goal was rejected!')
			self.currently_navigating = False
			return	

		self.currently_navigating = True
		self.result_future = self.goal_handle.get_result_async()
		self.result_future.add_done_callback(self.get_result_callback)
		return

	def get_result_callback(self, future):
		self.currently_navigating = False
		status = future.result().status

		# print(f"get result callback: status: {status}, vals of canceled: {GoalStatus.STATUS_CANCELED}, succed: {GoalStatus.STATUS_SUCCEEDED}")
		self.waiting_for_cancellation = False
		if(status == GoalStatus.STATUS_CANCELED):
			print("Goal cancelled.")
			return

		#if(self.currently_greeting):
		#	self.greet()

		if(status == GoalStatus.STATUS_SUCCEEDED):
			# self.get_logger().info(f'Goal reached (according to Nav2).')
			if(self.greeting_state == GreetState.DRIVING_TO_KEYPOINT):	
				self.greeting_state = GreetState.TALKING
				self.lookat_face()
			else:
				self.lookaround()
		else:
			self.get_logger().info(f'Goal failed with status code: {status}')
			#V tem primeru vzamemo prejsni goal in poskusimo do tja...
			
			world_x, world_y, orientation = self.get_prev_keypoint()
			goal_pose = self.generate_goal_message(world_x, world_y, orientation)
			self.go_to_pose(goal_pose)
		
		return

	def greet(self):
		req = Trigger.Request()
		self.future = self.client.call_async(req)
		return

	def yaw_to_quaternion(self, angle_z = 0.):
		quat_tf = quaternion_from_euler(0, 0, angle_z)
		quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3]) # for tf_turtle
		return quat_msg
	
	def get_rotation_matrix(self, theta):
		c = np.cos(theta)
		s = np.sin(theta)
		rot = np.array([[c, -s],
						[s , c]])
		return rot

	def add_face(self, goal_keypoint): #Add or merge
		if(goal_keypoint.face_id in self.greeted_faces):
			return


		if(self.current_greet_face_id == goal_keypoint.face_id):
			if(self.greeting_state == GreetState.DRIVING_TO_KEYPOINT and self.looking_around_state == LookaroundState.IDLE):
				#popravi current goal position
				print("Improve position")
				self.stop_following_keypoints()
				self.face_keypoints.insert(0, [[goal_keypoint.position[0], goal_keypoint.position[1], goal_keypoint.yaw], goal_keypoint.face_id]) # Stack
				self.greeting_state = GreetState.NOT_GREETING
				return
			pass
		else:
			index = -1
			for i in range(len(self.face_keypoints)):
				if(self.face_keypoints[i][1] == goal_keypoint.face_id):
					index = i
					break
			if(index >= 0):
				self.face_keypoints[index] = ([[goal_keypoint.position[0], goal_keypoint.position[1], goal_keypoint.yaw], goal_keypoint.face_id]) #Update
			else: # Add
				self.face_keypoints.append([[goal_keypoint.position[0], goal_keypoint.position[1], goal_keypoint.yaw], goal_keypoint.face_id]) # Stack
				if(self.greeting_state == GreetState.NOT_GREETING and self.looking_around_state == LookaroundState.IDLE):
					print("Fast stop")
					self.stop_following_keypoints()
					self.get_prev_keypoint()
		return

	def stop_following_keypoints(self):
		if(not self.currently_navigating):
			return

		self.currently_navigating = False
		while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
			self.get_logger().info("'NavigateToPose' action server not available, waiting...")

		if(self.goal_handle_inited):
			self.nav_to_pose_client._cancel_goal_async(self.goal_handle)
			self.waiting_for_cancellation = True
		return

	def lookaround_pre_update(self):
		self.looking_around_preinit = True
		self.start_yaw = self.yaw
		self.prev_yaw = self.yaw
		self.angle_traveled = 0
		return

	def lookaround_update(self): #TODO look for face routine
		if(not self.looking_around_preinit):
			self.lookaround_pre_update()

		if(self.looking_around_state == LookaroundState.LOOKING_AROUND):
			delta = sharp_angle(self.yaw, self.prev_yaw)
			self.prev_yaw = self.yaw
			self.angle_traveled += delta

			print(f"Angle: {self.angle_traveled}")

			if(abs(self.angle_traveled) >= deg2rad(360)):
				self.lookaround_end()
				return

			cmd_msg = Twist()
			cmd_msg.angular.z = 0.5
			cmd_msg.linear.x = 0.
			self.teleop_pub.publish(cmd_msg)
		else: #Accurate rotation to face_yaw
			error = sharp_angle(self.yaw, self.face_yaw)	#TODO

			#ce je error vec kot 2 sekundi < 0.1 pol smo gut
			if(abs(error) > deg2rad(10)):
				self.t1 = millis()

			if(millis() - self.t1 > 2000):
				self.lookaround_end()
				self.greet()
				return

			kp = 0.1
			ki = 0.00

			self.yaw_error_integral += error * ki
			self.yaw_error_integral = clamp(self.yaw_error_integral, -0.5, 0.5)
		
			vel = error * kp + self.yaw_error_integral * ki
			cmd_msg = Twist()
			cmd_msg.angular.z = -vel
			cmd_msg.linear.x = 0.
			self.teleop_pub.publish(cmd_msg)

			print(f"error: {error}, p: {kp*error}, i: {ki*self.yaw_error_integral}, integral: {self.yaw_error_integral}")

		return

	def lookaround_end(self):
		self.looking_around_state = LookaroundState.IDLE	
		self.looking_around_preinit = False
		return

	def lookaround(self):
		# self.looking_around_state = LookaroundState.LOOKING_AROUND
		return
	
	def lookat_face(self):
		self.looking_around_state = LookaroundState.LOOKING_FOR_FACE
		return

def main():
	rclpy.init(args=None)
	node = MapGoals()
	rclpy.spin(node)
	rclpy.shutdown()

if __name__ == '__main__':
	main()
