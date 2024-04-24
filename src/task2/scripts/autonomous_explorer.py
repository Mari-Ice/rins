#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Quaternion, PoseStamped

from visualization_msgs.msg import Marker

from tf_transformations import quaternion_from_euler, euler_from_quaternion

from std_srvs.srv import Trigger

import cv2
import numpy as np
import math

#import tf_transformations

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

import math
import numpy as np

STOP_AFTER_THREE = True

qos_profile = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)
	   
class MapGoals(Node):
	def __init__(self):
		super().__init__('map_goals')

		self.keypoints = []
		self.last_keypoint = [0,0,0]
		self.face_keypoints = []
		self.keypoint_index = 0
		self.forward = True

		self.future = None

		# Basic ROS stuff
		timer_frequency = 10
		map_topic = "/map"
		timer_period = 1/timer_frequency

		# Functional variables
		self.pending_goal = True
		self.result_future = None
		self.currently_navigating = False
		self.currently_greeting = False
		self.clicked_x = None
		self.clicked_y = None
		self.ros_occupancy_grid = None
		self.map_np = None
		self.map_data = {"map_load_time":None,
						 "resolution":None,
						 "width":None,
						 "height":None,
						 "origin":None} # origin will be in the format [x,y,theta]
		self.face_count = 0
		# Subscribe to map, and create an action client for sending goals
		self.occupancy_grid_sub = self.create_subscription(OccupancyGrid, map_topic, self.map_callback, qos_profile)
		self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
		self.faces = self.create_subscription(Marker, '/detected_faces', self.add_face, 10)
		# Create a timer, to do the main work.
		self.timer = self.create_timer(timer_period, self.timer_callback)

		self.client = self.create_client(Trigger, '/say_hello')

		self.calculate_keypoints()

	def map_callback(self, msg):
		self.get_logger().info(f"Read a new Map (Occupancy grid) from the topic.")
		# reshape the message vector back into a map
		self.map_np = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
		# fix the direction of Y (origin at top for OpenCV, origin at bottom for ROS2)
		self.map_np = np.flipud(self.map_np)
		# change the colors so they match with the .pgm image
		self.map_np[self.map_np==0] = 127
		self.map_np[self.map_np==100] = 0
		# load the map parameters
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
								 tf_transformations.euler_from_quaternion(quat_list)[-1]]
		#self.get_logger().info(f"Read a new Map (Occupancy grid) from the topic.")

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
		print(f"YAW: {q}, direction: {direction}")
		
		self.last_keypoint = new_kp
		return new_kp

	def get_next_keypoint(self):
		if(len(self.face_keypoints) > 0):
			self.currently_greeting = True
			return self.face_keypoints.pop(0)
	 
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
		print(f"YAW: {q}, direction: {direction}")
		
		self.last_keypoint = new_kp
		return new_kp

	def timer_callback(self): 
		if STOP_AFTER_THREE and self.face_count >= 3:
			self.get_logger().info(f"\n\nDone, i found and greeted with 3 faces.\n\n")
			rclpy.shutdown()
			exit(0)

		if self.future and self.future.done():
			result = self.future.result()
			self.get_logger().info(f"greeting result: {result}")
			self.currently_greeting = False
			self.future = None
			self.face_count += 1
		
		# If the robot is not currently navigating to a goal, and there is a goal pending
		if not self.currently_navigating and self.pending_goal and not self.currently_greeting:
			world_x, world_y, orientation = self.get_next_keypoint()
			goal_pose = self.generate_goal_message(world_x, world_y, orientation)
			self.go_to_pose(goal_pose)

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
		self.pending_goal = False

		while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
			self.get_logger().info("'NavigateToPose' action server not available, waiting...")

		goal_msg = NavigateToPose.Goal()
		goal_msg.pose = pose
		goal_msg.behavior_tree = ""

		self.get_logger().info('Attempting to navigate to goal: ' + str(pose.pose.position.x) + ' ' + str(pose.pose.position.y) + '...')
		self.send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
		
		# Call this function when the Action Server accepts or rejects a goal
		self.send_goal_future.add_done_callback(self.goal_accepted_callback)

	def goal_accepted_callback(self, future):
		goal_handle = future.result()

		if not goal_handle.accepted:
			self.get_logger().error('Goal was rejected!')
			self.currently_greeting = False
			return	

		self.currently_navigating = True
		self.pending_goal = False
		self.result_future = goal_handle.get_result_async()
		self.result_future.add_done_callback(self.get_result_callback)

	def get_result_callback(self, future):
		if(self.currently_greeting):
			self.greet()

		self.currently_navigating = False
		# self.pending_goal = False
		self.pending_goal = True
		status = future.result().status

		if status != GoalStatus.STATUS_SUCCEEDED:
			self.get_logger().info(f'Goal failed with status code: {status}')
			#V tem primeru vzamemo prejsni goal in poskusimo do tja...
			
			world_x, world_y, orientation = self.get_prev_keypoint()
			goal_pose = self.generate_goal_message(world_x, world_y, orientation)
			self.go_to_pose(goal_pose)

			return
		 
		self.get_logger().info(f'Goal reached (according to Nav2).')
		
		# self.pending_goal = True

	def greet(self):
		req = Trigger.Request()
		self.future = self.client.call_async(req)

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

	def add_face(self, marker):
		x = marker.pose.orientation.x
		y = marker.pose.orientation.y
		z = marker.pose.orientation.z
		w = marker.pose.orientation.w
		_, _, theta = euler_from_quaternion((x, y, z, w))
		self.face_keypoints.append([marker.pose.position.x, marker.pose.position.y, theta])

	def calculate_keypoints(self):
		image = self.map_np
		original_image = image.copy()

		# convert the image to grayscale
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# threshold the image
		_, image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)

		dst = cv2.Canny(image, 50, 200, None, 3)

		lines = cv2.HoughLines(dst, 1, np.pi / 180, 25, None, 0, 0)

		if lines is not None:
			for i in range(0, len(lines)):
				rho = lines[i][0][0]
				theta = lines[i][0][1]
				a = math.cos(theta)
				b = math.sin(theta)
				x0 = a * rho
				y0 = b * rho
				pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
				pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
				cv2.line(image, pt1, pt2, (0,0,0), 3, cv2.LINE_AA)

		# erode
		kernel = np.ones((10,10), np.uint8)
		image = cv2.erode(image, kernel, iterations=1)

		# find connected components and their centroids
		num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

		for i in range(1, num_labels):
			# print(f"Centroid {i}: {centroids[i]}")
			world_keypoint = self.map_pixel_to_world(centroids[i][0], centroids[i][1])
			self.keypoints.append([world_keypoint[0], world_keypoint[1], 0])

if __name__ == '__main__':
	main()
