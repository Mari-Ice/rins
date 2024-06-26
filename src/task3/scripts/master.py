#!/usr/bin/python3

import math
import random
import time
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros
import os

from enum import Enum
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import ColorRGBA
from std_msgs.msg import String

from geometry_msgs.msg import PointStamped, Point, Twist
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, PoseStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from task3.msg import RingInfo, Waypoint, AnomalyInfo, FaceInfo, Park, CylinderInfo
from task3.srv import Color
from std_srvs.srv import Trigger
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from statemachine import State
from statemachine import StateMachine

MIN_DETECTED_RINGS = 4

qos_profile = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

"""
	Vsi nodi publish-ajo svoje podatke, skupaj z neko kvaliteto podatkov
"""

class MasterState(Enum):
	INIT = 0
	CAMERA_SETUP_FOR_EXPLORATION = 1
	EXPLORING = 2
	MOVING_TO_PERSON = 3
	TALKING_TO_PERSON = 4
	VALIDATING_RING = 5
	CHECKING_INFO = 6
	MOVING_TO_RING_FOR_PARKING = 7
	CAMERA_SETUP_FOR_PARKING = 8
	PARKING = 9
	CAMERA_SETUP_FOR_CYLINDER = 10
	FINDING_CYLINDER = 11
	MOVING_TO_CYLINDER = 12
	CAMERA_SETUP_FOR_QR = 13
	READING_QR = 14
	DISPLAYING_PHOTO_FROM_QR = 15
	CAMERA_SETUP_FOR_PAINTINGS = 16
	SEARCHING_FOR_PAINTINGS = 17
	MOVING_TO_PAINTING = 18
	DETECTING_ANOMALIES = 19
	MOVING_TO_GENUINE_PAINTING = 20
	DONE = 21

def create_marker_point(position, color=[1.,0.,0.], size=0.1):
	marker = Marker()
	marker.type = 2
	marker.scale.x = size
	marker.scale.y = size
	marker.scale.z = size
	marker.color.r = float(color[0])
	marker.color.g = float(color[1])
	marker.color.b = float(color[2])
	marker.color.a = 1.0
	marker.pose.position.x = float(position[0])
	marker.pose.position.y = float(position[1])
	marker.pose.position.z = float(position[2])
	return marker

def argmin(arr, normal_fcn, params):
	c_min = 100001
	c_min_index = -1
	for i, e in enumerate(arr):
		dist = normal_fcn(e, params)
		if(dist < c_min):
			c_min = dist
			c_min_index = i
	return [c_min, c_min_index]

def dist_normal_fcn(elt, new):
	return np.linalg.norm(np.array(elt[0].position) - np.array(new.position))

def millis():
	return round(time.time() * 1000)	

def array2point(arr):
	p = Point()
	p.x = float(arr[0])
	p.y = float(arr[1])
	p.z = float(arr[2])
	return p

def compare_faces(face1, face2):
	dist1 = np.linalg.norm(face1.origin + face1.normal * 0.25  - face2.origin)
	dist2 = np.linalg.norm(face2.origin + face2.normal * 0.25  - face1.origin)
	dist = min(dist1, dist2)
	
	cosfi = face1.normal.dot(face2.normal)
	return (dist < 0.65) and (cosfi > -0.25)

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

class MasterNode(Node):
	def __init__(self):
		super().__init__('master_node')
  
		self.sm = MasterStateMachine(self)
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		
		self.inited = False
		self.timer1 = self.create_timer(3, self.init2)

		print("OK")
		return

	def init2(self):
		if(self.inited):
			return
		self.inited = True
		self.park_sub = self.create_subscription(Park, '/park_near_obj', self.parking_info_callback, QoSReliabilityPolicy.BEST_EFFORT)
		self.parking_stopped_sub = self.create_subscription(String, '/parking_stopped', self.parking_stopped_callback, QoSReliabilityPolicy.BEST_EFFORT)

		self.ring_sub = self.create_subscription(RingInfo, "/ring_info", self.ring_callback, qos_profile_sensor_data)
		self.ring_markers_pub = self.create_publisher(MarkerArray, "/rings_markers", QoSReliabilityPolicy.BEST_EFFORT)
		self.cylinder_sub = self.create_subscription(CylinderInfo, "/cylinder_info", self.cylinder_callback, qos_profile_sensor_data)

		self.arm_pos_pub = self.create_publisher(String, "/arm_command", QoSReliabilityPolicy.BEST_EFFORT)
		self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

		self.teleop_pub = self.create_publisher(Twist, "cmd_vel", 10)

		self.park_srv = self.create_client(Trigger, '/park_cmd')
		self.enable_exploration_srv = self.create_client(Trigger, '/enable_navigation')
		self.disable_exploration_srv = self.create_client(Trigger, '/disable_navigation')
		self.priority_keypoint_pub = self.create_publisher(Waypoint, '/priority_keypoint', QoSReliabilityPolicy.BEST_EFFORT)

		self.anomaly_info_sub = self.create_subscription(AnomalyInfo, "/anomaly_info", self.anomaly_callback, qos_profile_sensor_data)
		self.face_info_sub = self.create_subscription(FaceInfo, "/face_info", self.people_callback, qos_profile_sensor_data)

		self.color_talker_srv = self.create_client(Color, '/say_color')
		self.greet_srv = self.create_client(Trigger, '/say_hello')
		self.read_qr_srv = self.create_client(Trigger, '/read_qr')

		pwd = os.getcwd()
		gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]
		self.costmap = cv2.cvtColor(cv2.imread(f"{gpath}/src/dis_tutorial3/maps/costmap.pgm"), cv2.COLOR_BGR2GRAY)
		self.timer = self.create_timer(0.1, self.on_update)
		self.ring_quality_threshold = 0.3
		self.person_quality_threshold = 0.1 # TODO: determine threshold
		self.cylinder_threshold = 0.1 # TODO: determine threshold

		self.t1 = millis()

		self.ros_occupancy_grid = None
		self.map_np = None
		self.map_data = {"map_load_time":None, "resolution":None, "width":None, "height":None, "origin":None}
		self.occupancy_grid_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_profile)

		self.clock_sub = self.create_subscription(Clock, "/clock", self.clock_callback, qos_profile_sensor_data)
		self.time = rclpy.time.Time() #T0

		self.go_to_person = False
		self.talk_future = None

		self.go_to_mona = False

		self.potential_parking_spots = None

		self.last_person_seen = None
		self.last_mona_seen = None

		self.ring_position = {}
		self.mona_quality_threshold = 0.1
		self.exploration_active = False
		self.parking_ring_position = [0,0]
		self.parking_ring_found = False
		self.cylinder_position = [0,0]
		self.cylinder_found = False
		self.genuine_mona_lisa_position = [0,0]
		self.genuine_mona_lisa_found = False
		self.qr_code_read = False
		self.ring_count = 0	
		self.rings = []
		self.people_count = 0
		self.people = []
		self.monas_count = 0
		self.monas = []
		self.last_cylinder = None
		self.last_cylinder_dist = None

		return


	def clock_callback(self, msg):
		self.time = msg.clock
		return

	def relative_to_global(self, relative, link="oakd_link"):
		time_now = rclpy.time.Time()
		timeout = Duration(seconds=0.5)
		transform = self.tf_buffer.lookup_transform("map", link, time_now, timeout)	

		position_point = PointStamped() #robot global pos
		position_point.header.frame_id = "/map"
		position_point.header.stamp = self.get_clock().now().to_msg()
		position_point.point = array2point(relative)
		pp_global = tfg.do_transform_point(position_point, transform)
		return np.array([pp_global.point.x, pp_global.point.y, pp_global.point.z])

	def rotate_on_point(self, x, y):
		cmd_msg = Twist()
		cmd_msg.linear.x = 0.
		cmd_msg.angular.z = 1.8
		self.teleop_pub.publish(cmd_msg)
	
	def create_nav2_goal_msg(self, x,y, yaw):
		goal_pose = PoseStamped()
		goal_pose.header.frame_id = 'map'
		goal_pose.header.stamp = self.time
		goal_pose.pose.position.x = float(x)
		goal_pose.pose.position.y = float(y)

		quat_tf = quaternion_from_euler(0, 0, yaw)
		quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3]) # fo

		goal_pose.pose.orientation = quat_msg
		return goal_pose

	def go_to_pose(self, x,y, yaw = 0): #nav2
		fixed_x, fixed_y = self.get_valid_close_position(x, y)
		print(f"original: {x}, {y} -new-> {fixed_x}, {fixed_y}")
		
		pose = self.create_nav2_goal_msg(fixed_x,fixed_y, yaw)

		while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
			self.get_logger().info("'NavigateToPose' action server not available, waiting...")

		goal_msg = NavigateToPose.Goal()
		goal_msg.pose = pose
		goal_msg.behavior_tree = ""
		self.send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
		self.send_goal_future.add_done_callback(self.goal_accepted_callback)
		return
	
	def goal_accepted_callback(self, future):
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.get_logger().error('Goal was rejected!')
			return	
		self.result_future = goal_handle.get_result_async()
		self.result_future.add_done_callback(self.get_result_callback)
		return
	
	def get_result_callback(self, future):
		status = future.result().status
		if status != GoalStatus.STATUS_SUCCEEDED:
			print(f'Goal failed with status code: {status}')
		else:
			print(f'Goal reached (according to Nav2).')
		if(self.sm.moving_to_ring_for_parking.is_active):
			self.sm.setup_camera_for_parking()
		elif(self.sm.moving_to_cylinder.is_active):
			self.sm.setup_camera_for_qr()
		elif(self.sm.moving_to_genuine_painting.is_active):
			self.sm.stop()
		elif(self.sm.moving_to_person.is_active):
			self.sm.talk_to_person()
		elif(self.sm.moving_to_cylinder.is_active):
			self.sm.setup_camera_for_qr()
		else:
			raise ValueError("Unknown state")
		return

	def anomaly_callback(self, anomaly):
		# TODO: add monalisa to list and by quality determine if it is true monalisa
		pass	

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

	def on_update(self):
		self.send_ring_markers()

		m_time = millis()
		if(self.sm.init.is_active):
			if((m_time - self.t1) > 3000): #Wait for n_s na zacetku,da se zadeve inicializirajo...
				self.sm.setup_camera_for_exploration()
		elif(self.sm.camera_setup_for_exploration.is_active):
			#TODO: cakas, dokler ni potrjeno, da je kamera na pravem polozaju, ...
			#zaenkrat samo cakamo n_s
			if((m_time - self.t1) > 4000):
				if self.qr_code_read == False:
					self.sm.explore()
				else:
					self.sm.explore_paintings()
		elif(self.sm.exploring.is_active):
			if(self.found_all_people()):
				if((m_time - self.t1) > 2000):
					self.sm.move_to_ring_for_parking()
				else:
					self.disable_exploration()
			elif(self.go_to_person):
				if((m_time - self.t1) > 2000):
					self.sm.move_to_person()
				else:
					self.disable_exploration()
			# else: 
			# 	self.t1 = m_time

		elif(self.sm.camera_setup_for_parking.is_active):
			#TODO: cakas, dokler ni potrjeno, da je kamera na pravem polozaju, ...
			#zaenkrat samo cakamo 3s
			if((m_time - self.t1) > 3000):
				self.sm.park()

		elif(self.sm.camera_setup_for_cylinder.is_active):
			if((m_time - self.t1) > 3000):
				self.sm.find_cylinder()
    
		elif(self.sm.camera_setup_for_qr.is_active):
			if((m_time - self.t1) > 3000):
				self.sm.read_qr()

		elif(self.sm.reading_qr.is_active):
			# TODO: wait until qur code is actually read
			if((m_time - self.t1) > 3000):
				self.sm.setup_camera_for_exploration()

		elif(self.sm.finding_cylinder.is_active):
			self.rotate_on_point(None, None)
			if(self.cylinder_position):
				print("found at least one cylinder")
				if(self.last_cylinder_dist < 2.):
					print("Going to cylinder")
					self.sm.move_to_cylinder()
    
		elif(self.sm.searching_for_paintings.is_active):
			if(self.found_all_mona_lisas()):
				if((m_time - self.t1) > 2000):
					self.sm.move_to_genuine_painting()
				else:
					self.disable_exploration()
			# else: 
			# 	self.t1 = m_time 

			
    

		return

	def found_all_people(self):
		# TODO: implement (either "all people" or enough data to find parking spot)
		
		print(f"Potential Parking Spots: {self.potential_parking_spots}")
		print(f"Ring Positions: {self.ring_position}")

		# enough data:
		if self.potential_parking_spots is not None and len(self.potential_parking_spots) == 1:
			if list(self.potential_parking_spots)[0] in self.ring_position.keys():
				self.parking_ring_position = self.ring_position[list(self.potential_parking_spots)[0]]
				self.parking_ring_found = True
				return True
		return False

	def found_all_mona_lisas(self):
		# TODO: implement (either "all mona lisas" or enough data to find genuine painting)
		pass

	def found_new_ring(self, ring_info):
		self.ring_count += 1

		color_names = ["red", "green", "blue", "black"]
		print(f"Found new ring with color: {color_names[ring_info.color_index]}")
		self.say_color(ring_info.color_index)
		self.ring_position[color_names[ring_info.color_index]] = [ring_info.position[0], ring_info.position[1]]

		return
	
	def found_new_person(self, face_info):
		self.people_count += 1
		print(f"Found new person.")
		self.last_person_seen = face_info
		self.go_to_person = True

	def found_new_mona(self, face_info):
		self.monas_count += 1
		print(f"Found new mona.")
		self.last_mona_seen = face_info
		self.go_to_mona = True
	
	
	def talk_to_person(self):
		print("Talking to person")

		req = Trigger.Request()
		self.talk_future  = self.greet_srv.call_async(req)
		self.talk_future.add_done_callback(self.done_talking)
		

	def done_talking(self, future):
		res = future.result()

		print(f"Done talking: {res}")

		self.sm.explore()

	def send_ring_markers(self):
		ma = MarkerArray()

		for i, r in enumerate(self.rings):
			marker = None
			
			if(r[0].q > self.ring_quality_threshold):
				marker = create_marker_point(r[0].position)
				marker.id = i
			else:
				marker = create_marker_point(r[0].position, [0.8,0.8,0.8])
				marker.id = 100+i
			
			marker.header.frame_id = "/map"
			marker.lifetime = Duration(seconds=.2).to_msg()
			ma.markers.append(marker)
		
		self.ring_markers_pub.publish(ma)	
		self.cleanup_potential_rings()
		return

	def setup_camera_for_ring_detection(self):
		msg = String()
		msg.data = "look_for_rings"
		self.arm_pos_pub.publish(msg)
		return

	def setup_camera_for_parking(self):
		msg = String()
		msg.data = "look_for_parking2"
		self.arm_pos_pub.publish(msg)
		return

	def setup_camera_for_qr(self):
		msg = String()
		msg.data = "look_for_qr"
		self.arm_pos_pub.publish(msg)
		pass

	def start_parking(self):
		req = Trigger.Request()
		self.park_srv.call_async(req)
		return

	# TODO: call this when parking finishes
	def parking_ended(self):
		self.sm.setup_camera_for_cylinder()
		return

	def find_cylinder(self):
		# TODO: implement (rotate in place, run cylinder detection, once correct cylinder is found call self.sm.move_to_cylinder())
		print("finding cylinder")
		self.cylinder_position = None
		self.last_cylinder_dist = None

		pose = self.relative_to_global([0, 0, 0])
		self.rotate_on_point(pose[0], pose[1])

	def start_qr_reading(self):
		self.read_qr_srv.call_async(Trigger.Request())
		self.qr_code_read = True # TODO: call qr_reading_ended when qr reading finishes instead


 	# TODO: call this when qr reading finishes
	def qr_reading_ended(self):
		self.qr_code_read = True
		return

	def enable_exploration(self):
		if(not self.exploration_active):
			req = Trigger.Request()
			self.enable_exploration_srv.call_async(req)
			self.exploration_active = True
		return 

	def disable_exploration(self):
		if(self.exploration_active):
			req = Trigger.Request()
			self.disable_exploration_srv.call_async(req)
			self.exploration_active = False
		return

	def say_color(self, color_index):
		print(f"Say color: {color_index}")
		req = Color.Request()
		req.color = color_index
		self.color_talker_srv.call_async(req)
		return

	#TODO: to se da implementirat dosti lepse in hitrejse, ...
	#TODO: na potencialne tocke bi lahko dali tud nek timeout, po katerm joh brisemo...
	def cleanup_potential_rings(self): 
		for j, fi in enumerate(self.rings):
			if(fi[0].q < self.ring_quality_threshold):
				continue

			for i,ri in enumerate(self.rings):
				if(ri[0].q >= self.ring_quality_threshold):
					continue
				
				dist = np.linalg.norm(np.array(ri[0].position) - np.array(fi[0].position))
				if(dist < 1.0):
					self.rings.remove(ri)
		return

	def add_new_ring(self, ring_info):
		self.rings.append([ring_info, ring_info])
		self.cleanup_potential_rings()
		if(ring_info.q > self.ring_quality_threshold):
			self.found_new_ring(ring_info)
		else: #Pojdi od blizje pogledat. (Priority Keypoint)
			rx, ry = self.get_valid_close_position(ring_info.position[0], ring_info.position[1])
			if(rx == ring_info.position[0] and ry == ring_info.position[1]): #Ce ni blo najdene pametne tocke ki ni v steni...
				return

			self.rotate_on_point(rx, ry)
			
		return

	def merge_ring_with_target(self, target_index, ring):
		result = [self.rings[target_index][0], ring]
		if(ring.q > self.ring_quality_threshold and result[0].q <= self.ring_quality_threshold):
			self.found_new_ring(ring)
		if(ring.q > result[0].q):
			result = [ring, ring]
			# TODO: ne iščemo zelenega ringa ampak tistega ki nam ga povejo
			if(ring.color_index == 1 and ring.q > self.ring_quality_threshold): #Zelen, izboljsas... TODO
				self.parking_ring_position = ring.position
				
		self.rings[target_index] = result
		return
		
	def ring_callback(self, ring_info):
		if(not self.sm.exploring.is_active):
			return

		#XXX Tole je za tesk pariranja pod obroci drugih barv TO je treba ostranit
		#ring_info.color_index = (ring_info.color_index + 1) % 4
		ring_info.position = self.relative_to_global(np.array(ring_info.position_relative), "top_camera_link").tolist()
		min_dist, min_index = argmin(self.rings, dist_normal_fcn, ring_info)
		if(min_dist > 0.6): #TODO, threshold, glede na kvaliteto
			self.add_new_ring(ring_info)	
		else:
			self.merge_ring_with_target(min_index, ring_info)
		return
	
	def cleanup_potential_people(self): 
		for j, fi in enumerate(self.people):
			if(fi[0].quality < self.person_quality_threshold):
				continue

			for i,ri in enumerate(self.people):
				if(ri[0].quality >= self.person_quality_threshold):
					continue
				
				dist = np.linalg.norm(np.array(ri[0].position) - np.array(fi[0].position))
				if(dist < 1.0):
					self.people.remove(ri)
		return
	
	def add_new_person(self, face_info):
		self.people.append([face_info, face_info])
		self.cleanup_potential_people()
		if(face_info.quality > self.person_quality_threshold):
			self.found_new_person(face_info)
			
		return

	def merge_person_with_target(self, target_index, face_info):
		result = [self.people[target_index][0], face_info]
		if(face_info.quality > self.person_quality_threshold and result[0].quality <= self.person_quality_threshold):
			self.found_new_person(face_info)
		if(face_info.quality > result[0].quality):
			result = [face_info, face_info]
				
		self.people[target_index] = result
		return
	
	def cleanup_potential_monas(self): 
		for j, fi in enumerate(self.monas):
			if(fi[0].quality < self.mona_quality_threshold):
				continue

			for i,ri in enumerate(self.monas):
				if(ri[0].quality >= self.mona_quality_threshold):
					continue
				
				dist = np.linalg.norm(np.array(ri[0].position) - np.array(fi[0].position))
				if(dist < 1.0):
					self.monas.remove(ri)
		return
	
	def add_new_mona(self, face_info):
		self.monas.append([face_info, face_info])
		self.cleanup_potential_monas()
		if(face_info.quality > self.mona_quality_threshold):
			self.found_new_mona(face_info)
			
		return

	def merge_mona_with_target(self, target_index, face_info):
		result = [self.monas[target_index][0], face_info]
		if(face_info.quality > self.mona_quality_threshold and result[0].quality <= self.mona_quality_threshold):
			self.found_new_mona(face_info)
		if(face_info.quality > result[0].quality):
			result = [face_info, face_info]
				
		self.monas[target_index] = result
		return
	
	def people_callback(self, face_info : FaceInfo):
		# TODO: only go to non-mona-lisas during exploration and only go to mona-lisas during painting search
		global_position = self.relative_to_global(face_info.position_relative)
		global_normal = (self.relative_to_global(np.array(face_info.position_relative) + np.array(face_info.normal_relative)) - global_position)
		face_info.position = global_position.tolist()
		face_info.yaw_global = math.atan2(global_normal[1], global_normal[0])

		if(not face_info.is_mona and self.sm.exploring.is_active):
			min_dist, min_index = argmin(self.people, dist_normal_fcn, face_info)
			if(min_dist > 0.6): #TODO, threshold, glede na kvaliteto
				self.add_new_person(face_info)	
			else:
				self.merge_person_with_target(min_index, face_info)
			return
		
		if(face_info.is_mona and self.sm.searching_for_paintings.is_active):
			min_dist, min_index = argmin(self.monas, dist_normal_fcn, face_info)
			if(min_dist > 0.6): #TODO, threshold, glede na kvaliteto
				self.add_new_mona(face_info)	
			else:
				self.merge_mona_with_target(min_index, face_info)
			return
		
	def parking_info_callback(self, park_info):
		print(f"Park info: {park_info.colors}")
		if(self.potential_parking_spots is None):
			self.potential_parking_spots = set(park_info.colors)
		else:
			self.potential_parking_spots = self.potential_parking_spots.intersection(set(park_info.colors))

	
	def cylinder_callback(self, cylinder_info):
		if(cylinder_info.quality > self.cylinder_threshold):
			cylinder_info.position = self.relative_to_global(cylinder_info.position_relative).tolist()
			self.cylinder_position = cylinder_info.position
			self.last_cylinder_dist = np.linalg.norm(cylinder_info.position_relative)

	def parking_stopped_callback(self, msg):
		self.parking_ended()

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
		cv2.circle(mask1,(mx,my),10,255,-1)
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

			

class MasterStateMachine(StateMachine):
	init = State(MasterState.INIT, initial=True)
	camera_setup_for_exploration = State(MasterState.CAMERA_SETUP_FOR_EXPLORATION)
	exploring = State(MasterState.EXPLORING)
	moving_to_person = State(MasterState.MOVING_TO_PERSON)
	talking_to_person = State(MasterState.TALKING_TO_PERSON)
	# validating_ring = State(MasterState.VALIDATING_RING)
	# checking_info = State(MasterState.CHECKING_INFO)
	moving_to_ring_for_parking = State(MasterState.MOVING_TO_RING_FOR_PARKING)
	camera_setup_for_parking = State(MasterState.CAMERA_SETUP_FOR_PARKING)
	parking = State(MasterState.PARKING)
	camera_setup_for_cylinder = State(MasterState.CAMERA_SETUP_FOR_CYLINDER)
	finding_cylinder = State(MasterState.FINDING_CYLINDER)
	moving_to_cylinder = State(MasterState.MOVING_TO_CYLINDER)
	camera_setup_for_qr = State(MasterState.CAMERA_SETUP_FOR_QR)
	reading_qr = State(MasterState.READING_QR)
	# displaying_photo_from_qr = State(MasterState.DISPLAYING_PHOTO_FROM_QR)
	# camera_setup_for_paintings = State(MasterState.CAMERA_SETUP_FOR_PAINTINGS)
	searching_for_paintings = State(MasterState.SEARCHING_FOR_PAINTINGS)
	# moving_to_painting = State(MasterState.MOVING_TO_PAINTING)
	# detecting_anomalies = State(MasterState.DETECTING_ANOMALIES)
	moving_to_genuine_painting = State(MasterState.MOVING_TO_GENUINE_PAINTING)
	done = State(MasterState.DONE, final=True)
	
	setup_camera_for_exploration = init.to(camera_setup_for_exploration) | reading_qr.to(camera_setup_for_exploration)
	explore = camera_setup_for_exploration.to(exploring) | talking_to_person.to(exploring) # | validating_ring.to(exploring) | checking_info.to(exploring)
	explore_paintings = camera_setup_for_exploration.to(searching_for_paintings) # | detecting_anomalies.to(searching_for_paintings)
	
	move_to_person = exploring.to(moving_to_person)
	talk_to_person = moving_to_person.to(talking_to_person)
	
	# validate_ring = exploring.to(validating_ring)
	
	# check_info = talking_to_person.to(checking_info) | validating_ring.to(checking_info)
	
	move_to_ring_for_parking = exploring.to(moving_to_ring_for_parking) # checking_info.to(moving_to_ring_for_parking)
	setup_camera_for_parking = moving_to_ring_for_parking.to(camera_setup_for_parking)
	park = camera_setup_for_parking.to(parking)
	setup_camera_for_cylinder = parking.to(camera_setup_for_cylinder)
	find_cylinder = camera_setup_for_cylinder.to(finding_cylinder)
	move_to_cylinder = finding_cylinder.to(moving_to_cylinder)
	setup_camera_for_qr = moving_to_cylinder.to(camera_setup_for_qr)
	read_qr = camera_setup_for_qr.to(reading_qr)
	
	# move_to_painting = searching_for_paintings.to(moving_to_painting)
	# detect_anomalies = moving_to_painting.to(detecting_anomalies)
	
	move_to_genuine_painting = searching_for_paintings.to(moving_to_genuine_painting) # detecting_anomalies.to(moving_to_genuine_painting)
	stop = moving_to_genuine_painting.to(done)
	
	def __init__(self, master_node : MasterNode):
		super(MasterStateMachine, self).__init__()
		self.node = master_node
  
	def on_transition(self, event, state):
		print(f"Event: '{event}', previous state: '{state.id}'")
		self.node.t1 = millis()
		return "on_transition_return"
		
	def on_setup_camera_for_exploration(self):
		self.node.setup_camera_for_ring_detection()

	def on_setup_camera_for_parking(self):
		self.node.setup_camera_for_parking()
  
	def on_setup_camera_for_qr(self):
		self.node.setup_camera_for_qr()

	def on_setup_camera_for_cylinder(self):
		self.node.setup_camera_for_ring_detection()
  
	def on_enter_parking(self):
		self.node.start_parking()
  
	def on_find_cylinder(self):
		self.node.find_cylinder()
  
	def on_move_to_cylinder(self):
		self.node.go_to_pose(self.node.cylinder_position[0], self.node.cylinder_position[1])

	def on_move_to_person(self):
		# waypoint = self.node.relative_to_world_pos(np.array(self.node.last_person_seen.position_relative) + 0.3 * np.array(self.node.last_person_seen.normal_relative))
		self.node.go_to_pose(self.node.last_person_seen.position[0], self.node.last_person_seen.position[1], self.node.last_person_seen.yaw_global)

	def on_exit_talking_to_person(self):
		self.node.go_to_person = False

	def on_talk_to_person(self):
		self.node.talk_to_person()
  
	def on_read_qr(self):
		self.node.start_qr_reading()
  
	def on_explore(self):
		self.node.enable_exploration()
  
	def on_explore_paintings(self):
		self.node.enable_exploration()
  
	def on_exit_exploring(self):
		self.node.disable_exploration()
  
	def on_exit_searching_for_paintings(self):
		self.node.disable_exploration()
  
	def on_move_to_ring_for_parking(self):
		self.node.go_to_pose(self.node.parking_ring_position[0], self.node.parking_ring_position[1])
  
	def on_move_to_genuine_painting(self):
		self.node.go_to_pose(self.node.genuine_mona_lisa_position[0], self.node.genuine_mona_lisa_position[1])

def main():
	rclpy.init(args=None)
	rd_node = MasterNode()
	rclpy.spin(rd_node)
	cv2.destroyAllWindows()
	return

if __name__ == '__main__':
	main()
