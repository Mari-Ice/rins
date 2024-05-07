#!/usr/bin/python3

import time
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros

from enum import Enum
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import ColorRGBA
from std_msgs.msg import String

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose, PoseStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from task2.msg import RingInfo
from std_srvs.srv import Trigger

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
	EXPLORATION = 2
	MOVING_TO_GREEN = 3
	CAMERA_SETUP_FOR_PARKING = 4
	PARKING = 5
	DONE = 6

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

def ring_dist_normal_fcn(ring_elt, new_ring):
	return np.linalg.norm(np.array(ring_elt[0].position) - np.array(new_ring.position))

def millis():
    return round(time.time() * 1000)	

class MasterNode(Node):
	def __init__(self):
		super().__init__('master_node')

		self.clock_sub = self.create_subscription(Clock, "/clock", self.clock_callback, qos_profile_sensor_data)
		self.time = rclpy.time.Time() #T0
		
		self.ring_sub = self.create_subscription(RingInfo, "/ring_info", self.ring_callback, qos_profile_sensor_data)
		self.ring_markers_pub = self.create_publisher(MarkerArray, "/rings_markers", QoSReliabilityPolicy.BEST_EFFORT)

		self.arm_pos_pub = self.create_publisher(String, "/arm_command", QoSReliabilityPolicy.BEST_EFFORT)
		self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

		self.park_srv = self.create_client(Trigger, '/park_cmd')
		self.enable_exploration_srv = self.create_client(Trigger, '/enable_navigation')
		self.disable_exploration_srv = self.create_client(Trigger, '/disable_navigation')

		self.exploration_active = False
		self.green_ring_position = [0,0]
		self.green_ring_found = False
		self.ring_count = 0	
		self.rings = []
		self.timer = self.create_timer(0.1, self.on_update)
		self.ring_quality_threshold = 0.3
		self.t1 = millis()
		self.state = MasterState.INIT
		self.change_state(MasterState.INIT)

		print("OK")
		return

	def clock_callback(self, msg):
		self.time = msg.clock

	def create_nav2_goal_msg(self, x,y):
		goal_pose = PoseStamped()
		goal_pose.header.frame_id = 'map'
		goal_pose.header.stamp = self.time
		goal_pose.pose.position.x = float(x)
		goal_pose.pose.position.y = float(y)
		goal_pose.pose.orientation = Quaternion(x=0.,y=0.,z=0.,w=1.)
		return goal_pose

	def go_to_pose(self, x,y): #nav2
		pose = self.create_nav2_goal_msg(x,y)

		while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
			self.get_logger().info("'NavigateToPose' action server not available, waiting...")

		goal_msg = NavigateToPose.Goal()
		goal_msg.pose = pose
		goal_msg.behavior_tree = ""
		self.send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
		self.send_goal_future.add_done_callback(self.goal_accepted_callback)
	
	def goal_accepted_callback(self, future):
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.get_logger().error('Goal was rejected!')
			return	
		self.result_future = goal_handle.get_result_async()
		self.result_future.add_done_callback(self.get_result_callback)
	
	def get_result_callback(self, future):
		status = future.result().status
		if status != GoalStatus.STATUS_SUCCEEDED:
			print(f'Goal failed with status code: {status}')
		else:
			print(f'Goal reached (according to Nav2).')
		self.change_state(MasterState.CAMERA_SETUP_FOR_PARKING)

	def change_state(self, state):
		old_state = self.state
		self.state = state
		print(f"state -> {MasterState(state).name}")

		self.on_state_change(old_state, state)
	
	def on_state_change(self, old_state, new_state):
		m_time = millis()
		if(old_state == MasterState.MOVING_TO_GREEN):
			self.setup_camera_for_parking()
			self.t1 = m_time
		if(new_state == MasterState.PARKING):
			self.start_parking()
		if(new_state == MasterState.EXPLORATION):
			self.enable_exploration()

	def on_update(self):
		self.send_ring_markers()
		
		m_time = millis()
		if(self.state == MasterState.INIT):
			if((m_time - self.t1) > 5000): #Wait for 5s na zacetku,da se zadeve inicializirajo...
				self.setup_camera_for_ring_detection()
				self.change_state(MasterState.CAMERA_SETUP_FOR_EXPLORATION)
				self.t1 = m_time
		elif(self.state == MasterState.CAMERA_SETUP_FOR_EXPLORATION):
			#TODO: cakas, dokler ni potrjeno, da je kamera na pravem polozaju, ...
			#zaenkrat samo cakamo 3s
			if((m_time - self.t1) > 3000):
				self.change_state(MasterState.EXPLORATION)
				self.t1 = m_time
		elif(self.state == MasterState.EXPLORATION):
			if(self.green_ring_found):
				if((m_time - self.t1) > 2000):
					self.go_to_pose(self.green_ring_position[0], self.green_ring_position[1]) 
					self.change_state(MasterState.MOVING_TO_GREEN)
				else:
					self.disable_exploration()
			else:
				self.t1 = m_time


		elif(self.state == MasterState.CAMERA_SETUP_FOR_PARKING):
			#TODO: cakas, dokler ni potrjeno, da je kamera na pravem polozaju, ...
			#zaenkrat samo cakamo 3s
			if((m_time - self.t1) > 3000):
				self.change_state(MasterState.PARKING)
				self.t1 = m_time
		return

	def found_new_ring(self, ring_info):
		self.ring_count += 1
		#Tu lahko nacelom izrecemo barvo,...

		color_names = ["red", "green", "blue", "black"]
		print(f"Found new ring with color: {color_names[ring_info.color_index]}")

		if(ring_info.color_index == 1): #nasli smo zelen ring
			self.green_ring_found = True
			self.green_ring_position = [ring_info.position[0], ring_info.position[1]]
		return

	def send_ring_markers(self):
		ma = MarkerArray()

		for i, r in enumerate(self.rings):
			marker = None
			
			if(r[0].q > self.ring_quality_threshold):
				marker = create_marker_point(r[0].position, r[0].color)
				marker.id = i
			else:
				marker = create_marker_point(r[0].position, [0.8,0.8,0.8])
				marker.id = 100+i
			
			marker.header.frame_id = "/map"
			marker.lifetime = Duration(seconds=.2).to_msg()
			ma.markers.append(marker)
		
		self.ring_markers_pub.publish(ma)	
		self.cleanup_potential_rings()

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

	def start_parking(self):
		req = Trigger.Request()
		self.park_srv.call_async(req)
		return

	def enable_exploration(self):
		if(not self.exploration_active):
			req = Trigger.Request()
			self.enable_exploration_srv.call_async(req)
			self.exploration_active = True
	def disable_exploration(self):
		if(self.exploration_active):
			req = Trigger.Request()
			self.disable_exploration_srv.call_async(req)
			self.exploration_active = False

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
		return

	def merge_ring_with_target(self, target_index, ring):
		result = [self.rings[target_index][0], ring]
		if(ring.q > self.ring_quality_threshold and result[0].q <= self.ring_quality_threshold):
			self.found_new_ring(ring)
		if(ring.q > result[0].q):
			result = [ring, ring]
			if(ring.color_index == 1): #Zelen, izboljsas... TODO
				self.green_ring_position = ring.position
				
		self.rings[target_index] = result
		return
		
	def ring_callback(self, ring_info):
		if(self.state != MasterState.EXPLORATION):
			return

		min_dist, min_index = argmin(self.rings, ring_dist_normal_fcn, ring_info)
		if(min_dist > 0.5): #TODO, threshold, glede na kvaliteto
			self.add_new_ring(ring_info)	
		else:
			self.merge_ring_with_target(min_index, ring_info)
		return

def main():
	rclpy.init(args=None)
	rd_node = MasterNode()
	rclpy.spin(rd_node)
	cv2.destroyAllWindows()
	return

if __name__ == '__main__':
	main()
