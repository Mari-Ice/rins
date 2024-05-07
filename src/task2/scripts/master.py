#!/usr/bin/python3

import time
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros

from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import ColorRGBA
from std_msgs.msg import String

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from visualization_msgs.msg import Marker, MarkerArray
from task2.msg import RingInfo

qos_profile = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)


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

		self.green_ring_found = False
		self.ring_count = 0	
		self.rings = []
		self.timer = self.create_timer(0.1, self.update)
		self.ring_quality_threshold = 0.3
		self.state = MasterState.INIT
		self.start_millis = millis()

		print("OK")
		return

	def clock_callback(self, msg):
		self.time = msg.clock

	def update(self):
		self.send_ring_markers()
		
		m_time = millis()
		if(self.state == MasterState.INIT):
			if((m_time - self.start_time) > 5000): #Wait for 5s na zacetku,da se zadeve inicializirajo...
				self.setup_camera_for_ring_detection()
				self.state = MasterState.CAMERA_SETUP_FOR_EXPLORATION
				return
		elif(self.state == MasterState.CAMERA_SETUP_FOR_EXPLORATION):
			#cakas, dokler ni potrjeno, da je kamera na pravem polozaju, ...
			pass
		elif(self.state == MasterState.EXPLORATION):
			#Naceloma tu ustvljas in startas autonomous explorerja vmes se pa premikas do potencialnih obrocev.
			#In potem razisces potencialne obroce, dokler ne najdes kaj pravega
			#Ce najdes vse tri obroce in je obvezno eden izmed njih zelen gres v naslednji state, tj. premik do zelenega obroca
			pass
		elif(self.state == MasterState.MOVING_TO_GREEN):
			#Cakas dokler ne prides do tja.
			#potem premaknes poslejs req, za premik kamere v polozaj za parking...
			pass
		elif(self.state == MasterState.CAMERA_SETUP_FOR_PARKING):
			#cakas, dokler ni potrjeno, da je kamera na pravem polozaju, ...
			#potem posljes req v node za parking...
			pass
		elif(self.state == MasterState.PARKING):
			#cakas, dokler ni potrjeno, da je robot parkiran
			#ko robot enkrat je parkiran, potem ustavis in koncas. TO je to...			
		return

				
	def found_new_ring(self, ring_info):
		self.ring_count += 1
		#Tu lahko nacelom izrecemo barvo,...

		color_names = ["red", "green", "blue", "black"]
		print(f"Found new ring with color: {color_names[ring_info.color_index]}")

		if(ring_info.color_index == 1): #nasli smo zelen ring
			#spremenis objective v get_to_green_ring
			#ko prides do tja premaknes kamero, da gleda na parkplac
			#potem se parkiras.
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
		return
	def setup_camera_for_parking(self):
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
		return

	def merge_ring_with_target(self, target_index, ring):
		result = [self.rings[target_index][0], ring]
		if(ring.q > self.ring_quality_threshold and result[0].q <= self.ring_quality_threshold):
			self.found_new_ring(ring)
		if(ring.q > result[0].q):
			result = [ring, ring]
		self.rings[target_index] = result
		return
		
	def ring_callback(self, ring_info):
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
