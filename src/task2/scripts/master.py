#!/usr/bin/python3

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

"""
	Vsi nodi publish-ajo svoje podatke, skupaj z neko kvaliteto podatkov

	Ko vidis ring z dovolj veliko kvaliteto, ki ga se nisi videl (grouping bz position) reces njegovo barvo
	Ko vidis vsaj tri obroce in si ze videl zelenega gres do njega da se parkiras, ce zeleniega se nisi videl se voziz okoli in ga isces
	Potem gres do polozaja kjer je zelen in prizges parking node.	

	Za vsako ko dobis ring msg, pogledas najblizje kateremu ze znanemu si, slabsa kot je kvaliteta, daljso razdaljo dovolis.
	Ce je kvaliteta slaba: 
		* Ce ni nobenega blizu, ga dodas v seznam potencialnih obrocov.
		* Ce je kateri blizu, ga dodas pod njega
	Ce je kvaliteta dobra:
		* Ce je kateri ze zelo blizu, ga dodas pod njega
		* Ce je kateri srednje dalec, ustvaris not najden obroc
		* V vsakem primeru, odstranis potencialne obroce v blizini.

	Torej za vsak detected in potencialen ring hranis:
		best_quality_msg, last_received_msg
		best_quality se uporablja za poslijanje pozicije v rviz2, 
		last_received_msg se uporablja, za clustering, ker medtem ko se rebot premika, je lahko njegova pozicija nestabilna,
		zato rajsi uporabljamo clustering relativno na zadnji dober msg in na best_quality msg.

	Torej ko dobis msg, poisces najblizjo razdaljo med vse best_msgi in med vsemi last_msgi za vse najdene obroce in za vse potencialne obroce.

"""

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

class MasterNode(Node):
	def __init__(self):
		super().__init__('master_node')

		self.clock_sub = self.create_subscription(Clock, "/clock", self.clock_callback, qos_profile_sensor_data)
		self.time = rclpy.time.Time()
		
		self.ring_sub = self.create_subscription(RingInfo, "/ring_info", self.ring_callback, qos_profile_sensor_data)
		self.ring_markers_pub = self.create_publisher(MarkerArray, "/rings_markers", QoSReliabilityPolicy.BEST_EFFORT)

		self.arm_pos_pub = self.create_publisher(String, "/arm_command", QoSReliabilityPolicy.BEST_EFFORT)

		self.found_rings = []
		self.potential_rings = []
		self.timer = self.create_timer(0.1, self.send_ring_markers)

		print("OK")
		return

	def clock_callback(self, msg):
		self.time = msg.clock

	def send_ring_markers(self):
		ma = MarkerArray()

		for i, r in enumerate(self.found_rings):
			marker = create_marker_point(r[0].position, r[0].color)
			marker.id = i
			marker.header.frame_id = "/map"
			marker.lifetime = Duration(seconds=.2).to_msg()
			ma.markers.append(marker)
		for i, r in enumerate(self.potential_rings):
			marker = create_marker_point(r[0].position, [0.3,0.3,0.3])
			marker.id = 100+i
			marker.header.frame_id = "/map"
			marker.lifetime = Duration(seconds=.2).to_msg()
			marker.header.stamp = self.time
			ma.markers.append(marker)

		self.ring_markers_pub.publish(ma)	
		#print(f"published., num  of potenital: {len(self.potential_rings)}")

	def cleanup_potential_rings(self):
		for j, fi in enumerate(self.found_rings):
			for i,ri in enumerate(self.potential_rings):
				dist = np.linalg.norm(np.array(ri[0].position) - np.array(fi[0].position))
				if(dist < 2.):
					self.potential_rings.remove(ri)	
		

	def add_new_ring(self, ring_info):
		if(ring_info.q > 0.5):
			self.found_rings.append([ring_info, ring_info])
			self.cleanup_potential_rings()
		else:
			self.potential_rings.append([ring_info, ring_info])
	def merge_ring_with_target(self, target, ring):
		result = [target[0], ring]
		if(ring.q > target[0].q):
			result = [ring, ring]
			self.cleanup_potential_rings()
		return result
		
	def ring_callback(self, ring_info):
		#okej, najprej najdeno najmanjso razdaljo do kaksnega obroca.	

		min_dist_to_found, min_found_index 		   = argmin(self.found_rings, ring_dist_normal_fcn, ring_info)
		min_dist_to_potential, min_potential_index = argmin(self.found_rings, ring_dist_normal_fcn, ring_info)

		min_dist = min(min_dist_to_potential, min_dist_to_found)	
		if(min_dist > 0.4): #TODO, threshold, glede na kvaliteto
			self.add_new_ring(ring_info)	
		else:
			if(min_dist_to_potential < min_dist_to_found):
				self.potential_rings[min_potential_index] = self.merge_ring_with_target(self.potential_rings[min_potential_index], ring_info)
			else:
				self.found_rings[min_found_index] = self.merge_ring_with_target(self.found_rings[min_found_index], ring_info)
		return


def main():
	rclpy.init(args=None)
	rd_node = MasterNode()
	rclpy.spin(rd_node)
	cv2.destroyAllWindows()
	return


if __name__ == '__main__':
	main()
