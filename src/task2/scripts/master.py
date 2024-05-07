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

class MasterNode(Node):
	def __init__(self):
		super().__init__('master_node')

		self.ring_sub = self.create_subscription(RingInfo, "/ring_info", self.ring_callback, qos_profile_sensor_data)
		self.ring_marker_pub = self.create_publisher(Marker, "/rings_marker", QoSReliabilityPolicy.BEST_EFFORT)

		self.arm_pos_pub = self.create_publisher(String, "/arm_command", QoSReliabilityPolicy.BEST_EFFORT)

		self.found_rings = []
		self.potential_rings = []

		print("OK")
		return

	def add_new_ring(self, ring_info):
		if(ring_info.q > 0.5):
			self.found_rings.append([ring_info, ring_info])

			#Zdej pa zbrisemo vse potencialne v blizini
			for i,ri in enumerate(self.potential_rings):
				dist = np.linalg.norm(np.array(ri[0].position) - np.array(ring_info.position))
				if(dist < 0.4):
					self.potential_rings.remove(ri)	
		else:
			self.potential_rings.append([ring_info, ring_info])
	def merge_ring_with_target(self, target, ring):
		if(ring.q > target[0].q):
			return [ring, ring]
		return [target[0], ring]
		

	def ring_callback(self, ring_info):
		#okej, najprej najdeno najmanjso razdaljo do kaksnega obroca.	

		min_dist_to_found = 1001
		min_found_index = -1

		min_dist_to_potential = 1000
		min_potential_index = -1
		
		for i,fr in enumerate(self.found_rings):
			fr_best, fr_last = fr
			
			dist = np.linalg.norm(np.array(fr_best.position) - np.array(ring_info.position))
			if(dist < min_dist_to_found):
				min_dist_to_found = dist
				min_found_index = i
			
			dist = np.linalg.norm(np.array(fr_last.position) - np.array(ring_info.position))
			if(dist < min_dist_to_found):
				min_dist_to_found = dist
				min_found_index = i

		for i,fr in enumerate(self.potential_rings):
			fr_best, fr_last = fr
			
			dist = np.linalg.norm(np.array(fr_best.position) - np.array(ring_info.position))
			if(dist < min_dist_to_potential):
				min_dist_to_potential = dist
				min_potential_index = i
			
			dist = np.linalg.norm(np.array(fr_last.position) - np.array(ring_info.position))
			if(dist < min_dist_to_potential):
				min_dist_to_potential = dist
				min_potential_index = i

		min_dist = min(min_dist_to_potential, min_dist_to_found)	
		print(f"min_dist: {min_dist}")
		print(f"fund_rings: {len(self.found_rings)}")

		if(min_dist > 0.4): #TODO, threshold, glede na kvaliteto
			self.add_new_ring(ring_info)	
		else:
			if(min_dist_to_potential < min_dist_to_found):
				self.potential_rings[min_potential_index] = self.merge_ring_with_target(self.potential_rings[min_potential_index], ring_info)
			else:
				self.found_rings[min_found_index] = self.merge_ring_with_target(self.found_rings[min_found_index], ring_info)
			
		#Na koncu posljes posodobljene markerje najdenih ringov.
		for i, r in enumerate(self.found_rings):
			marker = create_marker_point(r[0].position, r[0].color)
			marker.id = i
			marker.header.frame_id = "/map"
			marker.header.stamp = rclpy.time.Time().to_msg()
			self.ring_marker_pub.publish(marker)	
		return


def main():
	rclpy.init(args=None)
	rd_node = MasterNode()
	rclpy.spin(rd_node)
	cv2.destroyAllWindows()
	return


if __name__ == '__main__':
	main()
