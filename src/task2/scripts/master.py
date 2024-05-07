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

		self.ring_sub = self.create_subscription(RingInfo, "/rings_info", self.rings_callback, qos_profile_sensor_data)
		self.ring_marker_pub = self.create_publisher(Marker, "/rings_marker", QoSReliabilityPolicy.BEST_EFFORT)
		self.cnt = 0

		self.arm_pos_pub = self.create_publisher(String, "/arm_command", QoSReliabilityPolicy.BEST_EFFORT)
		print("OK")

	def rings_callback(self, rings_info):
		marker = create_marker_point(rings_info.position, rings_info.color)
		marker.id = self.cnt
		marker.header.frame_id = "/top_camera_link"
		marker.header.stamp = rclpy.time.Time().to_msg()

		self.ring_marker_pub.publish(marker)	
		
		self.cnt = (self.cnt + 1) % 100


def main():
	rclpy.init(args=None)
	rd_node = MasterNode()
	rclpy.spin(rd_node)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
