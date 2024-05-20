#!/usr/bin/env python3

import random
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
from geometry_msgs.msg import Point
from collections import deque
import time
import os
from nav_msgs.msg import OccupancyGrid

# import tf_transformations
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import quaternion_from_euler, euler_from_quaternion

qos_profile = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

def millis():
    return round(time.time() * 1000)
def mag(vec):
	return np.sqrt(vec.dot(vec))

class Face():
	face_id = 0
	tresh_xy = 0.3
	tresh_cos = math.cos(20 * 3.14/180)

	def __init__(self, marker):
		self.origin = np.array([
			marker.points[0].x,
			marker.points[0].y,
			marker.points[0].z
		])
		self.endpoint = np.array([
			marker.points[1].x,
			marker.points[1].y,
			marker.points[1].z
		])
		self.normal = self.endpoint - self.origin

		self.id = Face.face_id
		Face.face_id += 1

		self.num = 1
		self.num_tresh = 3
		self.visited = False

	def compare(self, face): #mogoce bi blo lazje primerjat ze izracunane keypointe
		kp1 = self.origin + 0.25 * self.normal
		kp2 = face.origin + 0.25 * face.normal
		return mag(kp1-kp2) < 0.5


class detect_faces(Node):
	last_n_faces = []
	last_marker_time = 0

	rolling_origin = np.array([0, 0, 0])

	def __init__(self):
		super().__init__('people_manager')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])
		
		self.faces = []

		marker_topic = "/people_marker"

		self.marker = self.create_subscription(Marker, marker_topic, self.marker_callback, 10)
		self.publisher = self.create_publisher(Marker, '/detected_faces', QoSReliabilityPolicy.BEST_EFFORT)
		
		self.ros_occupancy_grid = None
		self.map_np = None
		self.map_data = {"map_load_time":None, "resolution":None, "width":None, "height":None, "origin":None}
		self.occupancy_grid_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_profile)

		pwd = os.getcwd()
		gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]
		self.costmap = cv2.cvtColor(cv2.imread(f"{gpath}/src/dis_tutorial3/maps/non_sim/costmap.pgm"), cv2.COLOR_BGR2GRAY) 
		#self.costmap = cv2.cvtColor(cv2.imread(f"{gpath}/src/dis_tutorial3/maps/costmap.pgm"), cv2.COLOR_BGR2GRAY) #For simulation

		print("OK")
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

	def get_valid_close_position(self, ox,oy):
		mx,my = self.world_to_map_pixel(ox,oy)

		height, width = self.costmap.shape[:2]
		pixel_locations = np.full((height, width, 2), 0, dtype=np.uint16)
		for y in range(0, height):
			for x in range(0, width):
				pixel_locations[y][x] = [x,y]

		mask1 = np.full((height, width), 0, dtype=np.uint8)
		cv2.circle(mask1,(mx,my),6,255,-1) #TODO, odvisno od resolucije
		mask1[self.costmap < 200] = 0
		pts = pixel_locations[mask1==255]
		if(len(pts) > 0):
			center = random.choice(pts)
			return self.map_pixel_to_world(center[0], center[1])
		else:
			return [ox,oy]

	def marker_callback(self, marker):
		new_face = Face(marker)
		if not (np.isfinite(new_face.origin).all() and np.isfinite(new_face.normal).all()):
			return

		notFound = True
		for face in self.faces:
			if(face.compare(new_face)):  
				face.num += 1
				notFound = False
				if(not face.visited):
					if(face.num > face.num_tresh):
						point = Marker()
						point.type = 2
						point.id = face.id
						point.header.frame_id = "/map"
						point.header.stamp = marker.header.stamp
						
						point.scale.x = 0.15
						point.scale.y = 0.15
						point.scale.z = 0.15

						point.color.r = 0.0
						point.color.g = 1.0
						point.color.b = 0.0
						point.color.a = 1.0

						vx, vy = self.get_valid_close_position(face.origin[0] + (face.normal[0] * 0.5), face.origin[1] + (face.normal[1] * 0.5))
						point.pose.position.x = vx
						point.pose.position.y = vy
						point.pose.position.z = face.origin[2] + (face.normal[2] * 0.5)

						marker_normal = -face.normal
						#q = quaternion_from_euler(0, 0, math.atan2(marker_normal[1], marker_normal[0]))
						q = quaternion_from_euler(0, 0, math.atan2(face.origin[1]-vy, face.origin[0]-vx))
						point.pose.orientation.x = q[0]
						point.pose.orientation.y = q[1]
						point.pose.orientation.z = q[2]
						point.pose.orientation.w = q[3]

						self.publisher.publish(point)
						face.visited = True
				break 
		if(notFound):
			self.faces.append(new_face)
	
		self.get_logger().info(f"Got a marker {marker.points[0]} {marker.points[1]}")
		self.get_logger().info(f"FACES: {len(self.faces)}")
		print()

def main():
	print('People manager node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
