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
from task1.msg import PersonInfo, GoalKeypoint

qos_profile = QoSProfile(
		  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		  reliability=QoSReliabilityPolicy.RELIABLE,
		  history=QoSHistoryPolicy.KEEP_LAST,
		  depth=1)

def millis():
    return round(time.time() * 1000)

def array2point(arr):
	p = Point()
	p.x = float(arr[0])
	p.y = float(arr[1])
	p.z = float(arr[2])
	return p

class Face():
	face_id = 0
	tresh_xy = 0.3
	tresh_cos = math.cos(20 * 3.14/180)

	def __init__(self, person_info):
		self.origin = np.array(person_info.origin)
		self.normal = np.array(person_info.normal)

		self.quality = person_info.quality
		self.id = Face.face_id
		Face.face_id += 1

	def compare(self, face): 
		dist1 = np.linalg.norm(self.origin + self.normal * 0.25  - face.origin)
		dist2 = np.linalg.norm(face.origin + face.normal * 0.25  - self.origin)
		dist = min(dist1, dist2)
		
		cosfi = self.normal.dot(face.normal)
		return (dist < 0.65) and (cosfi > -0.25)

class detect_faces(Node):
	def __init__(self):
		super().__init__('people_manager')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', '')
		])
		
		self.faces = []

		self.person_sub = self.create_subscription(PersonInfo, "/people_info", self.person_found_callback, 10)
		self.marker_publisher = self.create_publisher(Marker, '/detected_faces', QoSReliabilityPolicy.BEST_EFFORT)
		self.goal_publisher = self.create_publisher(GoalKeypoint, '/face_keypoints', QoSReliabilityPolicy.BEST_EFFORT)
		
		self.ros_occupancy_grid = None
		self.map_np = None
		self.map_data = {"map_load_time":None, "resolution":None, "width":None, "height":None, "origin":None}
		self.occupancy_grid_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_profile)

		self.simulation = False

		pwd = os.getcwd()
		gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]

		if(self.simulation):
			self.park_search_circle_size = 10
			self.costmap = cv2.cvtColor(cv2.imread(f"{gpath}/src/dis_tutorial3/maps/costmap.pgm"), cv2.COLOR_BGR2GRAY)
		else:	
			self.park_search_circle_size = 7
			self.costmap = cv2.cvtColor(cv2.imread(f"{gpath}/src/dis_tutorial3/maps/non_sim/costmap.pgm"), cv2.COLOR_BGR2GRAY) 

		print(f"OK, simulation: {self.simulation}")
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

	def person_found_callback(self, person_info):
		new_face = Face(person_info)
		if not (np.isfinite(new_face.origin).all() and np.isfinite(new_face.normal).all()):
			return

		notFound = True
		for i in range(len(self.faces)):
			if(self.faces[i].compare(new_face)):  
				notFound = False
				if(new_face.quality > self.faces[i].quality):#merge and send
					new_face.id = self.faces[i].id #Merge
					self.faces[i] = new_face

					vx, vy = self.get_valid_close_position(self.faces[i].origin[0] + (self.faces[i].normal[0] * 0.3), self.faces[i].origin[1] + (self.faces[i].normal[1] * 0.3))

					goal = GoalKeypoint()
					goal.face_id = self.faces[i].id
					goal.position = [ vx, vy, 0. ]
					goal.yaw = math.atan2(self.faces[i].origin[1]-vy, self.faces[i].origin[0]-vx)
					self.goal_publisher.publish(goal)
			
					#Rviz Marker #TODO replace Task2 Sytle
					point = Marker() #Send
					point.type = 2
					point.id = self.faces[i].id
					point.header.frame_id = "/map"
					point.header.stamp = rclpy.time.Time().to_msg()
					
					point.scale.x = 0.15
					point.scale.y = 0.15
					point.scale.z = 0.15

					point.color.r = 0.0
					point.color.g = 1.0
					point.color.b = 0.0
					point.color.a = 1.0

					marker_normal = -self.faces[i].normal
					q = quaternion_from_euler(0, 0, goal.yaw)
					point.pose.orientation.x = q[0]
					point.pose.orientation.y = q[1]
					point.pose.orientation.z = q[2]
					point.pose.orientation.w = q[3]
					point.pose.position = array2point(goal.position)

					self.marker_publisher.publish(point)
				break 
		if(notFound):
			self.faces.append(new_face)
	
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
