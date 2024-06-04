#!/usr/bin/env python3

import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, Point
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.duration import Duration
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import message_filters
from ultralytics import YOLO
from task3.msg import FaceInfo

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
def vec_normalize(vec):
	norm = np.linalg.norm(vec)
	if(norm < 0.01):
		return vec
	return vec / norm

def array2point(arr):
	p = Point()
	p.x = float(arr[0])
	p.y = float(arr[1])
	p.z = float(arr[2])
	return p

def create_face_marker(position, header):
	marker = Marker()
	marker.header = header
	
	marker.type = 2
	
	scale = 0.1
	marker.scale.x = scale
	marker.scale.y = scale
	marker.scale.z = scale
	
	marker.color.r = 1.0
	marker.color.g = 1.0
	marker.color.b = 1.0
	marker.color.a = 1.0
	marker.pose.position = array2point(position)
	
	return marker

class face_detection(Node):
	def __init__(self):
		super().__init__('face_detection')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])
		
		self.device = self.get_parameter('device').get_parameter_value().string_value
		self.bridge = CvBridge()

		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.point_cloud_sub = message_filters.Subscriber(self, PointCloud2, "/oakd/rgb/preview/depth/points")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_image_sub, self.point_cloud_sub], 20, 0.03, allow_headerless=False) 
		self.ts.registerCallback(self.sensors_callback)

		self.marker_pub = self.create_publisher(Marker, "/face_position", QoSReliabilityPolicy.BEST_EFFORT)
		self.data_pub   = self.create_publisher(FaceInfo, "/face_info", QoSReliabilityPolicy.BEST_EFFORT)

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.model = YOLO("yolov8n.pt")
		self.received_any_data = False
		print("Init")
		return

	def sensors_callback(self, rgb_data, pc_data):
		if(not self.received_any_data):
			self.received_any_data = True
			print("Data\nOK")

		img = None
		try:
			img = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
		except CvBridgeError as e:
			print(e)
			return

		faces = []
		res = self.model.predict(img, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)
		for x in res:
			for box in x.boxes.xyxy:
				box = [ int(min(box[0],box[2])), int(min(box[1], box[3])), int(max(box[0],box[2])), int(max(box[1], box[3]))]
				if((box[2]-box[0]) < 10 or (box[3]-box[1]) < 10): #Slika obraza mora bit vsaj 10x10 pixlov.
					continue
				faces.append(box)

		height	   = pc_data.height
		width	   = pc_data.width
		pc_xyz = pc2.read_points_numpy(pc_data, field_names= ("x", "y", "z"))
		pc_xyz = pc_xyz.reshape((height,width,3))

		height_mask = np.zeros((height, width), dtype=np.uint8)
		height_mask[(pc_xyz[:,:,2] > -0.145) & (pc_xyz[:,:,2] < 0.088)] = 255

		marker_header = Header()
		marker_header.frame_id = "/oakd_link"
		marker_header.stamp = rgb_data.header.stamp

		for i, face in enumerate(faces):
			mask = np.zeros((height, width), dtype=np.uint8)
			mask = cv2.rectangle(mask, (face[0], face[1]), (face[2], face[3]), 255, -1) 
			mask[height_mask != 255] = 0

			contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
			if(len(contours) != 1):
				continue
		
			x,y,w,h = cv2.boundingRect(contours[0])
			p1 = pc_xyz[y,x]
			p2 = pc_xyz[y,x+w-1]
			p3 = pc_xyz[y+h-1,x]
			p4 = pc_xyz[y+h-1,x+w-1]

			v1 = vec_normalize(p2 - p1)
			v2 = vec_normalize(p3 - p1)

			#p_center = np.mean(pc_xyz[mask != 0], axis=0)
			p_center2 = 0.5*(p2 + p3)
			p_center3 = 0.5*(p1 + p4)
			p_center = p_center2

			if(np.linalg.norm(p_center2 - p_center) > 0.1 or np.linalg.norm(p_center3 - p_center) > 0.1):
				continue
		
			normal = np.cross(v1,v2)
			mag = np.linalg.norm(normal)
			if(abs(mag) < 0.0001):
				continue
			normal /= mag

			img = cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (0,0,255), 2)

			marker = create_face_marker(p_center, marker_header)
			marker.id = 10 * i + 0
			marker.lifetime = Duration(seconds=.2).to_msg()
			self.marker_pub.publish(marker)

			#FaceInfo
			fi = math.atan2(p_center[1], p_center[0])
			q_dist = math.exp(-np.linalg.norm(p_center))
			q_angle = 1.0 - clamp(abs(fi)/(math.pi), 0, 1)
			q_edges = 1.0 if (face[0] > 1 and face[2] < width-2) else 0.8
			
			finfo = FaceInfo()
			finfo.img_bounds_xyxy = face
			finfo.width  = (face[2] - face[0])
			finfo.height = (face[3] - face[1])
			finfo.position_relative = p_center.tolist()
			finfo.normal_relative = normal.tolist()
			finfo.yaw_relative = fi
			finfo.quality = q_dist * q_angle * q_edges
			
			self.data_pub.publish(finfo)

		#cv2.imshow("Image", img)
		#key = cv2.waitKey(1)
		return	

def main():
	rclpy.init(args=None)
	node = face_detection()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
