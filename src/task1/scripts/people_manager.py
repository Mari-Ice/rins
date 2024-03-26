#!/usr/bin/env python3

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

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult



class Face():
    def __init__(self, marker):
        
        self.origin = np.array([
            marker.points[0].x,
            marker.points[0].y,
            marker.points[0].z
        ])
        self.normal = np.array([
            marker.points[1].x - marker.points[0].x,
            marker.points[1].y - marker.points[0].y,
            marker.points[1].z - marker.points[0].z
        ])

        self.tresh = 0.2

        self.num = 1
        self.num_tresh = 100
        self.visited = False

    def compare(self, face):
        return np.linalg.norm(self.origin - face.origin) < self.tresh and np.dot(self.normal, face.normal) > 0.8


class detect_faces(Node):

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
        self.publisher = self.create_publisher(Point, '/detected_faces', QoSReliabilityPolicy.BEST_EFFORT)
        
        self.get_logger().info(f"Node has been initialized! Reading from {marker_topic}.")



    def marker_callback(self, marker):
        new_face = Face(marker)
        notFound = True
        for face in self.faces:
            if(face.compare(new_face)):
                face.origin = 0.9 * face.origin + 0.1 * new_face.origin
                face.normal = 0.9 * face.normal + 0.1 * new_face.normal
                face.num += 1
                notFound = False
                if(not face.visited):
                    if(face.num > face.num_tresh):
                        point = Point()
                        point.x = face.origin[0] + face.normal[0]
                        point.y = face.origin[1] + face.normal[1]
                        point.z = face.origin[2] + face.normal[2]
                        self.publisher.publish(point)
                        face.visited = True
                break
        if(notFound):
            self.faces.append(new_face)
        
        self.get_logger().info(f"Got a marker {marker.pose.position.x} {marker.pose.position.y} {marker.pose.position.z}")
        self.get_logger().info(f"FACES: {len(self.faces)}")




def main():
    print('People manager node starting.')

    rclpy.init(args=None)
    node = detect_faces()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
