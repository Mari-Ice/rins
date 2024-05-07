#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, PoseStamped

from visualization_msgs.msg import Marker, MarkerArray

from tf_transformations import quaternion_from_euler, euler_from_quaternion

from task2.srv import Color
from task2.msg import Waypoint
from std_srvs.srv import Trigger
import cv2
import numpy as np
import math

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

import math
import numpy as np

from nav2_simple_commander.robot_navigator import BasicNavigator

STOP_AFTER_THREE = True

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)
       
class MapGoals(Node):
    def __init__(self):
        super().__init__('map_goals')

        self.waypoints = []
        self.waypoint_markers = MarkerArray()
        self.last_waypoint = [0,0,0]
        self.priority_waypoints = []
        self.waypoint_index = 0
        self.forward = True
        self.future = None
        self.future_color = None
        self.face_count = 0
        self.ring_count = { # 0: red, 1: green, 2: blue, 3: black
            0: 0,
            1: 0,
            2: 0,
            3: 0,
        }

        self.navigator_ready = False

        # Basic ROS stuff
        timer_frequency = 10
        map_topic = "/map"
        timer_period = 1/timer_frequency

        # Functional variables
        self.enable_navigation = True
        self.result_future = None
        self.currently_navigating = False
        self.currently_navigating_priority = False
        self.clicked_x = None
        self.clicked_y = None
        self.ros_occupancy_grid = None
        self.map_np = None
        self.map_data = {"map_load_time":None,
                         "resolution":None,
                         "width":None,
                         "height":None,
                         "origin":None} # origin will be in the format [x,y,theta]
        
        # Subscribe to map, and create an action client for sending goals
        self.occupancy_grid_sub = self.create_subscription(OccupancyGrid, map_topic, self.map_callback, qos_profile)
        # self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.navigator = BasicNavigator()
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)

        self.final_waypoint_sub = self.create_subscription(Waypoint, '/final_waypoint', self.final_waypoint_callback, 10)
        self.priority_waypoint_sub = self.create_subscription(Waypoint, '/priority_waypoint', self.priority_waypoint_callback, 10)
        # Create a timer, to do the main work.
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.waypoint_publisher = self.create_publisher(MarkerArray, '/waypoints', QoSReliabilityPolicy.BEST_EFFORT)
        
        # Enable and disable navigation
        self.enable_navigation_sub = self.create_service(Trigger, '/enable_navigation', self.enable_navigation_callback)
        self.disable_navigation_sub = self.create_service(Trigger, '/disable_navigation', self.disable_navigation_callback)

    def odometry_callback(self, msg : Odometry):
        if self.navigator_ready:
            return
        
        pose = PoseStamped()
        pose.header.frame_id = "/map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = msg.pose.pose.position.x
        pose.pose.position.y = msg.pose.pose.position.y
        pose.pose.position.z = 0.0
        pose.pose.orientation = msg.pose.pose.orientation

        self.navigator.setInitialPose(pose)
        self.navigator.waitUntilNav2Active()
        self.navigator_ready = True

    def enable_navigation_callback(self, _, response):
        if not self.navigator_ready:
            response.success = False
            response.message = "Navigator not ready."
            return response

        if(self.waypoints == []):
            response.success = False
            response.message = "No waypoints found."
            return response

        self.navigator.followWaypoints(self.waypoints)
        response.success = True
        response.message = "Navigation enabled."
        return response
    
    def disable_navigation_callback(self, _, response):
        # TODO: don't stop if the robot is currently navigating to a priority waypoint
        # TODO: only return success if the robot actually stops

        if not self.navigator_ready:
            response.success = False
            response.message = "Navigator not ready."
            return response

        self.navigator.cancelTask()
        response.success = True
        response.message = "Navigation disabled."
        return response

    def map_callback(self, msg):
        self.get_logger().info(f"Read a new Map (Occupancy grid) from the topic.")
        # reshape the message vector back into a map
        self.map_np = np.array(msg.data, dtype=np.uint8).reshape(msg.info.height, msg.info.width)
        # fix the direction of Y (origin at top for OpenCV, origin at bottom for ROS2)
        self.map_np = np.flipud(self.map_np)
        # change the colors so they match with the .pgm image
        self.map_np[self.map_np==0] = 127
        self.map_np[self.map_np==100] = 0
        # load the map parameters
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
        #self.get_logger().info(f"Read a new Map (Occupancy grid) from the topic.")
  
        self.calculate_waypoints()
    
    def final_waypoint_callback(self, waypoint: PoseStamped):
        # # in case we change this into a service uncomment this
        # if not self.navigator_ready:
        #     response.success = False
        #     response.message = "Navigator not ready."
        #     return response

        self.navigator.cancelTask()

        self.navigator.goToPose(waypoint)

        while not self.navigator.isTaskComplete():
            pass

        rclpy.shutdown()
        exit(0)

    def priority_waypoint_callback(self, waypoint: PoseStamped):
        # # in case we change this into a service uncomment this
        # if not self.navigator_ready:
        #     response.success = False
        #     response.message = "Navigator not ready."
        #     return response

        # stop following the waypoints, move to the priority waypoint

        self.navigator.cancelTask()
        self.navigator.goToPose(waypoint)

        # TODO: return to following waypoints after reaching the priority waypoint

    def timer_callback(self): 
        if STOP_AFTER_THREE and self.face_count >= 3:
            self.get_logger().info(f"\n\nDone, i found and greeted with 3 faces.\n\n")
            rclpy.shutdown()
            exit(0)

        if not self.navigator_ready:
            return

        if self.navigator.isTaskComplete():
            self.get_logger().info(f"I visited all waypoints.\n\n")
            self.navigator.followWaypoints(self.waypoints)

    def generate_goal_message(self, x, y, theta=0.2):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation = self.yaw_to_quaternion(theta)
        
        return goal_pose

    def map_pixel_to_world(self, x, y, theta=0):
        ### Convert a pixel in an numpy image, to a real world location
        ### Works only for theta=0
        assert not self.map_data["resolution"] is None

        # Apply resolution, change of origin, and translation
        # 
        world_x = x*self.map_data["resolution"] + self.map_data["origin"][0]
        world_y = (self.map_data["height"]-y)*self.map_data["resolution"] + self.map_data["origin"][1]

        # Apply rotation
        return world_x, world_y

    def world_to_map_pixel(self, world_x, world_y, world_theta=0.2):
        ### Convert a real world location to a pixel in a numpy image
        ### Works only for theta=0
        assert self.map_data["resolution"] is not None

        # Apply resolution, change of origin, and translation
        # x is the first coordinate, which in opencv and numpy that is the matrix row - vertical
        # vertical
        x = int((world_x - self.map_data["origin"][0])/self.map_data["resolution"])
        y = int(self.map_data["height"] - (world_y - self.map_data["origin"][1])/self.map_data["resolution"] )
        
        # Apply rotation
        return x, y

    def yaw_to_quaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3]) # for tf_turtle
        return quat_msg
    
    def get_rotation_matrix(self, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s],
                        [s , c]])
        return rot

    def calculate_waypoints(self):
        # image = self.map_np
        image = cv2.imread("/home/theta/colcon_ws/rins/src/dis_tutorial3/maps/map.pgm")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_image = image.copy()

        # thin the image
        _, dst = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        dst = 255 - dst
        dst	= cv2.ximgproc.thinning(dst)
        # cv2.imshow("preprocessed", dst)

        # threshold image
        _, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

        lines = cv2.HoughLines(dst, 1, 10 * np.pi / 180, 20, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(image, pt1, pt2, (0,0,0), 3, cv2.LINE_AA)

        # erode
        kernel = np.ones((10,10), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)

        # cv2.imshow("rooms", image)

        # find connected components and their centroids
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

        for i in range(1, num_labels):
            cv2.circle(original_image, (int(centroids[i][0]), int(centroids[i][1])), 4, (0, 0, 255), -1)

            # print(f"Centroid {i}: {centroids[i]}")
            world_waypoint = self.map_pixel_to_world(centroids[i][0], centroids[i][1])
        
            pose = PoseStamped()
            pose.header.frame_id = "/map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = world_waypoint[0]
            pose.pose.position.y = world_waypoint[1]
            pose.pose.position.z = 0.0

            self.waypoints.append(pose)

            
            # add marker
            marker = Marker()
            marker.header.frame_id = "/oakd_link"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.id = i

            marker.type = 2

            # Set the scale of the marker
            scale = 0.1
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Set the color
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            # Set the pose of the marker
            marker.pose.position.x = float(world_waypoint[0])
            marker.pose.position.y = float(world_waypoint[1])
            marker.pose.position.z = float(0)

            self.waypoint_markers.markers.append(marker)
        
        self.waypoint_publisher.publish(self.waypoint_markers)
        # cv2.imshow("waypoints", original_image)
        #vcv2.waitKey(0)

def main():
    rclpy.init(args=None)
    node = MapGoals()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
