#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 20
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1
        
        # Initialize the image variables
        self.color = None
        self.depth = None

        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        
        # Start the timer that will trigger the callback
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Publiser for the visualization markers
        self.marker_pub = self.create_publisher(MarkerArray, "/ring", QoSReliabilityPolicy.BEST_EFFORT)

        # Object we use for transforming between coordinate frames
        # self.tf_buf = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        #cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)

        cv2.createTrackbar("ThreshA", "Detected rings" , 5, 100, self.on_trackbar1)
        cv2.createTrackbar("ThreshB", "Detected rings" , 5, 100, self.on_trackbar2)
        cv2.createTrackbar("Kernel_size", "Detected rings" , 1, 31, self.on_trackbar3)
        cv2.createTrackbar("Kernel2_size", "Detected rings" , 1, 31, self.on_trackbar4)
        cv2.createTrackbar("min radius", "Detected rings" , 1, 31, self.on_trackbar5)

        self.threshA = 5
        self.threshB = 5
        self.kernel_size = 1
        self.kernel2_size = 5
        self.min_radius = 6

    def on_trackbar1(self, val):
        if(val % 2 == 0):
            val += 1
        self.threshA = max(3,val)

    def on_trackbar2(self, val):
        self.threshB = max(1,val)

    def on_trackbar3(self, val):
        if(val % 2 == 0):
            val += 1
        self.kernel_size = max(1,val)

    def on_trackbar4(self, val):
        if(val % 2 == 0):
            val += 1
        self.kernel2_size = max(1,val)

    def on_trackbar5(self, val):
        self.min_radius = max(1,val)

    def detect_ring(self, cv_image, isDepth):
        gray = cv_image

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size,self.kernel_size))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel2_size,self.kernel2_size))
        #gray = cv2.erode(gray, kernel2)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel2)

        cv2.imshow(f"gray Image {isDepth}", gray)
        
        
        # Apply Gaussian Blur
        # gray = cv2.GaussianBlur(gray,(3,3),0)

        # Do histogram equalization
        # gray = cv2.equalizeHist(gray)

        # Binarize the image, there are different ways to do it
        #ret, thresh = cv2.threshold(img, 50, 255, 0)
        #ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.threshA, self.threshB)
        cv2.imshow(f"Binary Image {isDepth}", thresh)
        cv2.waitKey(1)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Example of how to draw the contours, only for visualization purposes
        cv2.drawContours(gray, contours, -1, (120, 0, 0), 3)
        cv2.imshow(f"Detected contours {isDepth}", gray)
        #cv2.waitKey(1)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 5:
                ellipse = cv2.fitEllipse(cnt)
                if(ellipse[1][1] > self.min_radius):
                    elps.append(ellipse)


        # Find two elipses with same centers
        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                # e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees
                
                e1 = elps[n]
                e2 = elps[m]
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                angle_diff = np.abs(e1[2] - e2[2])

                # The centers of the two elipses should be within 5 pixels of each other (is there a better treshold?)
                if dist >= 25:
                    continue

                # The rotation of the elipses should be whitin 4 degrees of eachother
                # if angle_diff>4:
                #     continue

                e1_minor_axis = e1[1][0]
                e1_major_axis = e1[1][1]

                e2_minor_axis = e2[1][0]
                e2_major_axis = e2[1][1]

                if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
                    le = e1 # e1 is larger ellipse
                    se = e2 # e2 is smaller ellipse
                elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
                    le = e2 # e2 is larger ellipse
                    se = e1 # e1 is smaller ellipse
                else:
                    continue # if one ellipse does not contain the other, it is not a ring
                
                # The widths of the ring along the major and minor axis should be roughly the same
                border_major = (le[1][1]-se[1][1])/2
                border_minor = (le[1][0]-se[1][0])/2
                border_diff = np.abs(border_major - border_minor)

                if border_diff>5:
                    continue
                    
                candidates.append((e1,e2))

        print("Processing is done! found", len(candidates), "candidates for rings")

        return candidates
    
    def timer_callback(self):
        # If we have both an image and a depth image, we can start processing
        if self.color is None or self.depth is None:
            self.get_logger().info("Waiting for both images...")
            return
        
        # Do the ring detection on both images and check if any of the candidates match
        candidates_color = self.detect_ring(cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY), False)
        candidates_depth = self.detect_ring(255 - self.depth, True)
        
        candidates = []
        
        # If we have candidates in both images, we can start comparing them
        if len(candidates_color) > 0 and len(candidates_depth) > 0:
            for c in candidates_color:
                for c_d in candidates_depth:
                    # the centers of the ellipses
                    e1 = c[0]
                    e2 = c[1]

                    e1_d = c_d[0]
                    e2_d = c_d[1]

                    # The centers of the ellipses should be within 5 pixels of each other (is there a better treshold?)
                    dist = np.sqrt(((e1[0][0] - e1_d[0][0]) ** 2 + (e1[0][1] - e1_d[0][1]) ** 2))
                    if dist >= 25:
                        continue

                    dist = np.sqrt(((e2[0][0] - e2_d[0][0]) ** 2 + (e2[0][1] - e2_d[0][1]) ** 2))
                    if dist >= 25:
                        continue

                    # # The rotation of the elipses should be whitin 4 degrees of eachother
                    # angle_diff = np.abs(e1[2] - e1_d[2])
                    # if angle_diff>4:
                    #     continue
                    #
                    # angle_diff = np.abs(e2[2] - e2_d[2])
                    # if angle_diff>4:
                    #     continue

                    # The widths of the ring along the major and minor axis should be roughly the same
                    border_major = (e1[1][1]-e1_d[1][1])/2
                    border_minor = (e1[1][0]-e1_d[1][0])/2
                    border_diff = np.abs(border_major - border_minor)
                    
                    if border_diff>5:
                        continue
                    
                    # Add the candidate to the list
                    candidates.append((e1,e2))
                    

        print(f"Match cnt: {len(candidates)}")

        # Publish the rings
        if len(candidates) > 0:
            self.publish_rings(candidates)
                    
        # Plot the rings on the image
        for c in candidates:

            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # drawing the ellipses on the image
            cv2.ellipse(self.color, e1, (0, 255, 0), 2)
            cv2.ellipse(self.color, e2, (0, 255, 0), 2)

            # Get a bounding box, around the first ellipse ('average' of both elipsis)
            size = (e1[1][0]+e1[1][1])/2
            center = (e1[0][1], e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<self.color.shape[0] else self.color.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < self.color.shape[1] else self.color.shape[1]

        cv2.imshow("Detected rings",self.color)
        #cv2.waitKey(1)                    
                
    def publish_rings(self, candidates):
        for c in candidates:
            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # Get a bounding box, around the first ellipse ('average' of both elipsis)
            size = (e1[1][0]+e1[1][1])/2
            center = (e1[0][1], e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<self.color.shape[0] else self.color.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < self.color.shape[1] else self.color.shape[1]

            # Create a marker
            marker = Marker()
            marker.header.frame_id = "oakd_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rings"
            marker.id = self.marker_num
            self.marker_num += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            self.marker_array.markers.append(marker)

        self.marker_pub.publish(self.marker_array)

    def image_callback(self, data):
        #self.get_logger().info(f"I got a new image! Will try to find rings...")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.color = cv_image

    def depth_callback(self,data):

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

        depth_image[depth_image==np.inf] = 0
        
        #self.depth = depth_image
        
        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_1 = image_1/np.max(image_1)*255

        image_viz = np.array(image_1, dtype= np.uint8)

        self.depth = image_viz

        cv2.imshow("Depth window", image_viz)
        #cv2.waitKey(1)


def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()