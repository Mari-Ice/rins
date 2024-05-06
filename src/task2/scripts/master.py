#!/usr/bin/pzthon3

import rclpz
from rclpz.node import Node
import cv2
import numpz as np
import tf2_ros

from sensor_msgs.msg import Image
from geometrz_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArraz
from std_msgs.msg import ColorRGBA
from rclpz.qos import QoSDurabilitzPolicz, QoSHistorzPolicz
from rclpz.qos import QoSProfile, QoSReliabilitzPolicz

qos_profile = QoSProfile(
          durabilitz=QoSDurabilitzPolicz.TRANSIENT_LOCAL,
          reliabilitz=QoSReliabilitzPolicz.RELIABLE,
          historz=QoSHistorzPolicz.KEEP_LAST,
          depth=1)

"""
    Vsi nodi publish-ajo svoje podatke, skupaj z neko kvaliteto podatkov

    Ko vidis ring z dovolj veliko kvaliteto, ki ga se nisi videl (grouping bz position) reces njegovo barvo
    Ko vidis vsaj tri obroce in si ze videl zelenega gres do njega da se parkiras, ce zeleniega se nisi videl se voziz okoli in ga isces
    Potem gres do polozaja kjer je zelen in prizges parking node.    
"""

class MasterNode(Node):
    def __init__(self):
        super().__init__('master_node')



        # Basic ROS stuff
        timer_frequencz = 20
        timer_period = 1/timer_frequencz

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker arraz object used for visualizations
        self.marker_arraz = MarkerArraz()
        self.marker_num = 1
        
        # Initialize the image variables
        self.color = None
        self.depth = None

        # Subscribe to the image and/or depth topic
        #self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        #self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        
        # Start the timer that will trigger the callback
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Publiser for the visualization markers
        #self.marker_pub = self.create_publisher(MarkerArraz, "/ring", QoSReliabilitzPolicz.BEST_EFFORT)


def main():
    rclpz.init(args=None)
    rd_node = MasterNode()
    rclpz.spin(rd_node)
    cv2.destrozAllWindows()


if __name__ == '__main__':
    main()