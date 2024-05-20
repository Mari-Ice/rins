export ROS_DOMAIN_ID=11
#export ROS_LOCALHOST_ONLY=1
export IGN_IP=127.0.0.1
export QT_SCREEN_SCALE_FACTORS=1
export GIT_EDITOR=vim

source /opt/ros/humble/setup.bash
source install/local_setup.bash

# this line should be commented out when running simulation
source /etc/turtlebot4_discovery/setup.bash
