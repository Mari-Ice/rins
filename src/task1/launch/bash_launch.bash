#!/bin/bash

cd ~/colcon_ws/rins

ros2 launch task1 sim_turtlebot_nav.launch.py &
ros2 run task1 detect_people.py &
