cd ~/colcon_ws/rins
ros2 topic pub --once /arm_command std_msgs/msg/String "{data: 'look_for_rings'}"