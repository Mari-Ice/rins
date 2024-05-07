import launch
from launch.substitutions import PathJoinSubstitution
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import time


from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    return launch.LaunchDescription([
        Node(
            package='dis_tutorial7',
            executable='arm_mover_actions.py',
            name='arm_mover',
        ),
		Node(
            package='task2',
            executable='master.py',
            name='master',
        ),
        Node(
            package='task2',
            executable='color_talker.py',
            name='color_talker',        
        ),
        Node(
            package='task2',
            executable='ring_detection.py',
            name='ring_detection',
        ),
        Node(
            package='task2',
            executable='autonomous_explorer.py',
            name='autonomous_explorer'
        ),
    ])

    
    

