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
            package='task3',
            executable='master.py',
            name='master',
        ),
        Node(
            package='task3',
            executable='talker.py',
            name='talker',        
        ),
        Node(
            package='task3',
            executable='ring_detection.py',
            name='ring_detection',
        ),
        Node(
            package='task3',
            executable='autonomous_explorer.py',
            name='autonomous_explorer'
        ),
        Node(
            package='task3',
            executable='park.py',
            name='visually_assisted_parking'
        ),
    ])

    
    

