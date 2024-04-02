import launch
from launch.substitutions import PathJoinSubstitution
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    package_dir = get_package_share_directory('task1')

    sim_turtlebot_nav_launch = PathJoinSubstitution(
        [package_dir, 'launch', 'sim_turtlebot_nav.launch.py'])
    
    return launch.LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_turtlebot_nav_launch),
        ),
        Node(
            package='task1',
            executable='detect_people.py',
            name='detect_people',
        ),
        Node(
            package='task1',
            executable='people_manager.py',
            name='people_manager',
        ),
        Node(
            package='task1',
            executable='talker.py',
            name='talker',
        )
		# ,
        # Node(
        #     package='task1',
        #     executable='keypoint_follower.py',
        #     name='keypoint_follower',
        # )
    ])
