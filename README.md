# DONE:
	* Task1
	* Task2:
		* Parkiranje, ko si v blizini parkirnega mesta
		* Zaznavanje obročkov in njihove barve
		* Govor barv
		* Avtomatsko raziskovanje
		* Ocena kvalitete zaznanega obroča
		* Štetje obročkov, premik k zelenemu obročku, da se tam lahko parkiraš 
# TODO:
	* Task1:
		* Nastavit, da pravi robot uporablja dis3 package, ne pa turtlebot4_navigation, ker so v dis3 pravilne nastavitve (nav2 config).
		* Napisat kak launch file za zaganjanje pravega robota.

# Poganjanje na robotu
	* ros2 launch turtlebot4_navigation localization.launch.py map:=/home/theta/colcon_ws/rins/src/dis_tutorial3/maps/non_sim/map.yaml
	* ros2 launch turtlebot4_navigation nav2.launch.py
	* set initial pose (rviz pose estimate)
	* TODO fix nav2 settings
