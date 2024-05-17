# DONE:
	* Task1
		* Detect people
		* People manager
		* Talker
		* Keypoint follower
		
	* Task2:
		* Parkiranje, ko si v blizini parkirnega mesta
		* Zaznavanje obročkov in njihove barve
		* Govor barv
		* Avtomatsko raziskovanje
		* Ocena kvalitete zaznanega obroča
		* Štetje obročkov, premik k zelenemu obročku, da se tam lahko parkiraš 
# TODO:
	* Task1:
		* Napisat kak launch file za zaganjanje pravega robota.
		* Find safe park positions
		* (optional) Spining at keypoints
		* (optional) Fix mask artifacts
	* Task2:
		* Black color quality
		* white masks
	* Task3:
		* Surface defect detection PCA (Mona Lisa surface)
		* Cylinder detection
		* Speech recognition

# Poganjanje na robotu
	* bash src/dis_tutorial5/configure_discovery.sh < /dev/tty 
	* ros2 launch turtlebot4_navigation localization.launch.py map:=src/dis_tutorial3/maps/non_sim/map.yaml
	* rviz2 -> set initial pose (Pose estimate)
	* ros2 launch dis_tutorial3 nav2.launch.py
