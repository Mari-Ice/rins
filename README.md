# DONE:
	* Task1
		* Detect people
		* People manager
		* Talker
		* Keypoint follower
		* Find safe park positions
		* Fixed mask artifacts
		
	* Task2:
		* Parkiranje, ko si v blizini parkirnega mesta
		* Zaznavanje obročkov in njihove barve
		* Govor barv
		* Avtomatsko raziskovanje
		* Ocena kvalitete zaznanega obroča
		* Štetje obročkov, premik k zelenemu obročku, da se tam lahko parkiraš 
	* Task3:
		* Mona Lisa training set
# TODO:
	* Task1:
		* Napisat kak launch file za zaganjanje pravega robota.
		* (optional) Spining at keypoints
		* (optional) Higher resolution map
		* rotate to face more accurately
	* Task2:
		* Black color quality
		* white masks
	* Task3:
		* Surface defect detection PCA (Mona Lisa surface)
		* Cylinder detection
		* Speech recognition

# Poganjanje na robotu
	* NUJNO NA TURTLEBOT WIFIJU
	* bash src/dis_tutorial5/configure_discovery.sh < /dev/tty 
	* ros2 launch turtlebot4_navigation localization.launch.py map:=src/dis_tutorial3/maps/non_sim/map.yaml
	* rviz2 -> set initial pose (Pose estimate)
	* ros2 launch dis_tutorial3 nav2.launch.py
