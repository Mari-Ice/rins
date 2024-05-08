cmake_minimum_required(VERSION 3.5)
project(task2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL 1.10 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})

set(dependencies rclcpp std_msgs geometry_msgs sensor_msgs visualization_msgs tf2 tf2_geometry_msgs pcl_conversions message_filters)

add_executable(cylinder_segmentation src/cylinder_segmentation.cpp)
ament_target_dependencies(cylinder_segmentation ${dependencies})

set(dependencies rclcpp std_msgs geometry_msgs sensor_msgs visualization_msgs tf2 tf2_geometry_msgs pcl_conversions rosidl_default_generators message_filters)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Color.srv"
  "msg/RingInfo.msg"
  "msg/CylinderInfo.msg"
  "msg/Waypoint.msg"
)

target_link_libraries(cylinder_segmentation
  ${Boost_SYSTEM_LIBRARY}
  ${PCL_LIBRARIES}
)

install(TARGETS
  cylinder_segmentation
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/autonomous_explorer.py
  scripts/color_talker.py
  scripts/ring_detection.py
  scripts/master.py
  scripts/park.py
  scripts/cylinder_detection.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)
ament_package()
