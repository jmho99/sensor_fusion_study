cmake_minimum_required(VERSION 3.8)
project(calib_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(PCL REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(calib_interfaces REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

add_subdirectory(../calib_utils ${CMAKE_CURRENT_BINARY_DIR}/calib_utils_build)

add_executable(one_cam_calib src/one_cam_calib.cpp)
ament_target_dependencies(
  one_cam_calib
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
)
target_include_directories(one_cam_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(one_cam_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(cv_random src/cv_random.cpp)
ament_target_dependencies(
  cv_random
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
)
target_include_directories(cv_random PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_random PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(stereo_cam_calib src/stereo_cam_calib.cpp)
ament_target_dependencies(
  stereo_cam_calib
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
)
target_include_directories(stereo_cam_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(stereo_cam_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(pcd_publisher src/pcd_publisher.cpp)
ament_target_dependencies(
  pcd_publisher
  "rclcpp"
  "std_msgs"
  "sensor_msgs"
  "PCL"
  "pcl_conversions"
)
target_include_directories(pcd_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(pcd_publisher PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(cam_lidar_calib src/cam_lidar_calib.cpp)
ament_target_dependencies(
  cam_lidar_calib
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
  "PCL"
  "visualization_msgs"
  "pcl_conversions"
  "calib_interfaces"
  "Eigen3"
)
target_link_libraries(cam_lidar_calib
  calib_utils
)
target_include_directories(cam_lidar_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cam_lidar_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(multi_lidar_calib src/multi_lidar_calib.cpp)
ament_target_dependencies(
  multi_lidar_calib
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
  "PCL"
  "visualization_msgs"
  "pcl_conversions"
)
target_include_directories(multi_lidar_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(multi_lidar_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)


add_executable(intensity src/intensity.cpp)
ament_target_dependencies(
  intensity
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
  "PCL"
  "visualization_msgs"
  "pcl_conversions"
  "calib_interfaces"
  "Eigen3"
)
target_include_directories(intensity PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(intensity PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

install(TARGETS
  one_cam_calib
  stereo_cam_calib
  cv_random
  pcd_publisher
  cam_lidar_calib
  multi_lidar_calib
  intensity
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
    DESTINATION include/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.hpp"
)

  #python module
ament_python_install_package(${PROJECT_NAME})
install (PROGRAMS
  scripts/intensity.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
