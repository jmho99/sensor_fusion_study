cmake_minimum_required(VERSION 3.8)
project(sensor_fusion_study)

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

add_executable(cv_callibration src/cv_callibration.cpp)
ament_target_dependencies(
  cv_callibration
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
)
target_include_directories(cv_callibration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_callibration PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

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

add_executable(cv_stereo src/cv_stereo.cpp)
ament_target_dependencies(
  cv_stereo
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
)
target_include_directories(cv_stereo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_stereo PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(cv_cam_lidar_cali src/cv_cam_lidar_cali.cpp)
ament_target_dependencies(
  cv_cam_lidar_cali
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
target_include_directories(cv_cam_lidar_cali PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cv_cam_lidar_cali PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

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

add_executable(cam_lidar_cali src/cam_lidar_cali.cpp)
ament_target_dependencies(
  cam_lidar_cali
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
target_include_directories(cam_lidar_cali PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cam_lidar_cali PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

install(TARGETS
  cv_callibration
  cv_stereo
  cv_random
  cv_cam_lidar_cali
  pcd_publisher
  cam_lidar_cali
  DESTINATION lib/${PROJECT_NAME})

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
