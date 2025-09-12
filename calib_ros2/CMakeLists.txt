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
find_package(message_filters REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(Ceres REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

add_subdirectory(../calib_utils ${CMAKE_CURRENT_BINARY_DIR}/calib_utils_build)

add_executable(a_one_cam_calib src/a_one_cam_calib.cpp)
ament_target_dependencies(
  a_one_cam_calib
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
)
target_link_libraries(a_one_cam_calib
  calib_utils
)
target_include_directories(a_one_cam_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(a_one_cam_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(b_stereo_cam_calib src/b_stereo_cam_calib.cpp)
ament_target_dependencies(
  b_stereo_cam_calib
  "rclcpp"
  "std_msgs"
  "OpenCV"
  "cv_bridge"
  "sensor_msgs"
  "image_transport"
)
target_link_libraries(b_stereo_cam_calib
  calib_utils
)
target_include_directories(b_stereo_cam_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(b_stereo_cam_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(c_cam_lidar_calib src/c_cam_lidar_calib.cpp)
ament_target_dependencies(
  c_cam_lidar_calib
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
  "Ceres"
)
target_link_libraries(c_cam_lidar_calib
  calib_utils
)
target_include_directories(c_cam_lidar_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(c_cam_lidar_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(d_multi_lidar_calib src/d_multi_lidar_calib.cpp)
ament_target_dependencies(
  d_multi_lidar_calib
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
target_link_libraries(d_multi_lidar_calib
  calib_utils
)
target_include_directories(d_multi_lidar_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(d_multi_lidar_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

add_executable(e_lidar_imu_calib src/e_lidar_imu_calib.cpp)
ament_target_dependencies(
  e_lidar_imu_calib
  "rclcpp"
  "sensor_msgs"
  "message_filters"
  "PCL"
  "pcl_conversions"
  "Eigen3"
  "rosbag2_cpp"
  "rosbag2_storage"
)
target_link_libraries(e_lidar_imu_calib
  calib_utils
)
target_include_directories(e_lidar_imu_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(e_lidar_imu_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

install(TARGETS
  a_one_cam_calib
  b_stereo_cam_calib
  c_cam_lidar_calib
  d_multi_lidar_calib
  e_lidar_imu_calib
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
    DESTINATION include/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.hpp"
)

  #python module
ament_python_install_package(${PROJECT_NAME})
install (PROGRAMS
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
