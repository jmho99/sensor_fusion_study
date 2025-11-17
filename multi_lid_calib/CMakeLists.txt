cmake_minimum_required(VERSION 3.8)
project(multi_lid_calib)

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
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_subdirectory(../jmh_utils ${CMAKE_CURRENT_BINARY_DIR}/jmh_utils_build)

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
target_link_libraries(multi_lidar_calib
  jmh_utils
)
target_include_directories(multi_lidar_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(multi_lidar_calib PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17)

install(TARGETS
  multi_lidar_calib
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
