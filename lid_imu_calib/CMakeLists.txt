cmake_minimum_required(VERSION 3.8)
project(lid_imu_calib)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)



# 정적 라이브러리 생성
add_library(lid_imu_calib STATIC
    src/save_ros2_pcd.cpp
)
ament_target_dependencies(lid_imu_calib
    sensor_msgs
)
# include 폴더 추가
target_include_directories(lid_imu_calib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # 빌드 시 소스 디렉토리의 include 폴더
    $<INSTALL_INTERFACE:include> # 설치 시 (선택 사항)
)
#include 폴더 내의 파일을 install/include에 설치
install(
  DIRECTORY include/
  DESTINATION include)
install(
  TARGETS lid_imu_calib
  EXPORT lid_imu_calib_export
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
ament_export_include_directories(include)
ament_export_targets(lid_imu_calib_export HAS_LIBRARY_TARGET)



#python module
ament_python_install_package(${PROJECT_NAME})
install (PROGRAMS
  python/ros2_lid_point_time.py
  python/ros2_lid_pcd_save.py
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
