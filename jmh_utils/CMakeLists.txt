cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(jmh_utils LANGUAGES C CXX) # 프로젝트 이름을 라이브러리 이름으로 변경

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3 라이브러리 찾기
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED NO_MODULE)
find_package(PCL REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(GSL REQUIRED)

# 코너 검출 로직을 위한 정적 라이브러리 생성
add_library(jmh_utils STATIC
    src/main_camera_calib.cpp
    src/main_lidar_calib.cpp   
    src/main_intensity_lidar_corner_detection.cpp
    src/main_ndt_transformation.cpp
    src/main_fusion_visualization.cpp
    src/calc_convert.cpp
    src/calc_compute.cpp
    src/serve_keyboard.cpp
    src/serve_save.cpp
    src/serve_load.cpp
)

# 헤더 파일 경로를 PUBLIC으로 설정하여 이 라이브러리를 사용하는 다른 타겟에 전파되도록 합니다.
# 이렇게 하면 이 라이브러리를 링크하는 모든 타겟은 'include' 폴더를 자신의 include 경로에 자동으로 추가합니다.
target_include_directories(jmh_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # 빌드 시 소스 디렉토리의 include 폴더
    $<INSTALL_INTERFACE:include> # 설치 시 (선택 사항)
)


# Eigen3 라이브러리를 새로 생성한 라이브러리에 링크
target_link_libraries(jmh_utils
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${GSL_LIBRARIES}
)

add_executable(fusion_output
  examples/fusion_output.cpp
)

target_link_libraries(fusion_output PRIVATE jmh_utils)

add_executable(cam_lidar_calibration
  examples/cam_lidar_calibration.cpp
)

target_link_libraries(cam_lidar_calibration PRIVATE jmh_utils ceres)

add_executable(lid_imu_init_rot_sol
  examples/lid_imu_init_rot_sol.cpp
)

target_link_libraries(lid_imu_init_rot_sol PRIVATE jmh_utils ceres)


include(GNUInstallDirs)
install(TARGETS jmh_utils
    EXPORT jmh_utilsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  # (공유 라이브러리인 경우)
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS 
    fusion_output
    cam_lidar_calibration
    lid_imu_init_rot_sol
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# (선택 사항) 이 라이브러리를 시스템에 설치하고 싶다면 install 명령을 추가할 수 있습니다.
# 하지만 sensor_fusion_study에서 add_subdirectory로 사용한다면 필요 없을 수 있습니다.
# install(TARGETS jmh_utils
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
# )
# install(DIRECTORY include/
#     DESTINATION include/${PROJECT_NAME}
#     FILES_MATCHING PATTERN "*.hpp"
# )
