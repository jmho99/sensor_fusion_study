cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sensor_fusion_study_utils CXX) # 프로젝트 이름을 라이브러리 이름으로 변경

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3 라이브러리 찾기
find_package(Eigen3 REQUIRED NO_MODULE)

# 코너 검출 로직을 위한 정적 라이브러리 생성
add_library(sensor_fusion_study_utils STATIC
    src/intensity_lidar_corner_detection.cpp
)

# 헤더 파일 경로를 PUBLIC으로 설정하여 이 라이브러리를 사용하는 다른 타겟에 전파되도록 합니다.
# 이렇게 하면 이 라이브러리를 링크하는 모든 타겟은 'include' 폴더를 자신의 include 경로에 자동으로 추가합니다.
target_include_directories(sensor_fusion_study_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # 빌드 시 소스 디렉토리의 include 폴더
    $<INSTALL_INTERFACE:include> # 설치 시 (선택 사항)
)

# Eigen3 라이브러리를 새로 생성한 라이브러리에 링크
target_link_libraries(sensor_fusion_study_utils
    Eigen3::Eigen
)

# (선택 사항) 이 라이브러리를 시스템에 설치하고 싶다면 install 명령을 추가할 수 있습니다.
# 하지만 sensor_fusion_study에서 add_subdirectory로 사용한다면 필요 없을 수 있습니다.
# install(TARGETS sensor_fusion_study_utils
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
# )
# install(DIRECTORY include/
#     DESTINATION include/${PROJECT_NAME}
#     FILES_MATCHING PATTERN "*.hpp"
# )
