#ifndef INTENSITY_LIDAR_CORNER_DETECTION_HPP
#define INTENSITY_LIDAR_CORNER_DETECTION_HPP

#include <Eigen/Dense> // Eigen 라이브러리 사용
#include <vector>
#include <string>
#include <tuple> // std::tuple 사용
#include <sstream> // std::stringstream 사용

// 사용자 정의 포인트 구조체: XYZ 좌표와 강도(Intensity)를 포함
struct PointXYZI {
    double x, y, z, intensity;
};

// =============================================================================
// 1. PCD 데이터 파싱 함수 (ASCII 문자열에서)
// =============================================================================
// ASCII 형식의 PCD(Point Cloud Data) 문자열을 파싱하여 PointXYZI 벡터로 변환합니다.
// 헤더 정보와 XYZI 데이터를 추출합니다.
std::vector<PointXYZI> parsePCDString(const std::string& pcd_string);

// =============================================================================
// 3. 강도 기반 흑백 분류 (회색 영역 포함)
// =============================================================================
// 강도 값을 흑색(0), 백색(1), 또는 회색 영역(-1)으로 분류합니다.
// epsilon_g는 회색 영역 정의를 위한 상수입니다.
// 반환: 분류된 색상 벡터, 하한 임계값(tau_l), 상한 임계값(tau_h)
std::tuple<Eigen::VectorXi, double, double> classifyIntensityColor(const Eigen::VectorXd& intensities, double epsilon_g = 4.0);

// =============================================================================
// 4. 체커보드 모델 정의 및 패턴 색상 확인
// =============================================================================
// 주어진 (x, y) 좌표에서 체커보드 패턴의 색상(흑색:0, 백색:1)을 결정합니다.
// 전체 체커보드의 좌측 하단 (0,0) 칸이 백색이라고 가정합니다.
// grid_size_x_squares: 가로 사각형 수, grid_size_y_squares: 세로 사각형 수, checker_size_m: 각 사각형의 크기(미터)
// 반환: 흑색(0), 백색(1), 또는 체커보드 범위 밖(-1)
int getCheckerboardPatternColor(double x, double y, int grid_size_x_squares, int grid_size_y_squares, double checker_size_m);

// =============================================================================
// 5. 비용 함수 (Cost Function)
// =============================================================================
// 체커보드 자세 최적화를 위한 비용 함수입니다.
// params: (tx, ty, theta_z) - 평면 상의 2D 이동 및 회전 각도
// points_in_pca_plane_2d: PCA 변환된 2D 포인트 배열
// classified_colors: 분류된 색상 배열 (0:흑색, 1:백색, -1:회색)
// grid_size_x_squares: 가로 사각형 수, grid_size_y_squares: 세로 사각형 수, checker_size_m: 각 사각형의 크기(미터)
// 반환: 총 비용
double costFunction(const Eigen::Vector3d& params,
                    const Eigen::MatrixXd& points_in_pca_plane_2d,
                    const Eigen::VectorXi& classified_colors,
                    int grid_size_x_squares, int grid_size_y_squares, double checker_size_m);

// =============================================================================
// 6. 논문 방식의 코너 검출 메인 함수
// =============================================================================
// LiDAR 포인트로부터 체커보드 코너를 추정합니다.
// lidar_points_full_vec: LiDAR 포인트 (x, y, z, 강도) 벡터
// internal_corners_x: 가로 내부 코너 수 (예: 7x8 보드의 경우 6)
// internal_corners_y: 세로 내부 코너 수 (예: 7x8 보드의 경우 7)
// checker_size_m: 각 체커 사각형의 크기(미터)
// flip_normal_direction: true이면 PCA Z축(법선 벡터)의 방향을 뒤집습니다.
// 반환: 원본 LiDAR 프레임에서의 3D 코너 좌표 (PointXYZI 벡터)
std::vector<PointXYZI> estimateChessboardCornersPaperMethod(
    const std::vector<PointXYZI>& lidar_points_full_vec,
    int internal_corners_x, int internal_corners_y, double checker_size_m,
    bool flip_normal_direction);

#endif // LIDAR_CORNER_DETECTION_HPP
