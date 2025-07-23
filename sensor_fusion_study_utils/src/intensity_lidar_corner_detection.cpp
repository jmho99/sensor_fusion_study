#include "intensity_lidar_corner_detection.hpp" // 새로 생성한 헤더 파일 포함

#include <iostream> // std::cout, std::cerr
#include <algorithm> // std::min, std::max, std::sort
#include <cmath>     // std::abs, std::sin, std::cos, M_PI
#include <limits>    // std::numeric_limits

// =============================================================================
// 1. PCD 데이터 파싱 함수 (ASCII 문자열에서)
// =============================================================================
std::vector<PointXYZI> parsePCDString(const std::string& pcd_string) {
    std::vector<PointXYZI> lidar_points;
    std::vector<std::string> fields;
    int intensity_field_index = -1;
    bool header_parsed = false;
    bool data_started = false;

    std::stringstream ss(pcd_string);
    std::string line;
    
    while (std::getline(ss, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (line.empty()) {
            continue;
        }

        if (!header_parsed) {
            if (line.rfind("FIELDS", 0) == 0) {
                std::stringstream line_ss(line);
                std::string token;
                line_ss >> token; 
                while (line_ss >> token) {
                    fields.push_back(token);
                }
                auto it = std::find(fields.begin(), fields.end(), "intensity");
                if (it != fields.end()) {
                    intensity_field_index = std::distance(fields.begin(), it);
                } else {
                    std::cerr << "Warning: 'intensity' field not found in PCD header. Using dummy intensity." << std::endl;
                }
            } else if (line.rfind("DATA", 0) == 0) {
                std::stringstream line_ss(line);
                std::string token;
                line_ss >> token; 
                line_ss >> token; 
                if (token != "ascii") {
                    std::cerr << "Error: Only ASCII PCD data is supported for manual intensity parsing." << std::endl;
                    return {};
                }
                header_parsed = true;
                data_started = true;
            }
        } else if (data_started) {
            std::stringstream line_ss(line);
            std::string part;
            std::vector<std::string> parts;
            while (line_ss >> part) {
                parts.push_back(part);
            }

            if (parts.size() != fields.size()) {
                continue;
            }

            try {
                PointXYZI p;
                p.x = std::stod(parts[std::distance(fields.begin(), std::find(fields.begin(), fields.end(), "x"))]);
                p.y = std::stod(parts[std::distance(fields.begin(), std::find(fields.begin(), fields.end(), "y"))]);
                p.z = std::stod(parts[std::distance(fields.begin(), std::find(fields.begin(), fields.end(), "z"))]);

                if (intensity_field_index != -1 && static_cast<size_t>(intensity_field_index) < parts.size()) {
                    p.intensity = std::stod(parts[intensity_field_index]);
                } else {
                    p.intensity = 128.0; 
                }
                lidar_points.push_back(p);

                if (lidar_points.size() <= 5) {
                    if (lidar_points.size() == 1) {
                         std::cout << "\nFirst 5 rows of parsed LiDAR points (XYZI):" << std::endl;
                    }
                    std::cout << "  X: " << p.x << ", Y: " << p.y << ", Z: " << p.z << ", I: " << p.intensity << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse numeric data in line: " << line << " - " << e.what() << std::endl;
                continue;
            }
        }
    }

    if (!header_parsed) {
        std::cerr << "Error: PCD header not found or incomplete in string." << std::endl;
        return {};
    }
    if (lidar_points.empty()) {
        std::cerr << "PCD string is empty or no points parsed." << std::endl;
        return {};
    }

    std::cout << "Parsed " << lidar_points.size() << " points from PCD string with XYZ and Intensity." << std::endl;
    return lidar_points;
}


// =============================================================================
// 3. 강도 기반 흑백 분류 (회색 영역 포함)
// =============================================================================
std::tuple<Eigen::VectorXi, double, double> classifyIntensityColor(const Eigen::VectorXd& intensities, double epsilon_g) {
    if (intensities.size() == 0) {
        return std::make_tuple(Eigen::VectorXi(), 0.0, 0.0);
    }

    double R_L = intensities.minCoeff();
    double R_H = intensities.maxCoeff();

    std::cout << "Minimum Intensity (R_L): " << R_L << std::endl;
    std::cout << "Maximum Intensity (R_H): " << R_H << std::endl;

    double tau_l = ((epsilon_g - 1.0) * R_L + R_H) / epsilon_g;
    double tau_h = (R_L + (epsilon_g - 1.0) * R_H) / epsilon_g;

    Eigen::VectorXi classified_colors = Eigen::VectorXi::Constant(intensities.size(), -1);
    int black_count = 0;
    int white_count = 0;
    int gray_count = 0;

    for (int i = 0; i < intensities.size(); ++i) {
        if (intensities[i] < tau_l) {
            classified_colors[i] = 0;
            black_count++;
        } else if (intensities[i] > tau_h) {
            classified_colors[i] = 1;
            white_count++;
        } else {
            gray_count++;
        }
    }
    std::cout << "Intensity classification complete. Black (<" << tau_l << "), White (>" << tau_h << "), Gray Zone: [" << tau_l << ", " << tau_h << "]" << std::endl;
    std::cout << "Classified points (Black: " << black_count << ", White: " << white_count << ", Gray: " << gray_count << ")" << std::endl;

    return std::make_tuple(classified_colors, tau_l, tau_h);
}

// =============================================================================
// 4. 체커보드 모델 정의 및 패턴 색상 확인
// =============================================================================
int getCheckerboardPatternColor(double x, double y, int grid_size_x_squares, int grid_size_y_squares, double checker_size_m) {
    int col = static_cast<int>(x / checker_size_m);
    int row = static_cast<int>(y / checker_size_m);

    if (!(col >= 0 && col < grid_size_x_squares && row >= 0 && row < grid_size_y_squares)) {
        return -1;
    }

    return (row + col) % 2 == 0 ? 1 : 0;
}

// =============================================================================
// 5. 비용 함수 (Cost Function)
// =============================================================================
double costFunction(const Eigen::Vector3d& params,
                    const Eigen::MatrixXd& points_in_pca_plane_2d,
                    const Eigen::VectorXi& classified_colors,
                    int grid_size_x_squares, int grid_size_y_squares, double checker_size_m) {
    double tx = params[0];
    double ty = params[1];
    double theta_z = params[2];
    double cost = 0.0;

    double cos_theta = std::cos(theta_z);
    double sin_theta = std::sin(theta_z);
    Eigen::Matrix2d R_z;
    R_z << cos_theta, -sin_theta,
           sin_theta, cos_theta;

    double board_width = grid_size_x_squares * checker_size_m;
    double board_height = grid_size_y_squares * checker_size_m;
    double model_min_x = -board_width / 2.0;
    double model_max_x = board_width / 2.0;
    double model_min_y = -board_height / 2.0;
    double model_max_y = board_height / 2.0;

    Eigen::MatrixXd transformed_points_2d = (R_z.transpose() * (points_in_pca_plane_2d.rowwise() - Eigen::RowVector2d(tx, ty)).transpose()).transpose();

    for (int i = 0; i < transformed_points_2d.rows(); ++i) {
        double point_intensity_color = classified_colors[i];

        if (point_intensity_color == -1) {
            continue;
        }

        double p_2d_transformed_for_pattern_lookup_x = transformed_points_2d(i, 0) + board_width / 2.0;
        double p_2d_transformed_for_pattern_lookup_y = transformed_points_2d(i, 1) + board_height / 2.0;

        int model_pattern_color_at_point = getCheckerboardPatternColor(
            p_2d_transformed_for_pattern_lookup_x, p_2d_transformed_for_pattern_lookup_y,
            grid_size_x_squares, grid_size_y_squares, checker_size_m
        );

        if (model_pattern_color_at_point == -1) {
            double dist_x = std::max({0.0, model_min_x - transformed_points_2d(i, 0), transformed_points_2d(i, 0) - model_max_x});
            double dist_y = std::max({0.0, model_min_y - transformed_points_2d(i, 1), transformed_points_2d(i, 1) - model_max_y});
            cost += (dist_x + dist_y) * 100.0;
        } else if (point_intensity_color != model_pattern_color_at_point) {
            cost += 1.0;
        }
    }
    return cost;
}

// =============================================================================
// 6. 논문 방식의 코너 검출 메인 함수
// =============================================================================
std::vector<PointXYZI> estimateChessboardCornersPaperMethod(
    const std::vector<PointXYZI>& lidar_points_full_vec,
    int internal_corners_x, int internal_corners_y, double checker_size_m,
    bool flip_normal_direction) {

    std::cout << "\n--- 논문 방식의 코너 검출 시작 ---" << std::endl;

    if (lidar_points_full_vec.empty()) {
        std::cerr << "Error: 입력 LiDAR 포인트 벡터가 비어 있습니다." << std::endl;
        return {};
    }

    Eigen::MatrixXd points_3d(lidar_points_full_vec.size(), 3);
    Eigen::VectorXd intensities(lidar_points_full_vec.size());

    for (size_t i = 0; i < lidar_points_full_vec.size(); ++i) {
        points_3d(i, 0) = lidar_points_full_vec[i].x;
        points_3d(i, 1) = lidar_points_full_vec[i].y;
        points_3d(i, 2) = lidar_points_full_vec[i].z;
        intensities(i) = lidar_points_full_vec[i].intensity;
    }

    if (points_3d.rows() < 4) {
        std::cerr << "Error: PCA 및 코너 검출을 위한 포인트가 충분하지 않습니다. 중단합니다. (최소 4개 필요, 현재 " << lidar_points_full_vec.size() << "개)" << std::endl;
        return {};
    }

    // =============================================================================
    // PCA
    // =============================================================================
    Eigen::RowVector3d centroid = points_3d.colwise().mean();
    Eigen::MatrixXd centered_points = points_3d.rowwise() - centroid;

    Eigen::Matrix3d covariance_matrix = (centered_points.transpose() * centered_points) / (centered_points.rows() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(covariance_matrix);
    Eigen::Vector3d eigenvalues = es.eigenvalues();
    Eigen::Matrix3d eigenvectors = es.eigenvectors();

    std::vector<std::pair<double, Eigen::Vector3d>> eigen_pairs;
    for (int i = 0; i < 3; ++i) {
        eigen_pairs.push_back({eigenvalues(i), eigenvectors.col(i)});
    }
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), [](const std::pair<double, Eigen::Vector3d>& a, const std::pair<double, Eigen::Vector3d>& b) {
        return a.first > b.first;
    });

    Eigen::Matrix3d principal_components;
    Eigen::Vector3d principal_eigenvalues;
    for (int i = 0; i < 3; ++i) {
        principal_components.col(i) = eigen_pairs[i].second;
        principal_eigenvalues[i] = eigen_pairs[i].first;
    }

    Eigen::Vector3d v_z_pca = principal_components.col(2); 

    if (v_z_pca.dot(-centroid.transpose()) < 0) {
        v_z_pca = -v_z_pca;
    }
    
    if (flip_normal_direction) {
        v_z_pca = -v_z_pca;
    }
    v_z_pca.normalize();

    Eigen::Vector3d v_x_pca_initial = principal_components.col(0);

    v_x_pca_initial = v_x_pca_initial - v_x_pca_initial.dot(v_z_pca) * v_z_pca;
    v_x_pca_initial.normalize();

    if (v_x_pca_initial.dot(Eigen::Vector3d(1, 0, 0)) < 0) {
        v_x_pca_initial = -v_x_pca_initial;
    }
    
    Eigen::Vector3d v_y_pca = v_z_pca.cross(v_x_pca_initial);
    v_y_pca.normalize();

    Eigen::Vector3d v_x_pca = v_y_pca.cross(v_z_pca);
    v_x_pca.normalize();

    principal_components.col(0) = v_x_pca;
    principal_components.col(1) = v_y_pca;
    principal_components.col(2) = v_z_pca;

    std::cout << "\nPCA 결과 (정렬된 축):" << std::endl;
    std::cout << "주성분 (고유 벡터):\n" << principal_components << std::endl;
    std::cout << "고유값:\n" << principal_eigenvalues.transpose() << std::endl;

    std::cout << "\nDEBUG: pca_axes 직교성 검증 (내적 결과):" << std::endl;
    std::cout << "  v_x . v_x = " << principal_components.col(0).dot(principal_components.col(0)) << std::endl;
    std::cout << "  v_y . v_y = " << principal_components.col(1).dot(principal_components.col(1)) << std::endl;
    std::cout << "  v_z . v_z = " << principal_components.col(2).dot(principal_components.col(2)) << std::endl;
    std::cout << "  v_x . v_y = " << principal_components.col(0).dot(principal_components.col(1)) << std::endl;
    std::cout << "  v_x . v_z = " << principal_components.col(0).dot(principal_components.col(2)) << std::endl;
    std::cout << "  v_y . v_z = " << principal_components.col(1).dot(principal_components.col(2)) << std::endl;

    Eigen::MatrixXd points_in_pca_frame = centered_points * principal_components;
    std::cout << "PCA 정렬 완료. 원본 중심: [" << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << "], PCA 평면의 점 수: " << points_in_pca_frame.rows() << std::endl;
    std::cout << "PCA 축 (R 행렬):\n" << principal_components << std::endl;
    std::cout << "원본 포인트 중심 (Centroid):\n" << centroid << std::endl;

    Eigen::MatrixXd projected_points_in_pca_plane_2d = points_in_pca_frame.leftCols(2);

    // --- 체커보드 바운딩 박스 외부 점 필터링 (적응형 크롭) ---
    double min_x_points_pca = projected_points_in_pca_plane_2d.col(0).minCoeff();
    double max_x_points_pca = projected_points_in_pca_plane_2d.col(0).maxCoeff();
    double min_y_points_pca = projected_points_in_pca_plane_2d.col(1).minCoeff();
    double max_y_points_pca = projected_points_in_pca_plane_2d.col(1).maxCoeff();

    double crop_margin = 0.05;
    double min_x_crop = min_x_points_pca - crop_margin;
    double max_x_crop = max_x_points_pca + crop_margin;
    double min_y_crop = min_y_points_pca - crop_margin;
    double max_y_crop = max_y_points_pca + crop_margin;

    std::vector<int> cropped_indices_vec;
    for (int i = 0; i < projected_points_in_pca_plane_2d.rows(); ++i) {
        if (projected_points_in_pca_plane_2d(i, 0) >= min_x_crop &&
            projected_points_in_pca_plane_2d(i, 0) <= max_x_crop &&
            projected_points_in_pca_plane_2d(i, 1) >= min_y_crop &&
            projected_points_in_pca_plane_2d(i, 1) <= max_y_crop) {
            cropped_indices_vec.push_back(i);
        }
    }

    if (cropped_indices_vec.empty()) {
        std::cerr << "Error: 체커보드 바운딩 박스로 크롭한 후 남은 포인트가 없습니다. 중단합니다." << std::endl;
        return {};
    }

    Eigen::MatrixXd points_for_optimization_full(cropped_indices_vec.size(), 3);
    Eigen::VectorXd intensities_for_optimization(cropped_indices_vec.size());
    Eigen::MatrixXd cropped_points_in_pca_plane_2d(cropped_indices_vec.size(), 2);

    for (size_t i = 0; i < cropped_indices_vec.size(); ++i) {
        int original_idx = cropped_indices_vec[i];
        points_for_optimization_full.row(i) = points_in_pca_frame.row(original_idx);
        intensities_for_optimization(i) = intensities(original_idx);
        cropped_points_in_pca_plane_2d.row(i) = projected_points_in_pca_plane_2d.row(original_idx);
    }
    std::cout << "체커보드 영역으로 필터링 완료. 남은 점 수: " << cropped_points_in_pca_plane_2d.rows() << std::endl;

    Eigen::VectorXi classified_colors;
    double tau_l, tau_h;
    std::tie(classified_colors, tau_l, tau_h) = classifyIntensityColor(intensities_for_optimization);

    std::vector<double> initial_theta_guesses = {0.0, M_PI / 2.0, M_PI, 3.0 * M_PI / 2.0};

    double best_cost = std::numeric_limits<double>::infinity();
    Eigen::Vector3d best_params;

    int num_squares_x_total = internal_corners_x + 1;
    int num_squares_y_total = internal_corners_y + 1;
    double board_width = num_squares_x_total * checker_size_m;
    double board_height = num_squares_y_total * checker_size_m;

    for (double initial_theta : initial_theta_guesses) {
        Eigen::Vector3d current_initial_guess;
        current_initial_guess[0] = cropped_points_in_pca_plane_2d.col(0).mean();
        current_initial_guess[1] = cropped_points_in_pca_plane_2d.col(1).mean();
        current_initial_guess[2] = initial_theta;

        std::cout << "최적화 시도: 초기 추정값 (tx, ty, theta_z): " << current_initial_guess[0] << ", " << current_initial_guess[1] << ", " << current_initial_guess[2] * 180.0 / M_PI << " deg" << std::endl;

        Eigen::Vector3d optimized_params_local = current_initial_guess;
        double current_cost = costFunction(optimized_params_local, cropped_points_in_pca_plane_2d, classified_colors,
                                           num_squares_x_total, num_squares_y_total, checker_size_m);

        double step = 0.01;
        for (int iter = 0; iter < 5; ++iter) {
            Eigen::Vector3d temp_params = optimized_params_local;
            for (int p_idx = 0; p_idx < 3; ++p_idx) {
                temp_params[p_idx] += step;
                double new_cost_plus = costFunction(temp_params, cropped_points_in_pca_plane_2d, classified_colors, num_squares_x_total, num_squares_y_total, checker_size_m);
                temp_params[p_idx] -= 2 * step;
                double new_cost_minus = costFunction(temp_params, cropped_points_in_pca_plane_2d, classified_colors, num_squares_x_total, num_squares_y_total, checker_size_m);
                temp_params[p_idx] += step;

                if (new_cost_plus < current_cost) {
                    current_cost = new_cost_plus;
                    optimized_params_local[p_idx] += step;
                } else if (new_cost_minus < current_cost) {
                    current_cost = new_cost_minus;
                    optimized_params_local[p_idx] -= step;
                }
            }
            step *= 0.5;
        }

        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_params = optimized_params_local;
            std::cout << "새로운 최적 결과 발견! 비용: " << best_cost << ", 파라미터: " << best_params[0] << ", " << best_params[1] << ", " << best_params[2] * 180.0 / M_PI << " deg" << std::endl;
        }
    }

    if (best_params.isZero() && best_cost == std::numeric_limits<double>::infinity()) {
        std::cerr << "Error: 모든 초기 추정값에 대해 최적화에 실패했습니다." << std::endl;
        return {};
    }

    double optimized_tx = best_params[0];
    double optimized_ty = best_params[1];
    double optimized_theta_z = best_params[2];
    std::cout << "최적화 완료. 최종 (tx, ty, theta_z): " << optimized_tx << ", " << optimized_ty << ", " << optimized_theta_z * 180.0 / M_PI << " deg" << std::endl;
    std::cout << "최종 비용: " << best_cost << std::endl;

    // 4. 최적화된 모델에서 3D 코너 추출
    std::vector<Eigen::Vector2d> ideal_2d_corners;
    for (int r = 0; r < internal_corners_y; ++r) {
        for (int c = 0; c < internal_corners_x; ++c) {
            double corner_x = (c + 1) * checker_size_m - board_width / 2.0;
            double corner_y = (r + 1) * checker_size_m - board_height / 2.0;
            ideal_2d_corners.push_back(Eigen::Vector2d(corner_x, corner_y));
        }
    }

    Eigen::Matrix2d R_opt;
    R_opt << std::cos(optimized_theta_z), -std::sin(optimized_theta_z),
             std::sin(optimized_theta_z),  std::cos(optimized_theta_z);

    Eigen::MatrixXd optimized_corners_in_pca_plane(ideal_2d_corners.size(), 2);
    for (size_t i = 0; i < ideal_2d_corners.size(); ++i) {
        optimized_corners_in_pca_plane.row(i) = R_opt * ideal_2d_corners[i] + Eigen::Vector2d(optimized_tx, optimized_ty);
    }
    
    Eigen::MatrixXd optimized_corners_3d_in_pca_plane(ideal_2d_corners.size(), 3);
    optimized_corners_3d_in_pca_plane.leftCols(2) = optimized_corners_in_pca_plane;
    optimized_corners_3d_in_pca_plane.col(2).setZero(); 
    
    Eigen::MatrixXd final_3d_corners_lidar_frame_matrix = (optimized_corners_3d_in_pca_plane * principal_components.transpose()).rowwise() + centroid;

    std::vector<PointXYZI> final_3d_corners_lidar_frame_vec;
    for (int i = 0; i < final_3d_corners_lidar_frame_matrix.rows(); ++i) {
        final_3d_corners_lidar_frame_vec.push_back({final_3d_corners_lidar_frame_matrix(i, 0),
                                                    final_3d_corners_lidar_frame_matrix(i, 1),
                                                    final_3d_corners_lidar_frame_matrix(i, 2),
                                                    0.0});
    }

    std::cout << "--- 논문 방식 코너 검출 완료 ---" << std::endl;
    std::cout << "\n최종 3D 코너 좌표 (라이다 센서 좌표계):" << std::endl;
    
    std::cout << "corners_x: [";
    for (size_t i = 0; i < final_3d_corners_lidar_frame_vec.size(); ++i) {
        std::cout << final_3d_corners_lidar_frame_vec[i].x;
        if (i < final_3d_corners_lidar_frame_vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "corners_y: [";
    for (size_t i = 0; i < final_3d_corners_lidar_frame_vec.size(); ++i) {
        std::cout << final_3d_corners_lidar_frame_vec[i].y;
        if (i < final_3d_corners_lidar_frame_vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "corners_z: [";
    for (size_t i = 0; i < final_3d_corners_lidar_frame_vec.size(); ++i) {
        std::cout << final_3d_corners_lidar_frame_vec[i].z;
        if (i < final_3d_corners_lidar_frame_vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "\n총 " << final_3d_corners_lidar_frame_vec.size() << "개의 코너가 검출되었습니다." << std::endl;

    double min_x_corner = std::numeric_limits<double>::max();
    double max_x_corner = std::numeric_limits<double>::lowest();
    double min_y_corner = std::numeric_limits<double>::max();
    double max_y_corner = std::numeric_limits<double>::lowest();
    double min_z_corner = std::numeric_limits<double>::max();
    double max_z_corner = std::numeric_limits<double>::lowest();

    Eigen::Vector3d normal_vector_for_distance_check = principal_components.col(2);
    double d_plane_for_distance_check = normal_vector_for_distance_check.dot(centroid.transpose());

    std::cout << "\nDEBUG: 각 코너 점의 PCA 평면으로부터의 거리:" << std::endl;

    for (size_t i = 0; i < final_3d_corners_lidar_frame_vec.size(); ++i) {
        const auto& p = final_3d_corners_lidar_frame_vec[i];
        min_x_corner = std::min(min_x_corner, p.x);
        max_x_corner = std::max(max_x_corner, p.x);
        min_y_corner = std::min(min_y_corner, p.y);
        max_y_corner = std::max(max_y_corner, p.y);
        min_z_corner = std::min(min_z_corner, p.z);
        max_z_corner = std::max(max_z_corner, p.z);

        Eigen::Vector3d corner_vec(p.x, p.y, p.z);
        double distance = normal_vector_for_distance_check.dot(corner_vec) - d_plane_for_distance_check;
        std::cout << "  코너 " << i << ": 평면으로부터의 거리 = " << distance << std::endl;
    }

    std::cout << "\n검출된 코너 점들의 좌표 범위 (LiDAR 프레임):" << std::endl;
    std::cout << "  X 범위: [" << min_x_corner << ", " << max_x_corner << "]" << std::endl;
    std::cout << "  Y 범위: [" << min_y_corner << ", " << max_y_corner << "]" << std::endl;
    std::cout << "  Z 범위: [" << min_z_corner << ", " << max_z_corner << "]" << std::endl;

    return final_3d_corners_lidar_frame_vec;
}
