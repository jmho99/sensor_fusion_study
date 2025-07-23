#include "rclcpp/rclcpp.hpp"
#include "sensor_fusion_study_interfaces/srv/intensity.hpp" // Custom service message (Intensity로 변경)
#include <Eigen/Dense> // For PCA and matrix operations
#include <vector>
#include <string>
#include <sstream> // For std::stringstream
#include <iostream>
#include <algorithm> // For std::min, std::max, std::sort
#include <cmath>     // For std::abs, std::sin, std::cos, M_PI
#include <limits>    // For std::numeric_limits

// Custom point structure to hold XYZ and Intensity
struct PointXYZI {
    double x, y, z, intensity;
};

// =============================================================================
// 1. PCD Data Parsing Function (from ASCII string)
// =============================================================================
std::vector<PointXYZI> parsePCDString(const std::string& pcd_string) {
    std::vector<PointXYZI> lidar_points;
    std::vector<std::string> fields;
    int intensity_field_index = -1;
    bool header_parsed = false;
    bool data_started = false;

    std::stringstream ss(pcd_string);
    std::string line;
    
    // Iterate through each line of the PCD string
    while (std::getline(ss, line)) {
        // Remove leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (line.empty()) {
            continue;
        }

        // Parse header information
        if (!header_parsed) {
            if (line.rfind("FIELDS", 0) == 0) { // Starts with "FIELDS"
                std::stringstream line_ss(line);
                std::string token;
                line_ss >> token; // Skip "FIELDS"
                while (line_ss >> token) {
                    fields.push_back(token);
                }
                // Find the index of the 'intensity' field
                auto it = std::find(fields.begin(), fields.end(), "intensity");
                if (it != fields.end()) {
                    intensity_field_index = std::distance(fields.begin(), it);
                } else {
                    RCLCPP_WARN(rclcpp::get_logger("LidarCornerDetectionService"), "Warning: 'intensity' field not found in PCD header. Using dummy intensity.");
                }
            } else if (line.rfind("DATA", 0) == 0) { // Starts with "DATA"
                std::stringstream line_ss(line);
                std::string token;
                line_ss >> token; // Skip "DATA"
                line_ss >> token; // Read data type
                if (token != "ascii") {
                    RCLCPP_ERROR(rclcpp::get_logger("LidarCornerDetectionService"), "Error: Only ASCII PCD data is supported for manual intensity parsing.");
                    return {}; // Return empty vector on error
                }
                header_parsed = true;
                data_started = true;
            }
        } else if (data_started) { // Parse point data after header
            std::stringstream line_ss(line);
            std::string part;
            std::vector<std::string> parts;
            while (line_ss >> part) {
                parts.push_back(part);
            }

            // Skip malformed data lines
            if (parts.size() != fields.size()) {
                // RCLCPP_WARN(rclcpp::get_logger("LidarCornerDetectionService"), "Warning: Skipping malformed data line (expected %lu fields, got %lu): %s", fields.size(), parts.size(), line.c_str());
                continue;
            }

            try {
                PointXYZI p;
                // Parse x, y, z coordinates
                p.x = std::stod(parts[std::distance(fields.begin(), std::find(fields.begin(), fields.end(), "x"))]);
                p.y = std::stod(parts[std::distance(fields.begin(), std::find(fields.begin(), fields.end(), "y"))]);
                p.z = std::stod(parts[std::distance(fields.begin(), std::find(fields.begin(), fields.end(), "z"))]);

                // Parse intensity, or use dummy if not found/malformed
                if (intensity_field_index != -1 && static_cast<size_t>(intensity_field_index) < parts.size()) { // 경고 수정: signed/unsigned 비교
                    p.intensity = std::stod(parts[intensity_field_index]);
                } else {
                    p.intensity = 128.0; // Dummy intensity
                }
                lidar_points.push_back(p);

                // Print first 5 rows for debugging
                if (lidar_points.size() <= 5) {
                    if (lidar_points.size() == 1) {
                         RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "\nFirst 5 rows of parsed LiDAR points (XYZI):");
                    }
                    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "  X: %.4f, Y: %.4f, Z: %.4f, I: %.4f", p.x, p.y, p.z, p.intensity);
                }
            } catch (const std::exception& e) {
                RCLCPP_WARN(rclcpp::get_logger("LidarCornerDetectionService"), "Warning: Could not parse numeric data in line: %s - %s", line.c_str(), e.what());
                continue;
            }
        }
    }

    if (!header_parsed) {
        RCLCPP_ERROR(rclcpp::get_logger("LidarCornerDetectionService"), "Error: PCD header not found or incomplete in string.");
        return {};
    }
    if (lidar_points.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("LidarCornerDetectionService"), "PCD string is empty or no points parsed.");
        return {};
    }

    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "Parsed %lu points from PCD string with XYZ and Intensity.", lidar_points.size());
    return lidar_points;
}


// =============================================================================
// 3. Intensity-based Black/White Classification (with Gray Zone)
// =============================================================================
std::tuple<Eigen::VectorXi, double, double> classifyIntensityColor(const Eigen::VectorXd& intensities, double epsilon_g = 4.0) {
    if (intensities.size() == 0) {
        return std::make_tuple(Eigen::VectorXi(), 0.0, 0.0);
    }

    double R_L = intensities.minCoeff(); // Minimum intensity
    double R_H = intensities.maxCoeff(); // Maximum intensity

    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "Minimum Intensity (R_L): %.4f", R_L);
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "Maximum Intensity (R_H): %.4f", R_H);

    // Calculate thresholds for black (tau_l) and white (tau_h) based on Equation (4)
    double tau_l = ((epsilon_g - 1.0) * R_L + R_H) / epsilon_g;
    double tau_h = (R_L + (epsilon_g - 1.0) * R_H) / epsilon_g;

    Eigen::VectorXi classified_colors = Eigen::VectorXi::Constant(intensities.size(), -1); // -1 for gray zone
    int black_count = 0;
    int white_count = 0;
    int gray_count = 0;

    // Classify each point's intensity
    for (int i = 0; i < intensities.size(); ++i) {
        if (intensities[i] < tau_l) {
            classified_colors[i] = 0; // Black
            black_count++;
        } else if (intensities[i] > tau_h) {
            classified_colors[i] = 1; // White
            white_count++;
        } else {
            // Remains -1 for gray zone
            gray_count++;
        }
    }
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "Intensity classification complete. Black (<%.2f), White (>%.2f), Gray Zone: [%.2f, %.2f]", tau_l, tau_h, tau_l, tau_h);
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "Classified points (Black: %d, White: %d, Gray: %d)", black_count, white_count, gray_count);

    return std::make_tuple(classified_colors, tau_l, tau_h);
}

// =============================================================================
// 4. Checkerboard Model Definition and Pattern Color Check
// =============================================================================
// Determines the color (0 for black, 1 for white) of the checkerboard pattern at a given (x, y) coordinate.
// Assumes (0,0) square (bottom-left of the entire checkerboard) is white.
int getCheckerboardPatternColor(double x, double y, int grid_size_x_squares, int grid_size_y_squares, double checker_size_m) {
    int col = static_cast<int>(x / checker_size_m);
    int row = static_cast<int>(y / checker_size_m);

    // Check if the point is within the checkerboard bounds
    if (!(col >= 0 && col < grid_size_x_squares && row >= 0 && row < grid_size_y_squares)) {
        return -1; // Outside checkerboard
    }

    // Checkerboard pattern: (row + col) % 2 == 0 for white, else black
    // Assuming (0,0) square is white
    return (row + col) % 2 == 0 ? 1 : 0;
}

// =============================================================================
// 5. Cost Function
// =============================================================================
// Cost function for optimizing checkerboard pose.
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
    Eigen::Matrix2d R_z; // 2D rotation matrix
    R_z << cos_theta, -sin_theta,
           sin_theta, cos_theta;

    // Define the bounding box of the checkerboard model in its local frame (centered at origin)
    double board_width = grid_size_x_squares * checker_size_m;
    double board_height = grid_size_y_squares * checker_size_m;
    double model_min_x = -board_width / 2.0;
    double model_max_x = board_width / 2.0;
    double model_min_y = -board_height / 2.0;
    double model_max_y = board_height / 2.0;

    // Transform all points from PCA-frame to checkerboard's local frame (centered)
    Eigen::MatrixXd transformed_points_2d = (R_z.transpose() * (points_in_pca_plane_2d.rowwise() - Eigen::RowVector2d(tx, ty)).transpose()).transpose();

    for (int i = 0; i < transformed_points_2d.rows(); ++i) {
        double point_intensity_color = classified_colors[i];

        if (point_intensity_color == -1) { // Ignore gray zone points
            continue;
        }

        // Convert point coordinates back to the (0,0) bottom-left origin for pattern lookup
        double p_2d_transformed_for_pattern_lookup_x = transformed_points_2d(i, 0) + board_width / 2.0;
        double p_2d_transformed_for_pattern_lookup_y = transformed_points_2d(i, 1) + board_height / 2.0;

        int model_pattern_color_at_point = getCheckerboardPatternColor(
            p_2d_transformed_for_pattern_lookup_x, p_2d_transformed_for_pattern_lookup_y,
            grid_size_x_squares, grid_size_y_squares, checker_size_m
        );

        if (model_pattern_color_at_point == -1) {
            // Penalize points outside the overall model bounds.
            double dist_x = std::max({0.0, model_min_x - transformed_points_2d(i, 0), transformed_points_2d(i, 0) - model_max_x});
            double dist_y = std::max({0.0, model_min_y - transformed_points_2d(i, 1), transformed_points_2d(i, 1) - model_max_y});
            cost += (dist_x + dist_y) * 100.0; // Large penalty
        } else if (point_intensity_color != model_pattern_color_at_point) {
            cost += 1.0; // Penalty for color mismatch
        }
    }
    return cost;
}

// =============================================================================
// 6. Main Corner Detection Function
// =============================================================================
std::vector<PointXYZI> estimateChessboardCornersPaperMethod(
    const std::vector<PointXYZI>& lidar_points_full_vec,
    int internal_corners_x, int internal_corners_y, double checker_size_m,
    bool flip_normal_direction) {

    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "\n--- Starting Paper Method Corner Detection ---");

    if (lidar_points_full_vec.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("LidarCornerDetectionService"), "Error: Input LiDAR points vector is empty.");
        return {};
    }

    Eigen::MatrixXd points_3d(lidar_points_full_vec.size(), 3);
    Eigen::VectorXd intensities(lidar_points_full_vec.size());

    // Convert std::vector<PointXYZI> to Eigen matrices
    for (size_t i = 0; i < lidar_points_full_vec.size(); ++i) {
        points_3d(i, 0) = lidar_points_full_vec[i].x;
        points_3d(i, 1) = lidar_points_full_vec[i].y;
        points_3d(i, 2) = lidar_points_full_vec[i].z;
        intensities(i) = lidar_points_full_vec[i].intensity;
    }

    if (points_3d.rows() < 4) { // Check minimum points for PCA
        RCLCPP_ERROR(rclcpp::get_logger("LidarCornerDetectionService"), "Error: Not enough points for PCA and corner detection. Aborting. (Need at least 4, got %zu)", lidar_points_full_vec.size());
        return {};
    }

    // =============================================================================
    // PCA (Principal Component Analysis)
    // =============================================================================
    // Center the data
    Eigen::RowVector3d centroid = points_3d.colwise().mean();
    Eigen::MatrixXd centered_points = points_3d.rowwise() - centroid;

    // Calculate covariance matrix
    Eigen::Matrix3d covariance_matrix = (centered_points.transpose() * centered_points) / (centered_points.rows() - 1);

    // Compute eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(covariance_matrix);
    Eigen::Vector3d eigenvalues = es.eigenvalues();
    Eigen::Matrix3d eigenvectors = es.eigenvectors();

    // Sort eigenvalues and eigenvectors in descending order
    std::vector<std::pair<double, Eigen::Vector3d>> eigen_pairs;
    for (int i = 0; i < 3; ++i) {
        eigen_pairs.push_back({eigenvalues(i), eigenvectors.col(i)});
    }
    // Sort by eigenvalue (first element of pair) in descending order
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), [](const std::pair<double, Eigen::Vector3d>& a, const std::pair<double, Eigen::Vector3d>& b) {
        return a.first > b.first; // Sort by eigenvalue in descending order
    });

    Eigen::Matrix3d principal_components; // Corresponds to pca_axes in Python
    Eigen::Vector3d principal_eigenvalues;
    for (int i = 0; i < 3; ++i) {
        principal_components.col(i) = eigen_pairs[i].second;
        principal_eigenvalues[i] = eigen_pairs[i].first;
    }

    Eigen::Vector3d v_z_pca = principal_components.col(2); // Normal vector (smallest eigenvalue)

    // Force v_z_pca (normal) to point towards the LiDAR sensor origin (0,0,0)
    // This ensures the checkerboard plane's normal consistently faces the sensor direction.
    if (v_z_pca.dot(-centroid.transpose()) < 0) {
        v_z_pca = -v_z_pca;
    }
    
    // Flip Z-axis direction if requested (maintain right-hand rule)
    if (flip_normal_direction) {
        v_z_pca = -v_z_pca;
    }
    v_z_pca.normalize(); // Ensure unit vector after potential flip

    Eigen::Vector3d v_x_pca_initial = principal_components.col(0); // First principal component

    // Adjust v_x_pca to be strictly orthogonal to v_z_pca
    v_x_pca_initial = v_x_pca_initial - v_x_pca_initial.dot(v_z_pca) * v_z_pca;
    v_x_pca_initial.normalize();

    // Apply paper condition: angle between mu_1 and LiDAR X-axis should not exceed 90 degrees
    // This ensures the checkerboard X-axis roughly aligns with the LiDAR X-axis.
    if (v_x_pca_initial.dot(Eigen::Vector3d(1, 0, 0)) < 0) {
        v_x_pca_initial = -v_x_pca_initial;
    }
    
    // Reconstruct v_y_pca using right-hand rule: v_y_pca = cross(v_z_pca, v_x_pca)
    Eigen::Vector3d v_y_pca = v_z_pca.cross(v_x_pca_initial);
    v_y_pca.normalize(); // Ensure unit vector

    // Recalculate v_x_pca for numerical stability (v_x_pca = cross(v_y_pca, v_z_pca))
    Eigen::Vector3d v_x_pca = v_y_pca.cross(v_z_pca);
    v_x_pca.normalize(); // Ensure unit vector

    // Update the principal_components matrix with the aligned axes
    principal_components.col(0) = v_x_pca;
    principal_components.col(1) = v_y_pca;
    principal_components.col(2) = v_z_pca;

    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "\nPCA Results (Aligned Axes):");
    // Use stringstream to format Eigen objects for logging
    std::stringstream ss_components;
    ss_components << principal_components.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]"));
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "Principal Components (Eigenvectors):\n%s", ss_components.str().c_str());

    std::stringstream ss_eigenvalues;
    ss_eigenvalues << principal_eigenvalues.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]"));
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "Eigenvalues:\n%s", ss_eigenvalues.str().c_str());

    // Debug: Verify orthogonality
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "\nDEBUG: pca_axes orthogonality verification (dot products):");
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "  v_x . v_x = %.6f", principal_components.col(0).dot(principal_components.col(0)));
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "  v_y . v_y = %.6f", principal_components.col(1).dot(principal_components.col(1)));
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "  v_z . v_z = %.6f", principal_components.col(2).dot(principal_components.col(2)));
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "  v_x . v_y = %.6f", principal_components.col(0).dot(principal_components.col(1)));
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "  v_x . v_z = %.6f", principal_components.col(0).dot(principal_components.col(2)));
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "  v_y . v_z = %.6f", principal_components.col(1).dot(principal_components.col(2)));

    // Transform points to PCA coordinate system (relative to original centroid)
    Eigen::MatrixXd points_in_pca_frame = centered_points * principal_components;
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "PCA alignment complete. Original Centroid: [%.4f, %.4f, %.4f], Points in PCA plane: %zu", centroid[0], centroid[1], centroid[2], points_in_pca_frame.rows());
    
    std::stringstream ss_pca_axes_matrix;
    ss_pca_axes_matrix << principal_components.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]"));
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "PCA Axes (R matrix):\n%s", ss_pca_axes_matrix.str().c_str());
    
    std::stringstream ss_centroid_matrix;
    ss_centroid_matrix << centroid.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]"));
    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "Original Points Centroid:\n%s", ss_centroid_matrix.str().c_str());

    // Project points onto the 2D PCA plane (XY plane of the PCA frame)
    Eigen::MatrixXd projected_points_in_pca_plane_2d = points_in_pca_frame.leftCols(2);

    // --- Checkerboard Bounding Box Filtering (Adaptive Crop) ---
    // Define crop region based on the actual min/max of PCA-transformed points
    double min_x_points_pca = projected_points_in_pca_plane_2d.col(0).minCoeff();
    double max_x_points_pca = projected_points_in_pca_plane_2d.col(0).maxCoeff();
    double min_y_points_pca = projected_points_in_pca_plane_2d.col(1).minCoeff();
    double max_y_points_pca = projected_points_in_pca_plane_2d.col(1).maxCoeff();

    double crop_margin = 0.05; // meters
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
        RCLCPP_ERROR(rclcpp::get_logger("LidarCornerDetectionService"), "Error: No points remain after cropping to checkerboard bounding box. Aborting.");
        return {};
    }

    // Create new Eigen matrices for cropped points and their intensities
    Eigen::MatrixXd points_for_optimization_full(cropped_indices_vec.size(), 3);
    Eigen::VectorXd intensities_for_optimization(cropped_indices_vec.size());
    Eigen::MatrixXd cropped_points_in_pca_plane_2d(cropped_indices_vec.size(), 2);

    for (size_t i = 0; i < cropped_indices_vec.size(); ++i) {
        int original_idx = cropped_indices_vec[i];
        points_for_optimization_full.row(i) = points_in_pca_frame.row(original_idx);
        intensities_for_optimization(i) = intensities(original_idx);
        cropped_points_in_pca_plane_2d.row(i) = projected_points_in_pca_plane_2d.row(original_idx);
    }
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "Filtered to checkerboard area. Remaining points: %zu", cropped_points_in_pca_plane_2d.rows());

    // Classify intensities of the cropped points
    Eigen::VectorXi classified_colors;
    double tau_l, tau_h;
    std::tie(classified_colors, tau_l, tau_h) = classifyIntensityColor(intensities_for_optimization);

    // Initial guesses for rotation angle (0, 90, 180, 270 degrees)
    std::vector<double> initial_theta_guesses = {0.0, M_PI / 2.0, M_PI, 3.0 * M_PI / 2.0};

    double best_cost = std::numeric_limits<double>::infinity();
    Eigen::Vector3d best_params; // (tx, ty, theta_z)

    // Calculate total number of squares on the checkerboard
    int num_squares_x_total = internal_corners_x + 1;
    int num_squares_y_total = internal_corners_y + 1;
    double board_width = num_squares_x_total * checker_size_m;
    double board_height = num_squares_y_total * checker_size_m;

    // Simple iterative optimization (placeholder for a more robust optimizer like Ceres)
    // This performs a basic local search around initial angle guesses.
    for (double initial_theta : initial_theta_guesses) {
        Eigen::Vector3d current_initial_guess;
        current_initial_guess[0] = cropped_points_in_pca_plane_2d.col(0).mean(); // Initial tx is mean X of cropped points
        current_initial_guess[1] = cropped_points_in_pca_plane_2d.col(1).mean(); // Initial ty is mean Y of cropped points
        current_initial_guess[2] = initial_theta; // Initial theta_z

        RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "Attempting optimization: Initial guess (tx, ty, theta_z): %.4f, %.4f, %.2f deg",
                    current_initial_guess[0], current_initial_guess[1], current_initial_guess[2] * 180.0 / M_PI);

        Eigen::Vector3d optimized_params_local = current_initial_guess;
        double current_cost = costFunction(optimized_params_local, cropped_points_in_pca_plane_2d, classified_colors,
                                           num_squares_x_total, num_squares_y_total, checker_size_m);

        // Simple refinement: perturb tx, ty, theta_z and pick best
        double step = 0.01; // meters/radians
        for (int iter = 0; iter < 5; ++iter) { // A few iterations
            Eigen::Vector3d temp_params = optimized_params_local;
            for (int p_idx = 0; p_idx < 3; ++p_idx) {
                // Try increasing the parameter
                temp_params[p_idx] += step;
                double new_cost_plus = costFunction(temp_params, cropped_points_in_pca_plane_2d, classified_colors, num_squares_x_total, num_squares_y_total, checker_size_m);
                
                // Try decreasing the parameter
                temp_params[p_idx] -= 2 * step; // temp_params is now original - step
                double new_cost_minus = costFunction(temp_params, cropped_points_in_pca_plane_2d, classified_colors, num_squares_x_total, num_squares_y_total, checker_size_m);
                
                temp_params[p_idx] += step; // Reset temp_params to original for next perturbation

                if (new_cost_plus < current_cost) {
                    current_cost = new_cost_plus;
                    optimized_params_local[p_idx] += step;
                } else if (new_cost_minus < current_cost) {
                    current_cost = new_cost_minus;
                    optimized_params_local[p_idx] -= step;
                }
            }
            step *= 0.5; // Reduce step size for next iteration
        }

        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_params = optimized_params_local;
            RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "New best result found! Cost: %.4f, Parameters: %.4f, %.4f, %.2f deg",
                        best_cost, best_params[0], best_params[1], best_params[2] * 180.0 / M_PI);
        }
    }

    if (best_params.isZero() && best_cost == std::numeric_limits<double>::infinity()) { 
        RCLCPP_ERROR(rclcpp::get_logger("LidarCornerDetectionService"), "Optimization failed for all initial guesses.");
        return {};
    }

    double optimized_tx = best_params[0];
    double optimized_ty = best_params[1];
    double optimized_theta_z = best_params[2];
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "Optimization complete. Final (tx, ty, theta_z): %.4f, %.4f, %.2f deg",
                optimized_tx, optimized_ty, optimized_theta_z * 180.0 / M_PI);
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "Final Cost: %.4f", best_cost);

    // 4. Extract 3D corners from optimized model
    std::vector<Eigen::Vector2d> ideal_2d_corners;
    // Generate internal corner coordinates relative to the model's center
    for (int r = 0; r < internal_corners_y; ++r) {
        for (int c = 0; c < internal_corners_x; ++c) {
            // (c+1, r+1) are 1-based indices for internal corners
            double corner_x = (c + 1) * checker_size_m - board_width / 2.0;
            double corner_y = (r + 1) * checker_size_m - board_height / 2.0;
            ideal_2d_corners.push_back(Eigen::Vector2d(corner_x, corner_y));
        }
    }

    Eigen::Matrix2d R_opt; // Optimized 2D rotation matrix
    R_opt << std::cos(optimized_theta_z), -std::sin(optimized_theta_z),
             std::sin(optimized_theta_z),  std::cos(optimized_theta_z);

    Eigen::MatrixXd optimized_corners_in_pca_plane(ideal_2d_corners.size(), 2);
    for (size_t i = 0; i < ideal_2d_corners.size(); ++i) {
        // Rotate and translate ideal corners to their optimized position in the PCA plane
        optimized_corners_in_pca_plane.row(i) = R_opt * ideal_2d_corners[i] + Eigen::Vector2d(optimized_tx, optimized_ty);
    }
    
    // Add a zero Z-coordinate to make them 3D in the PCA plane
    Eigen::MatrixXd optimized_corners_3d_in_pca_plane(ideal_2d_corners.size(), 3);
    optimized_corners_3d_in_pca_plane.leftCols(2) = optimized_corners_in_pca_plane;
    optimized_corners_3d_in_pca_plane.col(2).setZero(); 
    
    // Revert corners from PCA frame to original LiDAR frame
    // This involves rotating back by the inverse of PCA axes and adding the original centroid.
    // In Eigen, the inverse of an orthogonal matrix (like principal_components) is its transpose.
    Eigen::MatrixXd final_3d_corners_lidar_frame_matrix = (optimized_corners_3d_in_pca_plane * principal_components.transpose()).rowwise() + centroid;

    std::vector<PointXYZI> final_3d_corners_lidar_frame_vec;
    for (int i = 0; i < final_3d_corners_lidar_frame_matrix.rows(); ++i) {
        final_3d_corners_lidar_frame_vec.push_back({final_3d_corners_lidar_frame_matrix(i, 0),
                                                    final_3d_corners_lidar_frame_matrix(i, 1),
                                                    final_3d_corners_lidar_frame_matrix(i, 2),
                                                    0.0}); // Intensity is not relevant for corners
    }

    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "--- Paper Method Corner Detection Complete ---");
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "\nFinal 3D Corner Coordinates (LiDAR Sensor Frame):");
    
    std::string corners_x_str = "corners_x: [";
    std::string corners_y_str = "corners_y: [";
    std::string corners_z_str = "corners_z: [";

    double min_x_corner = std::numeric_limits<double>::max();
    double max_x_corner = std::numeric_limits<double>::lowest();
    double min_y_corner = std::numeric_limits<double>::max();
    double max_y_corner = std::numeric_limits<double>::lowest();
    double min_z_corner = std::numeric_limits<double>::max();
    double max_z_corner = std::numeric_limits<double>::lowest();

    Eigen::Vector3d normal_vector_for_distance_check = principal_components.col(2); // The final, potentially flipped, v_z_pca
    double d_plane_for_distance_check = normal_vector_for_distance_check.dot(centroid.transpose());

    RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "\nDEBUG: Distance of each corner from PCA plane:");

    for (size_t i = 0; i < final_3d_corners_lidar_frame_vec.size(); ++i) {
        const auto& p = final_3d_corners_lidar_frame_vec[i];
        corners_x_str += std::to_string(p.x);
        corners_y_str += std::to_string(p.y);
        corners_z_str += std::to_string(p.z);
        if (i < final_3d_corners_lidar_frame_vec.size() - 1) {
            corners_x_str += ", ";
            corners_y_str += ", ";
            corners_z_str += ", ";
        }

        min_x_corner = std::min(min_x_corner, p.x);
        max_x_corner = std::max(max_x_corner, p.x);
        min_y_corner = std::min(min_y_corner, p.y);
        max_y_corner = std::max(max_y_corner, p.y);
        min_z_corner = std::min(min_z_corner, p.z);
        max_z_corner = std::max(max_z_corner, p.z);

        Eigen::Vector3d corner_vec(p.x, p.y, p.z);
        double distance = normal_vector_for_distance_check.dot(corner_vec) - d_plane_for_distance_check;
        RCLCPP_DEBUG(rclcpp::get_logger("LidarCornerDetectionService"), "  Corner %zu: Distance from plane = %.6f", i, distance);
    }
    corners_x_str += "]";
    corners_y_str += "]";
    corners_z_str += "]";

    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "%s", corners_x_str.c_str());
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "%s", corners_y_str.c_str());
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "%s", corners_z_str.c_str());
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "\nTotal %zu corners detected.", final_3d_corners_lidar_frame_vec.size());

    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "\nDetected Corner Points Coordinate Range (LiDAR Frame):");
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "  X Range: [%.4f, %.4f]", min_x_corner, max_x_corner);
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "  Y Range: [%.4f, %.4f]", min_y_corner, max_y_corner);
    RCLCPP_INFO(rclcpp::get_logger("LidarCornerDetectionService"), "  Z Range: [%.4f, %.4f]", min_z_corner, max_z_corner);

    return final_3d_corners_lidar_frame_vec;
}

// =============================================================================
// ROS2 Service Server Implementation
// =============================================================================
class LidarCornerDetectionService : public rclcpp::Node {
public:
    LidarCornerDetectionService() : Node("lidar_corner_detection_service") {
        // Declare the 'flip_normal_direction' parameter with a default value of false
        this->declare_parameter("flip_normal_direction", false);
        
        // Create the ROS2 service server
        srv_ = this->create_service<sensor_fusion_study_interfaces::srv::Intensity>(
            "detect_lidar_corners", // Service name
            std::bind(&LidarCornerDetectionService::detect_corners_callback, this,
                      std::placeholders::_1, std::placeholders::_2)); // Callback function
        RCLCPP_INFO(this->get_logger(), "Lidar Corner Detection Service Ready.");
    }

private:
    // Service callback function
    void detect_corners_callback(
        const std::shared_ptr<sensor_fusion_study_interfaces::srv::Intensity::Request> request,
        std::shared_ptr<sensor_fusion_study_interfaces::srv::Intensity::Response> response) {
        RCLCPP_INFO(this->get_logger(), "Received request for corner detection.");

        // Extract parameters from the service request
        std::string pcd_data_ascii = request->pcd_data_ascii;
        int internal_corners_x = request->grid_size_x;
        int internal_corners_y = request->grid_size_y;
        double checker_size_m = request->checker_size_m;
        
        // Get the value of the 'flip_normal_direction' parameter
        bool flip_normal_direction = this->get_parameter("flip_normal_direction").as_bool();

        try {
            // Parse the PCD string into a vector of PointXYZI
            std::vector<PointXYZI> lidar_points_full = parsePCDString(pcd_data_ascii);

            if (lidar_points_full.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to parse PCD data or no points received. Please ensure valid PCD data is sent.");
                response->corners_x.clear();
                response->corners_y.clear();
                response->corners_z.clear();
                return;
            }

            // Estimate chessboard corners using the paper method
            std::vector<PointXYZI> final_3d_corners_lidar_frame = estimateChessboardCornersPaperMethod(
                lidar_points_full,
                internal_corners_x,
                internal_corners_y,
                checker_size_m,
                flip_normal_direction
            );

            // Populate the service response with detected corners
            if (!final_3d_corners_lidar_frame.empty()) {
                response->corners_x.reserve(final_3d_corners_lidar_frame.size());
                response->corners_y.reserve(final_3d_corners_lidar_frame.size());
                response->corners_z.reserve(final_3d_corners_lidar_frame.size());
                for (const auto& p : final_3d_corners_lidar_frame) {
                    response->corners_x.push_back(p.x);
                    response->corners_y.push_back(p.y);
                    response->corners_z.push_back(p.z);
                }
                RCLCPP_INFO(this->get_logger(), "Detected %zu corners and sent response.", final_3d_corners_lidar_frame.size());
            } else {
                RCLCPP_WARN(this->get_logger(), "No corners detected after processing. Sending empty response.");
                response->corners_x.clear();
                response->corners_y.clear();
                response->corners_z.clear();
            }

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error during corner detection: %s", e.what());
            response->corners_x.clear();
            response->corners_y.clear();
            response->corners_z.clear();
        }
    }

    rclcpp::Service<sensor_fusion_study_interfaces::srv::Intensity>::SharedPtr srv_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv); // Initialize ROS2
    // Create and spin the LidarCornerDetectionService node
    rclcpp::spin(std::make_shared<LidarCornerDetectionService>());
    rclcpp::shutdown(); // Shutdown ROS2
    return 0;
}