#include <iostream>
#include <Eigen>

void optimizeLM()
{
    std::cout << "4. Global Calibration Optimization (Levenberg-Marquardt)");

    Eigen::Matrix3d R_optimized = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_optimized = Eigen::Vector3d::Zero();

    if (rotations_frame.empty())
    {
        RCLCPP_WARN(this->get_logger(), "No valid frames found for global calibration. Optimization skipped.");
    }
    else
    {
        for (const auto &t_val : translations_frame)
        {
            t_optimized += t_val.cast<double>();
        }
        t_optimized /= static_cast<double>(translations_frame.size());

        Eigen::Quaterniond q_init(0, 0, 0, 0);
        for (const auto &R_val : rotations_frame)
        {
            Eigen::Quaterniond q(R_val.cast<double>()); // Cast to double
            if (q.dot(q_init) < 0)
            {
                q.coeffs() *= -1.0;
            }
            q_init.coeffs() += q.coeffs();
        }
        q_init.normalize();
        R_optimized = q_init.toRotationMatrix();

#ifdef LOOK_DEBUG
        std::stringstream ss_initial_r, ss_initial_t;
        ss_initial_r << R_optimized.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", ""));
        ss_initial_t << t_optimized.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", ""));

        RCLCPP_INFO(this->get_logger(), "Initial R (from averaging SVD): \n%s", ss_initial_r.str().c_str());
        RCLCPP_INFO(this->get_logger(), "Initial t (from averaging SVD): \n%s", ss_initial_t.str().c_str());
#endif

        // Nonlinear Optimization (Levenberg-Marquardt)
        const int max_iterations = 100;
        const double convergence_threshold = 1e-6; // Threshold for parameter change
        double lambda = 1e-3;                      // Initial damping parameter
        double nu = 2.0;                           // Factor for increasing lambda

        Eigen::Matrix3d best_R = R_optimized; // Changed to double
        Eigen::Vector3d best_t = t_optimized; // Changed to double
        double best_error = std::numeric_limits<double>::max();

        for (int iter = 0; iter < max_iterations; ++iter)
        {
            Eigen::MatrixXd J(0, 6); // Jacobian matrix
            Eigen::VectorXd r(0);    // Residual vector

            // Calculate total number of residuals for dynamic resizing
            int total_residuals = 0;
            for (int frame_i = 0; frame_i < frame_count; ++frame_i)
            {
                if (all_corners[frame_i].size() >= 2 && all_corners[frame_i][0].size() == 3 && all_corners[frame_i][1].size() == 3)
                {
                    total_residuals += 3 * 3; // 3 corners * 3 dimensions
                }
            }
            J.resize(total_residuals, 6);
            r.resize(total_residuals);

            int row_idx = 0;
            for (int frame_i = 0; frame_i < frame_count; ++frame_i)
            {
                if (all_corners[frame_i].size() >= 2 && all_corners[frame_i][0].size() == 3 && all_corners[frame_i][1].size() == 3)
                {
                    const std::vector<Eigen::Vector3f> &ref_corners_frame = all_corners[frame_i][0]; // Lidar 0 corners
                    const std::vector<Eigen::Vector3f> &src_corners_frame = all_corners[frame_i][1]; // Lidar 1 corners

                    for (int i = 0; i < 3; ++i)
                    {                                                                      // Iterate over 3 corners
                        const Eigen::Vector3d P_ref = ref_corners_frame[i].cast<double>(); // Cast to double
                        const Eigen::Vector3d P_src = src_corners_frame[i].cast<double>(); // Cast to double

                        // Current transformed point
                        Eigen::Vector3d P_transformed = R_optimized * P_src + t_optimized;

                        // Residual vector for this point
                        Eigen::Vector3d residual_pt = P_ref - P_transformed;
                        r.segment<3>(row_idx) = residual_pt;

                        // Jacobian for this point (3x6 matrix)
                        Eigen::Matrix3d skew_P_transformed_src = Eigen::Matrix3d::Zero();
                        skew_P_transformed_src << 0, -P_transformed.z(), P_transformed.y(),
                            P_transformed.z(), 0, -P_transformed.x(),
                            -P_transformed.y(), P_transformed.x(), 0;

                        // Jacobian block for rotation (3x3)
                        J.block<3, 3>(row_idx, 0) = -skew_P_transformed_src;
                        // Jacobian block for translation (3x3)
                        J.block<3, 3>(row_idx, 3) = -Eigen::Matrix3d::Identity();

                        row_idx += 3;
                    }
                }
            }

            // Calculate current error before update
            double current_total_error = r.norm();
            if (iter == 0)
            {
                best_error = current_total_error;
            }

            // Solve the normal equations: (J^T * J + lambda * I) * delta_params = J^T * r
            Eigen::Matrix<double, 6, 6> Hessian = J.transpose() * J;
            Eigen::Matrix<double, 6, 1> gradient = J.transpose() * r;

            Eigen::Matrix<double, 6, 1> delta_params;
            Eigen::Matrix<double, 6, 6> damped_Hessian = Hessian + lambda * Eigen::Matrix<double, 6, 6>::Identity();
            delta_params = damped_Hessian.ldlt().solve(gradient); // Solve using LDLT decomposition

            Eigen::Vector3d delta_rotation_vec = delta_params.head<3>();    // Already double
            Eigen::Vector3d delta_translation_vec = delta_params.tail<3>(); // Already double

            // Evaluate the new parameters
            Eigen::Matrix3d R_new = Eigen::AngleAxisd(delta_rotation_vec.norm(), delta_rotation_vec.normalized()).toRotationMatrix() * R_optimized;
            Eigen::Vector3d t_new = t_optimized + delta_translation_vec;

            // Calculate error with new parameters
            Eigen::VectorXd r_new(total_residuals);
            int new_row_idx = 0;
            for (int frame_i = 0; frame_i < frame_count; ++frame_i)
            {
                if (all_corners[frame_i].size() >= 2 && all_corners[frame_i][0].size() == 3 && all_corners[frame_i][1].size() == 3)
                {
                    const std::vector<Eigen::Vector3f> &ref_corners_frame = all_corners[frame_i][0];
                    const std::vector<Eigen::Vector3f> &src_corners_frame = all_corners[frame_i][1];
                    for (int i = 0; i < 3; ++i)
                    {
                        const Eigen::Vector3d P_ref = ref_corners_frame[i].cast<double>();
                        const Eigen::Vector3d P_src = src_corners_frame[i].cast<double>();
                        Eigen::Vector3d P_transformed_new = R_new * P_src + t_new;
                        r_new.segment<3>(new_row_idx) = P_ref - P_transformed_new;
                        new_row_idx += 3;
                    }
                }
            }
            double new_total_error = r_new.norm();

            // Levenberg-Marquardt damping update
            // Calculate actual reduction vs. expected reduction
            double actual_reduction = current_total_error * current_total_error - new_total_error * new_total_error;
            double expected_reduction = (gradient.transpose() * delta_params)(0, 0) - 0.5 * (delta_params.transpose() * Hessian * delta_params)(0, 0);

            double gain_ratio = 0.0;
            if (expected_reduction > 1e-9)
            { // Avoid division by zero or very small expected reduction
                gain_ratio = actual_reduction / expected_reduction;
            }
            else
            {
                gain_ratio = (actual_reduction > 0) ? 1.0 : -1.0; // If expected is zero, check if actual improved
            }

#ifdef LOOK_DEBUG
            RCLCPP_INFO(this->get_logger(), "Iteration %d: Current Error = %.6f, New Error = %.6f, Gain Ratio = %.6f, Lambda = %.6f",
                        iter, current_total_error, new_total_error, gain_ratio, lambda);
#endif

            if (gain_ratio > 0)
            { // Actual reduction is positive, step is good
                R_optimized = R_new;
                t_optimized = t_new;
                lambda = std::max(lambda * 0.1, 1e-7); // Decrease lambda
                nu = 2.0;
                if (new_total_error < best_error)
                {
                    best_error = new_total_error;
                    best_R = R_optimized;
                    best_t = t_optimized;
                }
            }
            else
            {                 // Actual reduction is zero or negative, step is bad
                lambda *= nu; // Increase lambda
                nu *= 2.0;
            }

            if (delta_params.norm() < convergence_threshold || lambda > 1e10)
            { // Also add a max lambda to prevent explosion
                RCLCPP_INFO(this->get_logger(), "Step rejected. Increasing lambda to %.6f", lambda);
                RCLCPP_INFO(this->get_logger(), "Optimization converged or lambda exploded.");
                break;
            }
        }
        R_optimized = best_R; // Use the best parameters found
        t_optimized = best_t;

        std::stringstream ss_final_r, ss_final_t;
        ss_final_r << R_optimized.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", ""));
        ss_final_t << t_optimized.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", ""));
        RCLCPP_INFO(this->get_logger(), "\n-----------\nGlobal Calibrated Rotation Matrix (Lidar1 to Lidar0) - Optimized:\n%s", ss_final_r.str().c_str());
        RCLCPP_INFO(this->get_logger(), "Global Calibrated Translation Vector (Lidar1 to Lidar0) - Optimized:\n%s", ss_final_t.str().c_str());
        RCLCPP_INFO(this->get_logger(), "-----------\n");
    }
}