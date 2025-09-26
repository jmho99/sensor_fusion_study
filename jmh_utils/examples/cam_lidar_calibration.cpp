#include "jmh_utils/jmh_utils.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct RigidResidual
{
    RigidResidual(const Eigen::Vector3d &pref, const Eigen::Vector3d &psrc, double w = 1.0)
        : pref_(pref), psrc_(psrc), w_(w) {}

    template <typename T>
    bool operator()(const T *const angle_axis, const T *const trans, T *residuals) const
    {
        T psrc_T[3] = {T(psrc_(0)), T(psrc_(1)), T(psrc_(2))};
        T Rp[3];
        ceres::AngleAxisRotatePoint(angle_axis, psrc_T, Rp);
        Rp[0] += trans[0];
        Rp[1] += trans[1];
        Rp[2] += trans[2];
        residuals[0] = T(w_) * (T(pref_(0)) - Rp[0]);
        residuals[1] = T(w_) * (T(pref_(1)) - Rp[1]);
        residuals[2] = T(w_) * (T(pref_(2)) - Rp[2]);
        return true;
    }

    const Eigen::Vector3d pref_, psrc_;
    const double w_;
};

bool optimizeRtWithCeres(
    const std::vector<std::vector<Eigen::Vector3d>> &camera_3d_corners,
    const std::vector<std::vector<Eigen::Vector3d>> &lidar_3d_corners,
    const Eigen::Matrix3d &R_init, // <- 네가 평균으로 구한 초기 R
    const Eigen::Vector3d &t_init, // <- 네가 평균으로 구한 초기 t
    Eigen::Matrix3d &R_opt, Eigen::Vector3d &t_opt,
    double *rmse_out = nullptr,
    // 옵션들
    bool use_huber = true,
    double huber_delta = 1.0,
    int max_iters = 100)
{
    if (camera_3d_corners.size() != lidar_3d_corners.size())
        return false;

    // 초기값: 회전은 Angle-Axis로 변환
    double aa[3];
    {
        Eigen::AngleAxisd aa_eig(R_init);
        Eigen::Vector3d v = aa_eig.axis() * aa_eig.angle();
        aa[0] = v.x();
        aa[1] = v.y();
        aa[2] = v.z();
    }
    double trans[3] = {t_init.x(), t_init.y(), t_init.z()};

    ceres::Problem problem;

    size_t total_pairs = 0;
    for (size_t f = 0; f < camera_3d_corners.size(); ++f)
    {
        const auto &refs = camera_3d_corners[f];
        const auto &srcs = lidar_3d_corners[f];
        if (refs.size() != srcs.size() || refs.empty())
            continue;

        for (size_t i = 0; i < refs.size(); ++i)
        {
            ceres::CostFunction *cost =
                new ceres::AutoDiffCostFunction<RigidResidual, 3, 3, 3>(
                    new RigidResidual(refs[i], srcs[i], 1.0) // 필요하면 가중치 조절
                );
            ceres::LossFunction *loss_ptr =
                use_huber ? static_cast<ceres::LossFunction *>(new ceres::HuberLoss(huber_delta))
                          : nullptr;
            problem.AddResidualBlock(cost, loss_ptr, aa, trans);
            ++total_pairs;
        }
    }

    if (total_pairs == 0)
        return false;

    ceres::Solver::Options opts;
    opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    opts.linear_solver_type = ceres::DENSE_QR;
    
    opts.max_num_iterations = 100;
    opts.function_tolerance = 1e-12;
    opts.gradient_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-10;
    opts.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    // 결과 복원
    Eigen::Vector3d aa_vec(aa[0], aa[1], aa[2]);
    double angle = aa_vec.norm();
    R_opt = Eigen::Matrix3d::Identity();
    if (angle > 1e-12)
        R_opt = Eigen::AngleAxisd(angle, aa_vec.normalized()).toRotationMatrix();
    t_opt = Eigen::Vector3d(trans[0], trans[1], trans[2]);

    // RMSE 계산(옵션)
    if (rmse_out)
    {
        double sqerr = 0.0;
        size_t N = 0;
        for (size_t f = 0; f < camera_3d_corners.size(); ++f)
        {
            const auto &refs = camera_3d_corners[f];
            if (f >= lidar_3d_corners.size())
                break;
            const auto &srcs = lidar_3d_corners[f];
            if (refs.size() != srcs.size())
                continue;
            for (size_t k = 0; k < refs.size(); ++k)
            {
                Eigen::Vector3d pred = R_opt * srcs[k] + t_opt;
                sqerr += (refs[k] - pred).squaredNorm();
                ++N;
            }
        }
        *rmse_out = (N ? std::sqrt(sqerr / double(N)) : 0.0);
    }
    return summary.IsSolutionUsable();
}

int main(int argc, char **argv)
{

    std::string pcd_directory = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/c_cam_lidar_calib/pointclouds/";
    std::string image_directory = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/c_cam_lidar_calib/images/";

    std::vector<std::string> all_pcds = jmh_utils::loadFiles(".pcd", pcd_directory);
    std::vector<std::string> all_images = jmh_utils::loadFiles(".png", image_directory);

    jmh_utils::BoardParameter board_params;
    board_params.columns = 5;
    board_params.rows = 7;
    board_params.square_size = 0.1;
    board_params.frame_width = 1920;
    board_params.frame_height = 1200;
    std::vector<std::vector<double>> intrinsic = {{2.3531675448071378e+03, 0.0, 1.0273207575316267e+03},
                                                  {0.0, 2.3586343982493358e+03, 7.6010058320781366e+02},
                                                  {0.0, 0.0, 1.0}};

    std::vector<double> distortion = {-1.1939616059985084e-01, 4.0157218609309486e-01, 1.0824710229080765e-03, -2.3984687380279165e-03, -1.0518135116602727e+00};
    cv::Mat intrinsic_matrix_ = jmh_utils::vector2Mat(intrinsic);
    cv::Mat distortion_coeffs_ = jmh_utils::vector2Mat(distortion);

    std::vector<std::vector<Eigen::Vector3d>> camera_3d_corners = jmh_utils::runCameraPlane(board_params, intrinsic_matrix_, distortion_coeffs_, all_images);

    jmh_utils::ROI_PARAMS roi;
    roi.min_ROI = Eigen::Vector4f(-5.0, -1.1, -0.6, 1.0);
    roi.max_ROI = Eigen::Vector4f(0.0, 0.6, 3.0, 1.0);

    jmh_utils::RANSAC_PARAMS ransac;
    ransac.threshold = 0.02;
    ransac.iterations = 1000;

    jmh_utils::INTENSITY_PARAMS intensity;
    intensity.min_threshold = 1.0;
    intensity.max_threshold = 100000;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> lidar_plane_pcds = jmh_utils::runIntensityLidarPlane(all_pcds, intensity, roi, ransac);

    std::vector<std::vector<Eigen::Vector3d>> lidar_3d_corners;
    for (int i = 0; i < lidar_plane_pcds.size(); i++)
    {
        std::vector<Eigen::Vector4d> lidar_plane_4vector = jmh_utils::convertPcl2Vector(lidar_plane_pcds[i]);

        std::vector<Eigen::Vector4d> lidar_corner_4vector = jmh_utils::estimateChessboardCornersPaperMethod(
            lidar_plane_4vector,
            board_params.columns,    // internal_corners_x
            board_params.rows,       // internal_corners_y
            board_params.square_size // checker_size_m
        );

        std::vector<Eigen::Vector3d> lidar_corner_3vector;
        for (const auto &p : lidar_corner_4vector)
        {
            lidar_corner_3vector.emplace_back(p(0), p(1), p(2));
        }

        lidar_3d_corners.push_back(lidar_corner_3vector);
    }

    std::cout << "Successed [ " << lidar_3d_corners.size() << " ] lidar frames estimate corners" << std::endl;

    std::vector<Eigen::Matrix3d> all_rotation;
    std::vector<Eigen::Vector3d> all_translation;
    for (int i = 0; i < lidar_plane_pcds.size(); i++)
    {
        Eigen::MatrixXd rotation;
        Eigen::VectorXd translation;
        jmh_utils::computeTransformSVD(lidar_3d_corners[i], camera_3d_corners[i], rotation, translation);
        all_rotation.push_back(rotation);
        all_translation.push_back(translation);
    }

    std::string home_dir = std::getenv("HOME");
    std::string file_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data";
    std::filesystem::path file_path = file_dir + "/lid_cam_calib_cpp_result.yaml";
    std::filesystem::create_directories(file_path.parent_path());

    std::ofstream init_output(file_path, std::ios::out | std::ios::trunc);
    init_output.close();

    std::ofstream output(file_path, std::ios::out | std::ios::app);
    for (int i = 0; i < all_translation.size(); i++)
    {

        output << "rotation" << std::endl;
        output << std::fixed << std::setprecision(6) << all_rotation[i] << std::endl;
        output << "translation" << std::endl;
        output << std::fixed << std::setprecision(6) << all_translation[i].transpose() << std::endl;
    }
    output.close();
    //****************************************************************************************************
    Eigen::Matrix3d R_optimized = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_optimized = Eigen::Vector3d::Zero();

    if (all_rotation.empty())
    {
        std::cerr << "No valid frames found for global calibration. Optimization skipped." << std::endl;
    }

    for (const auto &t_val : all_translation)
    {
        t_optimized += t_val.cast<double>();
    }
    t_optimized /= static_cast<double>(all_translation.size());

    Eigen::Quaterniond q_init(0, 0, 0, 0);
    for (const auto &R_val : all_rotation)
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

    std::cout << "Nomalized, mean before LM optimize" << std::endl;
    std::cout << R_optimized << std::endl;
    std::cout << t_optimized << std::endl;

    Eigen::Matrix3d R_init = R_optimized /* 네 평균 R */;
    Eigen::Vector3d t_init = t_optimized /* 네 평균 t */;
    Eigen::Matrix3d R_opt;
    Eigen::Vector3d t_opt;
    double rmse = 0.0;

    bool ok = optimizeRtWithCeres(camera_3d_corners, lidar_3d_corners,
                                  R_init, t_init,
                                  R_opt, t_opt, &rmse,
                                  /*use_huber=*/true, /*delta=*/1.0, /*iters=*/100);

    if (ok)
    {
        std::cout << "R_opt=\n"
                  << R_opt << "\n";
        std::cout << "t_opt= " << t_opt.transpose() << "\n";
        std::cout << "RMSE = " << rmse << "\n";
    }
    else
    {
        std::cerr << "Optimization failed or insufficient correspondences.\n";
    }
    return 0;
}