#include "jmh_utils/jmh_utils.hpp"

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

    std::vector<Eigen::MatrixXd> all_rotation;
    std::vector<Eigen::VectorXd> all_translation;
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
    std::filesystem::path file_path = "/lid_cam_calib_cpp_result.yaml";
    std::filesystem::create_directories(file_path.parent_path());

    std::ofstream init_output(file_path, std::ios::out | std::ios::trunc);
    init_output.close();

    for (int i = 0; i < all_translation.size(); i++)
    {
        std::ofstream output(file_path, std::ios::out | std::ios::app);
        output << "rotation" << std::endl;
        output << std::fixed << std::setprecision(6) << all_rotation[i] << std::endl;
        output << "translation" << std::endl;
        output << std::fixed << std::setprecision(6) << all_translation[i].transpose() << std::endl;
    }
    return 0;
}