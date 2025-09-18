#include "jmh_utils/jmh_utils.hpp"

int main(int argc, char **argv)
{

    std::string pcd_path = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/c_cam_lidar_calib/pointclouds/pcd_0.pcd";
    std::string image_path = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/c_cam_lidar_calib/images/img_0.png";
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_{new pcl::PointCloud<pcl::PointXYZI>};
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *current_cloud_);
    cv::Mat current_frame_ = cv::imread(image_path, cv::IMREAD_COLOR);

    std::vector<std::vector<double>> intrinsic = {{2.3531675448071378e+03, 0.0, 1.0273207575316267e+03},
                                                  {0.0, 2.3586343982493358e+03, 7.6010058320781366e+02},
                                                  {0.0, 0.0, 1.0}};

    std::vector<double> distortion = {-1.1939616059985084e-01, 4.0157218609309486e-01, 1.0824710229080765e-03, -2.3984687380279165e-03, -1.0518135116602727e+00};

    std::array<std::array<double, 3>, 3> rotation = {{{1.0393276792571727e-02, 9.9972672212714619e-01, -2.0939457070910297e-02},
                                                      {-3.0532002083263361e-03, -2.0908762979334795e-02, -9.9977672587391331e-01},
                                                      {-9.9994132716174955e-01, 1.0454888597469817e-02, 2.8350552752334818e-03}}};

    std::array<double, 3> translation = {1.4641477985241055e-02, -1.0346761941224500e-01, -6.9225398306809405e-02};

    std::vector<std::vector<double>> extrinsic = {{rotation[0][0], rotation[0][1], rotation[0][2], translation[0]},
                                                  {rotation[1][0], rotation[1][1], rotation[1][2], translation[1]},
                                                  {rotation[2][0], rotation[2][1], rotation[2][2], translation[2]},
                                                  {0.0, 0.0, 0.0, 1.0}};

    cv::Mat intrinsic_matrix_ = jmh_utils::vector2Mat(intrinsic);
    cv::Mat distortion_coeffs_ = jmh_utils::vector2Mat(distortion);
    cv::Mat lidar2cam_T = jmh_utils::vector2Mat(extrinsic);

    cv::Mat lidar2cam_R_;
    lidar2cam_R_.create(3, 3, CV_64F);
    cv::Mat lidar2cam_t_;
    lidar2cam_t_.create(3, 1, CV_64F);

    for (int i = 0; i < lidar2cam_T.rows; i++)
    {
        for (int j = 0; j < lidar2cam_T.cols; j++)
        {
            double each = lidar2cam_T.at<double>(i, j);
            if (j < 3)
            {
                lidar2cam_R_.at<double>(i, j) = each;
            }
            else if (j >= 3)
            {
                lidar2cam_t_.at<double>(i, 0) = each;
            }
            if (i >= 4)
                continue;
        }
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformed_cloud->points.reserve(current_cloud_->points.size());

    for (const auto &pt_lidar : current_cloud_->points)
    {
        cv::Mat pt_mat = (cv::Mat_<double>(3, 1) << pt_lidar.x, pt_lidar.y, pt_lidar.z);
        cv::Mat pt_transformed = lidar2cam_R_ * pt_mat + lidar2cam_t_;

        pcl::PointXYZI p_transformed;
        p_transformed.x = pt_transformed.at<double>(0);
        p_transformed.y = pt_transformed.at<double>(1);
        p_transformed.z = pt_transformed.at<double>(2);
        p_transformed.intensity = pt_lidar.intensity;
        transformed_cloud->points.push_back(p_transformed);
    }

    cv::Mat image_with_lidar_projection_ = jmh_utils::resultCamLidarFusion(transformed_cloud, current_frame_, image_with_lidar_projection_,
                                                                           intrinsic_matrix_, distortion_coeffs_, false);

    cv::namedWindow("Lidar Projected on Image", cv::WINDOW_NORMAL); // Uncommented for display
    cv::resizeWindow("Lidar Projected on Image", 640, 480);
    cv::imshow("Lidar Projected on Image", image_with_lidar_projection_);
    cv::waitKey(0); // Keep window open briefly

    return 0;
}