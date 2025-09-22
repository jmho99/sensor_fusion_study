#ifndef MAIN_CAMERA_CALIB_HPP
#define MAIN_CAMERA_CALIB_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include "jmh_utils/serve_load.hpp"

namespace jmh_utils
{
    struct BoardParameter
    {
        int columns, rows, frame_width, frame_height;
        float square_size;
    };

    struct ResultIntrinsic
    {
        double rms;
        cv::Mat intrinsic_mat, distortion_coeffs;
        std::vector<cv::Mat> visualize_corners;
        std::vector<int> successed_index;
    };

    struct ResultUndistort
    {
        std::vector<cv::Mat> visualize_undistort;
        std::vector<int> undistroted_index;
    };

    struct FindCorners
    {
        std::vector<std::vector<cv::Point2f>> image_corners;
        std::vector<std::vector<cv::Point3f>> object_points;
        std::vector<cv::Mat> visual_corners;
        std::vector<int> successed_index;
    };

    struct ResultRmse
    {
        std::vector<double> error_each_frame;
        double rmse_overall;
    };

    jmh_utils::ResultIntrinsic runCalibrate(const jmh_utils::BoardParameter &params,
                                            std::vector<std::string> all_images);

    jmh_utils::ResultUndistort runUndistorted(const int frame_width, const int &frame_height,
                                              const cv::Mat &intinsic, const cv::Mat &distortion,
                                              std::vector<std::string> all_images);

    jmh_utils::ResultRmse runRMSE(const jmh_utils::BoardParameter &params,
                                  const cv::Mat &intinsic, const cv::Mat &distortion,
                                  std::vector<std::string> all_images);

    std::vector<std::vector<Eigen::Vector3d>> runCameraPlane(const jmh_utils::BoardParameter &params,
                                                const cv::Mat &intinsic, const cv::Mat &distortion,
                                                std::vector<std::string> all_images);

    static jmh_utils::FindCorners findCorners(const jmh_utils::BoardParameter &params,
                                              std::vector<std::string> all_images);
}
#endif