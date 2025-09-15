#ifndef MAIN_CAMERA_CALIB_HPP
#define MAIN_CAMERA_CALIB_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
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

    jmh_utils::ResultIntrinsic runCalibrate(const jmh_utils::BoardParameter &params,
                                            std::vector<std::string> all_images);

    jmh_utils::ResultUndistort runUndistorted(const int frame_width, const int &frame_height,
                                        const cv::Mat &intinsic, const cv::Mat &distortion,
                                        std::vector<std::string> all_images);
}
#endif