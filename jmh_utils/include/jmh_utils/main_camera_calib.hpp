#ifndef MAIN_CAMERA_CALIB_HPP
#define MAIN_CAMERA_CALIB_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace jmh_utils
{
    struct boardParameter;
    
    struct resultIntrinsic;

    jmh_utils::resultIntrinsic runCalibrate(const jmh_utils::boardParameter &params,
                                            std::vector<cv::String> all_images);
}
#endif