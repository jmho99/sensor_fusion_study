#ifndef SERVE_CONVERT_HPP
#define SERVE_CONVERT_HPP

#include <vector>
#include <opencv2/opencv.hpp>

namespace jmh_utils
{
    cv::Mat vector2Mat(const std::vector<std::vector<double>> &input);

    cv::Mat vector2Mat(const std::vector<double> &input);
}

#endif