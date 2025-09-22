#ifndef CALC_CONVERT_HPP
#define CALC_CONVERT_HPP

#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace jmh_utils
{
    cv::Mat vector2Mat(const std::vector<std::vector<double>> &input);

    cv::Mat vector2Mat(const std::vector<double> &input);

    std::vector<Eigen::Vector4d> convertPcl2Vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pointcloud);
}

#endif