#ifndef MAIN_FUSION_VISUALIZATION_HPP
#define MAIN_FUSION_VISUALIZATION_HPP

#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace jmh_utils
{
    cv::Mat resultCamLidarFusion(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_in_cam,
                                 const cv::Mat &image_in, cv::Mat &image_out,
                                 cv::Mat intrinsic, cv::Mat distortion,
                                 const bool use_undistorte_image);
}
#endif