#ifndef MAIN_MIDAR_CALIB_HPP
#define MAIN_MIDAR_CALIB_HPP

#include "jmh_utils/calc_convert.hpp"
#include "jmh_utils/serve_load.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

namespace jmh_utils
{
    struct ROI_PARAMS
    {
        Eigen::Vector4f min_ROI;
        Eigen::Vector4f max_ROI;
    };

    struct RANSAC_PARAMS
    {
        double threshold;
        int iterations;
    };

    struct INTENSITY_PARAMS
    {
        double min_threshold;
        double max_threshold;
    };

    struct PLANE_RESULT
    {
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> all_cloud_planes;
        std::vector<Eigen::Vector4d> lidar_plane_abcd;
        std::vector<Eigen::Vector3d> lidar_plane_centroid;
        std::vector<int> lidar_facing_flags;
    };

    jmh_utils::PLANE_RESULT runIntensityLidarPlane(std::vector<std::string> all_pointclouds,
                                                                             const jmh_utils::INTENSITY_PARAMS &intensity,
                                                                             const jmh_utils::ROI_PARAMS &ROI,
                                                                             const jmh_utils::RANSAC_PARAMS &RANSAC);
}

#endif