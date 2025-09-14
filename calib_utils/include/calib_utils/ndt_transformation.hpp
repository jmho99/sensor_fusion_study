#ifndef NDT_TRANSFORMATION_HPP
#define NDT_TRANSFORMATION_HPP

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

namespace calib_utils
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxelizedPcd(pcl::PointCloud<pcl::PointXYZ>::Ptr in,
                                                     float leaf);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcdCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr in,
                                                    Eigen::Vector4f &centroid_out);

    Eigen::Matrix3f orthonormalizedRotation(const Eigen::Matrix3f &R);

    Eigen::Matrix4f solvendt(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                             const Eigen::VectorXf &ndt_param);

    Eigen::VectorXf ndtRotation(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                                const float voxel_leaf,
                                const Eigen::VectorXf &ndt_param,
                                const std::string &result_type);
}
#endif