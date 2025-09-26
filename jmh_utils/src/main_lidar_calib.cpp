#include "jmh_utils/main_lidar_calib.hpp"

#include "jmh_utils/calc_convert.hpp"
#include "jmh_utils/serve_load.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

namespace jmh_utils
{
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> runIntensityLidarPlane(std::vector<std::string> all_pointclouds,
                                                                             const jmh_utils::INTENSITY_PARAMS &intensity,
                                                                             const jmh_utils::ROI_PARAMS &ROI,
                                                                             const jmh_utils::RANSAC_PARAMS &RANSAC)
    {
        std::cout << "Begin detecting checkerboard corners using intensity in the lidar pointclouds" << std::endl;
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> all_cloud_planes;

        for (int frame_index = 0; frame_index < all_pointclouds.size(); frame_index++)
        {
            const auto &frame_pcd = all_pointclouds[frame_index];
            const auto &frame_num = jmh_utils::extractNumber(frame_pcd);

            pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::io::loadPCDFile<pcl::PointXYZI>(frame_pcd, *current_cloud);

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_intensity(new pcl::PointCloud<pcl::PointXYZI>);

            pcl::PassThrough<pcl::PointXYZI> pass_intensity;
            pass_intensity.setInputCloud(current_cloud);
            pass_intensity.setFilterFieldName("intensity");
            // threshold 설정
            pass_intensity.setFilterLimits(intensity.min_threshold, intensity.max_threshold);
            pass_intensity.filter(*cloud_filtered_intensity);

            if (cloud_filtered_intensity->empty())
            {
                std::cout << "Cannot intensity filter [ " << frame_num << " ] pointcloud!!! Check file." << std::endl;
                continue;
            }

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::CropBox<pcl::PointXYZI> crop;
            crop.setInputCloud(cloud_filtered_intensity);
            // 파라미터로 설정된 ROI 제한 사용
            crop.setMin(ROI.min_ROI);
            crop.setMax(ROI.max_ROI);
            crop.filter(*cloud_roi);

            if (cloud_roi->empty())
            {
                std::cout << "Cannot set ROI [ " << frame_num << " ] pointcloud!!! Check file." << std::endl;
                continue;
            }

            pcl::SACSegmentation<pcl::PointXYZI> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            // 파라미터로 설정된 RANSAC 값 사용
            seg.setDistanceThreshold(RANSAC.threshold);
            seg.setMaxIterations(RANSAC.iterations);

            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

            seg.setInputCloud(cloud_roi);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.empty())
            {
                std::cout << "Cannot RANSAC [ " << frame_num << " ] pointcloud!!! Check file." << std::endl;
                continue;
            }

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::ExtractIndices<pcl::PointXYZI> extract;
            extract.setInputCloud(cloud_roi);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*cloud_plane);

            if (cloud_plane->empty())
            {
                std::cout << "Cannot extract [ " << frame_num << " ] pointcloud!!! Check file." << std::endl;
                continue;
            }

            all_cloud_planes.push_back(cloud_plane);
            std::cout << "Detected [ " << frame_num  << " ] [ " << cloud_plane->points.size() << " ] frames detecting planes" << std::endl;
        }

        std::cout << "Succesed [ " << all_cloud_planes.size() << " ] frames detecting planes" << std::endl;
        std::cout << "End detecting checkerboard corners using intensity in the lidar pointclouds" << std::endl;
        return all_cloud_planes;
    }
}