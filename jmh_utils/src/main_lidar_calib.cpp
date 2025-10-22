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
    // === [ADD] 평면 법선 정규화 및 d 정규화 ===
    // 입력: (a,b,c,d). 출력: 노멀=(a,b,c)를 단위화; d도 같은 비율로 나눔
    static Eigen::Vector4d normalize_plane(const Eigen::Vector4d &abcd)
    {
        Eigen::Vector3d n = abcd.head<3>();
        double nn = n.norm();
        if (nn <= 1e-12)
            return abcd;
        Eigen::Vector4d out = abcd;
        out.head<3>() /= nn;
        out[3] /= nn;
        return out;
    }

    // === [ADD] 평면이 원점을 향하는지 판정 ===
    // 규칙: (centroid · n) < 0  → "원점을 향함(0)" / 그렇지 않으면 1
    static int facing_origin_label(const Eigen::Vector3d &n_unit,
                                          const Eigen::Vector3d &centroid)
    {
        return (centroid.dot(n_unit) < 0.0) ? 0 : 1;
    }

    static Eigen::Vector3d centroid_from_cloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud)
    {
        if (!cloud || cloud->empty())
            return Eigen::Vector3d(0, 0, 0);
        Eigen::Vector3d c(0, 0, 0);
        for (const auto &p : cloud->points)
            c += Eigen::Vector3d(p.x, p.y, p.z);
        c /= static_cast<double>(cloud->points.size());
        return c;
    }

    jmh_utils::PLANE_RESULT runIntensityLidarPlane(std::vector<std::string> all_pointclouds,
                                                   const jmh_utils::INTENSITY_PARAMS &intensity,
                                                   const jmh_utils::ROI_PARAMS &ROI,
                                                   const jmh_utils::RANSAC_PARAMS &RANSAC)
    {
        std::cout << "Begin detecting checkerboard corners using intensity in the lidar pointclouds" << std::endl;
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> all_cloud_planes;
        std::vector<Eigen::Vector4d> lidar_plane_abcd;
        std::vector<Eigen::Vector3d> lidar_plane_centroid;
        std::vector<int> lidar_facing_flags;

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

            // (a,b,c,d) → 단위 노멀/정규화
            Eigen::Vector4d plane_function(coefficients->values[0],
                                           coefficients->values[1],
                                           coefficients->values[2],
                                           coefficients->values[3]);
            plane_function = normalize_plane(plane_function);

            // 인라이어 평면 포인트의 센트로이드
            Eigen::Vector3d plane_centroid = centroid_from_cloud(cloud_plane);

            // "원점 향함(0) / 아니면 1"
            Eigen::Vector3d normal_vec_dir = plane_centroid.head<3>();
            int lidar_label = facing_origin_label(normal_vec_dir, plane_centroid);

            // 이번 프레임용 캐시(나중에 R,t 후 카메라 좌표계 라벨링에 사용)
            lidar_plane_abcd.push_back(plane_function);
            lidar_plane_centroid.push_back(plane_centroid);
            lidar_facing_flags.push_back(lidar_label);

            all_cloud_planes.push_back(cloud_plane);
            std::cout << "Detected [ " << frame_num << " ] [ " << cloud_plane->points.size() << " ] frames detecting planes" << std::endl;
        }


        jmh_utils::PLANE_RESULT result;
        result.all_cloud_planes = all_cloud_planes;
        result.lidar_plane_abcd = lidar_plane_abcd;
        result.lidar_plane_centroid = lidar_plane_centroid;
        result.lidar_facing_flags = lidar_facing_flags;

        std::cout << "Succesed [ " << all_cloud_planes.size() << " ] frames detecting planes" << std::endl;
        std::cout << "End detecting checkerboard corners using intensity in the lidar pointclouds" << std::endl;
        return result;
    }
}