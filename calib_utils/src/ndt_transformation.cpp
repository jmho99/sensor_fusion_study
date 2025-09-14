#include "calib_utils/ndt_transformation.hpp"

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
                                                     float leaf)
    {
        if (leaf <= 0.f)
            return in;
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(in);
        vg.setLeafSize(leaf, leaf, leaf);
        auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        vg.filter(*out);
        return out;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcdCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr in,
                                                    Eigen::Vector4f &centroid_out)
    {
        pcl::compute3DCentroid(*in, centroid_out);
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 1>(0, 3) = -centroid_out.head<3>();
        auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::transformPointCloud(*in, *out, T);
        return out;
    }

    Eigen::Matrix3f orthonormalizedRotation(const Eigen::Matrix3f &R)
    {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V = svd.matrixV();
        Eigen::Matrix3f Rn = U * V.transpose();
        // 보정 후 det가 -1이면 마지막 축 반전
        if (Rn.determinant() < 0)
        {
            U.col(2) *= -1.0f;
            Rn = U * V.transpose();
        }
        return Rn;
    }

    Eigen::Matrix4f solvendt(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                             const Eigen::VectorXf &ndt_param)
    {
        // NDT 세팅
        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
        ndt.setResolution(ndt_param[0]);
        ndt.setStepSize(ndt_param[1]);
        ndt.setTransformationEpsilon(ndt_param[2]);
        ndt.setMaximumIterations(static_cast<int>(ndt_param[3]));
        ndt.setInputSource(source); // cur
        ndt.setInputTarget(target); // prev

        Eigen::AngleAxisf init_rotation(0.0, Eigen::Vector3f::UnitZ());
        Eigen::Translation3f init_translation(0.0, 0.0, 0.0);
        Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

        pcl::PointCloud<pcl::PointXYZ> aligned;
        ndt.align(aligned, init_guess);

        Eigen::Matrix4f T = ndt.getFinalTransformation();

        return T;
    }

    Eigen::VectorXf ndtRotation(pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                                const float voxel_leaf,
                                const Eigen::VectorXf &ndt_param,
                                const std::string &result_type)
    {
        auto source_vox = calib_utils::voxelizedPcd(source, voxel_leaf);
        auto target_vox = calib_utils::voxelizedPcd(target, voxel_leaf);

        Eigen::Vector4f c_source, c_target;
        auto source_cent = calib_utils::pcdCentroid(source_vox, c_source);
        auto target_cent = calib_utils::pcdCentroid(target_vox, c_target);

        auto T = calib_utils::solvendt(source_cent, target_cent, ndt_param);

        Eigen::Matrix3f R = T.block<3, 3>(0, 0);
        R = calib_utils::orthonormalizedRotation(R);

        Eigen::VectorXf res_xyz;

        if (result_type == "quarternion")
        {
            // 회전(Qurternion)
            Eigen::Quaternionf q(R);

            res_xyz.resize(4);
            res_xyz[0] = q.x();
            res_xyz[1] = q.y();
            res_xyz[2] = q.z();
            res_xyz[3] = q.w();
        }

        else if (result_type == "radian")
        {
            // 회전(Radian)
            double y_rad = std::asin(-R(2, 0));
            double z_rad = std::atan2(R(1, 0), R(0, 0));
            double x_rad = std::atan2(R(2, 1), R(2, 2));

            Eigen::Vector3f xyz_rad;
            xyz_rad[0] = x_rad;
            xyz_rad[1] = y_rad;
            xyz_rad[2] = z_rad;

            res_xyz.resize(3);
            res_xyz[0] = xyz_rad[2];
            res_xyz[1] = xyz_rad[1];
            res_xyz[2] = xyz_rad[0];
        }

        else if (result_type == "degree")
        {
            // 회전(Degree)
            double y_rad = std::asin(-R(2, 0));
            double z_rad = std::atan2(R(1, 0), R(0, 0));
            double x_rad = std::atan2(R(2, 1), R(2, 2));

            Eigen::Vector3f xyz_rad;
            xyz_rad[0] = x_rad;
            xyz_rad[1] = y_rad;
            xyz_rad[2] = z_rad;

            double PI = 3.14159265358979;
            Eigen::Vector3f xyz_deg = xyz_rad * (180.0 / PI);

            res_xyz.resize(3);
            res_xyz[0] = xyz_deg[0];
            res_xyz[1] = xyz_deg[1];
            res_xyz[2] = xyz_deg[2];
        }

        return res_xyz;
    }
}