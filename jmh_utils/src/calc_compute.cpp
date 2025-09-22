#include "jmh_utils/calc_compute.hpp"

#include <vector>
#include <Eigen/Dense>

namespace jmh_utils
{
    void computeTransformSVD(const std::vector<Eigen::Vector3d> &lid_corners, const std::vector<Eigen::Vector3d> &cam_corners,
                             Eigen::MatrixXd &rotation, Eigen::VectorXd &translation)
    {
        Eigen::Vector3d lid_corners_center = Eigen::Vector3d::Zero();
        Eigen::Vector3d cam_corners_center = Eigen::Vector3d::Zero();

        for (size_t i = 0; i < lid_corners.size(); ++i)
        {
            lid_corners_center += lid_corners[i];
            cam_corners_center += cam_corners[i];
        }

        lid_corners_center *= (1.0 / static_cast<double>(lid_corners.size()));
        cam_corners_center *= (1.0 / static_cast<double>(cam_corners.size()));

        Eigen::Matrix3d m_n_matrix = Eigen::Matrix3d::Zero();

        for (size_t i = 0; i < lid_corners.size(); ++i)
        {
            Eigen::Vector3d lid_corners_vec(lid_corners[i].x() - lid_corners_center.x(),
                                            lid_corners[i].y() - lid_corners_center.y(),
                                            lid_corners[i].z() - lid_corners_center.z());
            Eigen::Vector3d cam_corners_vec(cam_corners[i].x() - cam_corners_center.x(),
                                            cam_corners[i].y() - cam_corners_center.y(),
                                            cam_corners[i].z() - cam_corners_center.z());
            m_n_matrix += cam_corners_vec * lid_corners_vec.transpose();
        }

        Eigen::MatrixXd U, Sigma, V;
        int m = m_n_matrix.rows();
        int n = m_n_matrix.cols();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(m_n_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU();
        V = svd.matrixV();
        Sigma = Eigen::MatrixXd::Identity(m, n); // Compute Only Rotation
        Sigma(m - 1, n - 1) = (U * V.transpose()).determinant();
        rotation = U * Sigma * V.transpose();
        translation = cam_corners_center - rotation * lid_corners_center;
    }
}