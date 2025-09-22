#ifndef CALC_COMPUTE_HPP
#define CALC_COMPUTE_HPP

#include <vector>
#include <Eigen/Dense>

namespace jmh_utils
{
    void computeTransformSVD(const std::vector<Eigen::Vector3d> &lid_corners, const std::vector<Eigen::Vector3d> &cam_corners,
                             Eigen::MatrixXd &rotation, Eigen::VectorXd &translation);
}
#endif