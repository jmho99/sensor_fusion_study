#include "jmh_utils/jmh_utils.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>

struct RigidResidual
{
    RigidResidual(const Eigen::Vector3d &rot_i, const Eigen::Vector3d &rot_l)
        : rot_i_(rot_i), rot_l_(rot_l) {}

    template <typename T>
    bool operator()(const T *const rot_init, T *residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::ColMajor>> R(rot_init);
        // std::cout << R << std::endl;
        Eigen::Matrix<T, 3, 1> Ri = rot_i_.template cast<T>();
        // std::cout << Ri << std::endl;
        Eigen::Matrix<T, 3, 1> Rl = rot_l_.template cast<T>();
        // std::cout << Rl << std::endl;
        Eigen::Matrix<T, 3, 1> Diff = Ri - R * Rl;

        residuals[0] = Diff[0];
        residuals[1] = Diff[1];
        residuals[2] = Diff[2];
        return true;
    }

    const Eigen::Vector3d rot_i_, rot_l_;
};

class SO3MatrixManifold : public ceres::Manifold
{
public:
    // x : 9 doubles (col-major R)
    // delta : 3 doubles (so(3) 회전벡터)
    // x_plus_delta : 9 doubles (col-major)
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Matrix3d> R(x);
        const Eigen::Vector3d w(delta[0], delta[1], delta[2]);

        const double th = w.norm();
        Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
        if (th > 1e-12)
        {
            Eigen::Matrix3d wx;
            wx << 0, -w.z(), w.y(),
                w.z(), 0, -w.x(),
                -w.y(), w.x(), 0;
            dR = Eigen::Matrix3d::Identity() + (std::sin(th) / th) * wx + ((1.0 - std::cos(th)) / (th * th)) * (wx * wx);
        }

        Eigen::Map<Eigen::Matrix3d> Xp(x_plus_delta); // 변수로 바인딩 후 대입
        Xp = dR * R;                                  // 좌곱 업데이트
        return true;
    }

    // 9x3 자코비안 (간단 근사)
    bool PlusJacobian(const double * /*x*/, double *jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::ColMajor>> J(jacobian);
        J.setZero();
        J.block<3, 3>(0, 0).setIdentity();
        J.block<3, 3>(3, 0).setIdentity();
        J.block<3, 3>(6, 0).setIdentity();
        return true;
    }

    // y_minus_x ≈ log(Ry * Rx^T) (3x1)
    bool Minus(const double *y, const double *x, double *y_minus_x) const override
    {
        Eigen::Map<const Eigen::Matrix3d> Ry(y);
        Eigen::Map<const Eigen::Matrix3d> Rx(x);
        Eigen::Matrix3d Re = Ry * Rx.transpose();
        Eigen::AngleAxisd aa(Re);
        Eigen::Vector3d r = aa.angle() * aa.axis();
        y_minus_x[0] = r.x();
        y_minus_x[1] = r.y();
        y_minus_x[2] = r.z();
        return true;
    }

    // 3x9 자코비안 (간단 근사)
    bool MinusJacobian(const double * /*x*/, double *jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::ColMajor>> J(jacobian);
        J.setZero();
        J.block<3, 3>(0, 0).setIdentity();
        J.block<3, 3>(0, 3).setIdentity();
        J.block<3, 3>(0, 6).setIdentity();
        return true;
    }

    int AmbientSize() const override { return 9; } // R (3x3)
    int TangentSize() const override { return 3; } // so(3)
};

bool optimizeRtWithCeres(
    const std::vector<Eigen::Vector3d> &imu_data,
    const std::vector<Eigen::Vector3d> &lidar_data,
    Eigen::Matrix3d &R_init)
{
    if (imu_data.size() != lidar_data.size())
        return false;
    /*
        std::cout << R_init << std::endl;
        std::cout << R_init.data() << std::endl;
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R(R_init.data());
        std::cout << R << std::endl;
    */
    ceres::Problem problem;
    problem.AddParameterBlock(R_init.data(), 9, new SO3MatrixManifold());
    for (size_t f = 0; f < imu_data.size(); ++f)
    {
        const auto &rot_i = imu_data[f] * 3.141592 / 180.0;
        const auto &rot_l = lidar_data[f] * 3.141592 / 180.0;

        if (rot_i.size() != rot_l.size())
            continue;

        ceres::CostFunction *cost =
            new ceres::AutoDiffCostFunction<RigidResidual, 3, 9>(
                new RigidResidual(rot_i, rot_l));

        problem.AddResidualBlock(cost, nullptr, R_init.data());
    }

    ceres::Solver::Options opts;
    opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    opts.linear_solver_type = ceres::DENSE_QR;

    opts.max_num_iterations = 100;
    opts.function_tolerance = 1e-12;
    opts.gradient_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-10;
    opts.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    return summary.IsSolutionUsable();
}

void loadCSVAndConvertVector(const std::filesystem::path &file_path,
                             std::vector<Eigen::Vector3d> &data_vec)
{
    if (!std::filesystem::exists(file_path))
    {
        std::cerr << "File not found: " << file_path << std::endl;
        return;
    }

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Can't file" << std::endl;
        return;
    }

    std::string csv_line;
    while (std::getline(file, csv_line))
    {
        if (csv_line.empty())
            continue;

        std::stringstream data_string(csv_line);
        std::string cell;
        std::vector<double> values;
        while (std::getline(data_string, cell, ','))
        {
            try
            {
                values.push_back(std::stod(cell));
            }
            catch (...)
            {
                values.clear();
                break;
            }
        }

        if (values.size() == 3)
        {
            data_vec.emplace_back(values[0], values[1], values[2]);
        }
    }
}

int main(int argc, char **argv)
{
    std::string home_dir = std::getenv("HOME");
    std::string file_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_test/data";
    std::filesystem::path imu_file_path = file_dir + "/imu_rotation.csv";
    std::filesystem::path lidar_file_path = file_dir + "/lidar_rotation.csv";

    std::vector<Eigen::Vector3d> imu_rotation;
    std::vector<Eigen::Vector3d> lidar_rotation;

    std::cout << "convert vector" << std::endl;
    loadCSVAndConvertVector(imu_file_path, imu_rotation);
    loadCSVAndConvertVector(lidar_file_path, lidar_rotation);
    std::cout << "vector size " << imu_rotation.size() << " , " << lidar_rotation.size() << std::endl;
    Eigen::Matrix3d rotation_init;
    rotation_init << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;

    std::cout << "start optimize" << std::endl;
    bool ok = optimizeRtWithCeres(imu_rotation, lidar_rotation, rotation_init);

    if (ok)
    {
        std::cout << "Initial rotation =\n"
                  << rotation_init << "\n";
    }
    else
    {
        std::cerr << "Optimization failed.\n";
    }
    return 0;
}