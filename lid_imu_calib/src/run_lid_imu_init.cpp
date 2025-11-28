#include "jmh_utils/jmh_utils.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
// Skew
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3>
Skew(const Eigen::MatrixBase<Derived> &v_in)
{
    using T = typename Derived::Scalar;
    const Eigen::Matrix<T, 3, 1> v = v_in; // 평가
    Eigen::Matrix<T, 3, 3> wx;
    wx << T(0), -v.z(), v.y(),
        v.z(), T(0), -v.x(),
        -v.y(), v.x(), T(0);
    return wx;
}

// ExpSO3
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3>
ExpSO3(const Eigen::MatrixBase<Derived> &w_in)
{
    using T = typename Derived::Scalar;
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    const Eigen::Matrix<T, 3, 1> w = w_in; // 평가
    const T th = w.norm();
    const Mat3 I = Mat3::Identity();

    using std::cos;
    using std::sin; // Jet 대응

    if (th < T(1e-12))
    {
        return I + Skew(w); // 1차 근사
    }
    const Eigen::Matrix<T, 3, 1> a = w / th;
    const Mat3 ax = Skew(a);
    return I + sin(th) * ax + (T(1) - cos(th)) * (ax * ax);
}

// ===================== 기존 이름 유지: 잔차 =====================
struct RigidResidual
{
    RigidResidual(const Eigen::Vector3d &rot_i, const Eigen::Vector3d &rot_l)
        : rot_i_(rot_i), rot_l_(rot_l) {}

    template <typename T>
    bool operator()(const T *const rot_init, T *residuals) const
    {
        // rot_init: so(3) 3원소 → R = Exp(so3)
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> w(rot_init);
        Eigen::Matrix<T, 3, 3> R = ExpSO3(w);

        Eigen::Matrix<T, 3, 1> Ri = rot_i_.template cast<T>();
        Eigen::Matrix<T, 3, 1> Rl = rot_l_.template cast<T>();

        Eigen::Matrix<T, 3, 1> Diff = Ri - R * Rl;

        residuals[0] = Diff[0];
        residuals[1] = Diff[1];
        residuals[2] = Diff[2];
        return true;
    }

    const Eigen::Vector3d rot_i_, rot_l_;
};

// ===================== 기존 이름/시그니처 유지: 최적화 함수 =====================
bool optimizeRtWithCeres(
    const std::vector<Eigen::Vector3d> &imu_data,
    const std::vector<Eigen::Vector3d> &lidar_data,
    Eigen::Matrix3d &R_init)
{
    if (imu_data.size() != lidar_data.size() || imu_data.empty())
        return false;

    // 초기 so(3) 파라미터 (R_init의 로그)
    Eigen::AngleAxisd aa0(R_init);
    Eigen::Vector3d w0 = aa0.angle() * aa0.axis();

    ceres::Problem problem;
    // 파라미터 블록: 3개 (so3), 매니폴드 불필요
    problem.AddParameterBlock(w0.data(), 3);

    for (size_t f = 0; f < imu_data.size(); ++f)
    {
        const auto &rot_i = imu_data[f];
        const auto &rot_l = lidar_data[f];

        if (rot_i.size() != rot_l.size())
            continue;

        ceres::CostFunction *cost =
            new ceres::AutoDiffCostFunction<RigidResidual, 3, 3>(
                new RigidResidual(rot_i, rot_l));

        // (선택) 로버스트 손실
        // ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        // problem.AddResidualBlock(cost, loss, w0.data());
        problem.AddResidualBlock(cost, nullptr, w0.data());
    }

    ceres::Solver::Options opts;
    opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 100;
    opts.function_tolerance = 1e-12;
    opts.gradient_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-12;
    opts.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    // 최종 so3 → R 로 되돌려서 R_init에 기록 (in/out 유지)
    Eigen::Matrix3d R_final = ExpSO3(w0);
    R_init = R_final;

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
    std::string file_dir = home_dir + "/ROS2/calib_ws/src/calib_test/data";
    std::filesystem::path imu_file_path = file_dir + "/imu_rotation.csv";
    std::filesystem::path lidar_file_path = file_dir + "/lidar_rotation.csv";

    std::vector<Eigen::Vector3d> imu_rotation;
    std::vector<Eigen::Vector3d> lidar_rotation;

    std::cout << "convert vector" << std::endl;
    loadCSVAndConvertVector(imu_file_path, imu_rotation);
    loadCSVAndConvertVector(lidar_file_path, lidar_rotation);
    std::cout << "vector size " << imu_rotation.size() << " , " << lidar_rotation.size() << std::endl;
    Eigen::Matrix3d rotation_init;
    rotation_init << 1.0, 0, 0,
        0, 1.0, 0,
        0, 0, 1.0;

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