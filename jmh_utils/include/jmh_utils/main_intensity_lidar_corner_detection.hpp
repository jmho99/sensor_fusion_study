#ifndef MAIN_INTENSITY_LIDAR_CORNER_DETECTION_HPP
#define MAIN_INTENSITY_LIDAR_CORNER_DETECTION_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <numeric>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "jmh_utils/calc_compute.hpp"

namespace jmh_utils
{
    // 사용자 정의 포인트 구조체: XYZ 좌표와 강도(Intensity)를 포함
    struct DoubleXYZI
    {
        double x, y, z, intensity;
    };

    // =============================================================================
    // 1. PCD 데이터 파싱 함수 (ASCII 문자열에서)
    // =============================================================================
    // ASCII 형식의 PCD(Point Cloud Data) 문자열을 파싱하여 jmh_utils::DoubleXYZI 벡터로 변환합니다.
    // 헤더 정보와 XYZI 데이터를 추출합니다.
    std::vector<jmh_utils::DoubleXYZI> parsePCDString(const std::string &pcd_string);

    // =============================================================================
    // 3. 강도 기반 흑백 분류 (회색 영역 포함)
    // =============================================================================
    // 강도 값을 흑색(0), 백색(1), 또는 회색 영역(-1)으로 분류합니다.
    // epsilon_g는 회색 영역 정의를 위한 상수입니다.
    // 반환: 분류된 색상 벡터, 하한 임계값(tau_l), 상한 임계값(tau_h)
    std::tuple<Eigen::VectorXi, double, double> classifyIntensityColor(const Eigen::VectorXd &intensities, double epsilon_g = 4.0);

    // =============================================================================
    // 4. 체커보드 모델 정의 및 패턴 색상 확인
    // =============================================================================
    // 주어진 (x, y) 좌표에서 체커보드 패턴의 색상(흑색:0, 백색:1)을 결정합니다.
    // 전체 체커보드의 좌측 하단 (0,0) 칸이 백색이라고 가정합니다.
    // grid_size_x_squares: 가로 사각형 수, grid_size_y_squares: 세로 사각형 수, checker_size_m: 각 사각형의 크기(미터)
    // 반환: 흑색(0), 백색(1), 또는 체커보드 범위 밖(-1)
    int getCheckerboardPatternColor(double x, double y, int grid_size_x_squares, int grid_size_y_squares, double checker_size_m);

    // =============================================================================
    // 5. 비용 함수 (Cost Function)
    // =============================================================================
    // 체커보드 자세 최적화를 위한 비용 함수입니다.
    // params: (tx, ty, theta_z) - 평면 상의 2D 이동 및 회전 각도
    // points_in_pca_plane_2d: PCA 변환된 2D 포인트 배열
    // classified_colors: 분류된 색상 배열 (0:흑색, 1:백색, -1:회색)
    // grid_size_x_squares: 가로 사각형 수, grid_size_y_squares: 세로 사각형 수, checker_size_m: 각 사각형의 크기(미터)
    // 반환: 총 비용
    double costFunction(const Eigen::Vector3d &params,
                        const Eigen::MatrixXd &points_in_pca_plane_2d,
                        const Eigen::VectorXi &classified_colors,
                        int grid_size_x_squares, int grid_size_y_squares, double checker_size_m);

    // =============================================================================
    // 6. 논문 방식의 코너 검출 메인 함수
    // =============================================================================
    // LiDAR 포인트로부터 체커보드 코너를 추정합니다.
    // lidar_points_full_vec: LiDAR 포인트 (x, y, z, 강도) 벡터
    // internal_corners_x: 가로 내부 코너 수 (예: 7x8 보드의 경우 6)
    // internal_corners_y: 세로 내부 코너 수 (예: 7x8 보드의 경우 7)
    // checker_size_m: 각 체커 사각형의 크기(미터)
    // flip_normal_direction: true이면 PCA Z축(법선 벡터)의 방향을 뒤집습니다.
    // 반환: 원본 LiDAR 프레임에서의 3D 코너 좌표 (jmh_utils::DoubleXYZI 벡터)
    std::vector<Eigen::Vector4d> estimateChessboardCornersPaperMethod(
        const std::vector<Eigen::Vector4d> &lidar_points_full_vec,
        int internal_corners_x, int internal_corners_y, double checker_size_m);

    struct CompactPointResidual
    {
        CompactPointResidual(const Eigen::Vector2d &p, int color,
                             int grid_x, int grid_y, double checker_m,
                             double gate = 1.0, double out_scale = 1.0)
            : p_(p), c_(color), gx_(grid_x), gy_(grid_y),
              s_(checker_m), w_(gate), out_scale_(out_scale), phase_(0) {}

        // phase까지 지정하는 오버로드 (전역 변수 없이 parity 주입)
        CompactPointResidual(const Eigen::Vector2d &p, int color,
                             int grid_x, int grid_y, double checker_m,
                             double gate, double out_scale, int phase)
            : p_(p), c_(color), gx_(grid_x), gy_(grid_y),
              s_(checker_m), w_(gate), out_scale_(out_scale), phase_(phase) {}

        // params = [tx, ty, theta]
        bool operator()(const double *params, double *r) const
        {
            const double tx = params[0], ty = params[1], th = params[2];
            const double cs = std::cos(th), sn = std::sin(th);

            // q = R^T (p - t)
            const double px = p_.x() - tx;
            const double py = p_.y() - ty;
            const double x = cs * px + sn * py;
            const double y = -sn * px + cs * py;

            const double W = gx_ * s_, H = gy_ * s_;
            const double hw = 0.5 * W, hh = 0.5 * H;

            // 보드 경계 거리 + 내부여부
            double dx = 0.0, dy = 0.0;
            if (x < -hw)
                dx = (-hw - x);
            else if (x > hw)
                dx = (x - hw);
            if (y < -hh)
                dy = (-hh - y);
            else if (y > hh)
                dy = (y - hh);
            const bool inside = (dx == 0.0 && dy == 0.0);

            // 강한 OOB 페널티 (기존 스타일 복원)
            const double OUT_SCALE = std::max(W, H) * 10.0;

            double penalty = 0.0;

            if (c_ < 0)
            {
                penalty = 0.0; // gate-out
            }
            else if (!inside)
            {
                penalty = (dx + dy) * OUT_SCALE; // [1 - f_in] * f_d(p,G)
            }
            else
            {
                // 보드 안: 색상 및 타일 경계 거리
                const double X = x + hw, Y = y + hh;
                if (X < 0.0 || Y < 0.0 || X > W || Y > H)
                {
                    penalty = (dx + dy) * OUT_SCALE; // 경계 수치 보호
                }
                else
                {
                    const int ix = (int)std::floor(X / s_);
                    const int iy = (int)std::floor(Y / s_);
                    const int c_hat = ((ix + iy + phase_) & 1) ? 1 : 0; // parity 적용(전역 無)

                    if (c_ != c_hat)
                    {
                        const double lx = std::fmod(std::fabs(X), s_);
                        const double ly = std::fmod(std::fabs(Y), s_);
                        const double dx_edge = std::min(lx, s_ - lx);
                        const double dy_edge = std::min(ly, s_ - ly);
                        const double d_edge = std::min(dx_edge, dy_edge); // f_d(p, V)
                        penalty = d_edge;                                 // |c - c_hat|=1
                    }
                    else
                    {
                        penalty = 0.0;
                    }
                }
            }

            r[0] = std::sqrt(std::max(0.0, w_ * penalty));
            return true;
        }

        Eigen::Vector2d p_;
        int c_;
        int gx_, gy_;
        double s_, w_, out_scale_;
        int phase_; // ★ 전역 대신 멤버로 보관
    };

    // 센터 정규화(2 residual): sqrt(lambda)*(t - c) — 그대로
    struct CenterRegResidual
    {
        CenterRegResidual(double cx, double cy, double lambda)
            : cx_(cx), cy_(cy), sl_(std::sqrt(lambda)) {}

        template <typename T>
        bool operator()(const T *const params, T *res) const
        {
            res[0] = T(sl_) * (params[0] - T(cx_));
            res[1] = T(sl_) * (params[1] - T(cy_));
            return true;
        }

        double cx_, cy_, sl_;
    };

    struct SolveOpts
    {
        bool use_huber = true;
        double huber_delta = 0.02; // 데이터 스케일(미터)에 맞춰 조정
        int max_iters = 1000;
        double f_tol = 1e-10, g_tol = 1e-10, p_tol = 1e-12;
    };

    struct SolveIn
    {
        std::vector<Eigen::Vector2d> pts2d;
        std::vector<int> colors; // 0/1, -1=skip
        int gx = 7, gy = 5;
        double s = 0.01;
        double gate_default = 1.0;  // f_g
        double lambda_center = 0.0; // 정규화(옵션)
        Eigen::Vector2d center_reg = Eigen::Vector2d::Zero();
        double tx0 = 0, ty0 = 0, th0 = 0;
    };

    struct SolveOut
    {
        double tx, ty, th, final_cost;
        int iters;
        bool ok;
    };

    static inline SolveOut SolveBoardPoseDogleg(const SolveIn &in, const SolveOpts &opt = {})
    {
        auto wrap = [](double a)
        { return std::atan2(std::sin(a), std::cos(a)); };

        // NumericDiff step 확대
        ceres::NumericDiffOptions nopt;
        nopt.relative_step_size = 1e-3;

        // 여러 초기 θ 시드
        const std::vector<double> theta_seeds = {
            in.th0 - M_PI / 2.0,
            in.th0,
            in.th0 + M_PI / 2.0,
            in.th0 + M_PI};

        // 내부 solve: (phase, huber_on, iters, 초기값)
        auto solve_once = [&](int phase, bool huber_on, int max_iters,
                              double init_tx, double init_ty, double init_th) -> SolveOut
        {
            SolveOut out{};
            double params[3] = {init_tx, init_ty, init_th};
            ceres::Problem problem;

            // 포인트 residual
            for (size_t i = 0; i < in.pts2d.size(); ++i)
            {
                const int ci = (i < in.colors.size() ? in.colors[i] : -1);

                ceres::CostFunction *cf =
                    new ceres::NumericDiffCostFunction<CompactPointResidual,
                                                       ceres::CENTRAL, 1, 3>(
                        // ★ 전역 없이 phase 주입되는 오버로드 사용
                        new CompactPointResidual(in.pts2d[i], ci,
                                                 in.gx, in.gy, in.s,
                                                 /*gate*/ in.gate_default,
                                                 /*out_scale*/ 1.0,
                                                 /*phase*/ phase),
                        ceres::TAKE_OWNERSHIP,
                        /*num_residuals=*/1,
                        nopt);

                ceres::LossFunction *loss =
                    (huber_on && opt.use_huber)
                        ? static_cast<ceres::LossFunction *>(new ceres::HuberLoss(opt.huber_delta))
                        : nullptr;

                problem.AddResidualBlock(cf, loss, params);
            }

            // 센터 정규화(선택)
            if (in.lambda_center > 0.0)
            {
                ceres::CostFunction *reg =
                    new ceres::AutoDiffCostFunction<CenterRegResidual, 2, 3>(
                        new CenterRegResidual(in.center_reg.x(), in.center_reg.y(), in.lambda_center));
                problem.AddResidualBlock(reg, nullptr, params);
            }

            ceres::Solver::Options opts;
            //opts.trust_region_strategy_type = ceres::DOGLEG; // Powell’s Dogleg
            //opts.dogleg_type = ceres::TRADITIONAL_DOGLEG;
            opts.linear_solver_type = ceres::DENSE_QR;
            opts.max_num_iterations = max_iters;
            opts.function_tolerance = opt.f_tol;
            opts.gradient_tolerance = opt.g_tol;
            opts.parameter_tolerance = opt.p_tol;
            opts.minimizer_progress_to_stdout = false;

            ceres::Solver::Summary s;
            ceres::Solve(opts, &problem, &s);

            out.tx = params[0];
            out.ty = params[1];
            out.th = wrap(params[2]); // θ 래핑
            out.final_cost = s.final_cost;
            out.iters = (int)s.iterations.size();
            out.ok = s.IsSolutionUsable();
            return out;
        };

        // phase 0/1 × 다중 θ × 2단계(Huber OFF→ON)
        SolveOut best{};
        best.final_cost = std::numeric_limits<double>::infinity();
        best.ok = false;

        for (int phase : {0, 1})
        {
            for (double th_seed : theta_seeds)
            {
                // 1단계: Huber OFF (거칠게 수렴)
                SolveOut s1 = solve_once(phase, /*huber_on=*/false,
                                         /*iters=*/std::max(1, opt.max_iters / 2),
                                         in.tx0, in.ty0, th_seed);

                // 2단계: Huber ON (미세 정련)
                SolveOut s2 = solve_once(phase, /*huber_on=*/true,
                                         /*iters=*/opt.max_iters - std::max(1, opt.max_iters / 2),
                                         s1.tx, s1.ty, s1.th);

                const SolveOut &cur = (s2.ok ? s2 : s1);
                if (cur.final_cost < best.final_cost)
                    best = cur;
            }
        }

        return best;
    }
}
#endif // LIDAR_CORNER_DETECTION_HPP
