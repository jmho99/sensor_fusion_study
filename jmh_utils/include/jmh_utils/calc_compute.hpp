#ifndef CALC_COMPUTE_HPP
#define CALC_COMPUTE_HPP

#include <Eigen/Dense>
#include <functional>
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

namespace jmh_utils
{
    void computeTransformSVD(const std::vector<Eigen::Vector3d> &lid_corners, const std::vector<Eigen::Vector3d> &cam_corners,
                             Eigen::MatrixXd &rotation, Eigen::VectorXd &translation);

    struct PowellOptions
    {
        double xtol = 1e-4;                    // 위치 변화 허용오차 (유효자리 기준)
        double ftol = 1e-8;                    // 함수값 변화 허용오차
        int maxiter = 200;                     // 외부 반복 제한
        int maxfev = 20000;                    // 함수평가 제한
        double bracket_step = 1.0;             // 초기 브래킷 스텝
        double expand = 1.6180339887498948482; // golden ratio
        double brent_tol = 1e-6;               // Brent 1D 최적화 tol (alpha)
        bool verbose = false;
    };

    struct PowellResult
    {
        Eigen::VectorXd x; // argmin
        double fval;       // min value
        int nfev = 0;      // # function evaluations
        int nit = 0;       // # iterations
        bool success = false;
        std::string message;
    };

    static double brent_minimize(
        const std::function<double(double)> &phi,
        double a, double b, double tol, int &fev, int maxfev);

    static void bracket_minimum(
        const std::function<double(double)> &phi,
        double step, double expand, int &fev, int maxfev,
        double &a, double &b);
    struct LiteBracket
    {
        double a, b; // 반환용 [a,b]
        bool ok = false;
    };
    PowellResult computePowell(
        const std::function<double(const Eigen::VectorXd &)> &fun,
        const Eigen::VectorXd &x0,
        const PowellOptions &opts = PowellOptions());
}
#endif