#include "jmh_utils/calc_compute.hpp"

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
                             Eigen::MatrixXd &rotation, Eigen::VectorXd &translation)
    {
        std::cout << "Start computing SVD" << std::endl;
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
        std::cout << "End computing SVD" << std::endl;
    }
    // NaN/Inf 방지 + 평가 카운트
    static inline double safe_eval_phi(const std::function<double(double)> &phi,
                                       double x, int &fev, int maxfev)
    {
        if (fev >= maxfev)
            return 1e300;
        double v = phi(x);
        ++fev;
        return std::isfinite(v) ? v : 1e300;
    }

    // 0, +s0, (필요시 -s0) 및 1회 확장만으로 가벼운 브래킷 생성
    static LiteBracket bracket_lite(
        const std::function<double(double)> &phi,
        double s0, double expand, int &fev, int maxfev)
    {
        LiteBracket br{};
        // 중심 0, +s0
        double f0 = safe_eval_phi(phi, 0.0, fev, maxfev);
        double fp = safe_eval_phi(phi, +s0, fev, maxfev);

        if (fp >= f0)
        {
            // +방향 하강 아님 → -s0 검사
            double fm = safe_eval_phi(phi, -s0, fev, maxfev);
            if (fm >= f0)
            {
                // 0 근처가 더 낫다 → 라이트 브래킷 실패
                return br;
            }
            else
            {
                // -방향으로 한 번 확장
                double s1 = -s0 * expand;
                (void)safe_eval_phi(phi, s1, fev, maxfev); // 평가만 하며 구간 넓힘
                br.a = s1;
                br.b = 0.0;
                br.ok = true;
                return br;
            }
        }
        else
        {
            // +방향으로 한 번 확장
            double s1 = +s0 * expand;
            (void)safe_eval_phi(phi, s1, fev, maxfev);
            br.a = 0.0;
            br.b = s1;
            br.ok = true;
            return br;
        }
    }
    static double brent_minimize(
        const std::function<double(double)> &phi,
        double a, double b, double tol, int &fev, int maxfev)
    {
        const double CGOLD = 0.3819660112501051; // 1 - 1/phi
        const double ZEPS = std::numeric_limits<double>::epsilon() * 1e3;

        double x = a + CGOLD * (b - a);
        double w = x, v = x;
        double fx = phi(x);
        ++fev;
        double fw = fx, fv = fx;

        double d = 0.0, e = 0.0;

        while (true)
        {
            double m = 0.5 * (a + b);
            double tol1 = std::sqrt(ZEPS) * std::abs(x) + tol / 3.0;
            double tol2 = 2.0 * tol1;

            // 종료 조건: 구간이 충분히 작거나 반복 제한
            if (std::abs(x - m) <= (tol2 - 0.5 * (b - a)))
                break;
            if (fev >= maxfev)
                break;

            double p = 0.0, q = 0.0, r = 0.0;
            if (std::abs(e) > tol1)
            {
                // 포물선 보간
                r = (x - w) * (fx - fv);
                q = (x - v) * (fx - fw);
                p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if (q > 0.0)
                    p = -p;
                q = std::abs(q);
                double etemp = e;
                e = d;

                if ((std::abs(p) >= std::abs(0.5 * q * etemp)) ||
                    (p <= q * (a - x)) || (p >= q * (b - x)))
                {
                    // 황금분할
                    e = (x >= m) ? (a - x) : (b - x);
                    d = 0.3819660112501051 * e;
                }
                else
                {
                    d = p / q;
                    double u = x + d;
                    if ((u - a) < tol2 || (b - u) < tol2)
                        d = (x < m) ? tol1 : -tol1;
                }
            }
            else
            {
                // 황금분할
                e = (x >= m) ? (a - x) : (b - x);
                d = 0.3819660112501051 * e;
            }

            double u = (std::abs(d) >= tol1) ? x + d : x + ((d > 0) ? tol1 : -tol1);
            double fu = phi(u);
            ++fev;

            if (fu <= fx)
            {
                if (u >= x)
                    a = x;
                else
                    b = x;
                v = w;
                fv = fw;
                w = x;
                fw = fx;
                x = u;
                fx = fu;
            }
            else
            {
                if (u < x)
                    a = u;
                else
                    b = u;
                if (fu <= fw || w == x)
                {
                    v = w;
                    fv = fw;
                    w = u;
                    fw = fu;
                }
                else if (fu <= fv || v == x || v == w)
                {
                    v = u;
                    fv = fu;
                }
            }
        }
        return x;
    }

    static void bracket_minimum(
        const std::function<double(double)> &phi,
        double step, double expand, int &fev, int maxfev,
        double &a, double &b)
    {
        // 시작: [0, step]
        a = 0.0;
        b = step;
        double fa = phi(a);
        ++fev;
        double fb = phi(b);
        ++fev;

        // 감소 방향으로 확장 (함수값이 줄어드는 쪽으로)
        if (fb > fa)
        {
            // 반대 방향으로
            b = -step;
            fb = phi(b);
            ++fev;
            if (fb > fa)
            {             // 어차피 0 근처 최소
                b = step; // fallback
                fb = phi(b);
                ++fev;
                return;
            }
        }

        // f 계속 감소하는 동안 확장
        while (fb < fa && fev < maxfev)
        {
            double nb = b * expand;
            double fnb = phi(nb);
            ++fev;
            a = b;
            fa = fb;
            b = nb;
            fb = fnb;
            // 안전장치: 너무 크게 벌어지면 중단
            if (std::abs(b) > 1e8)
                break;
        }
    }

    // ------------------------------------------------------------
    // Powell (빠르고 안정, 세이프가드 전체 포함)
    //   - 기존 brent_minimize / bracket_minimum 재사용
    //   - PowellOptions: xtol, ftol, maxiter, maxfev, bracket_step, expand, brent_tol, verbose
    // ------------------------------------------------------------
    PowellResult computePowell(
        const std::function<double(const Eigen::VectorXd &)> &fun,
        const Eigen::VectorXd &x0,
        const PowellOptions &opts)
    {
        PowellResult res;
        res.x = x0;
        res.fval = fun(res.x);
        res.nfev = 1;

        const int n = static_cast<int>(x0.size());
        Eigen::MatrixXd direc = Eigen::MatrixXd::Identity(n, n);

        // 방향별 스텝 힌트(EMA). 초기값: bracket_step
        std::vector<double> step_hint(n, std::max(1e-9, opts.bracket_step));
        // 방향별 “연속 소개선” 카운트 → 나쁜 힌트 리셋용
        std::vector<int> small_drop_count(n, 0);

        auto line_search_along = [&](const Eigen::VectorXd &x,
                                     const Eigen::VectorXd &d,
                                     int idir, // 방향 인덱스(EMA/리셋용)
                                     double &fval,
                                     Eigen::VectorXd &x_out)
        {
            // 0-방향 스킵
            const double dnorm = d.norm();
            if (!(dnorm > 0.0) || d.isZero(0))
            {
                x_out = x; /* fval 그대로 */
                return;
            }

            // φ(λ)
            auto phi = [&](double a) -> double
            {
                Eigen::VectorXd xt = x + a * d;
                double v = fun(xt);
                return std::isfinite(v) ? v : 1e300;
            };

            // 빠른 종료(마진 포함): φ(0) ≤ min{φ(±s0)} - τ * max(1, |φ(0)|)
            const double s0 = std::max(1e-10, step_hint[idir]);
            double f0 = phi(0.0);
            ++res.nfev;
            double fp = phi(+s0);
            ++res.nfev;
            double fm = phi(-s0);
            ++res.nfev;

            const double margin = 1e-8 * std::max(1.0, std::abs(f0)); // τ = 1e-3
            if ((f0 <= fp - margin) && (f0 <= fm - margin))
            {
                x_out = x;
                fval = f0;
                return;
            }

            // 경량 브래킷 시도
            LiteBracket br = bracket_lite(phi, s0, opts.expand, res.nfev, opts.maxfev);

            double alpha = 0.0;
            bool did_search = false;

            if (br.ok)
            {
                // 라이트 브래킷 성공 → Brent
                const double a = std::min(br.a, br.b);
                const double b = std::max(br.a, br.b);
                alpha = brent_minimize(phi, a, b, opts.brent_tol, res.nfev, opts.maxfev);
                did_search = true;
            }
            else
            {
                // 라이트 실패 → 풀 브래킷 1회 폴백
                double a_full, b_full;
                bracket_minimum(phi, s0, opts.expand, res.nfev, opts.maxfev, a_full, b_full);
                if (std::isfinite(a_full) && std::isfinite(b_full) && a_full != b_full)
                {
                    const double a = std::min(a_full, b_full);
                    const double b = std::max(a_full, b_full);
                    alpha = brent_minimize(phi, a, b, opts.brent_tol, res.nfev, opts.maxfev);
                    did_search = true;
                }
            }

            if (did_search)
            {
                x_out = x + alpha * d;
                fval = fun(x_out);
                ++res.nfev;
                // EMA 업데이트
                step_hint[idir] = 0.5 * step_hint[idir] + 0.5 * std::abs(alpha);
                // 드롭 크기 보정은 바깥 루프에서
            }
            else
            {
                // 정말 브래킷 안 잡히면 보수적으로 정지
                x_out = x;
                fval = f0;
            }
        };

        Eigen::VectorXd x = res.x;

        for (int it = 0; it < std::max(1, opts.maxiter); ++it)
        {
            if (opts.verbose)
                std::cout << "[powell-safe] iter " << it
                          << " f=" << res.fval
                          << " fev=" << res.nfev << "\n";

            const Eigen::VectorXd x_start = x;
            const double f_start = res.fval;

            int bigind = 0;
            double fbigdrop = 0.0;

            // 각 방향 라인서치
            for (int i = 0; i < n; ++i)
            {
                if (res.nfev >= opts.maxfev)
                    break;

                const double f_before = res.fval;
                Eigen::VectorXd x_new(n);

                line_search_along(x, direc.col(i), i, res.fval, x_new);

                const double drop = f_before - res.fval;
                x = x_new;

                if (drop > fbigdrop)
                {
                    fbigdrop = drop;
                    bigind = i;
                }

                // 드롭이 매우 작으면 카운트 증가 → 나쁜 힌트 리셋
                if (drop <= std::max(1e-16, 0.5 * opts.ftol))
                {
                    small_drop_count[i] += 1;
                    if (small_drop_count[i] >= 2)
                    {
                        step_hint[i] = std::max(1e-9, opts.bracket_step); // 힌트 리셋
                        small_drop_count[i] = 0;
                    }
                }
                else
                {
                    small_drop_count[i] = 0; // 개선 충분하면 리셋
                }

                // **개선 미미 시 남은 방향 스킵** (보수적: 0.02*ftol)
                if (drop <= std::max(1e-16, 0.01 * opts.ftol))
                    break;
            }

            // 사이클 종료 조건 (coarse 기준)
            const double xchg = (x - x_start).cwiseAbs().maxCoeff();
            const double fchg = std::abs(f_start - res.fval);
            const bool tol_hit = (xchg <= opts.xtol) || (fchg <= opts.ftol);
            const bool budget_hit = (res.nfev >= opts.maxfev);

            // --- 품질 보증 패스(정밀 2회): bigind 방향 + 합성방향 ---
            //    라이트가 놓친 경우 보정 (Brent tol 10배 엄격)
            auto quality_pass = [&](void)
            {
                if (budget_hit)
                    return;

                // 1) bigind 방향 (현재 x 기준)
                {
                    const Eigen::VectorXd &d = direc.col(bigind);
                    if (d.norm() > 0.0 && !d.isZero(0))
                    {
                        auto phi1 = [&](double a) -> double
                        { return fun(x + a * d); };
                        double aa, bb;
                        bracket_minimum(phi1, std::max(step_hint[bigind], 1e-9), opts.expand,
                                        res.nfev, opts.maxfev, aa, bb);
                        if (res.nfev < opts.maxfev && aa != bb)
                        {
                            const double a = std::min(aa, bb), b = std::max(aa, bb);
                            double aopt = brent_minimize(phi1, a, b, std::max(1e-16, 0.1 * opts.brent_tol),
                                                         res.nfev, opts.maxfev);
                            Eigen::VectorXd x_try = x + aopt * d;
                            double f_try = fun(x_try);
                            ++res.nfev;
                            if (f_try < res.fval)
                            {
                                x = x_try;
                                res.fval = f_try;
                            }
                        }
                    }
                }

                // 2) 합성방향 (x - x_start)
                if (res.nfev < opts.maxfev)
                {
                    Eigen::VectorXd d_new = x - x_start;
                    const double dn = d_new.norm();
                    if (dn > 0.0 && !d_new.isZero(0))
                    {
                        auto phi2 = [&](double a) -> double
                        { return fun(x + a * d_new); };
                        double aa, bb;
                        bracket_minimum(phi2, std::max(dn, 1e-9), opts.expand,
                                        res.nfev, opts.maxfev, aa, bb);
                        if (res.nfev < opts.maxfev && aa != bb)
                        {
                            const double a = std::min(aa, bb), b = std::max(aa, bb);
                            double aopt = brent_minimize(phi2, a, b, std::max(1e-16, 0.1 * opts.brent_tol),
                                                         res.nfev, opts.maxfev);
                            Eigen::VectorXd x_try = x + aopt * d_new;
                            double f_try = fun(x_try);
                            ++res.nfev;
                            if (f_try < res.fval)
                            {
                                x = x_try;
                                res.fval = f_try;
                            }
                        }
                    }
                }
            };

            quality_pass();

            // 수렴/예산 확인 후 종료
            if (tol_hit || budget_hit)
            {
                res.success = true;
                res.message = budget_hit ? "Coarse stop: maxfev" : "Coarse tol reached";
                res.x = x;
                res.nit = it + 1;
                return res;
            }

            // 합성(conjugate) 방향 (정규화 금지)
            Eigen::VectorXd d_new = x - x_start;
            if (!(d_new.norm() > 0.0) || d_new.isZero(0))
            {
                res.success = true;
                res.message = "No progress in composite direction";
                res.x = x;
                res.nit = it + 1;
                return res;
            }

            // 합성방향으로 한 번 더 (라이트→폴백 가능)
            const double f_before = res.fval;
            {
                Eigen::VectorXd x_after(n);
                // 합성방향은 bigind의 힌트를 재활용 (EMA 수렴 가속)
                line_search_along(x, d_new, bigind, res.fval, x_after);
                x = x_after;
            }

            // 방향 집합 갱신: **정규화 금지**
            direc.col(bigind) = d_new;

            // 합성 후 개선 작으면 종료
            if (std::abs(f_before - res.fval) <= opts.ftol)
            {
                res.success = true;
                res.message = "Small improvement after conjugate";
                res.x = x;
                res.nit = it + 1;
                return res;
            }

            if (res.nfev >= opts.maxfev)
            {
                res.success = true;
                res.message = "Coarse stop: maxfev";
                res.x = x;
                res.nit = it + 1;
                return res;
            }
        }

        res.success = true; // coarse 완료
        res.message = "Coarse stop: max outer iterations";
        res.nit = std::max(1, opts.maxiter);
        res.x = x;
        return res;
    }
}