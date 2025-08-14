#include "intensity_lidar_corner_detection.hpp"

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using std::cout;
using std::cerr;
using std::endl;

// Helper: clamp
template <typename T>
static T clamp_val(T v, T lo, T hi) {
    return std::min(hi, std::max(lo, v));
}

// 1) PCD (ASCII) 파싱: 문자열에서 x,y,z,intensity 읽기
std::vector<PointXYZI> parsePCDString(const std::string& pcd_string) {
    std::vector<PointXYZI> lidar_points;
    std::vector<std::string> fields;
    int intensity_field_index = -1;
    bool header_parsed = false;
    bool data_started = false;

    std::stringstream ss(pcd_string);
    std::string line;

    while (std::getline(ss, line)) {
        // trim both ends
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;

        if (!header_parsed) {
            if (line.rfind("FIELDS", 0) == 0) {
                std::stringstream ls(line);
                std::string token;
                ls >> token;
                while (ls >> token) fields.push_back(token);
                auto it = std::find(fields.begin(), fields.end(), "intensity");
                if (it != fields.end()) {
                    intensity_field_index = static_cast<int>(std::distance(fields.begin(), it));
                } else {
                    cerr << "[parsePCDString] Warning: intensity field not found; using dummy intensity\n";
                }
            } else if (line.rfind("DATA", 0) == 0) {
                std::stringstream ls(line);
                std::string token;
                ls >> token;
                if (!(ls >> token)) {
                    cerr << "[parsePCDString] Error: malformed DATA line\n";
                    return {};
                }
                if (token != "ascii") {
                    cerr << "[parsePCDString] Error: only ascii PCD supported here\n";
                    return {};
                }
                header_parsed = true;
                data_started = true;
            }
        } else if (data_started) {
            std::stringstream ls(line);
            std::vector<std::string> parts;
            std::string part;
            while (ls >> part) parts.push_back(part);
            if (fields.size() > 0 && parts.size() != fields.size()) {
                continue;
            }
            try {
                PointXYZI p;
                if (fields.size() >= 3) {
                    auto itx = std::find(fields.begin(), fields.end(), "x");
                    auto ity = std::find(fields.begin(), fields.end(), "y");
                    auto itz = std::find(fields.begin(), fields.end(), "z");
                    if (itx != fields.end() && ity != fields.end() && itz != fields.end()) {
                        p.x = std::stod(parts[std::distance(fields.begin(), itx)]);
                        p.y = std::stod(parts[std::distance(fields.begin(), ity)]);
                        p.z = std::stod(parts[std::distance(fields.begin(), itz)]);
                    } else {
                        p.x = std::stod(parts[0]);
                        p.y = std::stod(parts[1]);
                        p.z = std::stod(parts[2]);
                    }
                } else {
                    p.x = std::stod(parts[0]);
                    p.y = std::stod(parts[1]);
                    p.z = std::stod(parts[2]);
                }

                if (intensity_field_index != -1 && static_cast<size_t>(intensity_field_index) < parts.size()) {
                    p.intensity = std::stod(parts[intensity_field_index]);
                } else {
                    p.intensity = 128.0;
                }
                lidar_points.push_back(p);
            } catch (const std::exception &e) {
                continue;
            }
        }
    }

    if (!header_parsed) {
        cerr << "[parsePCDString] Error: no header or DATA not ascii\n";
        return {};
    }
    cout << "[parsePCDString] Parsed points: " << lidar_points.size() << endl;
    return lidar_points;
}

// intensity 분류: black(0) / white(1) / gray(-1)
std::tuple<Eigen::VectorXi, double, double> classifyIntensityColor(const Eigen::VectorXd& intensities, double epsilon_g) {
    Eigen::VectorXi classified = Eigen::VectorXi::Constant(intensities.size(), -1);
    double tau_l = 0.0, tau_h = 0.0;
    if (intensities.size() == 0) return std::make_tuple(classified, tau_l, tau_h);

    double R_L = intensities.minCoeff();
    double R_H = intensities.maxCoeff();
    if (std::abs(R_H - R_L) < 1e-9) {
        return std::make_tuple(classified, R_L, R_H);
    }

    tau_l = ((epsilon_g - 1.0) * R_L + R_H) / epsilon_g;
    tau_h = (R_L + (epsilon_g - 1.0) * R_H) / epsilon_g;

    for (int i = 0; i < intensities.size(); ++i) {
        if (intensities[i] < tau_l) classified[i] = 0;
        else if (intensities[i] > tau_h) classified[i] = 1;
        else classified[i] = -1;
    }

    cout << "[classifyIntensityColor] R_L=" << R_L << " R_H=" << R_H << " tau_l=" << tau_l << " tau_h=" << tau_h << endl;
    return std::make_tuple(classified, tau_l, tau_h);
}

// checkerboard model color lookup
int getCheckerboardPatternColor(double x, double y, int grid_size_x_squares, int grid_size_y_squares, double checker_size_m) {
    if (checker_size_m <= 0.0) return -1;
    int col = static_cast<int>(std::floor(x / checker_size_m));
    int row = static_cast<int>(std::floor(y / checker_size_m));
    if (!(col >= 0 && col < grid_size_x_squares && row >= 0 && row < grid_size_y_squares)) return -1;
    return ((row + col) % 2 == 0) ? 1 : 0;
}

// cost function (with center regularization)
double costFunction(const Eigen::Vector3d& params,
                    const Eigen::MatrixXd& points_in_pca_plane_2d,
                    const Eigen::VectorXi& classified_colors,
                    int grid_size_x_squares, int grid_size_y_squares, double checker_size_m,
                    const Eigen::Vector2d &center_reg, double lambda_center)
{
    double tx = params[0];
    double ty = params[1];
    double theta_z = params[2];

    double cos_t = std::cos(theta_z), sin_t = std::sin(theta_z);
    Eigen::Matrix2d Rz;
    Rz << cos_t, -sin_t,
          sin_t,  cos_t;

    double board_w = grid_size_x_squares * checker_size_m;
    double board_h = grid_size_y_squares * checker_size_m;
    double model_min_x = -board_w / 2.0;
    double model_max_x =  board_w / 2.0;
    double model_min_y = -board_h / 2.0;
    double model_max_y =  board_h / 2.0;

    Eigen::MatrixXd shifted = points_in_pca_plane_2d.rowwise() - Eigen::RowVector2d(tx, ty);
    Eigen::MatrixXd transformed = (Rz.transpose() * shifted.transpose()).transpose();

    double cost = 0.0;
    const double out_scale = std::max(board_w, board_h) * 10.0;

    for (int i = 0; i < transformed.rows(); ++i) {
        int ci = classified_colors[i];
        if (ci == -1) continue;
        double px = transformed(i,0);
        double py = transformed(i,1);
        double lookup_x = px + board_w / 2.0;
        double lookup_y = py + board_h / 2.0;
        int model_color = getCheckerboardPatternColor(lookup_x, lookup_y, grid_size_x_squares, grid_size_y_squares, checker_size_m);
        if (model_color == -1) {
            double dx = 0.0, dy = 0.0;
            if (px < model_min_x) dx = model_min_x - px;
            else if (px > model_max_x) dx = px - model_max_x;
            if (py < model_min_y) dy = model_min_y - py;
            else if (py > model_max_y) dy = py - model_max_y;
            cost += (dx + dy) * out_scale;
        } else {
            if (ci != model_color) {
                double local_x = std::fmod(std::fabs(lookup_x), checker_size_m);
                double local_y = std::fmod(std::fabs(lookup_y), checker_size_m);
                double dx_side = std::min(local_x, checker_size_m - local_x);
                double dy_side = std::min(local_y, checker_size_m - local_y);
                double prox = std::min(dx_side, dy_side);
                double weight = 1.0 + (checker_size_m - prox) / checker_size_m;
                cost += 1.0 * weight;
            }
        }
    }

    double dx = tx - center_reg[0];
    double dy = ty - center_reg[1];
    cost += lambda_center * (dx*dx + dy*dy);

    return cost;
}

// Orientation & center refinement using intensity-pattern cross-correlation
Eigen::Vector3d refineOrientationAndCenter(
    const Eigen::MatrixXd &points2d,
    const Eigen::VectorXd &intensities2d,
    double checker_size,
    int num_squares_x, int num_squares_y)
{
    Eigen::Vector2d centroid = points2d.colwise().mean();

    int bins = 60;
    Eigen::VectorXd hist_x = Eigen::VectorXd::Zero(bins);
    Eigen::VectorXd hist_y = Eigen::VectorXd::Zero(bins);
    double min_x = points2d.col(0).minCoeff(), max_x = points2d.col(0).maxCoeff();
    double min_y = points2d.col(1).minCoeff(), max_y = points2d.col(1).maxCoeff();
    double eps = 1e-6;
    double range_x = std::max(eps, max_x - min_x);
    double range_y = std::max(eps, max_y - min_y);

    for (int i = 0; i < points2d.rows(); ++i) {
        int ix = clamp_val<int>( static_cast<int>(((points2d(i,0) - min_x) / range_x) * (bins-1)), 0, bins-1 );
        int iy = clamp_val<int>( static_cast<int>(((points2d(i,1) - min_y) / range_y) * (bins-1)), 0, bins-1 );
        hist_x(ix) += intensities2d(i);
        hist_y(iy) += intensities2d(i);
    }

    auto periodicityScore = [&](const Eigen::VectorXd &h, double range)->double {
        Eigen::VectorXd mean_sub = h.array() - h.mean();
        Eigen::VectorXd ac = Eigen::VectorXd::Zero(h.size());
        for (int lag = 0; lag < (int)h.size(); ++lag) {
            double s = 0;
            for (int k = 0; k + lag < h.size(); ++k) s += mean_sub(k) * mean_sub(k + lag);
            ac(lag) = s;
        }
        double expected_period_bins = std::max(1.0, (checker_size / (range / h.size())));
        int center = clamp_val<int>(static_cast<int>(std::round(expected_period_bins)), 1, h.size()-1);
        int w = std::max(1, int(h.size()*0.1));
        int start = std::max(1, center-w);
        int len = std::min(w*2+1, (int)h.size()-start);
        double peak = ac.segment(start, len).maxCoeff();
        return peak;
    };

    double score_x = periodicityScore(hist_x, range_x);
    double score_y = periodicityScore(hist_y, range_y);
    bool rows_along_x = (score_x > score_y);

    std::vector<double> angle_candidates;
    for (double a = -10.0*M_PI/180.0; a <= 10.0*M_PI/180.0; a += 1.0*M_PI/180.0) angle_candidates.push_back(a);

    int img_res = 200;
    Eigen::MatrixXd img = Eigen::MatrixXd::Zero(img_res, img_res);
    Eigen::MatrixXd cnt = Eigen::MatrixXd::Zero(img_res, img_res);
    double pad = 0.1;
    double minX = min_x - pad, maxX = max_x + pad;
    double minY = min_y - pad, maxY = max_y + pad;

    for (int i = 0; i < points2d.rows(); ++i) {
        int gx = clamp_val<int>( static_cast<int>(((points2d(i,0)-minX)/(maxX-minX))*(img_res-1)), 0, img_res-1 );
        int gy = clamp_val<int>( static_cast<int>(((points2d(i,1)-minY)/(maxY-minY))*(img_res-1)), 0, img_res-1 );
        img(gy,gx) += intensities2d(i);
        cnt(gy,gx) += 1.0;
    }
    for (int r=0;r<img_res;++r) for (int c=0;c<img_res;++c) if (cnt(r,c) > 0) img(r,c) /= cnt(r,c);
    Eigen::MatrixXd img_norm = img.array() - img.mean();

    double best_corr = -1e12;
    double best_angle = 0.0;
    for (double ang : angle_candidates) {
        double cang = std::cos(ang), sang = std::sin(ang);
        Eigen::MatrixXd pattern = Eigen::MatrixXd::Zero(img_res, img_res);
        for (int r = 0; r < img_res; ++r) {
            for (int c = 0; c < img_res; ++c) {
                double x = minX + (maxX-minX) * (double(c)/(img_res-1));
                double y = minY + (maxY-minY) * (double(r)/(img_res-1));
                double xr = cang * x + sang * y;
                double yr = -sang * x + cang * y;
                double bx = xr + (num_squares_x+1)*checker_size*0.5;
                double by = yr + (num_squares_y+1)*checker_size*0.5;
                int col = static_cast<int>(std::floor(bx / checker_size));
                int row = static_cast<int>(std::floor(by / checker_size));
                if (col >= 0 && col <= num_squares_x && row >= 0 && row <= num_squares_y) {
                    pattern(r,c) = ((row + col) % 2 == 0) ? 1.0 : -1.0;
                } else {
                    pattern(r,c) = 0.0;
                }
            }
        }
        Eigen::MatrixXd pat_norm = pattern.array() - pattern.mean();
        double corr = (img_norm.array() * pat_norm.array()).sum();
        if (corr > best_corr) {
            best_corr = corr;
            best_angle = ang;
        }
    }

    Eigen::Vector3d out;
    out[0] = centroid[0];
    out[1] = centroid[1];
    out[2] = best_angle;
    if (!rows_along_x) out[2] += M_PI_2;
    return out;
}

// 메인 함수: 논문 방식의 코너 추정 함수 (외부에서 호출됨)
std::vector<PointXYZI> estimateChessboardCornersPaperMethod(
    const std::vector<PointXYZI>& lidar_points_full_vec,
    int internal_corners_x, int internal_corners_y, double checker_size_m)
{
    cout << "[estimateChessboardCornersPaperMethod] start. Npoints=" << lidar_points_full_vec.size() << endl;
    std::vector<PointXYZI> out_corners;

    if (lidar_points_full_vec.empty()) {
        cerr << "[estimateChessboardCornersPaperMethod] empty input" << endl;
        return out_corners;
    }
    if (checker_size_m <= 0.0) {
        cerr << "[estimateChessboardCornersPaperMethod] invalid checker size" << endl;
        return out_corners;
    }

    int N = static_cast<int>(lidar_points_full_vec.size());
    Eigen::MatrixXd pts3d(N, 3);
    Eigen::VectorXd intens(N);
    for (int i = 0; i < N; ++i) {
        pts3d(i,0) = lidar_points_full_vec[i].x;
        pts3d(i,1) = lidar_points_full_vec[i].y;
        pts3d(i,2) = lidar_points_full_vec[i].z;
        intens(i)   = lidar_points_full_vec[i].intensity;
    }

    // Centroid 계산
    Eigen::Vector3d centroid = pts3d.colwise().mean().transpose();
    
    // Covariance Matrix 계산
    Eigen::MatrixXd centered = pts3d.rowwise() - centroid.transpose();
    Eigen::Matrix3d cov = (centered.transpose() * centered) / double(std::max(1, N-1));
    
    // PCA를 위한 고유값/고유벡터 계산
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d eigvals = es.eigenvalues();
    Eigen::Matrix3d eigvecs = es.eigenvectors();

    std::vector<std::pair<double, Eigen::Vector3d>> pairs;
    for (int i = 0; i < 3; ++i) pairs.emplace_back(eigvals(i), eigvecs.col(i));
    std::sort(pairs.begin(), pairs.end(), [](auto &a, auto &b){ return a.first > b.first; });

    Eigen::Matrix3d pcs;
    for (int i = 0; i < 3; ++i) pcs.col(i) = pairs[i].second;
    
    // -----------------------------------------------------------
    // PCA 축 안정화를 위한 보정 로직
    // -----------------------------------------------------------
    Eigen::Vector3d v_z = pcs.col(2);
    Eigen::Vector3d v_y_temp = pcs.col(1);
    Eigen::Vector3d v_x_temp = pcs.col(0);
    
    // 이전에 선언되지 않아 오류가 발생했던 v_x와 v_y를 여기서 선언합니다.
    Eigen::Vector3d v_x;
    Eigen::Vector3d v_y;
    
    // 1. 법선 벡터(v_z) 방향 고정 (사용자분께서 유지하길 원하는 로직)
    // LiDAR 원점에서 평면 중심까지의 벡터와 v_z의 내적을 이용해 방향을 고정합니다.
    if (v_z.dot(-centroid) < 0) {
        v_z = -v_z;
    }
    
    // 2. PCA의 두 주축(v_x, v_y)이 체커보드의 가로/세로 방향과 일치하도록 정렬
    // 여기에서 오류가 발생했던 코드를 수정합니다.
    Eigen::VectorXd projections_x = centered * v_x_temp;
    Eigen::VectorXd projections_y = centered * v_y_temp;
    
    double pca_len_x = projections_x.maxCoeff() - projections_x.minCoeff();
    double pca_len_y = projections_y.maxCoeff() - projections_y.minCoeff();
    
    double model_len_x = (internal_corners_x) * checker_size_m;
    double model_len_y = (internal_corners_y) * checker_size_m;

    if ((pca_len_x > pca_len_y && model_len_x > model_len_y) ||
        (pca_len_y > pca_len_x && model_len_y > model_len_x)) {
        v_x = v_x_temp;
        v_y = v_y_temp;
    } else {
        v_x = v_y_temp;
        v_y = v_x_temp;
    }

    // 3. X, Y축의 방향성을 더 안정적인 기준으로 고정
    // 기존의 LiDAR Y축 대신, 체커보드의 수직축(v_y)이 LiDAR Z축과 같은 방향을 향하도록 합니다.
    Eigen::Vector3d lidar_z_axis(0.0, 0.0, 1.0);
    if (v_y.dot(lidar_z_axis) < 0) {
        v_y = -v_y;
    }

    // 4. 오른손 좌표계 규칙을 강제하기 위해 v_x를 재계산
    v_x = v_y.cross(v_z);

    v_z.normalize();
    v_x.normalize();
    v_y.normalize();
    
    // -----------------------------------------------------------

    Eigen::Matrix3d Rpcs;
    Rpcs.col(0) = v_x;
    Rpcs.col(1) = v_y;
    Rpcs.col(2) = v_z;

    Eigen::MatrixXd pts_pca = (centered * Rpcs).eval();
    Eigen::MatrixXd pts_pca_2d = pts_pca.leftCols(2);

    double min_x = pts_pca_2d.col(0).minCoeff();
    double max_x = pts_pca_2d.col(0).maxCoeff();
    double min_y = pts_pca_2d.col(1).minCoeff();
    double max_y = pts_pca_2d.col(1).maxCoeff();
    double margin = std::max(checker_size_m * 0.05, 0.01);
    double min_x_c = min_x - margin, max_x_c = max_x + margin;
    double min_y_c = min_y - margin, max_y_c = max_y + margin;

    std::vector<int> keep_idx;
    keep_idx.reserve(N);
    for (int i = 0; i < pts_pca_2d.rows(); ++i) {
        double x = pts_pca_2d(i,0), y = pts_pca_2d(i,1);
        if (x >= min_x_c && x <= max_x_c && y >= min_y_c && y <= max_y_c) keep_idx.push_back(i);
    }
    if (keep_idx.empty()) {
        cerr << "[estimateChessboardCornersPaperMethod] no points remain after crop\n";
        return out_corners;
    }

    Eigen::MatrixXd pts_opt(keep_idx.size(), 2);
    Eigen::VectorXd intens_opt(keep_idx.size());
    for (size_t i = 0; i < keep_idx.size(); ++i) {
        pts_opt.row(i) = pts_pca_2d.row(keep_idx[i]);
        intens_opt(i) = intens(keep_idx[i]);
    }

    Eigen::VectorXi classified;
    double tau_l = 0.0, tau_h = 0.0;
    std::tie(classified, tau_l, tau_h) = classifyIntensityColor(intens_opt, 2.0);

    int num_squares_x = internal_corners_x + 1;
    int num_squares_y = internal_corners_y + 1;

    Eigen::Vector3d refined = refineOrientationAndCenter(pts_opt, intens_opt, checker_size_m, num_squares_x, num_squares_y);
    std::vector<double> theta_guesses = { refined[2] - M_PI/2.0, refined[2], refined[2] + M_PI/2.0, refined[2] + M_PI };

    double lambda_center = 10.0;
    double best_cost = std::numeric_limits<double>::infinity();
    Eigen::Vector3d best_params = refined;

    for (double tg : theta_guesses) {
        Eigen::Vector3d params(refined[0], refined[1], tg);
        double current_cost = costFunction(params, pts_opt, classified, num_squares_x, num_squares_y, checker_size_m,
                                          Eigen::Vector2d(refined[0], refined[1]), lambda_center);

        double tx_step = std::max(checker_size_m * 0.5, 0.05);
        double ty_step = tx_step;
        double th_step = M_PI/8.0;
        for (int iter = 0; iter < 6; ++iter) {
            bool improved = false;
            std::vector<Eigen::Vector3d> tries = {
                params,
                params + Eigen::Vector3d(tx_step,0,0),
                params + Eigen::Vector3d(-tx_step,0,0),
                params + Eigen::Vector3d(0,ty_step,0),
                params + Eigen::Vector3d(0,-ty_step,0),
                params + Eigen::Vector3d(0,0,th_step),
                params + Eigen::Vector3d(0,0,-th_step)
            };
            for (auto &t : tries) {
                double c = costFunction(t, pts_opt, classified, num_squares_x, num_squares_y, checker_size_m,
                                       Eigen::Vector2d(refined[0], refined[1]), lambda_center);
                if (c + 1e-9 < current_cost) {
                    current_cost = c;
                    params = t;
                    improved = true;
                }
            }
            if (!improved) {
                tx_step *= 0.5; ty_step *= 0.5; th_step *= 0.5;
            }
        }
        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_params = params;
        }
    }

    {
        Eigen::Vector3d base = best_params;
        double base_cost = best_cost;
        for (double dth = -5.0*M_PI/180.0; dth <= 5.0*M_PI/180.0; dth += 1.0*M_PI/180.0) {
            Eigen::Vector3d cand = base;
            cand[2] = base[2] + dth;
            double step = checker_size_m * 0.1;
            for (int it = 0; it < 8; ++it) {
                bool local_imp = false;
                std::vector<Eigen::Vector3d> tries = {
                    cand,
                    cand + Eigen::Vector3d(step,0,0),
                    cand + Eigen::Vector3d(-step,0,0),
                    cand + Eigen::Vector3d(0,step,0),
                    cand + Eigen::Vector3d(0,-step,0)
                };
                double best_local_c = costFunction(cand, pts_opt, classified, num_squares_x, num_squares_y, checker_size_m,
                                                  Eigen::Vector2d(refined[0], refined[1]), lambda_center);
                for (auto &t : tries) {
                    double c = costFunction(t, pts_opt, classified, num_squares_x, num_squares_y, checker_size_m,
                                           Eigen::Vector2d(refined[0], refined[1]), lambda_center);
                    if (c + 1e-9 < best_local_c) {
                        best_local_c = c;
                        cand = t;
                        local_imp = true;
                    }
                }
                if (!local_imp) step *= 0.5;
            }
            double cand_cost = costFunction(cand, pts_opt, classified, num_squares_x, num_squares_y, checker_size_m,
                                           Eigen::Vector2d(refined[0], refined[1]), lambda_center);
            if (cand_cost < base_cost) {
                base_cost = cand_cost;
                base = cand;
            }
        }
        best_params = base;
        best_cost = base_cost;
    }

    cout << "[estimateChessboardCornersPaperMethod] best_params tx,ty,theta(deg)=("
         << best_params[0] << ", " << best_params[1] << ", " << best_params[2]*180.0/M_PI << ") cost=" << best_cost << endl;

    std::vector<Eigen::Vector3d> lidar_pts_world;
    lidar_pts_world.reserve(N);
    for (const auto &p : lidar_points_full_vec) lidar_pts_world.emplace_back(Eigen::Vector3d(p.x, p.y, p.z));

    std::vector<Eigen::Vector2d> model_internal_corners_2d;
    double half_w = (internal_corners_x - 1) * checker_size_m / 2.0;
    double half_h = (internal_corners_y - 1) * checker_size_m / 2.0;
    double start_x = -half_w;
    double start_y = -half_h;
    
    for (int r = 0; r < internal_corners_y; ++r) {
        for (int c = 0; c < internal_corners_x; ++c) {
            double cx = start_x + c * checker_size_m;
            double cy = start_y + r * checker_size_m;
            model_internal_corners_2d.emplace_back(cx, cy);
        }
    }

    Eigen::Matrix2d Rz_best;
    double ct = std::cos(best_params[2]), st = std::sin(best_params[2]);
    Rz_best << ct, -st, st, ct;

    double max_corner_dist = checker_size_m * 0.6;
    struct CornerData {
        PointXYZI world_point;
        Eigen::Vector2d pca_point;
    };
    std::vector<CornerData> corners_data;

    for (const auto &mc : model_internal_corners_2d) {
        Eigen::Vector2d p_pca2 = Rz_best * mc + Eigen::Vector2d(best_params[0], best_params[1]);
        Eigen::Vector3d p_pca3(p_pca2[0], p_pca2[1], 0.0);
        Eigen::Vector3d p_world = Rpcs * p_pca3 + centroid;

        bool has_nearby = false;
        for (const auto &lp : lidar_pts_world) {
            double d2 = (lp - p_world).squaredNorm();
            if (d2 <= max_corner_dist * max_corner_dist) {
                has_nearby = true;
                break;
            }
        }
        if (has_nearby) {
            CornerData cd;
            cd.world_point.x = p_world[0];
            cd.world_point.y = p_world[1];
            cd.world_point.z = p_world[2];
            cd.world_point.intensity = 0.0;
            cd.pca_point = p_pca2;
            corners_data.push_back(cd);
        }
    }
    
    cout << "[estimateChessboardCornersPaperMethod] detected corners (nearby-filtered): " << corners_data.size() << endl;

    if (!corners_data.empty()) {
        std::sort(corners_data.begin(), corners_data.end(), [&](const CornerData& a, const CornerData& b) {
            Eigen::Vector3d a_vec(a.world_point.x, a.world_point.y, a.world_point.z);
            Eigen::Vector3d b_vec(b.world_point.x, b.world_point.y, b.world_point.z);
            if (std::abs(a_vec.dot(v_y) - b_vec.dot(v_y)) < checker_size_m * 0.5) {
                return a_vec.dot(v_x) < b_vec.dot(v_x);
            }
            return a_vec.dot(v_y) > b_vec.dot(v_y);
        });
    }

    for (const auto& cd : corners_data) {
        out_corners.push_back(cd.world_point);
    }

    if (!out_corners.empty()) {
        out_corners[0].intensity = 255.0;
    }
    
    cout << "[estimateChessboardCornersPaperMethod] detected corners (sorted): " << out_corners.size() << endl;
    return out_corners;
}
