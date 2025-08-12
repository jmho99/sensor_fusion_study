// intensity_lidar_corner_detection.cpp
// 수정: PCA 축 보정, 초기 회전 각도 refinement, tx/ty 정규화 항 추가
// 원저자 구현 기반에 기능 강화 및 안정성 보강
#include "intensity_lidar_corner_detection.hpp"

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Helper - safe clamp
template <typename T>
static T clamp_val(T v, T lo, T hi) {
    return std::min(hi, std::max(lo, v));
}

// ---------------------------------------------------------------------------
// 1) PCD 문자열 파싱 (ASCII PCD 지원)
// (원본 구현 유지, 디버깅/안정성 메시지 보강)
std::vector<PointXYZI> parsePCDString(const std::string& pcd_string) {
    std::vector<PointXYZI> lidar_points;
    std::vector<std::string> fields;
    int intensity_field_index = -1;
    bool header_parsed = false;
    bool data_started = false;

    std::stringstream ss(pcd_string);
    std::string line;

    while (std::getline(ss, line)) {
        // trim
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        if (line.empty()) continue;

        if (!header_parsed) {
            if (line.rfind("FIELDS", 0) == 0) {
                std::stringstream ls(line);
                std::string token;
                ls >> token; // "FIELDS"
                while (ls >> token) fields.push_back(token);
                auto it = std::find(fields.begin(), fields.end(), "intensity");
                if (it != fields.end()) {
                    intensity_field_index = static_cast<int>(std::distance(fields.begin(), it));
                } else {
                    std::cerr << "[parsePCDString] Warning: intensity field not found; will use dummy value.\n";
                }
            } else if (line.rfind("DATA", 0) == 0) {
                std::stringstream ls(line);
                std::string tok;
                ls >> tok; // "DATA"
                if (!(ls >> tok)) {
                    std::cerr << "[parsePCDString] Error: invalid DATA line.\n";
                    return {};
                }
                if (tok != "ascii") {
                    std::cerr << "[parsePCDString] Error: only ascii PCD supported by this parser.\n";
                    return {};
                }
                header_parsed = true;
                data_started = true;
                // continue reading data lines
            }
        } else if (data_started) {
            std::stringstream ls(line);
            std::vector<std::string> parts;
            std::string part;
            while (ls >> part) parts.push_back(part);
            if (fields.size() > 0 && parts.size() != fields.size()) {
                // not necessarily fatal (some PCD variations) -> skip
                continue;
            }
            try {
                PointXYZI p;
                // find indices for x,y,z
                auto itx = std::find(fields.begin(), fields.end(), "x");
                auto ity = std::find(fields.begin(), fields.end(), "y");
                auto itz = std::find(fields.begin(), fields.end(), "z");
                if (itx == fields.end() || ity == fields.end() || itz == fields.end()) {
                    // fallback: assume first three columns are x y z
                    p.x = std::stod(parts[0]);
                    p.y = std::stod(parts[1]);
                    p.z = std::stod(parts[2]);
                } else {
                    p.x = std::stod(parts[std::distance(fields.begin(), itx)]);
                    p.y = std::stod(parts[std::distance(fields.begin(), ity)]);
                    p.z = std::stod(parts[std::distance(fields.begin(), itz)]);
                }
                if (intensity_field_index != -1 && static_cast<size_t>(intensity_field_index) < parts.size()) {
                    p.intensity = std::stod(parts[intensity_field_index]);
                } else {
                    p.intensity = 128.0; // dummy
                }
                lidar_points.push_back(p);
            } catch (...) {
                continue;
            }
        }
    }

    if (!header_parsed) {
        std::cerr << "[parsePCDString] Error: header not found or DATA != ascii.\n";
        return {};
    }
    if (lidar_points.empty()) {
        std::cerr << "[parsePCDString] Warning: no points parsed.\n";
    } else {
        std::cout << "[parsePCDString] Parsed " << lidar_points.size() << " points.\n";
    }
    return lidar_points;
}

// ---------------------------------------------------------------------------
// 2) Intensity 분류: black / gray / white (Eigen 기반)
// 반환: classified vector (0 black, 1 white, -1 gray), tau_l, tau_h
std::tuple<Eigen::VectorXi, double, double> classifyIntensityColor(const Eigen::VectorXd& intensities, double epsilon_g) {
    if (intensities.size() == 0) return std::make_tuple(Eigen::VectorXi(), 0.0, 0.0);

    double R_L = intensities.minCoeff();
    double R_H = intensities.maxCoeff();

    // 안전: 같을 때 대비
    if (std::abs(R_H - R_L) < 1e-9) {
        Eigen::VectorXi cls = Eigen::VectorXi::Constant(intensities.size(), -1);
        return std::make_tuple(cls, R_L, R_H);
    }

    double tau_l = ((epsilon_g - 1.0) * R_L + R_H) / epsilon_g;
    double tau_h = (R_L + (epsilon_g - 1.0) * R_H) / epsilon_g;

    Eigen::VectorXi classified = Eigen::VectorXi::Constant(intensities.size(), -1);
    for (int i = 0; i < intensities.size(); ++i) {
        if (intensities[i] < tau_l) classified[i] = 0;
        else if (intensities[i] > tau_h) classified[i] = 1;
        else classified[i] = -1;
    }

    std::cout << "[classifyIntensityColor] R_L=" << R_L << " R_H=" << R_H
              << " tau_l=" << tau_l << " tau_h=" << tau_h << "\n";
    return std::make_tuple(classified, tau_l, tau_h);
}

// ---------------------------------------------------------------------------
// 3) Checkerboard coloring model: given x,y in board local coords -> pattern color
// grid_size_x_squares, grid_size_y_squares : number of squares (not internal corners)
// checker_size_m : square side length
int getCheckerboardPatternColor(double x, double y, int grid_size_x_squares, int grid_size_y_squares, double checker_size_m) {
    if (checker_size_m <= 0) return -1;
    int col = static_cast<int>(std::floor(x / checker_size_m));
    int row = static_cast<int>(std::floor(y / checker_size_m));
    if (!(col >= 0 && col < grid_size_x_squares && row >= 0 && row < grid_size_y_squares)) return -1;
    return ((row + col) % 2 == 0) ? 1 : 0; // convention: even -> white(1), odd -> black(0)
}

// ---------------------------------------------------------------------------
// 4) costFunction - 보강: tx/ty 정규화 항 추가
// params: [tx, ty, theta_z]
// points_in_pca_plane_2d: Nx2 (x,y in PCA plane coordinates centered at origin)
// classified_colors: N (0 black, 1 white, -1 gray)
// grid_size_x_squares/grid_size_y_squares = number of squares along axes (internal_corners +1)
// checker_size_m : square length
double costFunction(const Eigen::Vector3d& params,
                    const Eigen::MatrixXd& points_in_pca_plane_2d,
                    const Eigen::VectorXi& classified_colors,
                    int grid_size_x_squares, int grid_size_y_squares, double checker_size_m) {
    double tx = params[0];
    double ty = params[1];
    double theta_z = params[2];

    double cos_t = std::cos(theta_z), sin_t = std::sin(theta_z);
    Eigen::Matrix2d Rz;
    Rz << cos_t, -sin_t,
          sin_t,  cos_t;

    // model extents (centered at origin): we'll consider model coordinate origin centered
    double board_w = grid_size_x_squares * checker_size_m;
    double board_h = grid_size_y_squares * checker_size_m;
    double model_min_x = -board_w / 2.0;
    double model_max_x =  board_w / 2.0;
    double model_min_y = -board_h / 2.0;
    double model_max_y =  board_h / 2.0;

    // transform points by tx,ty and rotation (apply translation first then rotate relative to model origin)
    // but we want to transform points to model coordinates, so subtract tx,ty then rotate by Rz^T
    Eigen::MatrixXd shifted = points_in_pca_plane_2d.rowwise() - Eigen::RowVector2d(tx, ty);
    Eigen::MatrixXd transformed = (Rz.transpose() * shifted.transpose()).transpose(); // Nx2

    double cost = 0.0;
    const double out_of_bounds_penalty_scale = std::max(board_w, board_h) * 10.0; // larger penalty for being outside

    for (int i = 0; i < transformed.rows(); ++i) {
        int ci = classified_colors[i]; // -1 gray, 0 black, 1 white
        if (ci == -1) continue; // ignore gray

        double px = transformed(i, 0);
        double py = transformed(i, 1);

        double lookup_x = px + board_w / 2.0;
        double lookup_y = py + board_h / 2.0;

        int model_color = getCheckerboardPatternColor(lookup_x, lookup_y, grid_size_x_squares, grid_size_y_squares, checker_size_m);

        if (model_color == -1) {
            // point falls outside the board -> penalize proportionally to distance to nearest edge
            double dx = 0.0, dy = 0.0;
            if (px < model_min_x) dx = model_min_x - px;
            else if (px > model_max_x) dx = px - model_max_x;
            if (py < model_min_y) dy = model_min_y - py;
            else if (py > model_max_y) dy = py - model_max_y;
            double dist_pen = (dx + dy) * out_of_bounds_penalty_scale;
            cost += dist_pen;
        } else {
            // point falls inside board: if color mismatch -> small penalty weighted by distance to pattern center
            if (ci != model_color) {
                // compute distance to nearest pattern side to weight the penalty (closer -> less confident)
                double local_x = fmod(std::fabs(lookup_x), checker_size_m);
                double local_y = fmod(std::fabs(lookup_y), checker_size_m);
                double dx_side = std::min(local_x, checker_size_m - local_x);
                double dy_side = std::min(local_y, checker_size_m - local_y);
                double prox = std::min(dx_side, dy_side);
                double base_pen = 1.0;
                double weight = 1.0 + (checker_size_m - prox) / checker_size_m; // more weight if point near center
                cost += base_pen * weight;
            }
        }
    }

    // --- tx/ty normalization penalty ---
    // prefer solutions where tx,ty are within a reasonable fraction of the board extents.
    // This prevents solutions where the optimizer shifts the model far away and gets lower mismatch by accident.
    double tx_limit = board_w * 0.6; // allowed ~ +-60% of board width
    double ty_limit = board_h * 0.6;
    double tx_pen = 0.0;
    if (std::abs(tx) > tx_limit) tx_pen = (std::abs(tx) - tx_limit) * out_of_bounds_penalty_scale;
    double ty_pen = 0.0;
    if (std::abs(ty) > ty_limit) ty_pen = (std::abs(ty) - ty_limit) * out_of_bounds_penalty_scale;
    cost += (tx_pen + ty_pen);

    return cost;
}

// ---------------------------------------------------------------------------
// 5) Main function: 논문 방식 corner estimation (Powell-style greedy search + local refinement)
//    - PCA 축 보정: 축을 직교화하고, 법선/방향성 고정
//    - 영역 크롭, intensity 분류
//    - coarse optimization (grid-like) + local refinement (small angle perturbation)
//    - 모델에서 3D 코너들을 추출해서 리턴
// 참고: 함수 시그니처는 원본과 동일하게 유지
std::vector<PointXYZI> estimateChessboardCornersPaperMethod(
    const std::vector<PointXYZI>& lidar_points_full_vec,
    int internal_corners_x, int internal_corners_y, double checker_size_m,
    bool flip_normal_direction) {

    std::cout << "[estimateChessboardCornersPaperMethod] 시작: points=" << lidar_points_full_vec.size()
              << " internal_corners_x=" << internal_corners_x << " internal_corners_y=" << internal_corners_y
              << " checker_size_m=" << checker_size_m << "\n";

    if (lidar_points_full_vec.empty()) {
        std::cerr << "[estimateChessboardCornersPaperMethod] Error: empty input\n";
        return {};
    }
    if (checker_size_m <= 0) {
        std::cerr << "[estimateChessboardCornersPaperMethod] Error: invalid checker_size_m\n";
        return {};
    }

    const int Npts = static_cast<int>(lidar_points_full_vec.size());
    Eigen::MatrixXd pts3d(Npts, 3);
    Eigen::VectorXd intens(Npts);
    for (int i = 0; i < Npts; ++i) {
        pts3d(i, 0) = lidar_points_full_vec[i].x;
        pts3d(i, 1) = lidar_points_full_vec[i].y;
        pts3d(i, 2) = lidar_points_full_vec[i].z;
        intens(i)   = lidar_points_full_vec[i].intensity;
    }

    if (Npts < 4) {
        std::cerr << "[estimateChessboardCornersPaperMethod] Error: not enough points\n";
        return {};
    }

    // --- PCA ---
    Eigen::RowVector3d centroid = pts3d.colwise().mean();
    Eigen::MatrixXd centered = pts3d.rowwise() - centroid;
    Eigen::Matrix3d cov = (centered.transpose() * centered) / double(std::max(1, Npts - 1));
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d eigvals = es.eigenvalues();
    Eigen::Matrix3d eigvecs = es.eigenvectors();

    // sort eigenpairs descending
    std::vector<std::pair<double, Eigen::Vector3d>> pairs;
    for (int i = 0; i < 3; ++i) pairs.emplace_back(eigvals(i), eigvecs.col(i));
    std::sort(pairs.begin(), pairs.end(), [](auto &a, auto &b){ return a.first > b.first; });

    Eigen::Matrix3d pcs;
    Eigen::Vector3d sorted_vals;
    for (int i = 0; i < 3; ++i) {
        pcs.col(i) = pairs[i].second;
        sorted_vals[i] = pairs[i].first;
    }

    // principal axes: pcs.col(0)=major direction in plane, col(1)=2nd, col(2)=normal
    Eigen::Vector3d v_z = pcs.col(2);
    // ensure normal direction roughly points toward lidar origin (centroid vector)
    if (v_z.dot(-centroid.transpose()) < 0) v_z = -v_z;
    if (flip_normal_direction) v_z = -v_z;
    v_z.normalize();

    // make v_x orthogonal to v_z and oriented consistently
    Eigen::Vector3d v_x = pcs.col(0);
    v_x = v_x - v_x.dot(v_z) * v_z;
    if (v_x.norm() < 1e-6) {
        // fallback: pick arbitrary axis orthogonal to v_z
        Eigen::Vector3d arbitrary(1,0,0);
        if (std::abs(v_z.dot(arbitrary)) > 0.9) arbitrary = Eigen::Vector3d(0,1,0);
        v_x = arbitrary - arbitrary.dot(v_z) * v_z;
    }
    v_x.normalize();

    // ensure v_x roughly pointing to +X in LiDAR frame (reduce directional ambiguity)
    if (v_x.dot(Eigen::Vector3d(1,0,0)) < 0) v_x = -v_x;

    Eigen::Vector3d v_y = v_z.cross(v_x);
    v_y.normalize();
    // re-orthonormalize to be safe
    v_x = v_y.cross(v_z);
    v_x.normalize();

    Eigen::Matrix3d Rpcs;
    Rpcs.col(0) = v_x;
    Rpcs.col(1) = v_y;
    Rpcs.col(2) = v_z;

    std::cout << "[estimateChessboardCornersPaperMethod] PCA 축 (정규화후):\n" << Rpcs << "\n";

    // transform points into PCA frame and drop z
    Eigen::MatrixXd pts_pca = (centered * Rpcs).eval(); // Nx3
    Eigen::MatrixXd pts_pca_2d = pts_pca.leftCols(2); // Nx2

    // cropping to bounding box (with small margin)
    double min_x = pts_pca_2d.col(0).minCoeff();
    double max_x = pts_pca_2d.col(0).maxCoeff();
    double min_y = pts_pca_2d.col(1).minCoeff();
    double max_y = pts_pca_2d.col(1).maxCoeff();
    double margin = std::max(checker_size_m * 0.05, 0.01);
    double min_x_c = min_x - margin, max_x_c = max_x + margin;
    double min_y_c = min_y - margin, max_y_c = max_y + margin;

    std::vector<int> indices_keep;
    indices_keep.reserve(Npts);
    for (int i = 0; i < pts_pca_2d.rows(); ++i) {
        double x = pts_pca_2d(i,0), y = pts_pca_2d(i,1);
        if (x >= min_x_c && x <= max_x_c && y >= min_y_c && y <= max_y_c) indices_keep.push_back(i);
    }
    if (indices_keep.empty()) {
        std::cerr << "[estimateChessboardCornersPaperMethod] Error: no points after PCA-crop\n";
        return {};
    }

    Eigen::MatrixXd pts_opt(indices_keep.size(), 2);
    Eigen::VectorXd intens_opt(indices_keep.size());
    for (size_t i = 0; i < indices_keep.size(); ++i) {
        pts_opt.row(i) = pts_pca_2d.row(indices_keep[i]);
        intens_opt(i) = intens(indices_keep[i]);
    }

    // classify intensities
    Eigen::VectorXi classified;
    double tau_l, tau_h;
    std::tie(classified, tau_l, tau_h) = classifyIntensityColor(intens_opt, 2.0); // epsilon_g=2 default

    // prepare optimization
    int num_squares_x = internal_corners_x + 1;
    int num_squares_y = internal_corners_y + 1;

    // initial guesses: center at mean point in PCA 2D coords; theta guesses several quadrants
    Eigen::Vector3d initial_guess;
    initial_guess[0] = pts_opt.col(0).mean();
    initial_guess[1] = pts_opt.col(1).mean();
    std::vector<double> theta_candidates = {0.0, M_PI/2.0, M_PI, 3.0*M_PI/2.0};

    double best_cost = std::numeric_limits<double>::infinity();
    Eigen::Vector3d best_params = initial_guess;
    // coarse multi-start search (small iterative coordinate descent style)
    for (double theta0 : theta_candidates) {
        Eigen::Vector3d params = initial_guess;
        params[2] = theta0;

        // greedy coordinate-descent with decreasing steps
        double tx_step = std::max(checker_size_m * 0.5, 0.05);
        double ty_step = tx_step;
        double th_step = M_PI/8.0; // 22.5 deg
        for (int outer = 0; outer < 6; ++outer) {
            bool improved = false;
            // try small moves in tx, ty, theta
            std::vector<Eigen::Vector3d> trials;
            trials.push_back(params);
            trials.push_back(params + Eigen::Vector3d(tx_step, 0, 0));
            trials.push_back(params + Eigen::Vector3d(-tx_step, 0, 0));
            trials.push_back(params + Eigen::Vector3d(0, ty_step, 0));
            trials.push_back(params + Eigen::Vector3d(0, -ty_step, 0));
            trials.push_back(params + Eigen::Vector3d(0, 0, th_step));
            trials.push_back(params + Eigen::Vector3d(0, 0, -th_step));

            for (auto &tp : trials) {
                double c = costFunction(tp, pts_opt, classified, num_squares_x, num_squares_y, checker_size_m);
                if (c < best_cost - 1e-9) {
                    best_cost = c;
                    best_params = tp;
                    improved = true;
                }
            }
            if (improved) {
                params = best_params;
            } else {
                tx_step *= 0.5;
                ty_step *= 0.5;
                th_step *= 0.5;
            }
        }
    }

    // Local refinement around best_params: small angle sweep to fix slight skew (예: 기울어짐 문제 보정)
    {
        Eigen::Vector3d base = best_params;
        double best_local_cost = best_cost;
        Eigen::Vector3d best_local = best_params;

        // perturb theta in small increments and re-optimize tx/ty locally (few iterations)
        for (double dth = -5.0*M_PI/180.0; dth <= 5.0*M_PI/180.0; dth += 1.0*M_PI/180.0) {
            Eigen::Vector3d cand = base;
            cand[2] = base[2] + dth;
            // local refine tx/ty using small coordinate descent
            double txs = checker_size_m * 0.1;
            double tys = txs;
            for (int it = 0; it < 6; ++it) {
                bool any = false;
                std::vector<Eigen::Vector3d> trylist = {
                    cand,
                    cand + Eigen::Vector3d(txs,0,0),
                    cand + Eigen::Vector3d(-txs,0,0),
                    cand + Eigen::Vector3d(0,tys,0),
                    cand + Eigen::Vector3d(0,-tys,0)
                };
                for (auto &tst : trylist) {
                    double c = costFunction(tst, pts_opt, classified, num_squares_x, num_squares_y, checker_size_m);
                    if (c + 1e-9 < costFunction(cand, pts_opt, classified, num_squares_x, num_squares_y, checker_size_m)) {
                        cand = tst;
                        any = true;
                    }
                }
                if (!any) {
                    txs *= 0.5;
                    tys *= 0.5;
                }
            }
            double cand_cost = costFunction(cand, pts_opt, classified, num_squares_x, num_squares_y, checker_size_m);
            if (cand_cost < best_local_cost) {
                best_local_cost = cand_cost;
                best_local = cand;
            }
        }
        best_params = best_local;
        best_cost = best_local_cost;
    }

    std::cout << "[estimateChessboardCornersPaperMethod] 최종 최적 파라미터 (tx,ty,theta_deg)=("
              << best_params[0] << ", " << best_params[1] << ", " << best_params[2]*180.0/M_PI << ")\n";
    std::cout << "[estimateChessboardCornersPaperMethod] 최종 비용=" << best_cost << "\n";

    // --- 3) 모델에서 2D 코너 좌표 생성 (PCA 평면 좌표계) ---
    // internal_corners -> corners are at grid intersections
    std::vector<Eigen::Vector2d> corners_2d_model;
    int nx = internal_corners_x;
    int ny = internal_corners_y;
    // model origin is centered -> corner positions must reflect that
    double board_w = (nx + 1) * checker_size_m;
    double board_h = (ny + 1) * checker_size_m;
    double x0 = -board_w / 2.0;
    double y0 = -board_h / 2.0;
    for (int r = 0; r < ny; ++r) {
        for (int c = 0; c < nx; ++c) {
            // corners counted in same convention as image detection (lower-left origin)
            double cx = x0 + (c+1) * checker_size_m; // note: corners correspond to internal intersections
            double cy = y0 + (r+1) * checker_size_m;
            corners_2d_model.emplace_back(cx, cy);
        }
    }

    // apply best transform to bring model corners into PCA coordinate frame
    Eigen::Matrix2d Rz_best;
    double ct = std::cos(best_params[2]), st = std::sin(best_params[2]);
    Rz_best << ct, -st, st, ct;
    std::vector<Eigen::Vector3d> corners_3d_in_lidar;
    corners_3d_in_lidar.reserve(corners_2d_model.size());
    for (auto &c2 : corners_2d_model) {
        // the model corner coordinate in model frame -> inverse transform to PCA frame:
        // we have transformed points -> transformed = Rz^T * (pts - t)
        // so to map model->pca frame: p_pca = Rz * model + t
        Eigen::Vector2d p_pca = Rz_best * c2 + Eigen::Vector2d(best_params[0], best_params[1]);
        // p_pca is x,y in PCA frame; z comes from centroid z (0 in plane)
        Eigen::Vector3d p_pca3;
        p_pca3[0] = p_pca[0];
        p_pca3[1] = p_pca[1];
        p_pca3[2] = 0.0;
        // convert back to original LiDAR coords: pts_original = Rpcs * p_pca3 + centroid'
        Eigen::Vector3d p_world = Rpcs * p_pca3 + centroid.transpose();
        PointXYZI outp;
        outp.x = p_world[0];
        outp.y = p_world[1];
        outp.z = p_world[2];
        outp.intensity = 0.0;
        corners_3d_in_lidar.emplace_back(p_world);
    }

    // convert to return type vector<PointXYZI>
    std::vector<PointXYZI> out_corners;
    out_corners.reserve(corners_3d_in_lidar.size());
    for (const auto &v : corners_3d_in_lidar) {
        PointXYZI p;
        p.x = v[0]; p.y = v[1]; p.z = v[2]; p.intensity = 0.0;
        out_corners.push_back(p);
    }

    std::cout << "[estimateChessboardCornersPaperMethod] 검출된 코너 수: " << out_corners.size() << "\n";

    return out_corners;
}
