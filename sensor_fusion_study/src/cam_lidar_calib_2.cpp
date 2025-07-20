#include <rclcpp/rclcpp.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <fstream>
#include <string>
#include <filesystem>
#include "sensor_msgs/msg/image.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <cv_bridge/cv_bridge.h>
#include <numeric>               // for std::accumulate
#include <algorithm>             // for std::min/max
#include <pcl/common/centroid.h> // For computing centroid
#include <pcl/common/pca.h>      // For PCA
#include <limits>                // For std::numeric_limits
#include <pcl/point_cloud.h>     // For pcl::PointCloud
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <visualization_msgs/msg/marker_array.hpp> // For RViz visualization
#include <opencv2/core/eigen.hpp> // Required for cv::cv2eigen

namespace fs = std::filesystem;

// Powell 최적화에 필요한 데이터 구조체
// 이 구조체는 비용 함수가 PointCloud 데이터 및 체스보드 파라미터에 접근할 수 있도록 합니다.
struct CostFunctionData {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_points_M; // P_M (체스보드에서 추출된 LiDAR 포인트)
    double tau_l; // 강도 임계값 (흑백 분류 기준)
    double square_size; // 체스보드 한 칸의 크기
    int pattern_size_cols; // 체스보드 내부 코너 열 개수
    int pattern_size_rows; // 체스보드 내부 코너 행 개수
    Eigen::Vector3d plane_normal; // 체스보드 평면의 법선 벡터 (PCA Mu3)
    Eigen::Vector3d board_x_axis; // 체스보드 평면의 X축 (PCA Mu1)
    Eigen::Vector3d board_y_axis; // 체스보드 평면의 Y축 (PCA Mu2)
    Eigen::Vector3d chessboard_origin_3d_base; // 체스보드 모델의 (0,0,0) 기준점 (LiDAR 프레임)
};

class CamLidarCalibNode : public rclcpp::Node
{
public:
    CamLidarCalibNode()
        : Node("cam_lidar_calib")
    {
        initializedParameters();

        std::string connect = "aa"; // "flir" 대신 "aa"로 설정되어 있어 파일 로드 모드
        if (connect == "flir")
        {
            sub_cam_ = this->create_subscription<sensor_msgs::msg::Image>("/flir_camera/image_raw", rclcpp::SensorDataQoS(),
                                                                          std::bind(&CamLidarCalibNode::imageCallback, this, std::placeholders::_1));

            sub_lidar_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/ouster/points", rclcpp::SensorDataQoS(),
                                                                                  std::bind(&CamLidarCalibNode::pcdCallback, this, std::placeholders::_1));
        }
        else
        {
            // 키보드 입력을 처리하는 타이머
            timer__ = this->create_wall_timer(
                std::chrono::milliseconds(500),
                std::bind(&CamLidarCalibNode::timerCallback, this));
        }

        pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_points", 10);
        pub_checker_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("checker_points", 10);
        pub_lidar2cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar2cam_points", 10);
        pub_cb_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cb_object_points", 10);
        pub_cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cam_points", 10);
        pub_binarized_intensity_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("binarized_intensity_points", 10); // New publisher for binarized intensity
        pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("chessboard_markers", 10); // For visualization

        // PointCloud2 메시지를 주기적으로 퍼블리시하는 타이머 (주석 해제됨)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30), // 30ms 주기로 퍼블리시
            std::bind(&CamLidarCalibNode::pcdTimerCallback, this));
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_cam_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
    cv::Mat current_frame_;
    cv::Mat last_image_;                                                                   // Latest image data
    pcl::PointCloud<pcl::PointXYZI>::Ptr last_cloud_{new pcl::PointCloud<pcl::PointXYZI>}; // Latest point cloud data
    std::string img_path_;
    std::string pcd_path_;
    std::string absolute_path_;
    std::string one_cam_result_path_;
    int frame_counter_ = 0;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_plane_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_checker_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_lidar2cam_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cb_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_tf_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cam_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_binarized_intensity_; // New publisher
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_; // RViz markers publisher
    sensor_msgs::msg::PointCloud2 plane_msg_;
    sensor_msgs::msg::PointCloud2 checker_msg_;
    sensor_msgs::msg::PointCloud2 lidar2cam_points_;
    sensor_msgs::msg::PointCloud2 cb_points_msg_;
    sensor_msgs::msg::PointCloud2 cam_points_msg_;
    sensor_msgs::msg::PointCloud2 binarized_intensity_msg_; // New message for binarized intensity
    rclcpp::TimerBase::SharedPtr timer_;                    // pcdTimerCallback을 위한 타이머
    rclcpp::TimerBase::SharedPtr timer__;                   // timerCallback을 위한 타이머
    cv::Size board_size_;                                   // Chessboard parameters (internal corners)
    double square_size_;                                    // Chessboard parameters
    cv::Mat intrinsic_matrix_, distortion_coeffs_;          // Camera intrinsics
    cv::Mat cb2cam_rvec_, cb2cam_tvec_;
    cv::Mat lidar2cam_R_, lidar2cam_t_;
    std::vector<int> successful_indices_;
    std::vector<cv::String> image_files_;
    std::vector<std::vector<cv::Point2f>> img_points_;
    std::vector<std::vector<cv::Point3f>> obj_points_;
    std::vector<cv::Mat> rvecs_, tvecs_;
    double rms_;

    // 이미지에서 찾은 2D 코너
    std::vector<cv::Point2f> image_corners_latest_;
    // 라이다에서 찾은 체스보드 평면 포인트
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_plane_points_latest_{new pcl::PointCloud<pcl::PointXYZI>};

    // Intensity Thresholding Parameters
    double intensity_min_threshold_; // 파라미터로 설정된 최소 강도 (R_L_est 역할)
    double intensity_max_threshold_; // 파라미터로 설정된 최대 강도 (R_H_est 역할)
    double tau_l_;                   // 계산된 단일 임계값 (흑백 분류 기준)

    // ROI and RANSAC parameters
    double roi_min_x_, roi_max_x_;
    double roi_min_y_, roi_max_y_;
    double roi_min_z_, roi_max_z_;
    double ransac_distance_threshold_;
    int ransac_max_iterations_;

    // 체커보드 파라미터
    int pattern_size_cols_; // 내부 코너 개수
    int pattern_size_rows_; // 내부 코너 개수

    // 이미 인식된 평면 정보
    Eigen::Vector3d plane_normal_;
    Eigen::Vector3d plane_center_;
    // 추가: 체스보드 평면의 X, Y 축 및 원점 (클래스 멤버로 선언)
    Eigen::Vector3d board_x_axis_;
    Eigen::Vector3d board_y_axis_;
    Eigen::Vector3d chessboard_origin_3d_base_;
    
    // 체커보드 코너를 저장할 벡터
    std::vector<Eigen::Vector3d> chessboard_corners_3d_;

    // 강도 분류를 위한 enum (GRAY 제거)
    enum PatternColor
    {
        BLACK = 0,
        WHITE = 1,
        UNKNOWN = 2 // Add UNKNOWN state for cases with no data
    };

    // 강도 값을 흑색 또는 백색으로 분류하는 도우미 함수 (극단화)
    PatternColor classifyIntensity(float intensity)
    {
        if (intensity < tau_l_)
        {
            return BLACK;
        }
        else
        { // 기준보다 높으면 최대값과 같은 색 (WHITE)
            return WHITE;
        }
    }

    // PointCloud의 강도 분포를 기반으로 흑백 패턴 분류 임계값을 추정하는 함수
    // 사용자 요청에 따라 intensity_min_threshold_와 intensity_max_threshold_의 중간값을 단일 임계값으로 사용
    void calculateIntensityThresholds(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud)
    {
        if (cloud->empty())
        {
            tau_l_ = (intensity_min_threshold_ + intensity_max_threshold_) / 2.0;
            RCLCPP_WARN(this->get_logger(), "Input cloud is empty for intensity threshold calculation. Using default mid-point.");
            return;
        }

        // 파라미터로 설정된 최소/최대 강도 값을 사용하여 단일 임계값 계산
        // 논문 4.2.2 섹션에 따르면, 코너 추정 시 epsilon_g = 2 (회색 영역 없음)
        tau_l_ = (intensity_min_threshold_ + intensity_max_threshold_) / 2.0;

        RCLCPP_INFO(this->get_logger(), "Calculated single intensity threshold (tau_l_): %.2f", tau_l_);
    }

    // 로컬 PointCloud 내에서 강도(색상) 변화 횟수를 확인하는 함수
    // 코너 주변에서 색상 변화가 최소 3번 일어나는지 확인
    int checkIntensityTransitions(const pcl::PointCloud<pcl::PointXYZI>::Ptr &local_cloud,
                                  const Eigen::Vector3d &corner_center,
                                  const Eigen::Vector3d &board_x_axis,
                                  const Eigen::Vector3d &board_y_axis)
    {
        if (local_cloud->empty())
        {
            return 0;
        }

        // 코너 중심을 기준으로 점들을 2D 평면에 투영하고 각도와 색상 정보를 저장
        std::vector<std::pair<double, PatternColor>> projected_points_with_color;
        projected_points_with_color.reserve(local_cloud->points.size());

        for (const auto &pt : local_cloud->points)
        {
            Eigen::Vector3d vec = pt.getVector3fMap().cast<double>() - corner_center;
            // 평면 내 2D 좌표로 투영
            double proj_x = vec.dot(board_x_axis);
            double proj_y = vec.dot(board_y_axis);

            // 코너 중심으로부터의 각도를 계산하여 정렬에 사용
            double angle = std::atan2(proj_y, proj_x);
            projected_points_with_color.push_back({angle, classifyIntensity(pt.intensity)});
        }

        // 각도 기준으로 정렬
        std::sort(projected_points_with_color.begin(), projected_points_with_color.end());

        int transitions = 0;
        if (projected_points_with_color.size() < 2)
        {
            return 0; // 점이 충분하지 않으면 변화 없음
        }

        // 정렬된 점들을 순회하며 색상 변화 횟수 계산
        PatternColor prev_color = projected_points_with_color[0].second;

        for (size_t i = 1; i < projected_points_with_color.size(); ++i)
        {
            PatternColor current_color = projected_points_with_color[i].second;
            // 이전 색상과 다르면 변화로 간주
            if (current_color != prev_color)
            {
                transitions++;
            }
            prev_color = current_color;
        }
        // 마지막 점과 첫 점 사이의 변화도 확인 (원형으로 연결된 것으로 간주)
        if (projected_points_with_color.back().second != projected_points_with_color[0].second)
        {
            transitions++;
        }

        return transitions;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
            cv::namedWindow("FLIR View", cv::WINDOW_NORMAL);
            cv::resizeWindow("FLIR View", 640, 480);
            cv::imshow("FLIR View", current_frame_);
            inputKeyboard(current_frame_);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void pcdCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::fromROSMsg(*msg, *last_cloud_);
    }

    void pcdTimerCallback()
    {
        // RCLCPP_INFO(this->get_logger(), "pcdTimerCallback is actively running."); // 디버깅용 로그
        pub_plane_->publish(plane_msg_);
        pub_checker_->publish(checker_msg_);
        pub_lidar2cam_points_->publish(lidar2cam_points_);
        pub_cb_points_->publish(cb_points_msg_);
        pub_cam_points_->publish(cam_points_msg_);
        pub_binarized_intensity_->publish(binarized_intensity_msg_); // Publish binarized intensity message
        // pub_markers_ is published directly when needed, not in timer callback
    }

    void initializedParameters()
    {
        std::string where = "company";
        readWritePath(where);

        cv::FileStorage fs(one_cam_result_path_ + "one_cam_calib_result.yaml", cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            RCLCPP_WARN(rclcpp::get_logger("initializedParameters"), "Failed open one_cam_calib_result.yaml file! Shutting down node.");
            rclcpp::shutdown();
        }
        else
        {
            int cols_squares, rows_squares;
            // Assuming checkerboard_cols and checkerboard_rows in YAML refer to the number of squares.
            fs["checkerboard_cols"] >> cols_squares;
            fs["checkerboard_rows"] >> rows_squares;
            fs["square_size"] >> square_size_;
            fs["intrinsic_matrix"] >> intrinsic_matrix_;
            fs["distortion_coefficients"] >> distortion_coeffs_;
            fs.release();

            // Convert number of squares to number of internal corners for OpenCV and LiDAR logic.
            // 사용자 요청에 따라 YAML의 cols_squares, rows_squares를 내부 코너 개수로 직접 사용
            pattern_size_cols_ = cols_squares;
            pattern_size_rows_ = rows_squares;

            board_size_ = cv::Size(pattern_size_cols_, pattern_size_rows_); // For OpenCV functions like findChessboardCorners
        }
        RCLCPP_INFO(this->get_logger(), "Chessboard internal corner pattern size: %dx%d, square size: %.3f m",
                    pattern_size_cols_, pattern_size_rows_, square_size_);

        // Parameterization - Declare and get parameters for filters and RANSAC
        this->declare_parameter<double>("intensity_min_threshold", 1.0);
        this->declare_parameter<double>("intensity_max_threshold", 5747.0);
        this->declare_parameter<double>("roi_min_x", -3.0);
        this->declare_parameter<double>("roi_max_x", 0.0);
        this->declare_parameter<double>("roi_min_y", -0.8);
        this->declare_parameter<double>("roi_max_y", 0.5);
        this->declare_parameter<double>("roi_min_z", -0.63);
        this->declare_parameter<double>("roi_max_z", 3.0);
        this->declare_parameter<double>("ransac_distance_threshold", 0.02);
        this->declare_parameter<int>("ransac_max_iterations", 1000);

        this->get_parameter("intensity_min_threshold", intensity_min_threshold_);
        this->get_parameter("intensity_max_threshold", intensity_max_threshold_);
        this->get_parameter("roi_min_x", roi_min_x_);
        this->get_parameter("roi_max_x", roi_max_x_);
        this->get_parameter("roi_min_y", roi_min_y_);
        this->get_parameter("roi_max_y", roi_max_y_);
        this->get_parameter("roi_min_z", roi_min_z_);
        this->get_parameter("roi_max_z", roi_max_z_);
        this->get_parameter("ransac_distance_threshold", ransac_distance_threshold_);
        this->get_parameter("ransac_max_iterations", ransac_max_iterations_);

        RCLCPP_INFO(this->get_logger(), "Loaded intensity filter: [%.1f, %.1f]", intensity_min_threshold_, intensity_max_threshold_);
        RCLCPP_INFO(this->get_logger(), "Loaded ROI X: [%.1f, %.1f], Y: [%.1f, %.1f], Z: [%.1f, %.1f]",
                    roi_min_x_, roi_max_x_, roi_min_y_, roi_max_y_, roi_min_z_, roi_max_z_);
        RCLCPP_INFO(this->get_logger(), "Loaded RANSAC: dist_thresh=%.3f, max_iter=%d", ransac_distance_threshold_, ransac_max_iterations_);
    }

    void readWritePath(std::string where)
    {
        std::string change_path;
        if (where == "company")
        {
            change_path = "/antlab/sensor_fusion_study_ws";
        }
        else if (where == "home")
        {
            change_path = "/icrs/sensor_fusion_study_ws";
        }

        absolute_path_ = "/home" + change_path + "/src/sensor_fusion_study/cam_lidar_calib/";
        img_path_ = absolute_path_ + "images/";
        pcd_path_ = absolute_path_ + "pointclouds/";
        one_cam_result_path_ = "/home" + change_path + "/src/sensor_fusion_study/one_cam_calib/";

        fs::create_directories(img_path_);
        fs::create_directories(pcd_path_);
    }

    void timerCallback()
    {
        if (last_image_.empty())
        {
            cv::Mat dummy = cv::Mat::zeros(480, 640, CV_8UC3);
            cv::putText(dummy, "No camera image", cv::Point(50, 240),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Camera Image", dummy);
        }

        inputKeyboard(last_image_);
    }

    void inputKeyboard(const cv::Mat &frame)
    {
        int key = cv::waitKey(1); // Non-blocking waitKey
        if (key == 's')
        {
            saveFrame(frame.clone());
        }
        else if (key == 'c')
        {
            RCLCPP_INFO(this->get_logger(), "C key pressed. Starting calibration process.");
            try
            {
                findData();
                solveCameraPlane();
                detectLidarPlane(); // This function will call corner estimation.
                RCLCPP_INFO(this->get_logger(), "Calibration process finished successfully.");
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Caught C++ exception during calibration: %s", e.what());
            }
            catch (...)
            {
                RCLCPP_ERROR(this->get_logger(), "Caught unknown exception during calibration.");
            }
        }
        else if (key == 'e')
        {
            RCLCPP_INFO(this->get_logger(), "E key pressed. Shutting down node.");
            rclcpp::shutdown();
        }
    }

    void saveFrame(const cv::Mat &frame)
    {
        if (frame.empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("saveFrame"), "No current camera frame captured yet! Cannot save image.");
            return;
        }

        std::string img_filename = img_path_ + "img_" + std::to_string(frame_counter_) + ".png";
        cv::imwrite(img_filename, frame);

        if (last_cloud_->empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("saveFrame"), "No point cloud captured yet! Cannot save PCD.");
            return;
        }
        std::string pcd_filename = pcd_path_ + "pcd_" + std::to_string(frame_counter_) + ".pcd";
        pcl::io::savePCDFileBinary(pcd_filename, *last_cloud_);
        RCLCPP_INFO(this->get_logger(), "Saved image and pointcloud.");
        frame_counter_++;
    }

    void findData()
    {
        std::vector<cv::String> image_files;
        cv::glob(img_path_ + "*.png", image_files, false);

        if (image_files.empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("findData"), "SHUTDOWN_CAUSE: No image files found in %s. Shutting down node!", img_path_.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Found %zu image files.", image_files.size());

        std::string first_image_path = img_path_ + "img_6.png"; // index 6 고정
        last_image_ = cv::imread(first_image_path, cv::IMREAD_COLOR);

        if (last_image_.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: Failed to load the first image from %s! Shutting down node.", first_image_path.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded first image: %s", first_image_path.c_str());

        std::vector<cv::String> pcd_files;
        cv::glob(pcd_path_ + "*.pcd", pcd_files, false);

        if (pcd_files.empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("findData"), "SHUTDOWN_CAUSE: No PCD files found in %s. Shutting down node!", pcd_path_.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Found %zu PCD files.", pcd_files.size());

        std::string first_pcd_path = pcd_path_ + "pcd_6.pcd"; // index 6 고정
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(first_pcd_path, *last_cloud_) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: Failed to load the first PCD from %s! Shutting down node.", first_pcd_path.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded first pointcloud: %s", first_pcd_path.c_str());

        RCLCPP_INFO(this->get_logger(), "Successfully loaded calibration data (image and pointcloud).");
    }

    int extractNumber(const std::string &filename)
    {
        std::string number;
        for (char c : filename)
        {
            if (std::isdigit(c))
            {
                number += c;
            }
        }
        return number.empty() ? -1 : std::stoi(number);
    }

    void runCalibrateFromFolder()
    {
        RCLCPP_INFO(this->get_logger(), "Start calibration...");

        cv::glob(img_path_ + "*.png", image_files_);
        if (image_files_.size() < 5)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough image (%lu)", image_files_.size());
            return;
        }
        std::sort(image_files_.begin(), image_files_.end(),
                  [this](const std::string &a, const std::string &b)
                  {
                      return extractNumber(a) < extractNumber(b);
                  });

        std::vector<cv::Point3f> objp;

        // objp for camera calibration should also use internal corner counts (board_size_)
        for (int i = 0; i < board_size_.height; ++i)
            for (int j = 0; j < board_size_.width; ++j)
                objp.emplace_back(j * square_size_, i * square_size_, 0.0f);

        successful_indices_.clear();

        for (size_t idx = 0; idx < image_files_.size(); ++idx)
        {
            const auto &file = image_files_[idx];
            cv::Mat img = cv::imread(file);
            if (img.empty())
                continue;

            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(img, board_size_, corners, // board_size_ is now (cols-1, rows-1)
                                                   cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

            if (found)
            {
                cv::Mat gray;
                cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

                img_points_.push_back(corners);
                obj_points_.push_back(objp);

                rms_ = cv::calibrateCamera(obj_points_, img_points_, board_size_,
                                           intrinsic_matrix_, distortion_coeffs_, rvecs_, tvecs_);

                RCLCPP_INFO(this->get_logger(), "RMS error: %.4f", rms_);
                successful_indices_.push_back(idx);
            }

            if (img_points_.empty())
            {
                RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: Failed calibration. No image points found. Shutting down node.");
                rclcpp::shutdown();
                return;
            }
        }
        std::cout << "[Camera calib] intrinsic_matrix:\n"
                  << intrinsic_matrix_ << std::endl;
        std::cout << "[Camera calib] distortion_coeffs:\n"
                  << distortion_coeffs_ << std::endl;
    }

    void solveCameraPlane()
    {
        cv::Mat img_color = last_image_;
        if (img_color.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: Failed to load saved image! Shutting down node.");
            rclcpp::shutdown();
        }

        cv::Mat img_gray;
        cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img_gray, board_size_, corners); // board_size_ is now (cols-1, rows-1)
        if (!found)
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: Chessboard not found in image! Shutting down node.");
            rclcpp::shutdown();
        }

        cv::cornerSubPix(img_gray, corners, cv::Size(5, 5),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        std::vector<cv::Point3f> object_points;
        // Object points for camera calibration: (0,0) at top-left internal corner, X-right, Y-down
        for (int i = 0; i < board_size_.height; i++) // row index (increases downwards)
        {
            for (int j = 0; j < board_size_.width; j++) // column index (increases rightwards)
            {
                object_points.emplace_back(j * square_size_, i * square_size_, 0.0);
            }
        }

        bool success = cv::solvePnP(object_points, corners,
                                    intrinsic_matrix_, distortion_coeffs_,
                                    cb2cam_rvec_, cb2cam_tvec_);

        if (!success)
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: solvePnP failed! Shutting down node.");
            rclcpp::shutdown();
        }

        cv::Mat R;
        cv::Rodrigues(cb2cam_rvec_, R);
        std::vector<cv::Point3f> cb2cam_points;
        cv::transform(object_points, cb2cam_points, R);
        for (auto &pt : cb2cam_points)
        {
            pt.x += cb2cam_tvec_.at<double>(0);
            pt.y += cb2cam_tvec_.at<double>(1);
            pt.z += cb2cam_tvec_.at<double>(2);
        }

        point3fVectorToPointCloud2(cb2cam_points, cam_points_msg_, "map", this->now());

        RCLCPP_INFO(this->get_logger(), "PnP rvec: [%f %f %f]", cb2cam_rvec_.at<double>(0), cb2cam_rvec_.at<double>(1), cb2cam_rvec_.at<double>(2));
        RCLCPP_INFO(this->get_logger(), "PnP tvec: [%f %f %f]", cb2cam_tvec_.at<double>(0), cb2cam_tvec_.at<double>(1), cb2cam_tvec_.at<double>(2));
    }

    void detectLidarPlane()
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_intensity(new pcl::PointCloud<pcl::PointXYZI>);

        // 1. 반사 강도 기반 필터링 (Reflectance Intensity Assisted)
        pcl::PassThrough<pcl::PointXYZI> pass_intensity;
        pass_intensity.setInputCloud(last_cloud_);
        pass_intensity.setFilterFieldName("intensity");
        // 파라미터로 설정된 강도 제한 사용
        pass_intensity.setFilterLimits(intensity_min_threshold_, intensity_max_threshold_);
        pass_intensity.filter(*cloud_filtered_intensity);

        if (cloud_filtered_intensity->empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("detectLidarPlane"), "SHUTDOWN_CAUSE: No points after intensity filtering. Shutting down node.");
            lidar_plane_points_latest_->clear();
            rclcpp::shutdown();
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZI>);
        // 2. ROI 필터링 (CropBox) - 체스보드가 있을 것으로 예상되는 영역
        pcl::CropBox<pcl::PointXYZI> crop;
        crop.setInputCloud(cloud_filtered_intensity);
        // 파라미터로 설정된 ROI 제한 사용
        crop.setMin(Eigen::Vector4f(roi_min_x_, roi_min_y_, roi_min_z_, 1.0));
        crop.setMax(Eigen::Vector4f(roi_max_x_, roi_max_y_, roi_max_z_, 1.0));
        crop.filter(*cloud_roi);

        if (cloud_roi->empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("detectLidarPlane"), "SHUTDOWN_CAUSE: No points after ROI filtering. Shutting down node.");
            lidar_plane_points_latest_->clear();
            rclcpp::shutdown();
        }

        // 3. RANSAC을 이용한 평면 분할
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        // 파라미터로 설정된 RANSAC 값 사용
        seg.setDistanceThreshold(ransac_distance_threshold_);
        seg.setMaxIterations(ransac_max_iterations_);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        seg.setInputCloud(cloud_roi);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("detectLidarPlane"), "SHUTDOWN_CAUSE: No planar model found in filtered LiDAR data. Shutting down node.");
            lidar_plane_points_latest_->clear();
            rclcpp::shutdown();
        }

        RCLCPP_INFO(this->get_logger(), "Detected Lidar Plane coefficients: %f %f %f %f",
                    coefficients->values[0], coefficients->values[1],
                    coefficients->values[2], coefficients->values[3]);

        // 4. 평면 내 포인트 추출
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(cloud_roi);
        extract.setIndices(inliers);
        extract.setNegative(false); // 인라이어(평면 내 점)만 추출
        extract.filter(*lidar_plane_points_latest_);

        if (lidar_plane_points_latest_->empty())
        {
            RCLCPP_WARN(this->get_logger(), "SHUTDOWN_CAUSE: Extracted LiDAR plane points are empty. Shutting down node.");
            rclcpp::shutdown();
        }

        pcl::toROSMsg(*lidar_plane_points_latest_, plane_msg_);
        plane_msg_.header.frame_id = "map"; // RViz에서 사용하는 좌표계 설정

        RCLCPP_INFO(this->get_logger(), "Detected %zu points in LiDAR plane.", lidar_plane_points_latest_->points.size());

        // 강도 임계값 계산 (전체 평면 PointCloud 기반)
        calculateIntensityThresholds(lidar_plane_points_latest_);

        // 이진화된 강도 포인트 시각화
        publishBinarizedIntensity(lidar_plane_points_latest_); // Call the new function here

        // --- 논문 4.2 코너 추정 시작 ---
        // 1. 평면의 중심점 및 법선 벡터 획득
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*lidar_plane_points_latest_, centroid);
        this->plane_center_ = Eigen::Vector3d(centroid.x(), centroid.y(), centroid.z());

        // PCA를 사용하여 평면의 주축을 다시 계산합니다.
        // 이전에 사용된 coefficients->values[0-2]는 RANSAC 평면의 법선이며,
        // PCA는 데이터의 분산 방향을 기반으로 주성분을 제공합니다.
        // 논문에서는 PCA를 통해 얻은 basis vectors (mu1, mu2, mu3)를 사용합니다.
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(lidar_plane_points_latest_);
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors(); // 열 벡터가 주성분입니다.

        // 초기 PCA 축
        Eigen::Vector3d temp_pca_axis1 = eigen_vectors.col(0).cast<double>().normalized();
        Eigen::Vector3d temp_pca_axis2 = eigen_vectors.col(1).cast<double>().normalized();
        Eigen::Vector3d temp_plane_normal = eigen_vectors.col(2).cast<double>().normalized(); // PCA의 세 번째 축을 법선으로 가정

        // 논문 4.2.1에 따라, mu3 (법선)는 LiDAR 좌표계의 원점 방향을 향해야 합니다.
        // LiDAR 원점은 (0,0,0)으로 가정합니다.
        // 평면 중심에서 원점까지의 벡터와 법선 벡터의 내적을 통해 방향을 확인합니다.
        Eigen::Vector3d vec_center_to_origin = -this->plane_center_;
        if (temp_plane_normal.dot(vec_center_to_origin) < 0) {
            temp_plane_normal *= -1.0; // 법선이 원점 반대 방향을 향하면 뒤집습니다.
        }
        this->plane_normal_ = temp_plane_normal; // 최종 법선 할당

        RCLCPP_INFO(this->get_logger(), "Initial PCA Axis 1: %.3f, %.3f, %.3f", temp_pca_axis1.x(), temp_pca_axis1.y(), temp_pca_axis1.z());
        RCLCPP_INFO(this->get_logger(), "Initial PCA Axis 2: %.3f, %.3f, %.3f", temp_pca_axis2.x(), temp_pca_axis2.y(), temp_pca_axis2.z());
        RCLCPP_INFO(this->get_logger(), "Adjusted Plane Normal: %.3f, %.3f, %.3f", this->plane_normal_.x(), this->plane_normal_.y(), this->plane_normal_.z());


        // Determine which PCA axis corresponds to the chessboard's width and height
        // Project points onto both PCA axes to find their extents
        double extent1_min = std::numeric_limits<double>::max();
        double extent1_max = std::numeric_limits<double>::min();
        double extent2_min = std::numeric_limits<double>::max();
        double extent2_max = std::numeric_limits<double>::min();

        for (const auto &pt : lidar_plane_points_latest_->points)
        {
            Eigen::Vector3d vec = pt.getVector3fMap().cast<double>() - this->plane_center_;
            double proj1 = vec.dot(temp_pca_axis1);
            double proj2 = vec.dot(temp_pca_axis2);

            if (proj1 < extent1_min) extent1_min = proj1;
            if (proj1 > extent1_max) extent1_max = proj1;
            if (proj2 < extent2_min) extent2_min = proj2;
            if (proj2 > extent2_max) extent2_max = proj2;
        }

        double len1 = extent1_max - extent1_min;
        double len2 = extent2_max - extent2_min;

        double chessboard_physical_width = (pattern_size_cols_ - 1) * square_size_;
        double chessboard_physical_height = (pattern_size_rows_ - 1) * square_size_;

        // Assign board_x_axis_ and board_y_axis_ based on which PCA axis better matches the physical dimensions
        if (std::abs(len1 - chessboard_physical_width) < std::abs(len2 - chessboard_physical_width)) {
            // temp_pca_axis1 is closer to width, temp_pca_axis2 is closer to height
            this->board_x_axis_ = temp_pca_axis1;
            this->board_y_axis_ = temp_pca_axis2;
        } else {
            // temp_pca_axis2 is closer to width, temp_pca_axis1 is closer to height
            this->board_x_axis_ = temp_pca_axis2;
            this->board_y_axis_ = temp_pca_axis1;
        }

        // Ensure board_x_axis_ and board_y_axis_ form a right-handed system with plane_normal_
        // and are orthogonal to plane_normal_
        this->board_x_axis_ = this->board_x_axis_.normalized();
        this->board_y_axis_ = this->plane_normal_.cross(this->board_x_axis_).normalized();
        this->board_x_axis_ = this->board_y_axis_.cross(this->plane_normal_).normalized(); // Re-orthogonalize x-axis

        RCLCPP_INFO(this->get_logger(), "Assigned Board X axis (before intensity check): %.3f, %.3f, %.3f", this->board_x_axis_.x(), this->board_x_axis_.y(), this->board_x_axis_.z());
        RCLCPP_INFO(this->get_logger(), "Assigned Board Y axis (before intensity check): %.3f, %.3f, %.3f", this->board_y_axis_.x(), this->board_y_axis_.y(), this->board_y_axis_.z());

        // Now, resolve 180-degree ambiguities using intensity patterns.
        // We assume the (0,0) square of the chessboard is WHITE.
        // Define the temporary chessboard origin for intensity checks as the top-left internal corner.
        Eigen::Vector3d temp_chessboard_origin_3d_base_for_intensity = this->plane_center_ - (chessboard_physical_width / 2.0) * this->board_x_axis_ + (chessboard_physical_height / 2.0) * this->board_y_axis_;
        
        // Define a reference point (e.g., center of the first square (0,0))
        // The (0,0) square is the one immediately to the right and down from the top-left internal corner.
        // Its center is at (square_size/2, square_size/2) in the local chessboard frame.
        Eigen::Vector3d ref_point_3d = temp_chessboard_origin_3d_base_for_intensity +
                                       this->board_x_axis_ * (square_size_ / 2.0) +
                                       (-this->board_y_axis_) * (square_size_ / 2.0); // Y-axis for local coords is downwards (using negative of PCA Y-axis)

        // Sample a point in the square to the right (col + 1, row)
        Eigen::Vector3d sample_x_3d = ref_point_3d + this->board_x_axis_ * square_size_;
        // Sample a point in the square below (col, row + 1)
        Eigen::Vector3d sample_y_3d = ref_point_3d + (-this->board_y_axis_) * square_size_; // Y-axis for local coords is downwards

        pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
        kdtree.setInputCloud(lidar_plane_points_latest_);
        double search_radius_color = square_size_ * 0.3; // Search within 30% of square size

        auto get_avg_intensity_and_classify = [&](const Eigen::Vector3d& point_3d) {
            pcl::PointXYZI s_pt;
            s_pt.x = point_3d.x(); s_pt.y = point_3d.y(); s_pt.z = point_3d.z();
            std::vector<int> idx_search;
            std::vector<float> dist_sq;
            if (kdtree.radiusSearch(s_pt, search_radius_color, idx_search, dist_sq) > 0) {
                float avg_int = 0.0;
                for (int idx : idx_search) avg_int += lidar_plane_points_latest_->points[idx].intensity;
                return classifyIntensity(avg_int / idx_search.size());
            }
            return UNKNOWN; // Return UNKNOWN if no points found
        };

        PatternColor color_ref = get_avg_intensity_and_classify(ref_point_3d);
        PatternColor color_x = get_avg_intensity_and_classify(sample_x_3d);
        PatternColor color_y = get_avg_intensity_and_classify(sample_y_3d);

        // Only proceed if reference color is known
        if (color_ref != UNKNOWN) {
            // Adjust for global pattern inversion if (0,0) square is detected as BLACK
            // This means the entire intensity pattern is inverted (e.g., black squares are high intensity)
            if (color_ref == BLACK) {
                RCLCPP_INFO(this->get_logger(), "Detected (0,0) square as BLACK. Inverting expected pattern for axis checks.");
                // If the reference square is black, then the adjacent squares should be white for comparison
                // Only invert if the color is known
                if (color_x != UNKNOWN) color_x = (color_x == WHITE) ? BLACK : WHITE; 
                if (color_y != UNKNOWN) color_y = (color_y == WHITE) ? BLACK : WHITE; 
            }

            // Check and potentially flip board_x_axis_
            // If color_x is the same as color_ref, it means the axis is pointing in the wrong direction
            // because adjacent squares should have alternating colors.
            if (color_x != UNKNOWN && color_x == color_ref) { 
                this->board_x_axis_ *= -1.0;
                RCLCPP_INFO(this->get_logger(), "Flipped board_x_axis_ due to intensity pattern mismatch.");
            }

            // Check and potentially flip board_y_axis_
            // Note: board_y_axis_ here is the PCA axis direction (conceptually "up").
            // If color_y is the same as color_ref, it means the *downwards* direction (which is -board_y_axis_)
            // is actually pointing to the same color. So, we need to flip board_y_axis_ to make -board_y_axis_
            // point to the alternating color.
            if (color_y != UNKNOWN && color_y == color_ref) { 
                this->board_y_axis_ *= -1.0;
                RCLCPP_INFO(this->get_logger(), "Flipped board_y_axis_ due to intensity pattern mismatch (Y-downwards check).");
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "Reference point for intensity pattern check is UNKNOWN. Cannot resolve 180-degree ambiguity for in-plane axes.");
        }

        // Re-orthogonalize after potential flips to maintain a perfect right-handed system
        this->board_y_axis_ = this->plane_normal_.cross(this->board_x_axis_).normalized();
        this->board_x_axis_ = this->board_y_axis_.cross(this->plane_normal_).normalized();


        RCLCPP_INFO(this->get_logger(), "Final Board X axis: %.3f, %.3f, %.3f", this->board_x_axis_.x(), this->board_x_axis_.y(), this->board_x_axis_.z());
        RCLCPP_INFO(this->get_logger(), "Final Board Y axis: %.3f, %.3f, %.3f", this->board_y_axis_.x(), this->board_y_axis_.y(), this->board_y_axis_.z());

        // Recalculate chessboard_origin_3d_base_ with the final, correctly oriented axes
        // Now setting the origin to the top-left internal corner of the chessboard.
        // Assuming board_x_axis_ points right and board_y_axis_ points up (from center).
        this->chessboard_origin_3d_base_ = this->plane_center_ - (chessboard_physical_width / 2.0) * this->board_x_axis_ + (chessboard_physical_height / 2.0) * this->board_y_axis_;

        // Publish chessboard plane coordinate axes to RViz
        Eigen::Matrix4d chessboard_pose_matrix = Eigen::Matrix4d::Identity();
        chessboard_pose_matrix.block<3,1>(0,0) = this->board_x_axis_;
        chessboard_pose_matrix.block<3,1>(0,1) = -this->board_y_axis_; // Y-axis of marker points downwards
        chessboard_pose_matrix.block<3,1>(0,2) = this->plane_normal_;
        chessboard_pose_matrix.block<3,1>(0,3) = this->chessboard_origin_3d_base_;
        publishCoordinateAxes(chessboard_pose_matrix, "map", "chessboard_plane_axes", 0.1, 0.005);

        // --- NEW VISUALIZATION ADDITIONS ---
        // Visualize temporary origin for intensity checks
        publishCornerMarkers({temp_chessboard_origin_3d_base_for_intensity}, "map", "temp_origin_intensity_check", 0.02, 1.0, 0.0, 1.0); // Magenta sphere for temp origin

        // Visualize sample points for intensity check with their classified colors
        std::vector<Eigen::Vector3d> sample_points_for_intensity_check = {ref_point_3d, sample_x_3d, sample_y_3d};
        std::vector<std::tuple<double, double, double>> sample_colors;
        PatternColor classified_ref = get_avg_intensity_and_classify(ref_point_3d);
        PatternColor classified_x = get_avg_intensity_and_classify(sample_x_3d);
        PatternColor classified_y = get_avg_intensity_and_classify(sample_y_3d);

        // Assign colors based on classification
        auto get_color_for_pattern = [](PatternColor pc) {
            if (pc == BLACK) return std::make_tuple(0.0, 0.0, 0.0); // Black
            if (pc == WHITE) return std::make_tuple(1.0, 1.0, 1.0); // White
            return std::make_tuple(0.5, 0.5, 0.5); // Gray for unknown
        };

        sample_colors.push_back(get_color_for_pattern(classified_ref));
        sample_colors.push_back(get_color_for_pattern(classified_x));
        sample_colors.push_back(get_color_for_pattern(classified_y));

        publishColoredCornerMarkers(sample_points_for_intensity_check, sample_colors, "map", "intensity_sample_points", 0.015);

        // Visualize initial chessboard grid (before Powell optimization)
        std::vector<Eigen::Vector3d> initial_grid_corners;
        for (int r = 0; r < pattern_size_rows_; ++r)
        {
            for (int c = 0; c < pattern_size_cols_; ++c)
            {
                Eigen::Vector3d local_corner_2d(c * square_size_, r * square_size_, 0.0);
                Eigen::Matrix3d local_chessboard_frame_to_lidar_initial;
                local_chessboard_frame_to_lidar_initial.col(0) = this->board_x_axis_;
                local_chessboard_frame_to_lidar_initial.col(1) = -this->board_y_axis_; // Y-axis points downwards
                local_chessboard_frame_to_lidar_initial.col(2) = this->plane_normal_;
                
                Eigen::Vector3d initial_corner_3d = local_chessboard_frame_to_lidar_initial * local_corner_2d + this->chessboard_origin_3d_base_;
                initial_grid_corners.push_back(initial_corner_3d);
            }
        }
        publishCornerMarkers(initial_grid_corners, "map", "initial_chessboard_grid", 0.008, 0.0, 1.0, 1.0); // Cyan small spheres for initial grid

        // --- END NEW VISUALIZATION ADDITIONS ---


        // Powell 최적화에 사용할 데이터 준비
        CostFunctionData cost_data;
        cost_data.cloud_points_M = lidar_plane_points_latest_;
        cost_data.tau_l = tau_l_;
        cost_data.square_size = square_size_;
        cost_data.pattern_size_cols = pattern_size_cols_;
        cost_data.pattern_size_rows = pattern_size_rows_;
        cost_data.plane_normal = this->plane_normal_; 
        cost_data.board_x_axis = this->board_x_axis_; 
        cost_data.board_y_axis = this->board_y_axis_; 
        cost_data.chessboard_origin_3d_base = this->chessboard_origin_3d_base_; 

        // Powell 최적화 파라미터: [theta_z_M, t_x_M, t_y_M]
        // 논문에 따라 초기값은 모두 0으로 설정
        std::vector<double> initial_params = {0.0, 0.0, 0.0}; // theta_z (rad), tx (m), ty (m)
        double h = 0.01; // 초기 검색 증분
        double tolerate = 1.0e-6; // 수렴 허용 오차
        int maxit = 100; // 최대 반복 횟수

        RCLCPP_INFO(this->get_logger(), "Starting Powell optimization for chessboard pose...");
        std::vector<double> optimized_params = min_powell(initial_params, h, tolerate, maxit, &cost_data);
        RCLCPP_INFO(this->get_logger(), "Powell optimization finished. Optimized params: [theta_z: %.4f rad, tx: %.4f m, ty: %.4f m]",
                    optimized_params[0], optimized_params[1], optimized_params[2]);

        double opt_theta_z = optimized_params[0];
        double opt_tx = optimized_params[1];
        double opt_ty = optimized_params[2];

        // 최적화된 파라미터를 사용하여 최종 체스보드 모델의 3D 코너 계산
        chessboard_corners_3d_.clear();
        
        // 최적화된 변환 행렬 (체스보드 평면 내에서)
        Eigen::Matrix3d R_opt_z;
        R_opt_z << std::cos(opt_theta_z), -std::sin(opt_theta_z), 0,
                   std::sin(opt_theta_z),  std::cos(opt_theta_z), 0,
                   0, 0, 1;
        Eigen::Vector3d t_opt(opt_tx, opt_ty, 0.0);

        // 체스보드 모델의 2D 코너 (local chessboard plane)
        // (0,0)을 기준으로 내부 코너들을 생성 (X-right, Y-down)
        for (int r = 0; r < pattern_size_rows_; ++r)
        {
            for (int c = 0; c < pattern_size_cols_; ++c)
            {
                // 체스보드 모델의 로컬 2D 좌표 (X-right, Y-down)
                Eigen::Vector3d local_corner_2d(c * square_size_, r * square_size_, 0.0);
                
                // 로컬 2D 좌표를 PCA 축에 맞게 회전 및 이동하여 3D 공간으로 변환
                Eigen::Vector3d transformed_local_corner = R_opt_z * local_corner_2d + t_opt;

                // PCA 축을 다시 LiDAR 좌표계로 변환 (X-right, Y-down)
                Eigen::Matrix3d local_chessboard_frame_to_lidar;
                local_chessboard_frame_to_lidar.col(0) = this->board_x_axis_;    // X-axis points right
                local_chessboard_frame_to_lidar.col(1) = -this->board_y_axis_;   // Y-axis points downwards (using negative of PCA Y-axis)
                local_chessboard_frame_to_lidar.col(2) = this->plane_normal_;   // Z-axis points out of the board

                Eigen::Vector3d final_corner_3d = local_chessboard_frame_to_lidar * transformed_local_corner + this->chessboard_origin_3d_base_;
                chessboard_corners_3d_.push_back(final_corner_3d);
            }
        }

        if (chessboard_corners_3d_.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: No valid chessboard corners detected after refinement. Shutting down node.");
            rclcpp::shutdown();
        }

        // Publish markers for estimated chessboard corners (LiDAR)
        publishCornerMarkers(chessboard_corners_3d_, "map", "chessboard_corners", 0.01, 1.0, 0.5, 0.0); // Orange corners

        // 추정된 코너들을 cv::Point3f 벡터로 변환 (calibrateLidarCameraFinal 함수를 위해)
        std::vector<cv::Point3f> estimated_cv_corners;
        for (const auto &eigen_pt : chessboard_corners_3d_)
        {
            estimated_cv_corners.emplace_back(eigen_pt.x(), eigen_pt.y(), eigen_pt.z());
        }
        
        // 캘리브레이션 함수 호출
        calibrateLidarCameraFinal(last_cloud_, estimated_cv_corners);
    }

    // --- 비용 함수 및 헬퍼 함수 시작 ---

    // f_g(r_i) 함수: 강도 r_i가 회색 영역에 속하는지 여부 (논문 Equation 6)
    // r_i가 [tau_l, tau_h] 밖에 있으면 1, 아니면 0
    // 현재 구현에서는 tau_l_ 하나만 사용하므로, r_i == tau_l_ 인 경우를 제외하고 1로 반환
    static double fg(float r_i, double tau_l) {
        // 논문에서는 tau_l과 tau_h가 정의되지만, 코너 추정 시에는 epsilon_g=2로 설정되어 회색 영역이 없다고 가정
        // 따라서 tau_l_ 하나로 흑백을 나눕니다.
        // 여기서는 논문의 정의에 따라 r_i가 회색 영역에 '속하지 않으면' 1을 반환합니다.
        // 즉, 흑색 또는 백색으로 명확히 분류되는 경우 1을 반환합니다.
        // 현재 tau_l_이 단일 임계값이므로, 모든 점은 흑색 또는 백색으로 분류됩니다.
        // 따라서 항상 1을 반환하도록 단순화합니다.
        return 1.0; 
    }

    // isPointInPolygon 함수: 점이 주어진 다각형 내부에 있는지 확인
    // point_2d: (x, y) 좌표
    // polygon_vertices_2d: 다각형의 꼭짓점 벡터 (2D)
    static bool isPointInPolygon(const Eigen::Vector2d& point_2d, const std::vector<Eigen::Vector2d>& polygon_vertices_2d) {
        // Ray casting algorithm (간단한 구현)
        bool inside = false;
        size_t num_vertices = polygon_vertices_2d.size();
        for (size_t i = 0, j = num_vertices - 1; i < num_vertices; j = i++) {
            const Eigen::Vector2d& vi = polygon_vertices_2d[i];
            const Eigen::Vector2d& vj = polygon_vertices_2d[j];

            if (((vi.y() > point_2d.y()) != (vj.y() > point_2d.y())) &&
                (point_2d.x() < (vj.x() - vi.x()) * (point_2d.y() - vi.y()) / (vj.y() - vi.y()) + vi.x())) {
                inside = !inside;
            }
        }
        return inside;
    }

    // getExpectedPatternColor 함수: 2D 체스보드 좌표에 따라 예상되는 패턴 색상 반환 (0: BLACK, 1: WHITE)
    // (0,0) 칸이 WHITE라고 가정합니다.
    static PatternColor getExpectedPatternColor(const Eigen::Vector2d& point_2d, double square_size) {
        int col_idx = static_cast<int>(std::floor(point_2d.x() / square_size));
        int row_idx = static_cast<int>(std::floor(point_2d.y() / square_size));

        // 체스보드 패턴: (0,0)이 WHITE라고 가정
        if ((col_idx + row_idx) % 2 == 0) {
            return WHITE;
        } else {
            return BLACK;
        }
    }

    // calculateFd 함수: 논문 Equation 8에 정의된 f_d 계산
    // point_2d: (x, y) 좌표
    // rect_min_x, rect_max_x, rect_min_y, rect_max_y: 사각형의 경계
    static double calculateFd(const Eigen::Vector2d& point_2d, 
                              double rect_min_x, double rect_max_x, 
                              double rect_min_y, double rect_max_y) {
        double dx1 = std::abs(point_2d.x() - rect_min_x);
        double dx2 = std::abs(point_2d.x() - rect_max_x);
        double dy1 = std::abs(point_2d.y() - rect_min_y);
        double dy2 = std::abs(point_2d.y() - rect_max_y);
        return std::min(dx1, dx2) + std::min(dy1, dy2);
    }

    // calculateCostFunction: Powell 최적화에 사용될 비용 함수 (논문 Equation 5)
    // params: [theta_z_M, t_x_M, t_y_M]
    // user_data: CostFunctionData 구조체 포인터
    static double calculateCostFunction(std::vector<double> params, void* user_data_ptr) {
        CostFunctionData* data = static_cast<CostFunctionData*>(user_data_ptr);

        double theta_z_M = params[0];
        double t_x_M = params[1];
        double t_y_M = params[2];

        // 체스보드 평면 내에서의 회전 변환 (Z축 회전)
        Eigen::Matrix3d R_M;
        R_M << std::cos(theta_z_M), -std::sin(theta_z_M), 0,
               std::sin(theta_z_M),  std::cos(theta_z_M), 0,
               0, 0, 1;
        Eigen::Vector3d t_M(t_x_M, t_y_M, 0.0);

        double total_cost = 0.0;

        // 전체 체스보드 모델의 경계 (2D 평면)
        double board_width = (data->pattern_size_cols - 1) * data->square_size;
        double board_height = (data->pattern_size_rows - 1) * data->square_size;
        
        // 전체 체스보드 모델의 2D 꼭짓점 (local chessboard plane)
        // (0,0), (board_width, 0), (board_width, board_height), (0, board_height)
        std::vector<Eigen::Vector2d> G_vertices_2d;
        G_vertices_2d.emplace_back(0.0, 0.0);
        G_vertices_2d.emplace_back(board_width, 0.0);
        G_vertices_2d.emplace_back(board_width, board_height);
        G_vertices_2d.emplace_back(0.0, board_height);

        // LiDAR 평면 포인트들을 순회하며 비용 계산
        for (const auto& p_M_raw : data->cloud_points_M->points) {
            // PointCloud의 3D 점을 Eigen::Vector3d로 변환
            Eigen::Vector3d p_M_3d(p_M_raw.x, p_M_raw.y, p_M_raw.z);

            // 1. LiDAR 좌표계의 점을 체스보드 평면의 로컬 좌표계로 변환 (PCA 변환 역적용)
            // 논문 4.2.1에 따라, PCA 변환 후 점들은 XOY 평면에 있고 Z=0.
            // 여기서는 이미 lidar_plane_points_latest_가 PCA 변환 및 중심 이동이 적용된 상태라고 가정하고,
            // 추가적인 theta_z_M, t_x_M, t_y_M 변환만 적용합니다.
            // 즉, p_M_3d는 이미 PCA 변환 후의 점입니다.

            // PCA 축을 다시 LiDAR 좌표계로 변환하는 행렬의 역행렬
            Eigen::Matrix3d board_axes_to_lidar;
            board_axes_to_lidar.col(0) = data->board_x_axis;
            board_axes_to_lidar.col(1) = data->board_y_axis;
            board_axes_to_lidar.col(2) = data->plane_normal;
            
            Eigen::Matrix3d lidar_to_board_axes = board_axes_to_lidar.inverse();

            // LiDAR 좌표계의 점을 체스보드 모델의 로컬 평면 좌표로 변환
            // 1. 중심 이동 역적용: LiDAR 점 - 체스보드 원점
            Eigen::Vector3d p_M_centered = p_M_3d - data->chessboard_origin_3d_base;
            // 2. PCA 축으로 회전: PCA 축 기준 로컬 좌표
            Eigen::Vector3d p_M_local_plane = lidar_to_board_axes * p_M_centered;

            // 최적화 파라미터 (theta_z_M, t_x_M, t_y_M) 적용
            // p_hat_i = T_r(theta_M, t_M, p_M)
            // p_M_local_plane은 이미 PCA 변환된 점이므로, 여기에 추가적인 평면 내 변환을 적용
            Eigen::Vector3d p_hat_i_3d = R_M * p_M_local_plane + t_M;
            Eigen::Vector2d p_hat_i_2d(p_hat_i_3d.x(), p_hat_i_3d.y()); // 2D로 투영

            // f_g(r_i) 계산 (논문 Equation 6)
            double fg_val = fg(p_M_raw.intensity, data->tau_l);
            if (fg_val == 0) continue; // 회색 영역에 속하면 무시

            // c_i: LiDAR 강도로부터 추정된 색상
            PatternColor c_i = (p_M_raw.intensity < data->tau_l) ? BLACK : WHITE;

            // f_in(p_hat_i, G): 점이 전체 체스보드 모델 내부에 있는지
            bool is_in_G = isPointInPolygon(p_hat_i_2d, G_vertices_2d);

            double term1_cost = 0.0;
            double term2_cost = 0.0;

            if (is_in_G) {
                // 점이 체스보드 내부에 있는 경우 (첫 번째 항 활성화)
                // hat_c_i: 점이 속한 패턴의 예상 색상
                PatternColor hat_c_i = getExpectedPatternColor(p_hat_i_2d, data->square_size);

                // V_i: 점 p_hat_i가 속해야 할 패턴의 꼭짓점
                int col_idx = static_cast<int>(std::floor(p_hat_i_2d.x() / data->square_size));
                int row_idx = static_cast<int>(std::floor(p_hat_i_2d.y() / data->square_size));

                double pattern_min_x = col_idx * data->square_size;
                double pattern_max_x = (col_idx + 1) * data->square_size;
                double pattern_min_y = row_idx * data->square_size;
                double pattern_max_y = (row_idx + 1) * data->square_size;

                // |c_i - hat_c_i|: 관측된 색상과 예상 색상 간의 차이 (0 또는 1)
                double color_diff = std::abs(static_cast<double>(c_i) - static_cast<double>(hat_c_i));
                
                // f_d(p_hat_i, V_i): 점과 해당 패턴 경계까지의 거리
                double fd_Vi = calculateFd(p_hat_i_2d, pattern_min_x, pattern_max_x, pattern_min_y, pattern_max_y);
                
                term1_cost = color_diff * fd_Vi;
            } else {
                // 점이 체스보드 외부에 있는 경우 (두 번째 항 활성화)
                // f_d(p_hat_i, G): 점과 전체 체스보드 경계까지의 거리
                term2_cost = calculateFd(p_hat_i_2d, 0.0, board_width, 0.0, board_height);
            }
            total_cost += fg_val * (term1_cost + term2_cost);
        }
        return total_cost;
    }

    // --- Powell 최적화 함수들 (powell.cpp에서 통합) ---

    // 전역 래퍼 함수 (Powell 라이브러리가 이 형태로 콜백을 기대할 때 사용)
    // calculateCostFunction이 이미 static이므로 직접 호출 가능하지만,
    // powell.cpp의 기존 구조를 유지하기 위해 래퍼를 사용합니다.
    static double globalFitnessWrapper(std::vector<double> vx, void* user_data) {
        return calculateCostFunction(vx, user_data);
    }

    // fitness_direction: 단일 방향 검색을 위한 비용 함수 인터페이스
    static double fitness_direction(std::vector<double>& x, double lam, std::vector<double>& v, void* user_data){
        int n = x.size();
        std::vector<double> x_new(n, 0.0);
        for (int i = 0; i < n; ++i){
            x_new[i] = x[i] + lam * v[i];
        }
        return globalFitnessWrapper(x_new, user_data);
    }

    // bracket: 최소점의 구간을 괄호로 묶는 함수
    static std::tuple<double, double> bracket(std::vector<double>& x,
                                              std::vector<double>& v,
                                              double x1,
                                              double h,
                                              void* user_data){
        double c = 1.618033989;
        double x2 = h + x1;
        double f1, f2;
        f1 = fitness_direction(x, x1, v, user_data);
        f2 = fitness_direction(x, x2, v, user_data);

        if(f2 > f1){
            h = -h;
            x2 = x1 + h;
            f2 = fitness_direction(x, x2, v, user_data);
            if(f2 > f1){
                return std::make_tuple(x2, x1-h);
            }
        }

        for (int i = 0; i < 100; ++i){
            h = c * h;
            double x3 = x2 + h;
            double f3 = fitness_direction(x, x3, v, user_data);
            if(f3 > f2){
                return std::make_tuple(x1, x3);
            }
            x1 = x2;
            x2 = x3;
            f1 = f2;
            f2 = f3;
        }
        RCLCPP_ERROR(rclcpp::get_logger("PowellBracket"), "Bracket did not find a minimum");
        return std::make_tuple(0.0, 0.0); // 오류 발생 시 기본값 반환
    }

    // golden_section_search: 황금 분할 검색을 통해 단일 방향에서 최소값 찾기
    static std::tuple<double, double> golden_section_search(std::vector<double>& x,
                                                            std::vector<double>& v,
                                                            double a,
                                                            double b,
                                                            double tol = 1.0e-9,
                                                            void* user_data = nullptr){
        int nIter = static_cast<int>(std::ceil(-2.078087*std::log(tol/std::abs(b-a))));
        double R = 0.618033989;   // golden ratio
        double C = 1.0 - R;
        
        double x1 = R * a + C * b;
        double x2 = C * a + R * b;
        double f1, f2;
        f1 = fitness_direction(x, x1, v, user_data);
        f2 = fitness_direction(x, x2, v, user_data);
        for (int i = 0; i < nIter; ++i){
            if (f1 > f2){
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = C * a + R * b;
                f2 = fitness_direction(x, x2, v, user_data);
            }
            else{
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = R * a + C * b;
                f1 = fitness_direction(x, x1, v, user_data);
            }
        }
        if(f1 < f2){
            return std::make_tuple(x1, f1);
        }
        else{
            return std::make_tuple(x2, f2);
        }
    }

    // mse: 평균 제곱 오차 계산
    static double mse(std::vector<double> v1, std::vector<double> v2){
        double len = 0.0;
        for (size_t i = 0; i < v1.size(); ++i){
            len += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        }
        return std::sqrt(len / v1.size());
    }

    // min_powell: Powell의 최적화 메서드
    static std::vector<double> min_powell(std::vector<double>& x, // initial value
                                          double h,
                                          double tolerate,
                                          int maxit,
                                          void* user_data)
    {
        int n = x.size();                 // 설계 변수의 수
        std::vector<double> df(n, 0);          // 비용 감소량 저장
        // 방향 벡터 u를 행별로 저장
        std::vector<std::vector<double>> u(n, std::vector<double>(n, 0.0));
        // 방향 벡터 설정 (단위 벡터)
        for (int i = 0; i < n; ++i){
            u[i][i] = 1.0;
        }

        // 주 반복 루프
        for (int j = 0; j < maxit; j++) { 
            std::vector<double> x_old = x;
            double fitness_old = globalFitnessWrapper(x_old, user_data);
            std::vector<double> fitness_dir_min(n+1, 0.0);
            fitness_dir_min[0] = fitness_old;
            for (int i = 0; i < n; ++i){
                std::vector<double> v = u[i];
                double a, b, s;
                std::tie(a, b) = bracket(x, v, 0.0, h, user_data);
                std::tie(s, fitness_dir_min[i+1]) = golden_section_search(x, v, a, b, 1.0e-9, user_data);
                for (int k = 0; k < n; ++k){ // 'i' 대신 'k' 사용
                    x[k] = x[k] + s * v[k];
                }
            }
            for (int i = 0; i < n; ++i){
                df[i] = fitness_dir_min[i] - fitness_dir_min[i+1];
            }
            // 사이클의 마지막 라인 황금 분할 검색
            std::vector<double> v(n);
            for (int i = 0; i < n; ++i){
                v[i] = x[i] - x_old[i];
            }
            double a, b, s, dummy;
            std::tie(a, b) = bracket(x, v, 0.0, h, user_data);
            std::tie(s, dummy) = golden_section_search(x, v, a, b, 1.0e-9, user_data);
            // 검색 방향 간의 의존성
            for (int i = 0; i < n; ++i){
                x[i] = x[i] + s * v[i];
            }
            // 수렴 확인
            if(mse(x, x_old) < tolerate){
                std::cout << "found minimize value at " << j+1 << " step" << " with value: " << globalFitnessWrapper(x, user_data) << std::endl;
                return x;
            }
            // 가장 큰 감소량 식별 및 검색 방향 업데이트
            int i_max = 0;
            for (int i = 1; i < n; ++i){
                if(df[i] > df[i_max]){
                    i_max = i;
                }
            }
            for (int i = i_max; i < n-1; ++i){
                u[i] = u[i+1];
            }
            u[n-1] = v;
        }
        RCLCPP_WARN(rclcpp::get_logger("PowellOptimizer"), "Powell did not converge within max iterations.");
        return x; // 수렴하지 못한 경우 마지막 x 값 반환
    }
    // --- Powell 최적화 함수들 끝 ---


    void publishBinarizedIntensity(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud)
    {
        if (input_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Input cloud for binarized intensity is empty. Not publishing.");
            binarized_intensity_msg_.data.clear(); // Clear previous data
            return;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr binarized_rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        binarized_rgb_cloud->width = input_cloud->width;
        binarized_rgb_cloud->height = input_cloud->height;
        binarized_rgb_cloud->is_dense = input_cloud->is_dense;
        binarized_rgb_cloud->points.resize(input_cloud->points.size());

        for (size_t i = 0; i < input_cloud->points.size(); ++i)
        {
            const auto &pt_i = input_cloud->points[i];
            pcl::PointXYZRGB &pt_rgb = binarized_rgb_cloud->points[i];

            pt_rgb.x = pt_i.x;
            pt_rgb.y = pt_i.y;
            pt_rgb.z = pt_i.z;

            PatternColor color = classifyIntensity(pt_i.intensity);
            if (color == BLACK)
            {
                pt_rgb.r = 0;
                pt_rgb.g = 0;
                pt_rgb.b = 0;
            }
            else
            { // WHITE
                pt_rgb.r = 255;
                pt_rgb.g = 255;
                pt_rgb.b = 255;
            }
        }

        pcl::toROSMsg(*binarized_rgb_cloud, binarized_intensity_msg_);
        binarized_intensity_msg_.header.frame_id = "map";
        binarized_intensity_msg_.header.stamp = this->now();
        // The message will be published by pcdTimerCallback
    }

    // Renamed calibrateLidarCamera function and modified parameters
    void calibrateLidarCameraFinal(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_all,
                                   const std::vector<cv::Point3f> &estimated_chessboard_corners_lidar)
    {
        // lidar_points now uses estimated_chessboard_corners_lidar
        std::vector<cv::Point3f> lidar_points = estimated_chessboard_corners_lidar;

        std::vector<cv::Point3f> cb_object_points;
        // The object points represent the 3D coordinates of the chessboard corners in its own coordinate system (origin at one corner, Z=0)
        // This should also use internal corner counts.
        for (int i = 0; i < board_size_.height; i++) // rows of internal corners
        {
            for (int j = 0; j < board_size_.width; j++) // columns of internal corners
            {
                cb_object_points.emplace_back(j * square_size_, i * square_size_, 0.0);
            }
        }

        cv::Mat R;
        cv::Rodrigues(cb2cam_rvec_, R);
        std::vector<cv::Point3f> cb2cam_points;
        cv::transform(cb_object_points, cb2cam_points, R);
        for (auto &pt : cb2cam_points)
        {
            pt.x += cb2cam_tvec_.at<double>(0);
            pt.y += cb2cam_tvec_.at<double>(1);
            pt.z += cb2cam_tvec_.at<double>(2);
        }

        // 3. Match point counts for SVD
        // The number of estimated LiDAR corners should match the number of camera-detected corners.
        // If they don't match, it indicates a problem in corner detection for either sensor.
        // For SVD, the input point sets must have the same number of points.
        size_t num_lidar_corners = lidar_points.size();
        size_t num_cam_corners = cb2cam_points.size();

        if (num_lidar_corners == 0 || num_cam_corners == 0)
        {
            RCLCPP_ERROR(this->get_logger(), "No corners detected from one or both sensors for SVD computation. LiDAR corners: %zu, Camera corners: %zu", num_lidar_corners, num_cam_corners);
            // TODO 2: Robust Error Handling & Logging - Consider more graceful recovery or specific error codes.
            return;
        }

        if (num_lidar_corners != num_cam_corners)
        {
            RCLCPP_WARN(this->get_logger(), "Mismatch in corner counts! LiDAR: %zu, Camera: %zu. SVD might be inaccurate.", num_lidar_corners, num_cam_corners);
            // If counts differ, SVD will only use the minimum number of points, which might be incorrect.
            // A more robust approach would be to try to match corresponding corners.
            // For now, we'll resize to the smaller set.
            size_t n_points_svd = std::min(num_lidar_corners, num_cam_corners);
            if (lidar_points.size() > n_points_svd)
            {
                lidar_points.resize(n_points_svd);
            }
            if (cb2cam_points.size() > n_points_svd)
            {
                cb2cam_points.resize(n_points_svd);
            }
            RCLCPP_WARN(this->get_logger(), "Resized point sets for SVD to %zu points.", n_points_svd);
        }

        point3fVectorToPointCloud2(lidar_points, checker_msg_, "map", this->now()); // Publish estimated corners from LiDAR
        // Publish markers for estimated chessboard corners (LiDAR)
        publishCornerMarkers(convertCvPointsToEigen(lidar_points), "map", "lidar_corners_svd_input", 0.01, 1.0, 0.0, 1.0); // Magenta corners

        // 4. Calculate R, t based on SVD
        computeRigidTransformSVD(lidar_points, cb2cam_points, lidar2cam_R_, lidar2cam_t_);

        std::cout << "[Lidar → Camera] R:\n"
                  << lidar2cam_R_ << std::endl;
        std::cout << "[Lidar → Camera] t:\n"
                  << lidar2cam_t_ << std::endl;

        // Save
        saveMultipleToFile("txt", "cam_lidar_calib_result",
                           "[Lidar → Camera] R:\n", lidar2cam_R_,
                           "[Lidar → Camera] t:\n", lidar2cam_t_);

        // TODO 8: Output and Reporting - Save to YAML
        saveCalibrationResultToYaml(lidar2cam_R_, lidar2cam_t_);

        // Publish LiDAR to Camera transform as coordinate axes in RViz
        Eigen::Matrix4d lidar_to_cam_pose_matrix = Eigen::Matrix4d::Identity();
        // Convert cv::Mat R to Eigen::Matrix3d
        Eigen::Matrix3d eigen_R;
        cv::cv2eigen(lidar2cam_R_, eigen_R);
        lidar_to_cam_pose_matrix.block<3,3>(0,0) = eigen_R;
        // Convert cv::Mat t to Eigen::Vector3d
        Eigen::Vector3d eigen_t;
        cv::cv2eigen(lidar2cam_t_, eigen_t);
        lidar_to_cam_pose_matrix.block<3,1>(0,3) = eigen_t;
        publishCoordinateAxes(lidar_to_cam_pose_matrix, "map", "lidar_to_camera_transform", 0.2, 0.01);


        point3fVectorToPointCloud2(cb2cam_points, cb_points_msg_, "map", this->now()); // Publish corners seen from camera
        // Publish markers for camera's estimated corners
        publishCornerMarkers(convertCvPointsToEigen(cb2cam_points), "map", "camera_corners", 0.01, 0.0, 0.5, 1.0); // Cyan corners


        std::vector<cv::Point3f> lidar_points_all;
        for (const auto &p : cloud_all->points)
        {
            lidar_points_all.emplace_back(
                p.x,
                p.y,
                p.z);
        }

        std::vector<cv::Point3f> lidar2cam_points_all;
        for (const auto &pt : lidar_points_all)
        {
            cv::Mat pt_mat = (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
            cv::Mat pt_transformed = lidar2cam_R_ * pt_mat + lidar2cam_t_;

            lidar2cam_points_all.emplace_back(
                pt_transformed.at<double>(0),
                pt_transformed.at<double>(1),
                pt_transformed.at<double>(2));
        }

        point3fVectorToPointCloud2(lidar2cam_points_all, lidar2cam_points_, "map", this->now()); // Publish all LiDAR points after transformation

        // Image projection
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in_cam(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto &pt : lidar2cam_points_all)
        {
            cloud_in_cam->emplace_back(pt.x, pt.y, pt.z);
        }

        std::string filename = img_path_ + "/img_6.png"; // Changed from img_0.png to img_6.png
        cv::Mat img_color = cv::imread(filename, cv::IMREAD_COLOR);
        if (img_color.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: Failed to load image. Shutting down node.");
            rclcpp::shutdown();
        }

        projectLidarToImage(cloud_in_cam, img_color, last_image_);

        cv::namedWindow("Lidar Overlaid (All points)", cv::WINDOW_NORMAL);
        cv::resizeWindow("Lidar Overlaid (All points)", 640, 480);
        cv::imshow("Lidar Overlaid (All points)", last_image_);
        // cv::waitKey(0); // This blocking function has been removed.
    }

    void computeRigidTransformSVD(
        const std::vector<cv::Point3f> &src,
        const std::vector<cv::Point3f> &dst,
        cv::Mat &R,
        cv::Mat &t)
    {
        cv::Point3f src_center(0, 0, 0);
        cv::Point3f dst_center(0, 0, 0);
        for (size_t i = 0; i < src.size(); ++i)
        {
            src_center += src[i];
            dst_center += dst[i];
        }
        src_center *= (1.0 / src.size());
        dst_center *= (1.0 / dst.size());

        cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
        for (size_t i = 0; i < src.size(); ++i)
        {
            cv::Mat src_vec = (cv::Mat_<double>(3, 1) << src[i].x - src_center.x,
                               src[i].y - src_center.y,
                               src[i].z - src_center.z);
            cv::Mat dst_vec = (cv::Mat_<double>(3, 1) << dst[i].x - dst_center.x,
                               dst[i].y - dst_center.y,
                               dst[i].z - dst_center.z);
            H += dst_vec * src_vec.t();
        }

        cv::Mat U, S, Vt;
        cv::SVD::compute(H, S, U, Vt);
        R = U * Vt;

        if (cv::determinant(R) < 0)
        {
            U.col(2) *= -1;
            R = U * Vt;
        }

        cv::Mat src_center_mat = (cv::Mat_<double>(3, 1) << src_center.x,
                                  src_center.y,
                                  src_center.z);
        cv::Mat dst_center_mat = (cv::Mat_<double>(3, 1) << dst_center.x,
                                  dst_center.y,
                                  dst_center.z);

        t = dst_center_mat - R * src_center_mat;
    }

    void projectLidarToImage(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_in_cam,
        const cv::Mat &image,
        cv::Mat &image_out)
    {
        image_out = image.clone();

        double fx = intrinsic_matrix_.at<double>(0, 0);
        double fy = intrinsic_matrix_.at<double>(1, 1);
        double cx = intrinsic_matrix_.at<double>(0, 2);
        double cy = intrinsic_matrix_.at<double>(1, 2);

        for (const auto &pt : cloud_in_cam->points)
        {
            if (pt.z <= 0.0)
                continue;

            double x = pt.x;
            double y = pt.y;
            double z = pt.z;

            int u = static_cast<int>((fx * x / z) + cx);
            int v = static_cast<int>((fy * y / z) + cy);

            if (u >= 0 && u < image_out.cols && v >= 0 && v < image_out.rows)
            {
                float depth = static_cast<float>(z);
                cv::Scalar color = cv::Scalar(0, 255, 0); // Green
                if (depth < 1.0)
                    color = cv::Scalar(0, 0, 255); // Red
                else if (depth < 2.0)
                    color = cv::Scalar(0, 255, 255); // Yellow

                cv::circle(image_out, cv::Point(u, v), 2, color, -1);
            }
        }
    }

    void point3fVectorToPointCloud2(
        const std::vector<cv::Point3f> &points,
        sensor_msgs::msg::PointCloud2 &msg,
        const std::string &frame_id,
        const rclcpp::Time &stamp)
    {
        // Create a PCL PointCloud from cv::Point3f vector
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl_cloud->width = static_cast<uint32_t>(points.size());
        pcl_cloud->height = 1;
        pcl_cloud->is_dense = false;
        pcl_cloud->points.resize(points.size());

        for (size_t i = 0; i < points.size(); ++i)
        {
            pcl_cloud->points[i].x = points[i].x;
            pcl_cloud->points[i].y = points[i].y;
            pcl_cloud->points[i].z = points[i].z;
        }

        // Convert PCL PointCloud to sensor_msgs::msg::PointCloud2
        pcl::toROSMsg(*pcl_cloud, msg);

        // Set header information
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id;
    }

    template <typename T>
    void saveToFile(const std::string &extension,
                    const std::string &filename,
                    const T &data)
    {
        std::string fullpath = absolute_path_ + filename + "." + extension;
        std::ofstream ofs(fullpath);

        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file: %s", fullpath.c_str());
            rclcpp::shutdown();
        }

        ofs << data << std::endl;
        ofs.close();

        RCLCPP_INFO(this->get_logger(), "File saved: %s", fullpath.c_str());
    }

    void saveToFile(const std::string &extension,
                    const std::string &filename,
                    const cv::Mat &mat)
    {
        std::string fullpath = absolute_path_ + filename + "." + extension;
        std::ofstream ofs(fullpath);

        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file: %s", fullpath.c_str());
            rclcpp::shutdown();
        }

        for (int i = 0; i < mat.rows; ++i)
        {
            for (int j = 0; j < mat.cols; ++j)
            {
                ofs << mat.at<double>(i, j) << " ";
            }
            ofs << "\n";
        }

        ofs.close();

        RCLCPP_INFO(this->get_logger(), "cv::Mat saved: %s", fullpath.c_str());
    }

    template <typename... Args>
    void saveMultipleToFile(const std::string &extension,
                            const std::string &filename,
                            const Args &...args)
    {
        std::string fullpath = absolute_path_ + filename + "." + extension;
        std::ofstream ofs(fullpath);

        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file: %s", fullpath.c_str());
            rclcpp::shutdown();
        }

        (writeData(ofs, args), ...);

        ofs.close();

        RCLCPP_INFO(this->get_logger(), "File saved: %s", fullpath.c_str());
    }

    template <typename T>
    void writeData(std::ofstream &ofs, const T &data)
    {
        ofs << data << "\n";
    }

    void writeData(std::ofstream &ofs, const cv::Mat &mat)
    {
        for (int i = 0; i < mat.rows; ++i)
        {
            for (int j = 0; j < mat.cols; ++j)
            {
                ofs << mat.at<double>(i, j) << " ";
            }
            ofs << "\n";
        }
    }

    // Output and Reporting - Save to YAML
    void saveCalibrationResultToYaml(const cv::Mat &R, const cv::Mat &t)
    {
        std::string filepath = absolute_path_ + "lidar_camera_extrinsic.yaml";
        cv::FileStorage fs(filepath, cv::FileStorage::WRITE);

        if (!fs.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open YAML file for saving extrinsic calibration: %s", filepath.c_str());
            return;
        }

        fs << "lidar_to_camera_rotation" << R;
        fs << "lidar_to_camera_translation" << t;
        fs.release();
        RCLCPP_INFO(this->get_logger(), "Extrinsic calibration results saved to %s", filepath.c_str());
    }

    // Function to publish coordinate axes as markers in RViz
    void publishCoordinateAxes(const Eigen::Matrix4d& pose_matrix, 
                               const std::string& frame_id, 
                               const std::string& ns, 
                               double scale_length, 
                               double scale_width)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        rclcpp::Time stamp = this->now();
        int id_counter = 0;

        // Extract translation
        Eigen::Vector3d translation = pose_matrix.block<3,1>(0,3);

        // Extract rotation matrix
        Eigen::Matrix3d rotation = pose_matrix.block<3,3>(0,0);

        // X-axis (Red)
        visualization_msgs::msg::Marker x_axis;
        x_axis.header.frame_id = frame_id;
        x_axis.header.stamp = stamp;
        x_axis.ns = ns;
        x_axis.id = id_counter++;
        x_axis.type = visualization_msgs::msg::Marker::ARROW;
        x_axis.action = visualization_msgs::msg::Marker::ADD;
        x_axis.pose.position.x = translation.x();
        x_axis.pose.position.y = translation.y();
        x_axis.pose.position.z = translation.z();
        
        // Calculate quaternion from rotation matrix
        Eigen::Quaterniond q(rotation);
        x_axis.pose.orientation.x = q.x();
        x_axis.pose.orientation.y = q.y();
        x_axis.pose.orientation.z = q.z();
        x_axis.pose.orientation.w = q.w();

        x_axis.scale.x = scale_length; // Length
        x_axis.scale.y = scale_width;  // Shaft diameter
        x_axis.scale.z = scale_width;  // Head diameter
        x_axis.color.a = 1.0;
        x_axis.color.r = 1.0; x_axis.color.g = 0.0; x_axis.color.b = 0.0;
        marker_array.markers.push_back(x_axis);

        // Y-axis (Green)
        visualization_msgs::msg::Marker y_axis = x_axis; // Copy from x_axis
        y_axis.id = id_counter++;
        Eigen::Matrix3d rot_y = rotation * Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix(); // Rotate X-axis to Y-axis
        Eigen::Quaterniond q_y(rot_y);
        y_axis.pose.orientation.x = q_y.x();
        y_axis.pose.orientation.y = q_y.y();
        y_axis.pose.orientation.z = q_y.z();
        y_axis.pose.orientation.w = q_y.w();
        y_axis.color.r = 0.0; y_axis.color.g = 1.0; y_axis.color.b = 0.0;
        marker_array.markers.push_back(y_axis);

        // Z-axis (Blue)
        visualization_msgs::msg::Marker z_axis = x_axis; // Copy from x_axis
        z_axis.id = id_counter++;
        Eigen::Matrix3d rot_z = rotation * Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitY()).toRotationMatrix(); // Rotate X-axis to Z-axis
        Eigen::Quaterniond q_z(rot_z);
        z_axis.pose.orientation.x = q_z.x();
        z_axis.pose.orientation.y = q_z.y();
        z_axis.pose.orientation.z = q_z.z();
        z_axis.pose.orientation.w = q_z.w();
        z_axis.color.r = 0.0; y_axis.color.g = 0.0; y_axis.color.b = 1.0;
        marker_array.markers.push_back(z_axis);

        pub_markers_->publish(marker_array);
    }

    // New function to publish individual corner markers
    void publishCornerMarkers(const std::vector<Eigen::Vector3d>& corners,
                              const std::string& frame_id,
                              const std::string& ns,
                              double scale,
                              double r, double g, double b)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        rclcpp::Time stamp = this->now();
        int id_counter = 1000; // Start with a high ID to avoid conflicts with axes

        for (const auto& corner : corners)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = stamp;
            marker.ns = ns;
            marker.id = id_counter++;
            marker.type = visualization_msgs::msg::Marker::SPHERE; // Or CUBE
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = corner.x();
            marker.pose.position.y = corner.y();
            marker.pose.position.z = corner.z();
            marker.pose.orientation.w = 1.0; // No rotation for spheres/cubes
            marker.scale.x = scale;
            marker.scale.y = scale;
            marker.scale.z = scale;
            marker.color.a = 1.0;
            marker.color.r = r;
            marker.color.g = g;
            marker.color.b = b;
            marker_array.markers.push_back(marker);
        }
        pub_markers_->publish(marker_array);
    }

    // New function to publish individual corner markers with specified colors
    void publishColoredCornerMarkers(const std::vector<Eigen::Vector3d>& corners,
                                     const std::vector<std::tuple<double, double, double>>& colors,
                                     const std::string& frame_id,
                                     const std::string& ns,
                                     double scale)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        rclcpp::Time stamp = this->now();
        int id_counter = 2000; // Start with a higher ID for this type of marker

        for (size_t i = 0; i < corners.size(); ++i)
        {
            const auto& corner = corners[i];
            const auto& color = colors[i];

            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = stamp;
            marker.ns = ns;
            marker.id = id_counter++;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = corner.x();
            marker.pose.position.y = corner.y();
            marker.pose.position.z = corner.z();
            marker.pose.orientation.w = 1.0;
            marker.scale.x = scale;
            marker.scale.y = scale;
            marker.scale.z = scale;
            marker.color.a = 1.0;
            marker.color.r = std::get<0>(color);
            marker.color.g = std::get<1>(color);
            marker.color.b = std::get<2>(color);
            marker_array.markers.push_back(marker);
        }
        pub_markers_->publish(marker_array);
    }

    // Helper to convert std::vector<cv::Point3f> to std::vector<Eigen::Vector3d>
    std::vector<Eigen::Vector3d> convertCvPointsToEigen(const std::vector<cv::Point3f>& cv_points)
    {
        std::vector<Eigen::Vector3d> eigen_points;
        eigen_points.reserve(cv_points.size());
        for (const auto& pt : cv_points)
        {
            eigen_points.emplace_back(pt.x, pt.y, pt.z);
        }
        return eigen_points;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CamLidarCalibNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
