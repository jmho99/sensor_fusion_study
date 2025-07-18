#include <rclcpp/rclcpp.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <fstream>
#include <string>
#include <filesystem>
#include "sensor_msgs/msg/image.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.h>
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

namespace fs = std::filesystem;

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
                                                                                                                            // pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("chessboard_markers", 10); // For visualization

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
    // 체커보드 코너를 저장할 벡터
    std::vector<Eigen::Vector3d> chessboard_corners_3d_;

    // 강도 분류를 위한 enum (GRAY 제거)
    enum PatternColor
    {
        BLACK,
        WHITE
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
        tau_l_ = (intensity_min_threshold_ + intensity_max_threshold_) / 2.0;

        RCLCPP_INFO(this->get_logger(), "Calculated single intensity threshold (tau_l_): %.2f", tau_l_);
    }

    // 로컬 PointCloud 내에서 강도(색상) 변화 횟수를 확인하는 함수
    // 코너 주변에서 색상 변화가 최소 3번 일어나는지 확인
    // NOTE: This function's name is misleading. It counts transitions but doesn't *refine* the corner.
    // It's used as a *validation* step.
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
        // This last check can be problematic if the angular range doesn't truly wrap around
        // e.g., if points are only on one side of the 0/2pi boundary.
        // For chessboard corners, transitions should generally be 4 (black->white->black->white or vice versa)
        // This transition count logic is more about *validating* a corner region than refining its position.
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
        image_corners_latest_ = corners;                                        // 이미지에서 감지된 2D 코너 저장
        if (!found)
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: Chessboard not found in image! Shutting down node.");
            rclcpp::shutdown();
        }

        cv::cornerSubPix(img_gray, corners, cv::Size(5, 5),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        std::vector<cv::Point3f> object_points;
        // Object points should also reflect the number of internal corners.
        for (int i = 0; i < board_size_.height; i++) // board_size_ is now (cols-1, rows-1)
        {
            for (int j = 0; j < board_size_.width; j++) // board_size_ is now (cols-1, rows-1)
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
        plane_center_ = Eigen::Vector3d(centroid.x(), centroid.y(), centroid.z());

        plane_normal_ = Eigen::Vector3d(coefficients->values[0], coefficients->values[1], coefficients->values[2]).normalized();

        RCLCPP_INFO(this->get_logger(), "Plane center: %.3f, %.3f, %.3f", plane_center_.x(), plane_center_.y(), plane_center_.z());
        RCLCPP_INFO(this->get_logger(), "Plane normal: %.3f, %.3f, %.3f", plane_normal_.x(), plane_normal_.y(), plane_normal_.z());

        // 2. 체커보드 평면 내 지역 좌표계 설정 (PCA 활용)
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(lidar_plane_points_latest_);
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();

        Eigen::Vector3d pca_axis1 = eigen_vectors.col(0).cast<double>();
        Eigen::Vector3d pca_axis2 = eigen_vectors.col(1).cast<double>();

        // Z축 (법선)과 X, Y축이 정확히 직교하는지 확인 및 조정
        pca_axis1.normalize();
        pca_axis2 = plane_normal_.cross(pca_axis1).normalized();
        pca_axis1 = pca_axis2.cross(plane_normal_).normalized();

        RCLCPP_INFO(this->get_logger(), "PCA Axis 1: %.3f, %.3f, %.3f", pca_axis1.x(), pca_axis1.y(), pca_axis1.z());
        RCLCPP_INFO(this->get_logger(), "PCA Axis 2: %.3f, %.3f, %.3f", pca_axis2.x(), pca_axis2.y(), pca_axis2.z());

        // 평면의 가로/세로 길이 측정
        double min_proj_x = std::numeric_limits<double>::max();
        double max_proj_x = std::numeric_limits<double>::min();
        double min_proj_y = std::numeric_limits<double>::max();
        double max_proj_y = std::numeric_limits<double>::min();

        for (const auto &pt : lidar_plane_points_latest_->points)
        {
            Eigen::Vector3d vec = pt.getVector3fMap().cast<double>() - plane_center_;
            double proj_x = vec.dot(pca_axis1);
            double proj_y = vec.dot(pca_axis2);

            if (proj_x < min_proj_x)
                min_proj_x = proj_x;
            if (proj_x > max_proj_x)
                max_proj_x = proj_x;
            if (proj_y < min_proj_y)
                min_proj_y = proj_y;
            if (proj_y > max_proj_y)
                max_proj_y = proj_y;
        }

        double plane_width = max_proj_x - min_proj_x;
        double plane_height = max_proj_y - min_proj_y;

        RCLCPP_INFO(this->get_logger(), "Estimated Plane Dimensions: Width=%.3f m, Height=%.3f m", plane_width, plane_height);

        // 체커보드 내부 코너의 물리적 크기
        double chessboard_physical_width = (pattern_size_cols_ - 1) * square_size_;
        double chessboard_physical_height = (pattern_size_rows_ - 1) * square_size_;

        Eigen::Vector3d board_x_axis;
        Eigen::Vector3d board_y_axis;

        // 평면의 가로/세로 길이와 체커보드의 행/열 개수를 매칭
        // 더 작은 길이에 더 적은 코너 개수를 매칭, 더 큰 길이에 더 많은 코너 개수를 매칭
        // This logic seems incorrect. The board axes should align with the physical dimensions of the board,
        // which are derived from `pattern_size_cols_` and `pattern_size_rows_`.
        // It's more about ensuring `board_x_axis` points along the 'width' of the pattern and `board_y_axis` along the 'height'.
        // If PCA axes are arbitrary, you might need to check dot products with expected directions, or rely on a fixed orientation.
        // For now, assuming PCA gives reasonable axes.
        if (plane_width < plane_height)
        {
            if (pattern_size_cols_ < pattern_size_rows_)
            {
                board_x_axis = pca_axis1;
                board_y_axis = pca_axis2;
            }
            else
            {
                board_x_axis = pca_axis2; // Swap axes
                board_y_axis = pca_axis1;
            }
        }
        else
        { // plane_width >= plane_height
            if (pattern_size_cols_ > pattern_size_rows_)
            {
                board_x_axis = pca_axis1;
                board_y_axis = pca_axis2;
            }
            else
            {
                board_x_axis = pca_axis2; // Swap axes
                board_y_axis = pca_axis1;
            }
        }

        RCLCPP_INFO(this->get_logger(), "Adjusted Board X axis: %.3f, %.3f, %.3f", board_x_axis.x(), board_x_axis.y(), board_x_axis.z());
        RCLCPP_INFO(this->get_logger(), "Adjusted Board Y axis: %.3f, %.3f, %.3f", board_y_axis.x(), board_y_axis.y(), board_y_axis.z());

        // 평면의 중심에서 체커보드 그리드의 좌상단 코너 (0,0) 위치 계산
        // (x_offset, y_offset)은 그리드 중심에서 (0,0)까지의 상대적인 거리
        // Note: The origin is usually the top-left corner of the chessboard pattern (0,0,0 in its local frame).
        // The plane_center_ is the centroid of the *entire detected plane*.
        // We need to shift from the plane_center_ to the (0,0) corner of the pattern.
        // The offset is half the *total physical dimension* of the pattern.
        Eigen::Vector3d chessboard_origin_3d = plane_center_ - (chessboard_physical_width / 2.0) * board_x_axis - (chessboard_physical_height / 2.0) * board_y_axis;

        chessboard_corners_3d_.clear();

        // 모든 내부 코너를 순회 (첫 행/열, 마지막 행/열 건너뛰기 로직 제거)
        for (int r = 0; r < pattern_size_rows_; ++r)
        {
            for (int c = 0; c < pattern_size_cols_; ++c)
            {
                // 3D 공간에서의 코너 위치 계산 (초기 추정)
                // 이제 chessboard_origin_3d를 기준으로 상대적인 위치를 더함
                Eigen::Vector3d initial_corner_3d = chessboard_origin_3d + (c * square_size_) * board_x_axis + (r * square_size_) * board_y_axis;

                // LiDAR Chessboard Corner Refinement (Intensity-based)
                // 각 초기 추정 코너에 대해 Intensity 정보를 활용한 정밀화 로직을 추가합니다.
                Eigen::Vector3d refined_corner_3d;
                bool is_valid_corner = refineCornerWithIntensity(
                    initial_corner_3d,
                    plane_normal_,
                    board_x_axis,
                    board_y_axis,
                    square_size_,
                    lidar_plane_points_latest_, // 평면 내의 모든 PointCloud
                    r, c,                       // 코너의 그리드 인덱스
                    refined_corner_3d);

                if (is_valid_corner)
                {
                    chessboard_corners_3d_.push_back(refined_corner_3d);
                    // RCLCPP_INFO(this->get_logger(), "Refined Corner (%d, %d): %.3f, %.3f, %.3f", r, c, refined_corner_3d.x(), refined_corner_3d.y(), refined_corner_3d.z());
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "Corner (%d, %d) has insufficient intensity transitions or could not be refined. Skipping.", r, c);
                }
            }
        }
        // --- 논문 4.2 코너 추정 끝 ---

        if (chessboard_corners_3d_.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: No valid chessboard corners detected after refinement. Shutting down node.");
            rclcpp::shutdown();
        }

        // TODO 5: Visualization Enhancements - Publish markers for estimated chessboard corners
        // publishChessboardMarkers(chessboard_corners_3d_); // Requires visualization_msgs/msg/Marker.hpp

        // 추정된 코너들을 cv::Point3f 벡터로 변환 (calibrateLidarCameraFinal 함수를 위해)
        std::vector<cv::Point3f> estimated_cv_corners;
        for (const auto &eigen_pt : chessboard_corners_3d_)
        {
            estimated_cv_corners.emplace_back(eigen_pt.x(), eigen_pt.y(), eigen_pt.z());
        }
        /*
                Eigen::MatrixXd corners_matrix(chessboard_corners_3d_.size(), 3);
                for (size_t i = 0; i < chessboard_corners_3d_.size(); ++i)
                {
                    corners_matrix.row(i) << chessboard_corners_3d_[i].x(), chessboard_corners_3d_[i].y(), chessboard_corners_3d_[i].z();
                }
                RCLCPP_INFO(this->get_logger(), "--- Detected Chessboard Corners (3D LiDAR - Matrix Form) ---");
                std::cout << corners_matrix << std::endl;

                cv::Mat cv_corners_matrix(estimated_cv_corners.size(), 3, CV_64F);
                for (int i = 0; i < estimated_cv_corners.size(); ++i)
                {
                    cv_corners_matrix.at<double>(i, 0) = estimated_cv_corners[i].x;
                    cv_corners_matrix.at<double>(i, 1) = estimated_cv_corners[i].y;
                    cv_corners_matrix.at<double>(i, 2) = estimated_cv_corners[i].z;
                }
                RCLCPP_INFO(this->get_logger(), "--- Estimated Chessboard Corners (cv::Point3f - Matrix Form) ---");
                std::cout << cv_corners_matrix << std::endl;
        */
        // 캘리브레이션 함수 호출
        calibrateLidarCameraFinal(last_cloud_, estimated_cv_corners);
    }

    // LiDAR Chessboard Corner Refinement (Intensity-based) - Modified to refine position
    bool refineCornerWithIntensity(
        const Eigen::Vector3d &initial_corner,
        const Eigen::Vector3d &plane_normal,
        const Eigen::Vector3d &board_x_axis,
        const Eigen::Vector3d &board_y_axis,
        double square_size,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &plane_points_cloud,
        int r_idx, int c_idx, // 그리드 인덱스 (코너의 위치 파악용)
        Eigen::Vector3d &refined_corner)
    {
        // 1. 초기 코너 주변의 PointCloud 데이터 추출 (ROI 설정)
        // 코너를 중심으로 약 한 칸 크기 정도의 작은 박스/영역 설정
        pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::CropBox<pcl::PointXYZI> local_crop;
        local_crop.setInputCloud(plane_points_cloud);

        // 검색 반경을 square_size의 절반 정도로 설정하여 4개의 인접한 사각형 영역을 포함하도록 함
        double search_half_size = square_size * 0.7; // Adjusted search radius for refinement
        local_crop.setMin(Eigen::Vector4f(initial_corner.x() - search_half_size, initial_corner.y() - search_half_size, initial_corner.z() - search_half_size, 1.0));
        local_crop.setMax(Eigen::Vector4f(initial_corner.x() + search_half_size, initial_corner.y() + search_half_size, initial_corner.z() + search_half_size, 1.0));
        local_crop.filter(*local_cloud);

        if (local_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No points in local cloud for corner (%d, %d) refinement. Skipping.", r_idx, c_idx);
            return false; // 이 코너는 정밀화할 수 없음
        }

        // 2. 색상 변화 횟수 확인 (Validation step)
        int transitions = checkIntensityTransitions(local_cloud, initial_corner, board_x_axis, board_y_axis);

        if (transitions < 3)
        {
            RCLCPP_WARN(this->get_logger(), "Corner (%d, %d) has only %d intensity transitions. Not considered a valid corner.", r_idx, c_idx, transitions);
            return false; // 색상 변화가 충분하지 않으면 유효하지 않은 코너
        }

        // --- Refinement Logic Starts Here ---
        // Your request: "색상이 변하는 바뀌는 경계부분의 절반을 경계선으로 정하고, 가로/세로 경계선의 교점을 코너로 정하려고 하고 있어."
        // This implies finding the edges and their intersection.
        // A simple way to approximate this without complex line fitting in 3D:
        // Identify points belonging to the four squares meeting at the corner and find their collective centroid.
        // Alternatively, project points to 2D plane, find 2D corners, then project back.

        // Let's implement a more refined approach:
        // 1. Project points in `local_cloud` onto the 2D plane (defined by board_x_axis, board_y_axis).
        // 2. Analyze the intensity variations along these 2D axes to find intensity boundaries.
        // 3. The intersection of these boundaries will be the refined 2D corner, then project back to 3D.

        std::vector<cv::Point2f> projected_2d_points;
        std::vector<float> intensities;
        std::vector<pcl::PointXYZI> original_points;

        for (const auto &pt : local_cloud->points)
        {
            Eigen::Vector3d vec = pt.getVector3fMap().cast<double>() - initial_corner; // Relative to initial corner
            double proj_x = vec.dot(board_x_axis);
            double proj_y = vec.dot(board_y_axis);
            projected_2d_points.emplace_back(proj_x, proj_y);
            intensities.push_back(pt.intensity);
            original_points.push_back(pt);
        }

        if (projected_2d_points.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No projected 2D points for corner (%d, %d). Skipping refinement.", r_idx, c_idx);
            return false;
        }

        // We can create an "intensity image" or grid to perform subpixel refinement,
        // similar to what `cv::cornerSubPix` does on images.
        // For simplicity and adherence to the request: Find transition points.

        // A more robust method would involve iterating around the corner in different directions,
        // finding where intensity changes significantly, and then averaging these transition points.

        // Approach: Define search lines (e.g., cross shape) through the initial corner,
        // sample intensities along these lines, and find intensity steps.
        // Or, more simply, find the centroid of points from the four quadrants around the corner,
        // but weighted by their "boundary-ness" (e.g., proximity to average intensity).

        // Let's try to find the "center of mass" of the intensity transitions.
        // This means, for points that are near the intensity threshold, we want to give them more weight.

        pcl::PointCloud<pcl::PointXYZI>::Ptr transition_points(new pcl::PointCloud<pcl::PointXYZI>);
        for (size_t i = 0; i < local_cloud->points.size(); ++i)
        {
            const auto &pt = local_cloud->points[i];
            // Check if this point is "near" the intensity transition threshold (tau_l_)
            // You might define a small epsilon for this "nearness"
            double epsilon_intensity = 0.1 * (intensity_max_threshold_ - intensity_min_threshold_); // 10% of range
            if (std::abs(pt.intensity - tau_l_) < epsilon_intensity)
            {
                transition_points->push_back(pt);
            }
        }

        if (transition_points->points.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No transition points found near corner (%d, %d). Using initial corner.", r_idx, c_idx);
            refined_corner = initial_corner; // Fallback
            return true;                     // Still consider it valid as transitions were counted
        }

        // Calculate centroid of these transition points
        Eigen::Vector4f refined_centroid_4f;
        pcl::compute3DCentroid(*transition_points, refined_centroid_4f);
        refined_corner = Eigen::Vector3d(refined_centroid_4f.x(), refined_centroid_4f.y(), refined_centroid_4f.z());

        // Ensure the refined corner stays on the detected plane, if desired, by projecting it.
        // The plane equation is ax + by + cz + d = 0. Normal (a,b,c), d = -(ax_0 + by_0 + cz_0).
        // Current plane_normal_ and plane_center_ define the plane.
        // Project refined_corner onto the plane:
        // p_proj = p - ( (p - p_0) . n ) * n
        // where p_0 is plane_center_ and n is plane_normal_
        Eigen::Vector3d vec_to_plane_center = refined_corner - plane_center_;
        double dist_to_plane = vec_to_plane_center.dot(plane_normal_);
        refined_corner = refined_corner - dist_to_plane * plane_normal_;

        RCLCPP_INFO(this->get_logger(), "Refined Corner (%d, %d): from (%.3f, %.3f, %.3f) to (%.3f, %.3f, %.3f)",
                    r_idx, c_idx, initial_corner.x(), initial_corner.y(), initial_corner.z(),
                    refined_corner.x(), refined_corner.y(), refined_corner.z());

        return true;
    }

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

        cv::Mat R_cb2cam; // Renamed to avoid clash with member variable lidar2cam_R_
        cv::Rodrigues(cb2cam_rvec_, R_cb2cam);
        std::vector<cv::Point3f> cb2cam_points;
        cv::transform(cb_object_points, cb2cam_points, R_cb2cam);
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

        point3fVectorToPointCloud2(cb2cam_points, cb_points_msg_, "map", this->now()); // Publish corners seen from camera

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

        // 재투영 에러 계산 함수 호출
        calculateReprojectionError();
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

    void calculateReprojectionError()
    {
        if (chessboard_corners_3d_.empty() || image_corners_latest_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Cannot calculate reprojection error: Chessboard corners from LiDAR or image are empty.");
            return;
        }

        if (chessboard_corners_3d_.size() != image_corners_latest_.size())
        {
            RCLCPP_WARN(this->get_logger(), "Cannot calculate reprojection error: Mismatch in number of corners. LiDAR: %zu, Camera: %zu. Skipping error calculation.",
                        chessboard_corners_3d_.size(), image_corners_latest_.size());
            return;
        }

        // 1. LiDAR 3D 코너 포인트를 cv::Point3f 벡터로 변환
        std::vector<cv::Point3f> lidar_3d_corners_cv;
        for (const auto &eigen_pt : chessboard_corners_3d_)
        {
            lidar_3d_corners_cv.emplace_back(eigen_pt.x(), eigen_pt.y(), eigen_pt.z());
        }

        // 2. LiDAR 3D 코너 포인트를 카메라 좌표계로 변환
        std::vector<cv::Point3f> transformed_lidar_3d_corners;
        for (const auto &pt_lidar : lidar_3d_corners_cv)
        {
            cv::Mat pt_mat = (cv::Mat_<double>(3, 1) << pt_lidar.x, pt_lidar.y, pt_lidar.z);
            cv::Mat pt_transformed = lidar2cam_R_ * pt_mat + lidar2cam_t_;
            transformed_lidar_3d_corners.emplace_back(
                pt_transformed.at<double>(0),
                pt_transformed.at<double>(1),
                pt_transformed.at<double>(2));
        }

        // 3. 변환된 3D LiDAR 포인트를 2D 이미지 평면에 투영
        std::vector<cv::Point2f> projected_lidar_2d_corners;
        cv::Mat dummy_rvec = cv::Mat::zeros(3, 1, CV_64F); // 3D 포인트가 이미 카메라 좌표계에 있으므로 회전 벡터는 0
        cv::Mat dummy_tvec = cv::Mat::zeros(3, 1, CV_64F); // 3D 포인트가 이미 카메라 좌표계에 있으므로 이동 벡터는 0

        cv::projectPoints(transformed_lidar_3d_corners,
                          dummy_rvec, // 3D 점이 이미 카메라 좌표계에 있으므로 0
                          dummy_tvec, // 3D 점이 이미 카메라 좌표계에 있으므로 0
                          intrinsic_matrix_,
                          distortion_coeffs_,
                          projected_lidar_2d_corners);

        // 4. 재투영 에러 계산 (RMS 에러)
        double sum_squared_error = 0.0;
        for (size_t i = 0; i < image_corners_latest_.size(); ++i)
        {
            double dx = image_corners_latest_[i].x - projected_lidar_2d_corners[i].x;
            double dy = image_corners_latest_[i].y - projected_lidar_2d_corners[i].y;
            sum_squared_error += (dx * dx + dy * dy);
        }

        double mean_reprojection_error = std::sqrt(sum_squared_error / image_corners_latest_.size());

        RCLCPP_INFO(this->get_logger(), "Mean Reprojection Error: %.4f pixels", mean_reprojection_error);

        // 선택적: 에러 결과를 파일에 저장
        saveToFile("txt", "reprojection_error", std::string("Mean Reprojection Error: ") + std::to_string(mean_reprojection_error) + " pixels");
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
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CamLidarCalibNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}