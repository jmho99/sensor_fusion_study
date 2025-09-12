#include <rclcpp/rclcpp.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <fstream>
#include <string>
#include <filesystem>
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cstdlib>

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
#include <cmath>                 // For std::acos, std::fabs
#include <random>                // For random number generation

#include "calib_utils/calib_utils.hpp"
#include "ceres/ceres.h"

namespace fs = std::filesystem;

class CamLidarCalibNode : public rclcpp::Node
{
public:
    CamLidarCalibNode()
        : Node("c_cam_lidar_calib")
    {
        initializedParameters();
        // 키보드 입력을 처리하는 타이머
        keboard_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&CamLidarCalibNode::keyboardCallback, this));

        pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_points", 10);
        pub_checker_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("checker_points", 10);
        pub_lidar2cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar2cam_points", 10);
        pub_cb_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cb_object_points", 10);
        pub_cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cam_points", 10);
        pub_binarized_intensity_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("binarized_intensity_points", 10); // New publisher for binarized intensity
        // 서비스에서 받은 코너 포인트를 퍼블리시할 새로운 퍼블리셔 (이제 직접 검출된 코너를 퍼블리시)
        pub_service_corners_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("detected_lidar_corners", 10);

        // PointCloud2 메시지를 주기적으로 퍼블리시하는 타이머 (주석 해제됨)
        pcd_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30), // 30ms 주기로 퍼블리시
            std::bind(&CamLidarCalibNode::pcdTimerCallback, this));

        // Removed ROS2 service client creation and waiting for service
    }

private:
    std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> sub_cam_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> sub_lidar_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_cam__;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar__;
    cv::Mat current_frame_;
    cv::Mat last_image_;                                                                   // Latest image data
    pcl::PointCloud<pcl::PointXYZI>::Ptr last_cloud_{new pcl::PointCloud<pcl::PointXYZI>}; // Latest point cloud data
    std::string img_path_;
    std::string pcd_path_;
    std::string cam_lidar_path_;
    std::string one_cam_result_path_;
    int frame_counter_ = 0;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_plane_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_checker_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_lidar2cam_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cb_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cam_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_binarized_intensity_; // New publisher
    // 서비스에서 받은 코너 포인트를 퍼블리시할 새로운 퍼블리셔 선언 (이제 직접 검출된 코너)
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_service_corners_;

    sensor_msgs::msg::PointCloud2 plane_msg_;
    sensor_msgs::msg::PointCloud2 checker_msg_;
    sensor_msgs::msg::PointCloud2 lidar2cam_points_;
    sensor_msgs::msg::PointCloud2 cb_points_msg_;
    sensor_msgs::msg::PointCloud2 cam_points_msg_;
    sensor_msgs::msg::PointCloud2 binarized_intensity_msg_; // New message for binarized intensity
    // 서비스에서 받은 코너 포인트를 담을 새로운 메시지 (이제 직접 검출된 코너)
    sensor_msgs::msg::PointCloud2 service_corners_msg_;

    rclcpp::TimerBase::SharedPtr pcd_timer_;       // pcdTimerCallback을 위한 타이머
    rclcpp::TimerBase::SharedPtr keboard_timer_;   // timerCallback을 위한 타이머
    cv::Size board_size_;                          // Chessboard parameters (internal corners)
    double square_size_;                           // Chessboard parameters
    cv::Mat intrinsic_matrix_, distortion_coeffs_; // Camera intrinsics
    cv::Mat cb2cam_rvec_, cb2cam_tvec_;
    cv::Mat lidar2cam_R_, lidar2cam_t_;
    std::vector<int> successful_indices_;
    std::vector<cv::String> image_files_;
    std::vector<std::vector<cv::Point2f>> img_points_;
    std::vector<std::vector<cv::Point3f>> obj_points_;
    std::vector<cv::Mat> rvecs_, tvecs_;
    double rms_;
    std::vector<double> all_frames_rms_;

    std::string img_file_;
    std::string pcd_file_;

    cv::Mat new_camera_matrix_, undistorted_image_;
    std::vector<cv::Point2f> image_corners_undistorted_latest_;

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

    // 랜덤으로 선택된 점 관련 멤버 변수
    cv::Point2f random_selected_image_point_;
    cv::Point3f random_selected_lidar_point_in_cam_frame_; // 카메라 좌표계로 변환된 라이다 점

    std::vector<cv::Mat> all_frame_rot_;
    std::vector<cv::Mat> all_frame_trans_;
    // --- 함수 선언 순서 조정 끝 ---

    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        cv::namedWindow("FLIR View", cv::WINDOW_NORMAL); // Uncommented for display
        cv::resizeWindow("FLIR View", 640, 480);         // Uncommented for display
        cv::imshow("FLIR View", current_frame_);         // Uncommented for display
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
        pub_service_corners_->publish(service_corners_msg_);         // 직접 검출된 코너 포인트 퍼블리시
    }

    void initializedParameters()
    {
        readWritePath();

        cv::FileStorage fs(one_cam_result_path_ + "a_one_cam_calib_result.yaml", cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            RCLCPP_WARN(rclcpp::get_logger("initializedParameters"), "Failed open a_one_cam_calib_result.yaml file! Shutting down node.");
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
        this->declare_parameter<double>("intensity_max_threshold", 100000);
        this->declare_parameter<double>("roi_min_x", -5.0);
        this->declare_parameter<double>("roi_max_x", 0.0);
        this->declare_parameter<double>("roi_min_y", -1.1);
        this->declare_parameter<double>("roi_max_y", 0.6);
        this->declare_parameter<double>("roi_min_z", -0.6);
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
    }

    void readWritePath()
    {
        std::string home_dir = std::getenv("HOME");
        std::string calibration_path = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data";

        cam_lidar_path_ = calibration_path + "/c_cam_lidar_calib/";
        img_path_ = cam_lidar_path_ + "images/";
        pcd_path_ = cam_lidar_path_ + "pointclouds/";
        one_cam_result_path_ = calibration_path + "/a_one_cam_calib/";

        fs::create_directories(img_path_);
        fs::create_directories(pcd_path_);
    }

    void keyboardCallback()
    {
        if (keyboardAvailable())
        {
            std::string input;
            std::getline(std::cin, input);

            if (!input.empty() && std::all_of(input.begin(), input.end(), ::isdigit))
            {
                int number = std::stoi(input);
                img_file_ = "img_" + std::to_string(number) + ".png";
                pcd_file_ = "pcd_" + std::to_string(number) + ".pcd";
            }
            if (input == "connect")
            {
                auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
                auto sub_cam = this->create_subscription<sensor_msgs::msg::CompressedImage>("/image_raw/compressed", qos,
                                                                                            std::bind(&CamLidarCalibNode::imageCallback, this, std::placeholders::_1));

                std::string lidar_topic = "/ouster/points";
                auto sub_lidar = this->create_subscription<sensor_msgs::msg::PointCloud2>(lidar_topic, rclcpp::SensorDataQoS(),
                                                                                          [this, lidar_topic](const sensor_msgs::msg::PointCloud2::SharedPtr msg)
                                                                                          { pcdCallback(msg); });

                sub_cam_.push_back(sub_cam);
                sub_lidar_.push_back(sub_lidar);
            }

            if (input == "s")
            {
                saveImageFile("png", img_path_, frame_counter_, current_frame_);
                savePcdFile("pcd", pcd_path_, frame_counter_, last_cloud_);
                frame_counter_++;
            }
            else if (input == "c")
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

            else if (input == "all")
            {
                int total_frames = 0;
                for (const auto &entry : fs::directory_iterator(img_path_))
                {
                    if (entry.path().extension() == ".png")
                        total_frames++;
                }
                RCLCPP_INFO(this->get_logger(), "Total frame is : %d", total_frames);

                for (int frame_i = 0; frame_i < total_frames; ++frame_i)
                {
                    img_file_ = "img_" + std::to_string(frame_i) + ".png";
                    pcd_file_ = "pcd_" + std::to_string(frame_i) + ".pcd";
#define LOOK_DEBUG
                    findData();
                    solveCameraPlane();
                    detectLidarPlane(); // This function will call corner estimation.
                }
                RCLCPP_INFO(this->get_logger(), "All frames calibration process finished successfully.");
            }
            else if (input == "e")
            {
                RCLCPP_INFO(this->get_logger(), "E key pressed. Shutting down node.");
                rclcpp::shutdown();
            }
        }
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

        std::string first_image_path = img_path_ + img_file_; // index 6 고정
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

        std::string first_pcd_path = pcd_path_ + pcd_file_; // index 6 고정
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(first_pcd_path, *last_cloud_) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: Failed to load the first PCD from %s! Shutting down node.", first_pcd_path.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded first pointcloud: %s", first_pcd_path.c_str());

        RCLCPP_INFO(this->get_logger(), "Successfully loaded calibration data (image and pointcloud).");
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

        image_corners_latest_ = corners;

        // 2) 새 카메라 행렬 계산 및 이미지 언디스토션
        new_camera_matrix_ = cv::getOptimalNewCameraMatrix(intrinsic_matrix_, distortion_coeffs_,
                                                           img_gray.size(), 0 /*alpha*/, img_gray.size());

        // 언디스토션된 이미지 생성
        cv::undistort(img_color, undistorted_image_, intrinsic_matrix_, distortion_coeffs_, new_camera_matrix_);

        // 3) 코너 픽셀도 언디스토션된 픽셀로 변환
        std::vector<cv::Point2f> undist_norm;
        cv::undistortPoints(image_corners_latest_, undist_norm,
                            intrinsic_matrix_, distortion_coeffs_,
                            cv::noArray(), new_camera_matrix_);

        // undistortPoints의 결과는 이미 새 K 픽셀 좌표계이므로 그대로 사용
        image_corners_undistorted_latest_.assign(undist_norm.begin(), undist_norm.end());

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

        // Convert PCL PointCloud to std::vector<PointXYZI> for the external function
        std::vector<PointXYZI> lidar_points_for_corner_detection;
        lidar_points_for_corner_detection.reserve(lidar_plane_points_latest_->points.size());
        for (const auto &p : lidar_plane_points_latest_->points)
        {
            lidar_points_for_corner_detection.push_back({p.x, p.y, p.z, p.intensity});
        }

        RCLCPP_INFO(this->get_logger(), "Calling external corner detection function...");

        // Call the external corner detection function directly, passing the parameter
        std::vector<PointXYZI> detected_corners_xyz_i = estimateChessboardCornersPaperMethod(
            lidar_points_for_corner_detection,
            pattern_size_cols_, // internal_corners_x
            pattern_size_rows_, // internal_corners_y
            square_size_        // checker_size_m
        );

        if (detected_corners_xyz_i.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "External corner detection function returned no corners.");
            return;
        }

        std::vector<cv::Point3f> estimated_cv_corners;
        chessboard_corners_3d_.clear(); // Clear existing corner data
        for (const auto &p : detected_corners_xyz_i)
        {
            estimated_cv_corners.emplace_back(p.x, p.y, p.z);
            // Store detected 3D corners as Eigen::Vector3d
            chessboard_corners_3d_.emplace_back(p.x, p.y, p.z);
        }
        RCLCPP_INFO(this->get_logger(), "Received %zu corners from external function and stored for reprojection error calculation.", estimated_cv_corners.size());

        // Publish detected corners to RViz
        point3fVectorToPointCloud2(estimated_cv_corners, service_corners_msg_, "map", this->now());

        pub_service_corners_->publish(service_corners_msg_);
        RCLCPP_INFO(this->get_logger(), "Published detected corners to /detected_lidar_corners topic.");

        // Proceed with final calibration using the detected corners
        calibrateLidarCameraFinal(last_cloud_, estimated_cv_corners);
    }

    // --- Definition of calibrateLidarCameraFinal function ---
    void calibrateLidarCameraFinal(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_all,
                                   const std::vector<cv::Point3f> &estimated_chessboard_corners_lidar)
    {
        // 1. Generate 3D chessboard points in the camera frame
        // These are the object points (chessboard corners in its own frame)
        std::vector<cv::Point3f> object_points_chessboard_frame;
        for (int i = 0; i < board_size_.height; i++)
        {
            for (int j = 0; j < board_size_.width; j++)
            {
                object_points_chessboard_frame.emplace_back(j * square_size_, i * square_size_, 0.0);
            }
        }

        // Transform object points from chessboard frame to camera frame
        std::vector<cv::Point3f> chessboard_3d_in_cam_frame;
        cv::Mat R_cb2cam;
        cv::Rodrigues(cb2cam_rvec_, R_cb2cam); // Convert rotation vector to rotation matrix

        for (const auto &pt_obj : object_points_chessboard_frame)
        {
            cv::Mat pt_mat = (cv::Mat_<double>(3, 1) << pt_obj.x, pt_obj.y, pt_obj.z);
            cv::Mat pt_transformed = R_cb2cam * pt_mat + cb2cam_tvec_;
            chessboard_3d_in_cam_frame.emplace_back(
                pt_transformed.at<double>(0),
                pt_transformed.at<double>(1),
                pt_transformed.at<double>(2));
        }

        // Ensure we have enough corresponding points
        if (estimated_chessboard_corners_lidar.size() != chessboard_3d_in_cam_frame.size() ||
            estimated_chessboard_corners_lidar.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Mismatch in number of 3D corners for extrinsic calibration or no corners found!");
            return;
        }

        // 2. Compute rigid transformation from LiDAR frame to Camera frame using SVD
        // src: estimated_chessboard_corners_lidar (3D points in LiDAR frame)
        // dst: chessboard_3d_in_cam_frame (3D points in Camera frame)
        computeRigidTransformSVD(estimated_chessboard_corners_lidar, chessboard_3d_in_cam_frame, lidar2cam_R_, lidar2cam_t_);

        all_frame_rot_.push_back(lidar2cam_R_);
        all_frame_trans_.push_back(lidar2cam_t_);

        RCLCPP_INFO(this->get_logger(), "Lidar to Camera Rotation Matrix (R):\n%f %f %f\n%f %f %f\n%f %f %f",
                    lidar2cam_R_.at<double>(0, 0), lidar2cam_R_.at<double>(0, 1), lidar2cam_R_.at<double>(0, 2),
                    lidar2cam_R_.at<double>(1, 0), lidar2cam_R_.at<double>(1, 1), lidar2cam_R_.at<double>(1, 2),
                    lidar2cam_R_.at<double>(2, 0), lidar2cam_R_.at<double>(2, 1), lidar2cam_R_.at<double>(2, 2));
        RCLCPP_INFO(this->get_logger(), "Lidar to Camera Translation Vector (t):\n%f\n%f\n%f",
                    lidar2cam_t_.at<double>(0), lidar2cam_t_.at<double>(1), lidar2cam_t_.at<double>(2));

        // 3. Project all LiDAR points to the image plane using the newly found transformation
        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        transformed_cloud->points.reserve(cloud_all->points.size());

        std::vector<cv::Point2f> projected_lidar_points_2d;                        // Store all projected 2D points
        std::vector<cv::Point3f> transformed_lidar_points_3d_for_random_selection; // Store corresponding 3D points

        double fx = intrinsic_matrix_.at<double>(0, 0);
        double fy = intrinsic_matrix_.at<double>(1, 1);
        double cx = intrinsic_matrix_.at<double>(0, 2);
        double cy = intrinsic_matrix_.at<double>(1, 2);

        for (const auto &pt_lidar : cloud_all->points)
        {
            cv::Mat pt_mat = (cv::Mat_<double>(3, 1) << pt_lidar.x, pt_lidar.y, pt_lidar.z);
            cv::Mat pt_transformed = lidar2cam_R_ * pt_mat + lidar2cam_t_;

            pcl::PointXYZI p_transformed;
            p_transformed.x = pt_transformed.at<double>(0);
            p_transformed.y = pt_transformed.at<double>(1);
            p_transformed.z = pt_transformed.at<double>(2);
            p_transformed.intensity = pt_lidar.intensity; // Preserve intensity
            transformed_cloud->points.push_back(p_transformed);

            // Project to 2D image plane for random selection
            if (p_transformed.z > 0) // Only project points in front of the camera
            {
                int u = static_cast<int>((fx * p_transformed.x / p_transformed.z) + cx);
                int v = static_cast<int>((fy * p_transformed.y / p_transformed.z) + cy);

                if (u >= 0 && u < last_image_.cols && v >= 0 && v < last_image_.rows)
                {
                    projected_lidar_points_2d.emplace_back(u, v);
                    transformed_lidar_points_3d_for_random_selection.emplace_back(p_transformed.x, p_transformed.y, p_transformed.z);
                }
            }
        }

        // Convert transformed PCL cloud to ROS2 message for visualization
        pcl::toROSMsg(*transformed_cloud, lidar2cam_points_);
        lidar2cam_points_.header.frame_id = "map"; // Or "map", depending on your RViz setup
        RCLCPP_INFO(this->get_logger(), "Transformed full LiDAR cloud to camera frame and published to /lidar2cam_points.");

        // 4. Calculate and report reprojection error
        calculateReprojectionError();
#ifdef LOOK_DEBUG
        // 5. Save calibration results to YAML
        saveCalibrationResultToYaml(lidar2cam_R_, lidar2cam_t_);

        // 6. Project LiDAR points onto the camera image for visual verification
        cv::Mat image_with_lidar_projection;
        projectLidarToImage(transformed_cloud, !undistorted_image_.empty() ? undistorted_image_ : last_image_,
                            image_with_lidar_projection);

        // 7. 랜덤 점 선택 및 방향 비교
        if (!projected_lidar_points_2d.empty())
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distrib(0, projected_lidar_points_2d.size() - 1);
            int random_idx = distrib(gen);

            random_selected_image_point_ = projected_lidar_points_2d[random_idx];
            random_selected_lidar_point_in_cam_frame_ = transformed_lidar_points_3d_for_random_selection[random_idx];

            RCLCPP_INFO(this->get_logger(), "Randomly selected image point: (%f, %f)", random_selected_image_point_.x, random_selected_image_point_.y);
            RCLCPP_INFO(this->get_logger(), "Corresponding LiDAR point (in camera frame): (%f, %f, %f)",
                        random_selected_lidar_point_in_cam_frame_.x, random_selected_lidar_point_in_cam_frame_.y, random_selected_lidar_point_in_cam_frame_.z);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "No valid projected LiDAR points to select a random point from.");
        }

        cv::imwrite(cam_lidar_path_ + "results/" + img_file_ + "projected_image.png", image_with_lidar_projection);
        cv::namedWindow("Lidar Projected on Image", cv::WINDOW_NORMAL); // Uncommented for display
        cv::resizeWindow("Lidar Projected on Image", 640, 480);
        cv::imshow("Lidar Projected on Image", image_with_lidar_projection);
        cv::waitKey(0); // Keep window open briefly
#endif
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

    void calculateReprojectionError()
    {
        // chessboard_corners_3d_는 이제 Python 서비스에서 받은 라이다 3D 코너를 포함합니다.
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

        const std::vector<cv::Point2f> *img_corners_ptr = nullptr;
        if (!image_corners_undistorted_latest_.empty())
        {
            img_corners_ptr = &image_corners_undistorted_latest_;
        }
        else if (!image_corners_latest_.empty())
        {
            img_corners_ptr = &image_corners_latest_;
        }
        if (img_corners_ptr == nullptr)
        {
            RCLCPP_ERROR(this->get_logger(), "No image corners available for error calculation.");
            return;
        }

        // LiDAR 3D → 카메라 좌표계
        std::vector<cv::Point3f> transformed_lidar_3d_corners;
        transformed_lidar_3d_corners.reserve(chessboard_corners_3d_.size());
        for (const auto &eigen_pt : chessboard_corners_3d_)
        {
            cv::Mat pt = (cv::Mat_<double>(3, 1) << eigen_pt.x(), eigen_pt.y(), eigen_pt.z());
            cv::Mat q = lidar2cam_R_ * pt + lidar2cam_t_;
            transformed_lidar_3d_corners.emplace_back(
                q.at<double>(0), q.at<double>(1), q.at<double>(2));
        }

        // 프로젝션: newK + no distortion
        std::vector<cv::Point2f> proj;
        cv::Mat zero_r = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat zero_t = cv::Mat::zeros(3, 1, CV_64F);
        const cv::Mat &K = new_camera_matrix_.empty() ? intrinsic_matrix_ : new_camera_matrix_;
        cv::projectPoints(transformed_lidar_3d_corners, zero_r, zero_t, K,
                          cv::noArray(), proj);

        // 에러 계산 (img_corners_ptr과 proj 크기 확인)
        if (proj.size() != img_corners_ptr->size())
        { /* ... */
            return;
        }

        double error_square = 0.0;
        for (size_t i = 0; i < proj.size(); ++i)
        {
            double dx = (*img_corners_ptr)[i].x - proj[i].x;
            double dy = (*img_corners_ptr)[i].y - proj[i].y;
            error_square += dx * dx + dy * dy;
        }
        double rms = std::sqrt(error_square / proj.size());
        RCLCPP_INFO(this->get_logger(), "Mean Reprojection Error (undistorted): %.4f px", rms);
        saveFile("txt", cam_lidar_path_, "reprojection_error",
                 std::string("Mean Reprojection Error (undistorted newK): ") + std::to_string(rms) + " pixels");
        all_frames_rms_.push_back(rms);

        // 사용 중인 카메라 행렬/초점거리 선택
        const cv::Mat &K_used = new_camera_matrix_.empty() ? intrinsic_matrix_ : new_camera_matrix_;
        const double fx_used = K_used.at<double>(0, 0);
        const double fy_used = K_used.at<double>(1, 1);

        double error_square_px = 0.0; // px^2 합
        double error_square_m = 0.0;  // m^2 합
        size_t N = proj.size();

        for (size_t i = 0; i < N; ++i)
        {
            // 2D 픽셀 오차 (언디스토트면 undist 좌표 vs newK 투영, 원본이면 원본 좌표 vs K+dist 투영)
            const double du = (*img_corners_ptr)[i].x - proj[i].x;
            const double dv = (*img_corners_ptr)[i].y - proj[i].y;
            error_square_px += du * du + dv * dv;

            // 대응 라이다 코너의 카메라 깊이 Zc (이미 L->C로 변환된 3D 사용)
            const cv::Point3f &Pc = transformed_lidar_3d_corners[i];
            const double Zc = static_cast<double>(Pc.z);

            if (std::isfinite(Zc) && Zc > 1e-6)
            {
                // 픽셀 오차를 해당 깊이에서의 미터 오차로 변환
                const double ex = (du / fx_used) * Zc; // X-축 방향 [m]
                const double ey = (dv / fy_used) * Zc; // Y-축 방향 [m]
                error_square_m += ex * ex + ey * ey;
            }
            else
            {
                // 깊이 비정상이면 그 샘플은 미터 RMSE에서 제외하려면 N_m을 따로 두세요.
                // 여기서는 단순히 무시: N_m 감소를 위해 카운팅 로직을 쓰는 게 정확합니다.
            }
        }

        double rmse_px = std::sqrt(error_square_px / static_cast<double>(N));
        double rmse_m = std::sqrt(error_square_m / static_cast<double>(N)); // 위 주석처럼 유효 Zc 개수로 나눌 수도 있음

        RCLCPP_INFO(this->get_logger(),
                    "Reprojection RMSE: %.4f px   (back-projected) %.6f m", rmse_px, rmse_m);
    }

    void optimizationAllFrames()
    {
        //if (all_frames_rms_<=1) return;

        
    }

    // Output and Reporting - Save to YAML
    void saveCalibrationResultToYaml(const cv::Mat &R, const cv::Mat &t)
    {
        std::string filepath = cam_lidar_path_ + "lidar_camera_extrinsic.yaml";
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

    void projectLidarToImage(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_in_cam,
        const cv::Mat &image_in,
        cv::Mat &image_out)
    {
        // 언디스토션 이미지가 만들어졌으면 그걸 사용
        const cv::Mat &base = !undistorted_image_.empty() ? undistorted_image_ : image_in;
        image_out = base.clone();

        const cv::Mat &K = !new_camera_matrix_.empty() ? new_camera_matrix_ : intrinsic_matrix_;
        double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);

        float min_horizontal_dist = std::numeric_limits<float>::max();
        float max_horizontal_dist = std::numeric_limits<float>::min();

        // 1. 거리 범위 계산 (수평 거리: sqrt(x^2 + y^2))
        for (const auto &pt : cloud_in_cam->points)
        {
            if (pt.z <= 0.0 || pt.z > 30.0f)
                continue;

            float current_horizontal_dist = std::sqrt(pt.x * pt.x + pt.y * pt.y);
            min_horizontal_dist = std::min(min_horizontal_dist, current_horizontal_dist);
            max_horizontal_dist = std::max(max_horizontal_dist, current_horizontal_dist);
        }

        if (max_horizontal_dist - min_horizontal_dist < 1e-6f)
        {
            min_horizontal_dist = 0.0f;
            max_horizontal_dist = 30.0f;
        }

        // 2. HSV 색상으로 표현
        for (const auto &pt : cloud_in_cam->points)
        {
            if (pt.z <= 0.0 || pt.z > 5.0f)
                continue;

            double x = pt.x;
            double y = pt.y;
            double z = pt.z;

            int u = static_cast<int>((fx * x / z) + cx);
            int v = static_cast<int>((fy * y / z) + cy);

            if (u >= 0 && u < image_out.cols && v >= 0 && v < image_out.rows)
            {
                float horizontal_dist = std::sqrt(x * x + z * z);
                float t = (horizontal_dist - min_horizontal_dist) / (max_horizontal_dist - min_horizontal_dist);
                t = std::clamp(t, 0.0f, 1.0f);

                // HSV -> RGB 수동 변환 (Hue: 0~360 → 무지개, S=1, V=1)
                float h = t * 240.0f; // Blue(240) → Red(0)
                float s = 1.0f, v_val = 1.0f;
                float c = v_val * s;
                float x_hsv = c * (1 - std::fabs(fmod(h / 60.0f, 2) - 1)); // Renamed to avoid clash with pt.x

                float m = v_val - c;

                float r = 0, g = 0, b = 0;
                if (h < 60)
                {
                    r = c;
                    g = x_hsv;
                    b = 0;
                }
                else if (h < 120)
                {
                    r = x_hsv;
                    g = c;
                    b = 0;
                }
                else if (h < 180)
                {
                    r = 0;
                    g = c;
                    b = x_hsv;
                }
                else if (h < 240)
                {
                    r = 0;
                    g = x_hsv;
                    b = c;
                }
                else
                {
                    r = 0, g = 0, b = 0;
                }

                uint8_t R = static_cast<uint8_t>((r + m) * 255);
                uint8_t G = static_cast<uint8_t>((g + m) * 255);
                uint8_t B = static_cast<uint8_t>((b + m) * 255);

                cv::Scalar color = cv::Scalar(B, G, R); // OpenCV는 BGR
                cv::circle(image_out, cv::Point(u, v), 5, color, -1);
            }
        }
    }
    // --- End of calibrateLidarCameraFinal function definition ---

    void point3fVectorToPointCloud2(
        const std::vector<cv::Point3f> &points,
        sensor_msgs::msg::PointCloud2 &msg,
        const std::string &frame_id,
        const rclcpp::Time &stamp)
    {
        // Create a PCL PointCloud from cv::Point3f vector
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl_cloud->width = static_cast<uint32_t>(points.size());
        pcl_cloud->height = 1;
        pcl_cloud->is_dense = false;
        pcl_cloud->points.resize(points.size());

        float min_d = std::numeric_limits<float>::max();
        float max_d = std::numeric_limits<float>::lowest();

        std::vector<float> distances;
        distances.reserve(points.size());

        for (const auto &pt : points)
        {
            float d = std::sqrt(pt.x * pt.x + pt.z * pt.z); // 수평 거리
            distances.push_back(d);
            if (d < min_d)
                min_d = d;
            if (d > max_d)
                max_d = d;
        }

        // 2. 그라데이션 컬러 함수 (Jet 스타일)
        auto hsvRainbowColormap = [](float t) -> uint32_t
        {
            t = std::clamp(t, 0.0f, 1.0f);

            float h = t * 240.0f; // Hue: 0 = red, 240 = blue (OpenCV는 0~180 but RGB는 0~360)
            float s = 1.0f;
            float v = 1.0f;

            float c = v * s;
            float x_hsv = c * (1 - std::fabs(fmod(h / 60.0f, 2) - 1)); // Renamed to avoid clash with pt.x
            float m = v - c;

            float r = 0, g = 0, b = 0;
            if (h < 60)
            {
                r = c, g = x_hsv, b = 0;
            }
            else if (h < 120)
            {
                r = x_hsv, g = c, b = 0;
            }
            else if (h < 180)
            {
                r = 0, g = c, b = x_hsv;
            }
            else if (h < 240)
            {
                r = 0, g = x_hsv, b = c;
            }
            else
            {
                r = 0, g = 0, b = 0;
            }

            uint8_t R = static_cast<uint8_t>((r + m) * 255);
            uint8_t G = static_cast<uint8_t>((g + m) * 255);
            uint8_t B = static_cast<uint8_t>((b + m) * 255);

            return (R << 16) | (G << 8) | B;
        };

        // 3. 포인트별로 컬러 지정
        for (size_t i = 0; i < points.size(); ++i)
        {
            float d = distances[i];
            float normalized_d = (d - min_d) / (max_d - min_d); // 0~1 정규화

            pcl_cloud->points[i].x = points[i].x;
            pcl_cloud->points[i].y = points[i].y;
            pcl_cloud->points[i].z = points[i].z;
            pcl_cloud->points[i].rgb = hsvRainbowColormap(normalized_d);
        }

        // Convert PCL PointCloud to sensor_msgs::msg::PointCloud2
        pcl::toROSMsg(*pcl_cloud, msg);

        // Set header information
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id;
    }
}; // End of CamLidarCalibNode class

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CamLidarCalibNode>();

    // 서비스 클라이언트가 없으므로 MultiThreadedExecutor 대신 rclcpp::spin() 사용
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
