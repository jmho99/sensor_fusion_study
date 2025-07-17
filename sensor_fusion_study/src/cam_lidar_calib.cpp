#include <rclcpp/rclcpp.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <fstream>
#include <string>
#include <filesystem>
#include "sensor_msgs/msg/image.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
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
#include <numeric> // for std::accumulate
#include <algorithm> // for std::min/max

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
            
            // PointCloud2 메시지를 주기적으로 퍼블리시하는 타이머 (주석 해제됨)
            timer_ = this->create_wall_timer(
                std::chrono::milliseconds(30), // 30ms 주기로 퍼블리시
                std::bind(&CamLidarCalibNode::pcdTimerCallback, this));
        }

        pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_points", 10);
        pub_checker_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("checker_points", 10);
        pub_lidar2cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar2cam_points", 10);
        pub_cb_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cb_object_points", 10);
        pub_cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cam_points", 10);

        // 초기 빈 메시지 발행 (타이머가 돌기 전까지는 이 메시지들이 유지됨)
        pub_plane_->publish(plane_msg_);
        pub_checker_->publish(checker_msg_);
        pub_lidar2cam_points_->publish(lidar2cam_points_);
        pub_cb_points_->publish(cb_points_msg_);
        pub_cam_points_->publish(cam_points_msg_);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_cam_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
    cv::Mat current_frame_;
    cv::Mat last_image_;                                                       // Latest image data
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
    sensor_msgs::msg::PointCloud2 plane_msg_;
    sensor_msgs::msg::PointCloud2 checker_msg_;
    sensor_msgs::msg::PointCloud2 lidar2cam_points_;
    sensor_msgs::msg::PointCloud2 cb_points_msg_;
    sensor_msgs::msg::PointCloud2 cam_points_msg_;
    rclcpp::TimerBase::SharedPtr timer_; // pcdTimerCallback을 위한 타이머
    rclcpp::TimerBase::SharedPtr timer__; // timerCallback을 위한 타이머
    cv::Size board_size_;                               // Chessboard parameters
    double square_size_;                                // Chessboard parameters
    cv::Mat intrinsic_matrix_, distortion_coeffs_; // Camera intrinsics
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

    // 4.2.2 Intensity Thresholding Parameters (now less critical for direct corner finding)
    double tau_l_ = 0.0; // Lower threshold for black pattern intensity
    double tau_h_ = 0.0; // Upper threshold for white pattern intensity
    double epsilon_g_ = 2.0; // Constant for gray zone definition (2 for corner estimation, 4 for error evaluation)

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
    }

    void initializedParameters()
    {
        std::string where = "home";
        readWritePath(where);

        cv::FileStorage fs(one_cam_result_path_ + "one_cam_calib_result.yaml", cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            RCLCPP_WARN(rclcpp::get_logger("initializedParameters"), "Failed open one_cam_calib_result.yaml file! Shutting down node.");
            rclcpp::shutdown();
        }
        else
        {
            int cols, rows;
            fs["checkerboard_cols"] >> cols;
            fs["checkerboard_rows"] >> rows;
            fs["square_size"] >> square_size_;
            fs["intrinsic_matrix"] >> intrinsic_matrix_;
            fs["distortion_coefficients"] >> distortion_coeffs_;
            fs.release();

            board_size_ = cv::Size(cols, rows);
        }
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
        if (!last_image_.empty())
        {
            cv::namedWindow("Camera Image", cv::WINDOW_NORMAL);
            cv::resizeWindow("Camera Image", 640, 480);
            cv::imshow("Camera Image", last_image_);
        }
        else
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
            try {
                findData();
                solveCameraPlane();
                detectLidarPlane(); // This function will call corner estimation.
                RCLCPP_INFO(this->get_logger(), "Calibration process finished successfully.");
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Caught C++ exception during calibration: %s", e.what());
            } catch (...) {
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

        std::string first_image_path = image_files[0];
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

        std::string first_pcd_path = pcd_files[0];
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
            bool found = cv::findChessboardCorners(img, board_size_, corners,
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
        bool found = cv::findChessboardCorners(img_gray, board_size_, corners);
        if (!found)
        {
            RCLCPP_ERROR(this->get_logger(), "SHUTDOWN_CAUSE: Chessboard not found in image! Shutting down node.");
            rclcpp::shutdown();
        }

        cv::cornerSubPix(img_gray, corners, cv::Size(5, 5),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        std::vector<cv::Point3f> object_points;
        for (int i = 0; i < board_size_.height; i++)
        {
            for (int j = 0; j < board_size_.width; j++)
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
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        std::string filename = pcd_path_ + "pcd_0.pcd"; // 현재는 첫 번째 PCD 파일만 로드
        pcl::io::loadPCDFile(filename, *cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_intensity(new pcl::PointCloud<pcl::PointXYZI>);

        // 1. 반사 강도 기반 필터링 (Reflectance Intensity Assisted)
        pcl::PassThrough<pcl::PointXYZI> pass_intensity;
        pass_intensity.setInputCloud(cloud);
        pass_intensity.setFilterFieldName("intensity");
        pass_intensity.setFilterLimits(1.0, 5747.0); // 예시: 50~255 강도 범위 (튜닝 필요)
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
        crop.setMin(Eigen::Vector4f(-3.0, -0.8, -0.63, 1.0)); // ROI 최소 x,y,z
        crop.setMax(Eigen::Vector4f(0.0, 0.5, 3.0, 1.0));     // ROI 최대 x,y,z
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
        seg.setDistanceThreshold(0.02); // 2cm 이내의 포인트
        seg.setMaxIterations(1000);

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
            RCLCPP_WARN(rclcpp::get_logger("detectLidarPlane"), "SHUTDOWN_CAUSE: Extracted LiDAR plane points are empty. Shutting down node.");
            rclcpp::shutdown();
        }

        pcl::toROSMsg(*lidar_plane_points_latest_, plane_msg_);
        plane_msg_.header.frame_id = "map"; // RViz에서 사용하는 좌표계 설정

        RCLCPP_INFO(this->get_logger(), "Detected %zu points in LiDAR plane.", lidar_plane_points_latest_->points.size());

        // --- 논문 4.2 코너 추정 시작 (개선된 방식) ---
        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_chessboard_points(new pcl::PointCloud<pcl::PointXYZI>);
        cv::Mat M_XOY_P, t_XOY_F; // Rotation matrix and translation vector for chessboard plane
        transformPointsToChessboardPlane(lidar_plane_points_latest_, transformed_chessboard_points, M_XOY_P, t_XOY_F);

        if (transformed_chessboard_points->empty()) {
            RCLCPP_ERROR(this->get_logger(), "Transformed chessboard points are empty after PCA transformation. Cannot estimate corners.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Transformed chessboard points count: %zu", transformed_chessboard_points->points.size());

        // 1. Transform LiDAR points on the plane to a 2D intensity image
        // Determine image resolution based on chessboard size and desired pixel density
        // For example, 10 pixels per square_size, or a fixed resolution
        double total_width_model = board_size_.width * square_size_;
        double total_height_model = board_size_.height * square_size_;
        
        // Define a margin around the chessboard to capture all points
        double margin = square_size_ * 0.5; // Half a square size margin
        double img_min_x = -total_width_model / 2.0 - margin;
        double img_max_x = total_width_model / 2.0 + margin;
        double img_min_y = -total_height_model / 2.0 - margin;
        double img_max_y = total_height_model / 2.0 + margin;

        // Choose a resolution, e.g., 100 pixels per meter or fixed pixel per square
        int pixels_per_meter = 200; // Adjust as needed for resolution
        int img_width = static_cast<int>((img_max_x - img_min_x) * pixels_per_meter);
        int img_height = static_cast<int>((img_max_y - img_min_y) * pixels_per_meter);

        // Ensure minimum image size
        img_width = std::max(img_width, board_size_.width * 20); // At least 20 pixels per square
        img_height = std::max(img_height, board_size_.height * 20);

        cv::Mat intensity_image = cv::Mat::zeros(img_height, img_width, CV_8U);
        cv::Mat point_count_image = cv::Mat::zeros(img_height, img_width, CV_32S); // To store count for averaging

        // Populate the intensity image
        for (const auto& p : transformed_chessboard_points->points) {
            // Map 3D point (x,y) from PCA plane to 2D image coordinates (u,v)
            int u = static_cast<int>((p.x - img_min_x) * pixels_per_meter);
            int v = static_cast<int>((p.y - img_min_y) * pixels_per_meter);

            if (u >= 0 && u < img_width && v >= 0 && v < img_height) {
                // Accumulate intensity and count for averaging
                intensity_image.at<uchar>(v, u) += static_cast<uchar>(p.intensity);
                point_count_image.at<int>(v, u)++;
            }
        }

        // Average intensities
        for (int r = 0; r < img_height; ++r) {
            for (int c = 0; c < img_width; ++c) {
                if (point_count_image.at<int>(r, c) > 0) {
                    intensity_image.at<uchar>(r, c) /= point_count_image.at<int>(r, c);
                }
            }
        }
        
        // Apply a blur to smooth the intensity image (optional but often helpful)
        cv::GaussianBlur(intensity_image, intensity_image, cv::Size(5, 5), 0);

        // Display the intensity image for debugging
        // cv::namedWindow("LiDAR Intensity Image", cv::WINDOW_NORMAL);
        // cv::resizeWindow("LiDAR Intensity Image", 640, 480);
        // cv::imshow("LiDAR Intensity Image", intensity_image);
        // cv::waitKey(1); // Non-blocking

        // 2. Find chessboard corners in the intensity image
        std::vector<cv::Point2f> image_corners_from_lidar_intensity;
        bool found_lidar_corners = cv::findChessboardCorners(intensity_image, board_size_, image_corners_from_lidar_intensity,
                                                             cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

        if (!found_lidar_corners) {
            RCLCPP_WARN(this->get_logger(), "Chessboard corners NOT found in LiDAR intensity image. Cannot proceed with LiDAR-Camera calibration.");
            // Fallback or shutdown if corners are critical
            return;
        }

        // Refine corner locations
        cv::cornerSubPix(intensity_image, image_corners_from_lidar_intensity, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

        RCLCPP_INFO(this->get_logger(), "Found %zu chessboard corners in LiDAR intensity image.", image_corners_from_lidar_intensity.size());

        // 3. Convert 2D image corners back to 3D points on the PCA plane, then to original LiDAR frame
        std::vector<cv::Point3f> estimated_chessboard_corners_lidar;
        // The calculateChessboardCorners function now directly takes the 2D corners and other parameters
        calculateChessboardCorners(M_XOY_P, t_XOY_F, image_corners_from_lidar_intensity,
                                   img_width, img_height, img_min_x, img_min_y, pixels_per_meter,
                                   estimated_chessboard_corners_lidar);
        
        RCLCPP_INFO(this->get_logger(), "Estimated %zu chessboard corners from LiDAR.", estimated_chessboard_corners_lidar.size());
        if (!estimated_chessboard_corners_lidar.empty()) {
            RCLCPP_INFO(this->get_logger(), "First estimated LiDAR corner: (%.3f, %.3f, %.3f)",
                        estimated_chessboard_corners_lidar[0].x, estimated_chessboard_corners_lidar[0].y, estimated_chessboard_corners_lidar[0].z);
        }

        // PointCloud2로 퍼블리시 (checker_msg_에 체스보드 코너 포인트)
        point3fVectorToPointCloud2(estimated_chessboard_corners_lidar, checker_msg_, "map", this->now());

        // --- 논문 4.2 코너 추정 끝 ---

        calibrateLidarCameraFinal(cloud, estimated_chessboard_corners_lidar); // 수정된 함수 호출
    }

    // 4.2.1 Model Formulation: PCA based alignment and centering
    void transformPointsToChessboardPlane(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_cloud,
        cv::Mat &M_XOY_P, // Rotation matrix (PCA vectors)
        cv::Mat &t_XOY_F) // Translation vector (mean)
    {
        transformed_cloud->clear();
        if (input_cloud->points.size() < 3) {
            RCLCPP_WARN(this->get_logger(), "Not enough points for PCA transformation.");
            M_XOY_P = cv::Mat::eye(3, 3, CV_64F);
            t_XOY_F = cv::Mat::zeros(3, 1, CV_64F);
            return;
        }

        // 1. Perform PCA
        Eigen::MatrixXd data(input_cloud->points.size(), 3);
        for (size_t i = 0; i < input_cloud->points.size(); ++i) {
            data(i, 0) = input_cloud->points[i].x;
            data(i, 1) = input_cloud->points[i].y;
            data(i, 2) = input_cloud->points[i].z;
        }

        Eigen::RowVectorXd mean = data.colwise().mean();
        Eigen::MatrixXd centered_data = data.rowwise() - mean;
        Eigen::MatrixXd cov = centered_data.transpose() * centered_data;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);

        Eigen::Matrix3d eigenvectors = eig.eigenvectors();
        Eigen::Vector3d eigenvalues = eig.eigenvalues();

        // PCA results (eigenvectors represent principal components as column vectors)
        Eigen::Vector3d mu1 = eigenvectors.col(2); // Largest eigenvalue
        Eigen::Vector3d mu2 = eigenvectors.col(1); // Second largest
        Eigen::Vector3d mu3 = eigenvectors.col(0); // Smallest (normal vector)

        // 2. Align principal component vectors (refer to paper 4.2.1)
        // mu3 (normal vector) direction: should point towards the LiDAR origin
        // Vector from LiDAR origin (0,0,0) to chessboard centroid
        Eigen::Vector3d centroid = mean.transpose();
        if (mu3.dot(centroid) > 0) { // If normal points away from LiDAR origin, flip it
            mu3 = -mu3;
        }
        
        // Right-hand rule (mu1 x mu2 = mu3)
        // Currently mu1, mu2, mu3 are sorted by magnitude, not necessarily following right-hand rule
        // Since mu3 is the normal, mu1 and mu2 can be freely chosen within the plane
        // Here, align mu1 closest to x-axis, and mu2 closest to y-axis.
        // Or simply compare mu1 x mu2 with mu3 to ensure right-hand rule.
        Eigen::Vector3d test_mu3_cross = mu1.cross(mu2);
        if (test_mu3_cross.dot(mu3) < 0) { // If not following right-hand rule, flip mu2
            mu2 = -mu2;
        }

        // Angle between mu1 and LiDAR x-axis is not more than 90 degrees (paper 4.2.1)
        Eigen::Vector3d lidar_x_axis(1.0, 0.0, 0.0);
        if (mu1.dot(lidar_x_axis) < 0) { // If mu1 is opposite to x-axis, flip it
            mu1 = -mu1;
            mu2 = mu3.cross(mu1); // If mu1 is flipped, re-calculate mu2 to maintain right-hand rule
        }

        M_XOY_P = (cv::Mat_<double>(3, 3) <<
            mu1(0), mu1(1), mu1(2),
            mu2(0), mu2(1), mu2(2),
            mu3(0), mu3(1), mu3(2));

        t_XOY_F = (cv::Mat_<double>(3, 1) << mean(0), mean(1), mean(2));

        // 3. Transform points (rotate then translate to center)
        cv::Mat mean_mat = (cv::Mat_<double>(3, 1) << mean(0), mean(1), mean(2));
        cv::Mat R_pca_forward = M_XOY_P; // M_XOY_P already acts as the rotation matrix

        for (const auto& p : input_cloud->points) {
            cv::Mat pt_orig = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
            cv::Mat pt_transformed_rotated = R_pca_forward * pt_orig; // Rotate
            cv::Mat pt_transformed_translated = pt_transformed_rotated - R_pca_forward * mean_mat; // Translate to center

            pcl::PointXYZI new_p;
            new_p.x = pt_transformed_translated.at<double>(0);
            new_p.y = pt_transformed_translated.at<double>(1);
            new_p.z = pt_transformed_translated.at<double>(2); // z should be close to 0
            new_p.intensity = p.intensity;
            transformed_cloud->push_back(new_p);
        }
        RCLCPP_INFO(this->get_logger(), "PCA transformation completed.");
    }

    // 4.2.2 Correspondence of Intensity and Color: Estimate intensity thresholds
    // This function is still present but its output is not directly used for findChessboardCorners.
    // It could be used for pre-filtering or validation.
    void estimateIntensityThresholds(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud)
    {
        if (cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Cannot estimate intensity thresholds: input cloud is empty.");
            tau_l_ = 0.0;
            tau_h_ = 0.0;
            return;
        }

        // 1. Extract intensity values
        std::vector<float> intensities;
        intensities.reserve(cloud->points.size());
        for (const auto& p : cloud->points) {
            intensities.push_back(p.intensity);
        }

        // 2. Create histogram (simple approach)
        // More sophisticated histogram binning might be needed in practice.
        // Assuming intensity range is 0-255, using 256 bins.
        int hist_size = 256;
        std::vector<int> hist(hist_size, 0);
        float max_intensity = 0.0;
        for (float intensity : intensities) {
            int bin = static_cast<int>(std::round(intensity));
            if (bin >= 0 && bin < hist_size) {
                hist[bin]++;
            }
            if (intensity > max_intensity) max_intensity = intensity;
        }

        // 3. Find two peaks (R_L, R_H)
        // Find the two highest peaks.
        int R_L_bin = -1, R_H_bin = -1;
        int max_count1 = 0, max_count2 = 0;

        for (int i = 0; i < hist_size; ++i) {
            if (hist[i] > max_count1) {
                max_count2 = max_count1;
                R_H_bin = R_L_bin; // Store the second peak
                max_count1 = hist[i];
                R_L_bin = i; // Store the first peak
            } else if (hist[i] > max_count2) {
                max_count2 = hist[i];
                R_H_bin = i;
            }
        }

        // Ensure R_L_bin and R_H_bin represent two distinct peaks
        if (R_L_bin == -1 || R_H_bin == -1 || R_L_bin == R_H_bin) {
            RCLCPP_WARN(this->get_logger(), "Could not find two distinct intensity peaks. Using default thresholds.");
            tau_l_ = 50.0; // Fallback
            tau_h_ = 150.0; // Fallback
            return;
        }

        // Sort R_L as lower intensity peak, R_H as higher intensity peak
        double R_L_val = std::min(static_cast<double>(R_L_bin), static_cast<double>(R_H_bin));
        double R_H_val = std::max(static_cast<double>(R_L_bin), static_cast<double>(R_H_bin));

        // 4. Calculate gray zone (tau_l, tau_h) (Equation (4))
        // epsilon_g_ is a class member variable set to 2.0 (corner estimation) or 4.0 (error evaluation)
        tau_l_ = ((epsilon_g_ - 1) * R_L_val + R_H_val) / epsilon_g_;
        tau_h_ = (R_L_val + (epsilon_g_ - 1) * R_H_val) / epsilon_g_;
    }

    // calculateCost and related optimization are removed as findChessboardCorners is used directly.
    // If fine-tuning is needed, a different optimization approach could be re-introduced.

    // Calculate chessboard corners using optimized pose (now refers to the result of findChessboardCorners)
    void calculateChessboardCorners(
        const cv::Mat &M_XOY_P,           // PCA Rotation matrix
        const cv::Mat &t_XOY_F,           // PCA Translation vector (mean)
        const std::vector<cv::Point2f> &image_corners_from_lidar_intensity, // New input: 2D corners from intensity image
        int img_width, int img_height, double img_min_x, double img_min_y, int pixels_per_meter, // Image parameters
        std::vector<cv::Point3f> &estimated_corners)
    {
        estimated_corners.clear();

        // 1. Convert 2D image corners back to 3D points on the PCA plane (Z=0)
        std::vector<cv::Point3f> transformed_model_corners_in_plane;
        transformed_model_corners_in_plane.reserve(image_corners_from_lidar_intensity.size());

        for (const auto& corner_2d : image_corners_from_lidar_intensity) {
            double x_pca = corner_2d.x / pixels_per_meter + img_min_x;
            double y_pca = corner_2d.y / pixels_per_meter + img_min_y;
            transformed_model_corners_in_plane.emplace_back(x_pca, y_pca, 0.0); // Z in PCA plane is 0
        }

        // 2. Apply inverse PCA transformation (reverse rotation and translation) to convert to LiDAR coordinate system
        cv::Mat R_pca_inv = M_XOY_P.t(); // Inverse of rotation matrix is its transpose
        cv::Mat mean_vec_original_coord_system = t_XOY_F; // Mean vector used during PCA transformation

        for (const auto& pt_transformed_pca : transformed_model_corners_in_plane) {
            cv::Mat pt_transformed_mat = (cv::Mat_<double>(3, 1) << pt_transformed_pca.x, pt_transformed_pca.y, pt_transformed_pca.z);
            
            // Inverse transformation: p_orig = R_pca_inv * (p_transformed_translated + M_XOY_P * mean_vec_original_coord_system);
            cv::Mat pt_original_coord = R_pca_inv * (pt_transformed_mat + M_XOY_P * mean_vec_original_coord_system);
            
            estimated_corners.emplace_back(
                pt_original_coord.at<double>(0),
                pt_original_coord.at<double>(1),
                pt_original_coord.at<double>(2)
            );
        }
    }


    // Renamed calibrateLidarCamera function and modified parameters
    void calibrateLidarCameraFinal(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_all,
                                   const std::vector<cv::Point3f> &estimated_chessboard_corners_lidar)
    {
        // lidar_points now uses estimated_chessboard_corners_lidar
        std::vector<cv::Point3f> lidar_points = estimated_chessboard_corners_lidar;

        std::vector<cv::Point3f> cb_object_points;
        // The object points represent the 3D coordinates of the chessboard corners in its own coordinate system (origin at one corner, Z=0)
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

        if (num_lidar_corners == 0 || num_cam_corners == 0) {
            RCLCPP_ERROR(this->get_logger(), "No corners detected from one or both sensors for SVD computation. LiDAR corners: %zu, Camera corners: %zu", num_lidar_corners, num_cam_corners);
            // Optionally, shutdown or return if calibration cannot proceed
            return;
        }

        if (num_lidar_corners != num_cam_corners) {
            RCLCPP_WARN(this->get_logger(), "Mismatch in corner counts! LiDAR: %zu, Camera: %zu. SVD might be inaccurate.", num_lidar_corners, num_cam_corners);
            // If counts differ, SVD will only use the minimum number of points, which might be incorrect.
            // A more robust approach would be to try to match corresponding corners.
            // For now, we'll resize to the smaller set.
            size_t n_points_svd = std::min(num_lidar_corners, num_cam_corners);
            if (lidar_points.size() > n_points_svd) {
                lidar_points.resize(n_points_svd);
            }
            if (cb2cam_points.size() > n_points_svd) {
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

        std::string filename = img_path_ + "/img_0.png";
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
        // RCLCPP_INFO(this->get_logger(), "Publishing PointCloud2 with frame_id: '%s' and %zu points.", frame_id.c_str(), points.size()); // Debug log

        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id.c_str();

        msg.height = 1;
        msg.width = static_cast<uint32_t>(points.size());
        msg.is_dense = false; // Set to false if NaN values might exist
        msg.is_bigendian = false;

        sensor_msgs::PointCloud2Modifier modifier(msg);
        modifier.setPointCloud2FieldsByString(1, "xyz"); // Add x, y, z fields only
        modifier.resize(points.size());

        sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");

        for (const auto &pt : points)
        {
            *iter_x = pt.x;
            *iter_y = pt.y;
            *iter_z = pt.z;

            ++iter_x;
            ++iter_y;
            ++iter_z;
        }
        // It's good practice to add an intensity field for RViz to visualize PointCloud2 (optional)
        // modifier.setPointCloud2FieldsByString(2, "xyz", "intensity");
        // sensor_msgs::PointCloud2Iterator<float> iter_intensity(msg, "intensity");
        // for (const auto &pt : points) { *iter_intensity = 1.0f; ++iter_intensity; } // Arbitrary intensity value
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
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CamLidarCalibNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
