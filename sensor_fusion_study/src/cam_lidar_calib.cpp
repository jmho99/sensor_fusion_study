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

namespace fs = std::filesystem;

class CamLidarCalibNode : public rclcpp::Node
{
public:
    CamLidarCalibNode()
        : Node("cam_lidar_calib")
    {
        initializedParameters();

        std::string connect = "flir";
        if (connect == "flir")
        {
            sub_cam_ = this->create_subscription<sensor_msgs::msg::Image>("/flir_camera/image_raw", rclcpp::SensorDataQoS(),
                                                                          std::bind(&CamLidarCalibNode::imageCallback, this, std::placeholders::_1));

            sub_lidar_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/ouster/points", rclcpp::SensorDataQoS(),
                                                                                  std::bind(&CamLidarCalibNode::pcdCallback, this, std::placeholders::_1));
        }
        else
        {
            timer__ = this->create_wall_timer(
                std::chrono::milliseconds(500),
                std::bind(&CamLidarCalibNode::timerCallback, this));
        }

        pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_points", 10);
        pub_checker_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("checker_points", 10);
        pub_lidar2cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar2cam_points", 10);
        pub_cb_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cb_object_points", 10);
        pub_cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cam_points", 10);

        // 타이머로 주기적 퍼블리시
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30),
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
    sensor_msgs::msg::PointCloud2 plane_msg_;
    sensor_msgs::msg::PointCloud2 checker_msg_;
    sensor_msgs::msg::PointCloud2 lidar2cam_points_;
    sensor_msgs::msg::PointCloud2 cb_points_msg_;
    sensor_msgs::msg::PointCloud2 cam_points_msg_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr timer__;
    cv::Size board_size_;                          // Chessboard parameters
    double square_size_;                           // Chessboard parameters
    cv::Mat intrinsic_matrix_, distortion_coeffs_; // Camera intrinsics
    cv::Mat cb2cam_rvec_, cb2cam_tvec_;
    cv::Mat lidar2cam_R_, lidar2cam_t_;

    // 이미지에서 찾은 2D 코너
    std::vector<cv::Point2f> image_corners_latest_;
    // 라이다에서 찾은 체스보드 평면 포인트
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_plane_points_latest_{new pcl::PointCloud<pcl::PointXYZI>};

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
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
        plane_msg_.header.stamp = this->now();
        pub_plane_->publish(plane_msg_);
        pub_checker_->publish(checker_msg_);
        pub_lidar2cam_points_->publish(lidar2cam_points_);
        pub_cb_points_->publish(cb_points_msg_);
        pub_cam_points_->publish(cam_points_msg_);
    }

    void initializedParameters()
    {
        std::string where = "company";
        readWritePath(where);

        cv::FileStorage fs(one_cam_result_path_ + "/one_cam_calib_result.yaml", cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            RCLCPP_WARN(rclcpp::get_logger("initializedParameters"), "Failed open one_cam_calib_result.yaml file!");
            rclcpp::shutdown();
        }
        else
        {
            int cols;
            int rows;
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

        absolute_path_ = "/home" + change_path + "/src/sensor_fusion_study/cam_lidar_calib";
        img_path_ = absolute_path_ + "/images";
        pcd_path_ = absolute_path_ + "/pointclouds";
        one_cam_result_path_ = "/home" + change_path + "/src/sensor_fusion_study/one_cam_calib";

        fs::create_directories(img_path_);
        fs::create_directories(pcd_path_);
    }

    void timerCallback()
    {
        if (!last_image_.empty())
        {
            cv::imshow("Camera Image", last_image_);
        }
        else
        {
            // 빈 화면이라도 띄우도록 할 수 있음
            cv::Mat dummy = cv::Mat::zeros(480, 640, CV_8UC3);
            cv::putText(dummy, "No camera image", cv::Point(50, 240),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Camera Image", dummy);
        }

        inputKeyboard(last_image_);
    }

    void inputKeyboard(const cv::Mat &frame)
    {
        int key = cv::waitKey(1);
        if (key == 's')
        {
            saveFrame(frame.clone());
        }
        else if (key == 'c')
        {
            findData();
            solveCameraPlane();
            detectLidarPlane();
        }
        else if (key == 'e')
        {
            rclcpp::shutdown();
        }
    }

    void saveFrame(const cv::Mat &frame)
    {
        // current_frame_이 비어있는지 확인 (카메라 데이터가 수신되었는지)
        if (frame.empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("saveFrame"), "No current camera frame captured yet! Cannot save image.");
            return; // 이미지 없으면 저장 안 함
        }

        std::string img_filename = img_path_ + "/image_" + std::to_string(frame_counter_) + ".png";
        cv::imwrite(img_filename, frame); // current_frame_을 저장

        // Save point cloud
        if (last_cloud_->empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("saveFrame"), "No point cloud captured yet! Cannot save PCD.");
            return; // 포인트 클라우드 없으면 저장 안 함
        }
        std::string pcd_filename = pcd_path_ + "/pointcloud_" + std::to_string(frame_counter_) + ".pcd";
        pcl::io::savePCDFileBinary(pcd_filename, *last_cloud_);
        RCLCPP_INFO(this->get_logger(), "Saved image and pointcloud.");
        frame_counter_++;
    }

    void findData()
    {
        // 1. 이미지 파일 로드
        std::vector<cv::String> image_files;
        // img_path_ 디렉토리의 모든 .png 파일을 찾습니다.
        cv::glob(img_path_ + "/*.png", image_files, false);

        // 이미지 파일 개수 확인
        if (image_files.empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("findData"), "No image files found in %s. Please capture some data first!", img_path_.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Found %zu image files.", image_files.size());

        // 첫 번째 이미지 파일을 로드합니다.
        std::string first_image_path = image_files[0];
        last_image_ = cv::imread(first_image_path, cv::IMREAD_COLOR);

        if (last_image_.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load the first image from %s!", first_image_path.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded first image: %s", first_image_path.c_str());

        // 2. PCD 파일 로드
        std::vector<cv::String> pcd_files;
        // pcd_path_ 디렉토리의 모든 .pcd 파일을 찾습니다.
        cv::glob(pcd_path_ + "/*.pcd", pcd_files, false);

        // PCD 파일 개수 확인
        if (pcd_files.empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("findData"), "No PCD files found in %s. Please capture some data first!", pcd_path_.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Found %zu PCD files.", pcd_files.size());

        // 첫 번째 PCD 파일을 로드합니다.
        std::string first_pcd_path = pcd_files[0];
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(first_pcd_path, *last_cloud_) == -1) // -1은 로드 실패를 의미
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load the first PCD from %s!", first_pcd_path.c_str());
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded first pointcloud: %s", first_pcd_path.c_str());

        RCLCPP_INFO(this->get_logger(), "Successfully loaded calibration data (image and pointcloud).");
    }

    void solveCameraPlane()
    {
        // Load color image
        cv::Mat img_color = last_image_;
        if (img_color.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load saved image!");
            rclcpp::shutdown();
        }

        // Convert to grayscale
        cv::Mat img_gray;
        cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);

        // Find corners
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img_gray, board_size_, corners);
        if (!found)
        {
            RCLCPP_ERROR(this->get_logger(), "Chessboard not found in image!");
            rclcpp::shutdown();
        }

        cv::cornerSubPix(img_gray, corners, cv::Size(5, 5),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        // 3D object points in chessboard coordinate system
        std::vector<cv::Point3f> object_points;
        for (int i = 0; i < board_size_.height; i++)
        {
            for (int j = 0; j < board_size_.width; j++)
            {
                object_points.emplace_back(j * square_size_, i * square_size_, 0.0);
            }
        }

        // Solve PnP
        bool success = cv::solvePnP(object_points, corners,
                                    intrinsic_matrix_, distortion_coeffs_,
                                    cb2cam_rvec_, cb2cam_tvec_);

        if (!success)
        {
            RCLCPP_ERROR(this->get_logger(), "solvePnP failed!");
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
        // Load pointcloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        std::string filename = pcd_path_ + "/pointcloud_0.pcd";
        pcl::io::loadPCDFile(filename, *cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_intensity(new pcl::PointCloud<pcl::PointXYZI>);

        // 1. 반사 강도 기반 필터링 (Reflectance Intensity Assisted)
        // 체스보드의 흰색 칸은 높은 강도, 검은색 칸은 낮은 강도를 가집니다.
        // 일반적인 체스보드 패턴의 반사 강도 범위에 맞춰 필터링합니다.
        // 이 범위는 실제 사용하는 체스보드와 LiDAR에 따라 튜닝해야 합니다.
        pcl::PassThrough<pcl::PointXYZI> pass_intensity;
        pass_intensity.setInputCloud(cloud);
        pass_intensity.setFilterFieldName("intensity");
        pass_intensity.setFilterLimits(50.0, 255.0); // 예시: 50~255 강도 범위 (튜닝 필요)
        pass_intensity.filter(*cloud_filtered_intensity);

        if (cloud_filtered_intensity->empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("detectLidarPlane"), "No points after intensity filtering.");
            lidar_plane_points_latest_->clear();
            rclcpp::shutdown();
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZI>);
        // 2. ROI 필터링 (CropBox) - 체스보드가 있을 것으로 예상되는 영역
        pcl::CropBox<pcl::PointXYZI> crop;
        crop.setInputCloud(cloud_filtered_intensity);
        // 이 값들도 실제 환경에 맞게 튜닝해야 합니다.
        crop.setMin(Eigen::Vector4f(-3.0, -1.0, -0.5, 1.0)); // min x,y,z
        crop.setMax(Eigen::Vector4f(3.0, 1.0, 1.5, 1.0));    // max x,y,z
        crop.filter(*cloud_roi);

        if (cloud_roi->empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("detectLidarPlane"), "No points after ROI filtering.");
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
            RCLCPP_WARN(rclcpp::get_logger("detectLidarPlane"), "No planar model found in filtered LiDAR data.");
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
            RCLCPP_WARN(rclcpp::get_logger("detectLidarPlane"), "Extracted LiDAR plane points are empty.");
            rclcpp::shutdown();
        }

        pcl::toROSMsg(*lidar_plane_points_latest_, plane_msg_);
        plane_msg_.header.frame_id = "map"; // RViz에서 사용하는 좌표계 설정

        RCLCPP_INFO(this->get_logger(), "Detected %zu points in LiDAR plane.", lidar_plane_points_latest_->points.size());

        calibrateLidarCamera(cloud, cloud_roi);
        // checkCenter(cloud_roi);
    }

    void calibrateLidarCamera(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_all,
                              const pcl::PointCloud<pcl::PointXYZI>::Ptr &plane_cloud_cLC)
    {
        std::vector<cv::Point3f> lidar_points;
        for (const auto &p : plane_cloud_cLC->points)
        {
            lidar_points.emplace_back(
                p.x,
                p.y,
                p.z);
        }

        std::vector<cv::Point3f> cb_object_points;
        for (int i = 0; i < board_size_.height; i++)
        {
            for (int j = 0; j < board_size_.width; j++)
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

        // 3. 점 개수 맞춤
        size_t n_points = std::min(lidar_points.size(), cb2cam_points.size());
        if (n_points < 3)
        {
            RCLCPP_INFO(this->get_logger(), "Not enough points for SVD computation!");
            rclcpp::shutdown();
        }
        // evenly sample points
        std::vector<cv::Point3f> sampled;
        size_t step = lidar_points.size() / n_points;
        if (step == 0)
            step = 1;

        for (size_t i = 0; i < lidar_points.size() && sampled.size() < n_points; i += step)
        {
            sampled.push_back(lidar_points[i]);
        }
        lidar_points = sampled;

        point3fVectorToPointCloud2(lidar_points, checker_msg_, "map", this->now());

        // 4. SVD 기반 R, t 계산
        computeRigidTransformSVD(lidar_points, cb2cam_points, lidar2cam_R_, lidar2cam_t_);

        std::cout << "[Lidar → Camera] R:\n"
                  << lidar2cam_R_ << std::endl;
        std::cout << "[Lidar → Camera] t:\n"
                  << lidar2cam_t_ << std::endl;

        // 저장
        saveMultipleToFile("txt", "cam_lidar_calib_result",
                           "[Lidar → Camera] R:\n", lidar2cam_R_,
                           "[Lidar → Camera] t:\n", lidar2cam_t_);

        point3fVectorToPointCloud2(cb2cam_points, cb_points_msg_, "map", this->now());

        std::vector<cv::Point3f> lidar2cam_points_transformed;
        for (const auto &pt : lidar_points)
        {
            cv::Mat pt_mat = (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
            cv::Mat pt_transformed = lidar2cam_R_ * pt_mat + lidar2cam_t_;

            lidar2cam_points_transformed.emplace_back(
                pt_transformed.at<double>(0),
                pt_transformed.at<double>(1),
                pt_transformed.at<double>(2));
        }

        // point3fVectorToPointCloud2(lidar2cam_points_transformed, lidar2cam_points_, "map", this->now());

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

        point3fVectorToPointCloud2(lidar2cam_points_all, lidar2cam_points_, "map", this->now());

        // 이미지 projection
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in_cam(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto &pt : lidar2cam_points_all)
        {
            cloud_in_cam->emplace_back(pt.x, pt.y, pt.z);
        }

        cv::Mat img_color = cv::imread(img_path_, cv::IMREAD_COLOR);
        if (img_color.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "이미지를 불러오지 못했습니다.");
            rclcpp::shutdown();
        }

        projectLidarToImage(cloud_in_cam, img_color, last_image_);

        cv::imshow("Lidar Overlaid (All points)", last_image_);
        cv::waitKey(0);
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
        // 이미지 복사
        image_out = image.clone();

        // 카메라 내부 파라미터
        double fx = intrinsic_matrix_.at<double>(0, 0);
        double fy = intrinsic_matrix_.at<double>(1, 1);
        double cx = intrinsic_matrix_.at<double>(0, 2);
        double cy = intrinsic_matrix_.at<double>(1, 2);

        for (const auto &pt : cloud_in_cam->points)
        {
            if (pt.z <= 0.0)
                continue; // 카메라 앞에 있는 점만 사용

            double x = pt.x;
            double y = pt.y;
            double z = pt.z;

            // 투영
            int u = static_cast<int>((fx * x / z) + cx);
            int v = static_cast<int>((fy * y / z) + cy);

            // 이미지 범위 체크
            if (u >= 0 && u < image_out.cols && v >= 0 && v < image_out.rows)
            {
                // 색상 결정 (e.g. 깊이에 따른 컬러맵)
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
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id;

        msg.height = 1;
        msg.width = points.size();
        msg.is_dense = true;
        msg.is_bigendian = false;

        sensor_msgs::PointCloud2Modifier modifier(msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");

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
    }

    template <typename T>
    void saveToFile(const std::string &extension,
                    const std::string &filename,
                    const T &data)
    {
        std::string fullpath = absolute_path_ + "/" + filename + "." + extension;
        std::ofstream ofs(fullpath);

        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "파일 열기 실패: %s", fullpath.c_str());
            rclcpp::shutdown();
        }

        ofs << data << std::endl;
        ofs.close();

        RCLCPP_INFO(this->get_logger(), "파일 저장 완료: %s", fullpath.c_str());
    }

    // cv::Mat 저장용 오버로드
    void saveToFile(const std::string &extension,
                    const std::string &filename,
                    const cv::Mat &mat)
    {
        std::string fullpath = absolute_path_ + "/" + filename + "." + extension;
        std::ofstream ofs(fullpath);

        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "파일 열기 실패: %s", fullpath.c_str());
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

        RCLCPP_INFO(this->get_logger(), "cv::Mat 저장 완료: %s", fullpath.c_str());
    }

    template <typename... Args>
    void saveMultipleToFile(const std::string &extension,
                            const std::string &filename,
                            const Args &...args)
    {
        std::string fullpath = absolute_path_ + "/" + filename + "." + extension;
        std::ofstream ofs(fullpath);

        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "파일 열기 실패: %s", fullpath.c_str());
            rclcpp::shutdown();
        }

        // 각 인자 처리
        (writeData(ofs, args), ...);

        ofs.close();

        RCLCPP_INFO(this->get_logger(), "파일 저장 완료: %s", fullpath.c_str());
    }

    // generic 타입 저장
    template <typename T>
    void writeData(std::ofstream &ofs, const T &data)
    {
        ofs << data << "\n";
    }

    // cv::Mat 특수화
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
