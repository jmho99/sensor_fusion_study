#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/crop_box.h>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class cv_cam_lidar_cali : public rclcpp::Node
{
public:
    cv_cam_lidar_cali()
    : Node("cv_cam_lidar_cali")
    {
        board_size_ = cv::Size(10, 7);
        square_size_ = 0.02; // m

        camera_matrix_ = (cv::Mat_<double>(3,3) <<
             1.0412562786381386e+03, 0., 6.8565540026982239e+02, 0.,
       1.0505976084535532e+03, 5.9778012298164469e+02, 0., 0., 1.);

        distortion_coeffs_ = (cv::Mat_<double>(1,5) <<
            -1.3724382468171908e-01, 4.9079709117302012e-01,
       8.2971299771431115e-03, -4.5215579888173568e-03,
       -7.7949268098546165e-01);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/flir_camera/image_raw", 10,
            std::bind(&cv_cam_lidar_cali::image_callback, this, std::placeholders::_1));

        pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ouster/points", rclcpp::SensorDataQoS(),
            std::bind(&cv_cam_lidar_cali::pc_callback, this, std::placeholders::_1));
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30),
            std::bind(&cv_cam_lidar_cali::timer_callback, this));
    }

private:
    // ROS subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;

    // Latest data
    cv::Mat last_image_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_cloud_{new pcl::PointCloud<pcl::PointXYZ>};

    // Chessboard parameters
    cv::Size board_size_;
    double square_size_;

    // Camera intrinsics
    cv::Mat camera_matrix_;
    cv::Mat distortion_coeffs_;

    rclcpp::TimerBase::SharedPtr timer_;

    void timer_callback()
{
    if (!last_image_.empty()) {
        cv::imshow("Camera Image", last_image_);
    } else {
        // 빈 화면이라도 띄우도록 할 수 있음
        cv::Mat dummy = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(dummy, "No camera image", cv::Point(50, 240),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255), 2);
        cv::imshow("Camera Image", dummy);
    }

    input_keyboard(last_image_);
}

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        last_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;

        if (!last_image_.empty()) {
            cv::imshow("Camera Image", last_image_);
            input_keyboard(last_image_);
        }
    }

    void pc_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::fromROSMsg(*msg, *last_cloud_);
    }

    void input_keyboard(const cv::Mat &frame)
    {
        int key = cv::waitKey(1);
        if (key == 's') {
            save_current_frame(frame.clone());
        } else if (key == 'c') {
            run_calibration_from_folder();
        } else if (key == 'e') {
            calibration_error();
        }
    }

    void save_current_frame(const cv::Mat &image)
    {
        // Save image
        fs::create_directories("capture");
        std::string img_path = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/capture/image.png";
        cv::imwrite(img_path, image);

        // Save point cloud
        if (last_cloud_->empty()) {
            RCLCPP_WARN(this->get_logger(), "No point cloud captured yet!");
            return;
        }
        std::string pc_path = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/capture/pointcloud.pcd";
        pcl::io::savePCDFileBinary(pc_path, *last_cloud_);
        RCLCPP_INFO(this->get_logger(), "Saved image and pointcloud.");
    }

    void run_calibration_from_folder()
    {
    // Check files exist
        std::string img_path = "/home/antlab/fusion_study_ws/src/cv_test/capture/image.png";
        std::string pc_path = "/home/antlab/fusion_study_ws/src/cv_test/capture/pointcloud.pcd";
        if (!fs::exists(img_path) || !fs::exists(pc_path)) {
            RCLCPP_WARN(this->get_logger(), "No saved data. Capture first!");
            return;
        }

        // Load color image
        cv::Mat img_color = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img_color.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load saved image!");
            return;
        }
            
        // Convert to grayscale
        cv::Mat img_gray;
        cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);
        
        // Find corners
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img_gray, board_size_, corners);
        if (!found) {
            RCLCPP_ERROR(this->get_logger(), "Chessboard not found in image!");
            return;
        }
            

        cv::cornerSubPix(img_gray, corners, cv::Size(11,11),
                         cv::Size(-1,-1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        // 3D object points in chessboard coordinate system
        std::vector<cv::Point3f> object_points;
        for (int i = 0; i < board_size_.height; i++) {
            for (int j = 0; j < board_size_.width; j++) {
                object_points.emplace_back(j * square_size_, i * square_size_, 0.0);
            }
        }

        // Solve PnP
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(object_points, corners,
                                    camera_matrix_, distortion_coeffs_,
                                    rvec, tvec);

        if (!success) {
            RCLCPP_ERROR(this->get_logger(), "solvePnP failed!");
            return;
        }

        // Load pointcloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile(pc_path, *cloud);

        // 추출 예시: 평면 검출
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud);
        crop.setMin(Eigen::Vector4f(-0.7, -0.329, -1.0, 1.0)); // ROI 최소 x,y,z
        crop.setMax(Eigen::Vector4f(-0.3, 0.005, 1.0, 1.0));   // ROI 최대 x,y,z

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZ>);
        crop.filter(*cloud_roi);

        // 평면 분할
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.02);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        seg.setInputCloud(cloud_roi);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No planar model found.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Plane coefficients: %f %f %f %f",
                    coefficients->values[0],
                    coefficients->values[1],
                    coefficients->values[2],
                    coefficients->values[3]);

        // 라이다 평면 상 점들 추출
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_roi);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_roi);

        // plane_cloud 중심 좌표를 chessboard 중심과 맞추는 식으로 R, t 추정 가능
        // plane_cloud 중심 좌표 계산
Eigen::Vector3d lidar_center(0,0,0);
for (const auto& p : cloud_roi->points) {
    lidar_center += Eigen::Vector3d(p.x, p.y, p.z);
}
lidar_center /= cloud_roi->size();

// 체커보드 중심 계산
double cb_center_x = (board_size_.width - 1) * 0.5 * square_size_;
double cb_center_y = (board_size_.height - 1) * 0.5 * square_size_;
double cb_center_z = 0.0;

cv::Mat cb_center_cb = (cv::Mat_<double>(3,1) <<
    cb_center_x, cb_center_y, cb_center_z);

// rvec → R
cv::Mat R;
cv::Rodrigues(rvec, R);

// 체커보드 중심을 camera 좌표계로
cv::Mat cb_center_cam = R * cb_center_cb + tvec;

// LiDAR 평면 중심을 camera 좌표계로
cv::Mat lidar_center_cam = (cv::Mat_<double>(3,1) <<
    lidar_center(0),
    lidar_center(1),
    lidar_center(2));

// 새로운 tvec 계산
cv::Mat new_tvec = lidar_center_cam - R * cb_center_cb;

RCLCPP_INFO(this->get_logger(), "Updated tvec for center alignment: [%f %f %f]",
    new_tvec.at<double>(0),
    new_tvec.at<double>(1),
    new_tvec.at<double>(2));

// tvec 교체
tvec = new_tvec;

        // 여기서는 예시로 그냥 출력
        RCLCPP_INFO(this->get_logger(), "PnP rvec: [%f %f %f]", rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
        RCLCPP_INFO(this->get_logger(), "PnP tvec: [%f %f %f]", tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        // save R, t
        save_rt(rvec, tvec);

        // 여기서 따로 만든 함수 호출
    compute_lidar_camera_extrinsic(cloud_roi, rvec, tvec, board_size_, square_size_);
    }

    void save_rt(const cv::Mat& rvec, const cv::Mat& tvec)
    {
        std::ofstream file("capture/extrinsic.txt");
        file << "rvec: "
             << rvec.at<double>(0) << " "
             << rvec.at<double>(1) << " "
             << rvec.at<double>(2) << "\n";
        file << "tvec: "
             << tvec.at<double>(0) << " "
             << tvec.at<double>(1) << " "
             << tvec.at<double>(2) << "\n";
        file.close();

        RCLCPP_INFO(this->get_logger(), "Saved extrinsic parameters.");
    }

    void compute_lidar_camera_extrinsic(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
    const cv::Mat &rvec,
    const cv::Mat &tvec,
    const cv::Size &board_size,
    double square_size)
{
    // 1. LiDAR 평면 점들을 체커보드 좌표계로 변환
    std::vector<cv::Point3f> lidar_points_cb =
        transformLidarPointsToChessboard(plane_cloud, rvec, tvec);

    // 2. 체커보드 기준 object points 생성
    std::vector<cv::Point3f> object_points;
    for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
            object_points.emplace_back(j * square_size, i * square_size, 0.0);
        }
    }

    // 3. 점 개수 맞춤
    size_t n_points = std::min(lidar_points_cb.size(), object_points.size());
    lidar_points_cb.resize(n_points);
    object_points.resize(n_points);

    if (n_points < 3) {
        std::cerr << "Not enough points for SVD computation!" << std::endl;
        return;
    }

    // 4. SVD 기반 R, t 계산
    cv::Mat R_lidar_to_cam, t_lidar_to_cam;
    computeRigidTransformSVD(lidar_points_cb, object_points, R_lidar_to_cam, t_lidar_to_cam);

    std::cout << "[Lidar → Camera] R:\n" << R_lidar_to_cam << std::endl;
    std::cout << "[Lidar → Camera] t:\n" << t_lidar_to_cam << std::endl;

    // 저장
    std::ofstream ofs("capture/extrinsic_lidar_to_camera.txt");
    ofs << "R:\n" << R_lidar_to_cam << "\nt:\n" << t_lidar_to_cam << std::endl;
    ofs.close();

    double residual = 0.0;
for (size_t i = 0; i < object_points.size(); ++i) {
    cv::Mat src_pt = (cv::Mat_<double>(3,1) <<
        lidar_points_cb[i].x,
        lidar_points_cb[i].y,
        lidar_points_cb[i].z);

    cv::Mat transformed = R_lidar_to_cam * src_pt + t_lidar_to_cam;

    double dx = transformed.at<double>(0) - object_points[i].x;
    double dy = transformed.at<double>(1) - object_points[i].y;
    double dz = transformed.at<double>(2) - object_points[i].z;
    residual += dx*dx + dy*dy + dz*dz;
}
residual = std::sqrt(residual / object_points.size());
std::cout << "[Lidar → Camera] RMS residual error: " << residual << " m" << std::endl;

// LiDAR points → camera frame 변환
std::vector<cv::Point3f> lidar_points_in_cam;
for (const auto& pt : lidar_points_cb) {
    cv::Mat pt_cb = (cv::Mat_<double>(3,1) <<
        pt.x, pt.y, pt.z);
    cv::Mat R;
cv::Rodrigues(rvec, R);
    cv::Mat pt_cam = R * pt_cb + tvec;
    lidar_points_in_cam.emplace_back(
        pt_cam.at<double>(0),
        pt_cam.at<double>(1),
        pt_cam.at<double>(2));
}

// project to image
std::vector<cv::Point2f> projected_points;
cv::projectPoints(
    lidar_points_in_cam,
    cv::Vec3d(0,0,0),
    cv::Vec3d(0,0,0),
    camera_matrix_,
    distortion_coeffs_,
    projected_points);

// load image
cv::Mat img_color = cv::imread(
    "/home/antlab/fusion_study_ws/src/cv_test/capture/image.png",
    cv::IMREAD_COLOR);
if (img_color.empty()) {
    std::cerr << "Failed to load saved image for overlay!" << std::endl;
    return;
}

// draw points
for (const auto& pt : projected_points) {
    if (pt.x >= 0 && pt.x < img_color.cols &&
        pt.y >= 0 && pt.y < img_color.rows) {
        cv::circle(img_color, pt, 3, cv::Scalar(0,0,255), -1);
    }
}

// save image
std::string save_path = "/home/antlab/fusion_study_ws/src/cv_test/capture/image_with_lidar_points.png";
cv::imwrite(save_path, img_color);
std::cout << "Saved overlay image: " << save_path << std::endl;


}

       void calibration_error()
    {
        // TODO: 재투영 오차 계산
        RCLCPP_INFO(this->get_logger(), "Calibration error check not implemented yet.");
    }

    bool extractPlaneFromPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_cloud,
    pcl::ModelCoefficients::Ptr &coefficients)
{
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        return false;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plane_cloud);

    return true;
}

std::vector<cv::Point3f> transformLidarPointsToChessboard(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_cloud,
    const cv::Mat &rvec, const cv::Mat &tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);  // 회전벡터 → 회전행렬 변환

    std::vector<cv::Point3f> transformed_points;
    for (const auto& pt : plane_cloud->points) {
        cv::Mat pt_mat = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
        cv::Mat pt_cb = R.t() * (pt_mat - tvec); // 카메라 좌표계 → 체커보드 좌표계 (역변환)
        transformed_points.emplace_back(pt_cb.at<double>(0), pt_cb.at<double>(1), pt_cb.at<double>(2));
    }
    return transformed_points;
}

void computeRigidTransformSVD(
    const std::vector<cv::Point3f> &src_points,
    const std::vector<cv::Point3f> &dst_points,
    cv::Mat &R, cv::Mat &t)
{
    assert(src_points.size() == dst_points.size());
    int N = (int)src_points.size();

    // Eigen 행렬 변환
    Eigen::MatrixXd src(3, N), dst(3, N);
    for (int i = 0; i < N; i++) {
        src(0, i) = src_points[i].x;
        src(1, i) = src_points[i].y;
        src(2, i) = src_points[i].z;

        dst(0, i) = dst_points[i].x;
        dst(1, i) = dst_points[i].y;
        dst(2, i) = dst_points[i].z;
    }

    // 평균 계산
    Eigen::Vector3d src_mean = src.rowwise().mean();
    Eigen::Vector3d dst_mean = dst.rowwise().mean();

    // 중심화
    src.colwise() -= src_mean;
    dst.colwise() -= dst_mean;

    // 공분산 행렬 계산
    Eigen::Matrix3d H = src * dst.transpose();

    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d R_eigen = V * U.transpose();

    // 반사 행렬 체크
    if (R_eigen.determinant() < 0) {
        V.col(2) *= -1;
        R_eigen = V * U.transpose();
    }

    Eigen::Vector3d t_eigen = dst_mean - R_eigen * src_mean;

    // Eigen → OpenCV 변환
    cv::eigen2cv(R_eigen, R);
    t = (cv::Mat_<double>(3,1) << t_eigen(0), t_eigen(1), t_eigen(2));
}
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<cv_cam_lidar_cali>());
    rclcpp::shutdown();
    return 0;
}
