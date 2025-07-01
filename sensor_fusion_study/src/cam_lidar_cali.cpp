#include <rclcpp/rclcpp.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <fstream>
#include <string>
#include <filesystem>
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

class CamLidarCali : public rclcpp::Node
{
public:
    CamLidarCali()
        : Node("cam_lidar_cali")
    {
        initializedParameters();

        pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_points", 10);
        pub_checker_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("checker_points", 10);
        pub_lidar2cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar2cam_points", 10);
        pub_cb_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cb_object_points", 10);
        pub_cam_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cam_points", 10);

        // 타이머로 주기적 퍼블리시
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30),
            std::bind(&CamLidarCali::timer_callback, this));

        timer__ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&CamLidarCali::timerCallback, this));
    }

private:
    cv::Mat last_image_;                                                                 // Latest image data
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_cloud_{new pcl::PointCloud<pcl::PointXYZ>}; // Latest point cloud data
    std::string img_path_;
    std::string pcd_path_;
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
    cv::Size board_size_;                       // Chessboard parameters
    double square_size_;                        // Chessboard parameters
    cv::Mat camera_matrix_, distortion_coeffs_; // Camera intrinsics
    cv::Mat cb2cam_rvec_, cb2cam_tvec_;
    cv::Mat lidar2cam_R_, lidar2cam_t_;

    void timerCallback()
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
        board_size_ = cv::Size(10, 7);
        square_size_ = 0.02;

        camera_matrix_ = (cv::Mat_<double>(3, 3) << 1.0412562786381386e+03, 0., 6.8565540026982239e+02, 0.,
                          1.0505976084535532e+03, 5.9778012298164469e+02, 0., 0., 1.);

        distortion_coeffs_ = (cv::Mat_<double>(1, 5) << -1.3724382468171908e-01, 4.9079709117302012e-01,
                              8.2971299771431115e-03, -4.5215579888173568e-03,
                              -7.7949268098546165e-01);
        std::string where = "home";
        std::string absolute_path;
        if (where == "company")
        {
            absolute_path = "/home/antlab/sensor_fusion_study/src/sensor_fusion_study";
        }
        else if (where == "home")
        {
            absolute_path = "/home/icrs/sensor_fusion_study_ws/src/sensor_fusion_study";
        }
        // auto absolute_path = ament_index_cpp::get_package_share_directory("/src/sensor_fusion_study");
        img_path_ = absolute_path + "/capture/image.png";
        pcd_path_ = absolute_path + "/capture/pointcloud.pcd";
    }

    void timer_callback()
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

        input_keyboard(last_image_);
    }

    void input_keyboard(const cv::Mat &frame)
    {
        int key = cv::waitKey(1);
        if (key == 's')
        {
            save_current_frame(frame.clone());
        }
        else if (key == 'c')
        {
            solveCameraPlane();
            detectLidarPlane();
        }
        else if (key == 'e')
        {
        }
    }

    void save_current_frame(const cv::Mat &image)
    {
        // Save image
        std::filesystem::create_directories("capture");
        cv::imwrite(img_path_, image);

        // Save point cloud
        if (last_cloud_->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No point cloud captured yet!");
            return;
        }
        pcl::io::savePCDFileBinary(pcd_path_, *last_cloud_);
        RCLCPP_INFO(this->get_logger(), "Saved image and pointcloud.");
    }

    void findData()
    {
        if (!std::filesystem::exists(img_path_) || !std::filesystem::exists(pcd_path_))
        {
            RCLCPP_WARN(this->get_logger(), "No saved data. Capture first!");
            return;
        }
    }

    void solveCameraPlane()
    {
        findData();

        // Load color image
        cv::Mat img_color = cv::imread(img_path_, cv::IMREAD_COLOR);
        if (img_color.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load saved image!");
            return;
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
            return;
        }

        cv::cornerSubPix(img_gray, corners, cv::Size(11, 11),
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
                                    camera_matrix_, distortion_coeffs_,
                                    cb2cam_rvec_, cb2cam_tvec_);

        if (!success)
        {
            RCLCPP_ERROR(this->get_logger(), "solvePnP failed!");
            return;
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
        findData();

        // Load pointcloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile(pcd_path_, *cloud);

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

        if (inliers->indices.empty())
        {
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

        pcl::toROSMsg(*cloud_roi, plane_msg_);
        plane_msg_.header.frame_id = "map"; // RViz에서 사용하는 좌표계 설정

        changeLidarCoordinates(cloud_roi);
        // checkCenter(cloud_roi);
    }

    void changeLidarCoordinates(const pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_cloud_cLC)
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
            return;
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
        saveMultipleToFile("txt", "lidar2cam_extrinsic",
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

        point3fVectorToPointCloud2(lidar2cam_points_transformed, lidar2cam_points_, "map", this->now());
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
                    const T &data,
                    const std::string &dir = "capture")
    {
        std::filesystem::create_directories(dir);

        std::string fullpath = dir + "/" + filename + "." + extension;
        std::ofstream ofs(fullpath);

        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "파일 열기 실패: %s", fullpath.c_str());
            return;
        }

        ofs << data << std::endl;
        ofs.close();

        RCLCPP_INFO(this->get_logger(), "파일 저장 완료: %s", fullpath.c_str());
    }

    // cv::Mat 저장용 오버로드
    void saveToFile(const std::string &extension,
                    const std::string &filename,
                    const cv::Mat &mat,
                    const std::string &dir = "capture")
    {
        std::filesystem::create_directories(dir);

        std::string fullpath = dir + "/" + filename + "." + extension;
        std::ofstream ofs(fullpath);

        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "파일 열기 실패: %s", fullpath.c_str());
            return;
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
        std::filesystem::create_directories("capture");

        std::string fullpath = "capture/" + filename + "." + extension;
        std::ofstream ofs(fullpath);

        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "파일 열기 실패: %s", fullpath.c_str());
            return;
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
    auto node = std::make_shared<CamLidarCali>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
