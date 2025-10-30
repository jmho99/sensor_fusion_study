#include <rclcpp/rclcpp.hpp>

#include <fstream>
#include <string>
#include <filesystem>
#include <Eigen/Dense>
#include <algorithm>

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/frustum_culling.h>

// #include "jmh_utils/jmh_utils.hpp"

using namespace std::chrono_literals;

using std::placeholders::_1;
using std::placeholders::_2;

class CamLidarFusionNode : public rclcpp::Node
{
public:
    CamLidarFusionNode(const std::vector<std::vector<double>> &intrins,
                       const std::vector<double> &distort,
                       const std::vector<std::vector<double>> &extrins)
        : Node("z_cam_lidar_fusion")
    {
        intrinsic_matrix_ = vector2Mat(intrins);
        distortion_coeffs_ = vector2Mat(distort);
        cv::Mat lidar2cam_T = vector2Mat(extrins);
        // std::cout << "finish convert to cv::Mat" << std::endl;
        // std::cout << intrinsic_matrix_ << std::endl;
        // std::cout << distortion_coeffs_ << std::endl;
        // std::cout << lidar2cam_T << std::endl;

        lidar2cam_R_.create(3, 3, CV_64F);
        lidar2cam_t_.create(3, 1, CV_64F);

        for (int i = 0; i < lidar2cam_T.rows; i++)
        {
            for (int j = 0; j < lidar2cam_T.cols; j++)
            {
                double each = lidar2cam_T.at<double>(i, j);
                if (j < 3)
                {
                    lidar2cam_R_.at<double>(i, j) = each;
                }
                else if (j >= 3)
                {
                    lidar2cam_t_.at<double>(i, 0) = each;
                }
                if (i >= 4)
                    continue;
            }
        }
        // std::cout << lidar2cam_R_ << std::endl;
        // std::cout << lidar2cam_t_ << std::endl;

        Eigen::Matrix3d R_eig_m;
        Eigen::Vector3d t_eig_v;
        cv::cv2eigen(lidar2cam_R_, R_eig_m);
        cv::cv2eigen(lidar2cam_t_, t_eig_v);
        R_eig_ = R_eig_m.cast<float>();
        t_eig_ = t_eig_v.cast<float>();

        T_eig_ = Eigen::Matrix4f::Identity();
        T_eig_.block<3, 3>(0, 0) = R_eig_;
        T_eig_.block<3, 1>(0, 3) = t_eig_;

        std::string what = "hard";
        auto cam_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
        auto lid_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();

        if (what == "img")
        {
            runprojection();
        }
        else if (what == "filter")
        {
            cam_out_ = this->create_publisher<sensor_msgs::msg::Image>(
                "/flir_camera/image_raw_shifted", cam_qos);
            cam_in_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/flir_camera/image_raw", cam_qos,
                [this](sensor_msgs::msg::Image::ConstSharedPtr m)
                {
                    auto out = std::make_shared<sensor_msgs::msg::Image>(*m); // 복사
                    out->header.stamp = rclcpp::Time(out->header.stamp) + rclcpp::Duration::from_seconds(cam_time_bias_sec_);
                    cam_out_->publish(*out);
                });

            cam_filter_sub_.subscribe(this, "/flir_camera/image_raw_shifted", cam_qos.get_rmw_qos_profile());
            lid_filter_sub_.subscribe(this, "/ouster/points", lid_qos.get_rmw_qos_profile());

            cam_filter_sub_.registerCallback([this](sensor_msgs::msg::Image::ConstSharedPtr m)
                                             { RCLCPP_INFO(this->get_logger(), "CAM %.3f", rclcpp::Time(m->header.stamp).seconds()); });

            lid_filter_sub_.registerCallback([this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m)
                                             { RCLCPP_INFO(this->get_logger(), "LID %.3f", rclcpp::Time(m->header.stamp).seconds()); });

            /*
                            if (cam_time && lid_time)
                            {
                                dt = (rclcpp::Time(cam_time->header.stamp) - rclcpp::Time(lid_time->header.stamp)).seconds();
                                RCLCPP_INFO(get_logger(), "dt(cam-lid)=%.3f s", dt);
                            }

                            else
                                dt = 0;
                    */
            uint32_t queue_size = 2;
            sync_ = std::make_shared<message_filters::Synchronizer<Policy>>(Policy(queue_size), cam_filter_sub_, lid_filter_sub_);
            sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.02));
            sync_->registerCallback(std::bind(&CamLidarFusionNode::syncCallback, this, _1, _2));
        }
        else if (what == "hard")
        {
            cam_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/flir_camera/image_raw", cam_qos,
                                                                          std::bind(&CamLidarFusionNode::imageCallback, this, std::placeholders::_1));
            lid_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/ouster/points", lid_qos,
                                                                                std::bind(&CamLidarFusionNode::lidarCallback, this, std::placeholders::_1));
            /*
                        // fusionOutput(current_frame_, current_cloud_);
                        fusion_timer_ = this->create_wall_timer(
                            std::chrono::milliseconds(50),
                            std::bind(&CamLidarFusionNode::fusionCallback, this));
                            */
        }
        fusion_pub_ = image_transport::create_publisher(this, "/calib/image");
        /*
        pub_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(100),
                std::bind(&CamLidarFusionNode::pubTimerCallback, this));
        */
    }

private:
    // 파라미터: cam_time_bias_sec (예: -0.200 → 200ms 당겨서 라이다와 맞춤)
    double cam_time_bias_sec_ = this->declare_parameter("cam_time_bias_sec", -0.150);
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_in_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cam_out_;

    sensor_msgs::msg::Image::ConstSharedPtr cam_time;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr lid_time;
    double dt;
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lid_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> cam_filter_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lid_filter_sub_;
    using Policy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>;
    std::shared_ptr<message_filters::Synchronizer<Policy>> sync_;

    image_transport::Publisher fusion_pub_;
    rclcpp::TimerBase::SharedPtr pub_timer_, fusion_timer_;

    cv::Mat current_frame_;
    cv::Mat new_camera_matrix_, intrinsic_matrix_, distortion_coeffs_;
    cv::Mat image_with_lidar_projection_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_{new pcl::PointCloud<pcl::PointXYZI>};

    cv::Mat lidar2cam_R_, lidar2cam_t_;
    Eigen::Matrix3f R_eig_;
    Eigen::Vector3f t_eig_;
    Eigen::Matrix4f T_eig_;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    }

    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::fromROSMsg(*msg, *current_cloud_);

        RCLCPP_INFO(this->get_logger(), "cam and lidar");
        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);

        pcl::transformPointCloud(*current_cloud_, *transformed_cloud, T_eig_);

        projectLidarToImage(transformed_cloud, current_frame_, image_with_lidar_projection_);

        pubTimerCallback();
    }

    void fusionCallback()
    {

        RCLCPP_INFO(this->get_logger(), "cam and lidar");

        Eigen::Matrix3d R_eig;
        Eigen::Vector3d t_eig;
        cv::cv2eigen(lidar2cam_R_, R_eig);
        cv::cv2eigen(lidar2cam_t_, t_eig);

        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 3>(0, 0) = R_eig.cast<float>();
        T.block<3, 1>(0, 3) = t_eig.cast<float>();

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cam(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*current_cloud_, *cloud_cam, T);

        pcl::FrustumCulling<pcl::PointXYZI> fc;
        fc.setInputCloud(cloud_cam);

        image_with_lidar_projection_ = current_frame_.clone();

        int W = image_with_lidar_projection_.cols;
        int H = image_with_lidar_projection_.rows;
        double PI = 3.141592;
        const double fx = intrinsic_matrix_.at<double>(0, 0);
        const double fy = intrinsic_matrix_.at<double>(1, 1);
        const double cx = intrinsic_matrix_.at<double>(0, 2);
        const double cy = intrinsic_matrix_.at<double>(1, 2);

        double hfov = 2.0 * std::atan(static_cast<double>(W) / (2.0 * intrinsic_matrix_.at<double>(0, 0))) * 180.0 / PI;
        double vfov = 2.0 * std::atan(static_cast<double>(H) / (2.0 * intrinsic_matrix_.at<double>(1, 1))) * 180.0 / PI;

        fc.setHorizontalFOV(static_cast<float>(hfov));
        fc.setVerticalFOV(static_cast<float>(vfov));

        fc.setNearPlaneDistance(0.1f);
        fc.setFarPlaneDistance(100.0f);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZI>());
        fc.filter(*cloud_roi);

        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);

        // 5) 이미 CAM 좌표계 → 직접 투영 (왜곡 무시; 왜곡 쓰면 projectPoints에 rvec=0,tvec=0)
        //    (lineType은 cv::LINE_AA 등으로 지정; COLOR_BGRA2BGR 사용 금지!)

        if (image_with_lidar_projection_.type() != CV_8UC3)
        {
            // 그리기 호환을 위해 BGR 8U로 강제
            cv::Mat tmp;
            if (image_with_lidar_projection_.channels() == 1)
            {
                image_with_lidar_projection_.convertTo(tmp, CV_8UC1);
                cv::cvtColor(tmp, image_with_lidar_projection_, cv::COLOR_GRAY2BGR);
            }
            else if (image_with_lidar_projection_.type() == CV_8UC4)
            {
                cv::cvtColor(image_with_lidar_projection_, image_with_lidar_projection_, cv::COLOR_BGRA2BGR);
            }
            else
            {
                image_with_lidar_projection_.convertTo(image_with_lidar_projection_, CV_8UC3);
            }
        }

        constexpr float Z_EPS = 1e-6f;
        for (const auto &p : cloud_roi->points)
        {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                continue;
            if (p.z <= Z_EPS)
                continue;

            const double u = fx * (p.x / p.z) + cx;
            const double v = fy * (p.y / p.z) + cy;
            if (u < 0 || v < 0 || u >= W || v >= H)
                continue;

            cv::circle(image_with_lidar_projection_, cv::Point((int)u, (int)v),
                       2, cv::Scalar(0, 255, 0), -1);
        }

        /*
                transformed_cloud->points.reserve(current_cloud_->points.size());

                for (const auto &pt_lidar : current_cloud_->points)
                {
                    cv::Mat pt_mat = (cv::Mat_<double>(3, 1) << pt_lidar.x, pt_lidar.y, pt_lidar.z);
                    cv::Mat pt_transformed = lidar2cam_R_ * pt_mat + lidar2cam_t_;

                    pcl::PointXYZI p_transformed;
                    p_transformed.x = pt_transformed.at<double>(0);
                    p_transformed.y = pt_transformed.at<double>(1);
                    p_transformed.z = pt_transformed.at<double>(2);
                    p_transformed.intensity = pt_lidar.intensity;
                    transformed_cloud->points.push_back(p_transformed);
                }

                projectLidarToImage(transformed_cloud, current_frame_, image_with_lidar_projection_);
        */
        pubTimerCallback();
    }

    void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pointcloud_msg)
    {

        RCLCPP_INFO(this->get_logger(), "---------------------------");
        RCLCPP_INFO(this->get_logger(), "sync cam and lidar");
        RCLCPP_INFO(this->get_logger(), "CAM %.3f", rclcpp::Time(image_msg->header.stamp).seconds());
        RCLCPP_INFO(this->get_logger(), "LID %.3f", rclcpp::Time(pointcloud_msg->header.stamp).seconds());
        RCLCPP_INFO(this->get_logger(), "---------------------------");
        pcl::fromROSMsg(*pointcloud_msg, *current_cloud_);
        current_frame_ = cv_bridge::toCvCopy(image_msg, "bgr8")->image;

        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        transformed_cloud->points.reserve(current_cloud_->points.size());

        for (const auto &pt_lidar : current_cloud_->points)
        {
            cv::Mat pt_mat = (cv::Mat_<double>(3, 1) << pt_lidar.x, pt_lidar.y, pt_lidar.z);
            cv::Mat pt_transformed = lidar2cam_R_ * pt_mat + lidar2cam_t_;

            pcl::PointXYZI p_transformed;
            p_transformed.x = pt_transformed.at<double>(0);
            p_transformed.y = pt_transformed.at<double>(1);
            p_transformed.z = pt_transformed.at<double>(2);
            p_transformed.intensity = pt_lidar.intensity;
            transformed_cloud->points.push_back(p_transformed);
        }

        projectLidarToImage(transformed_cloud, current_frame_, image_with_lidar_projection_);

        pubTimerCallback();
    }

    cv::Mat vector2Mat(const std::vector<std::vector<double>> &input)
    {
        std::cout << "convert vector<vector> to cv::Mat" << std::endl;
        std::size_t columns = input.front().size();
        std::size_t rows = input.size();

        cv::Mat output(rows, columns, CV_64F);
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < columns; j++)
            {
                output.at<double>(i, j) = input[i][j];
            }
        }
        return output;
    }

    cv::Mat vector2Mat(const std::vector<double> &input)
    {
        std::cout << "convert vector to cv::Mat" << std::endl;
        std::size_t nums = input.size();

        cv::Mat output(nums, 1, CV_64F);

        for (size_t j = 0; j < nums; j++)
        {
            output.at<double>(j, 0) = input[j];
        }

        return output;
    }

    void projectLidarToImage(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_in_cam,
        const cv::Mat &image_in,
        cv::Mat &image_out)
    {

        image_out = image_in.clone();

        const bool use_undistorte_image = false;
        if (use_undistorte_image == true)
        {
            intrinsic_matrix_ = cv::getOptimalNewCameraMatrix(intrinsic_matrix_, distortion_coeffs_,
                                                              image_out.size(), 0 /*alpha*/, image_out.size());
        }

        const cv::Mat &K = intrinsic_matrix_;
        double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);

        float min_horizontal_dist = std::numeric_limits<float>::max();
        float max_horizontal_dist = std::numeric_limits<float>::min();

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
            max_horizontal_dist = 120.0f;
        }

        for (const auto &pt : cloud_in_cam->points)
        {
            if (pt.z <= 0.0 || pt.z > 12.0f)
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

                float h = t * 240.0f;
                float s = 1.0f, v_val = 1.0f;
                float c = v_val * s;
                float x_hsv = c * (1 - std::fabs(fmod(h / 60.0f, 2) - 1));

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

                cv::Scalar color = cv::Scalar(B, G, R);
                cv::circle(image_out, cv::Point(u, v), 5, color, -1);
            }
        }
    }

    void pubTimerCallback()
    {
        if (image_with_lidar_projection_.empty())
        {
            return;
        }
        // cv::Mat -> sensor_msgs::msg::Image
        cv_bridge::CvImage cv_out;
        cv_out.header.stamp = this->now(); // ★ 가능하면 원본 헤더 사용
        cv_out.header.frame_id = "camera";
        // 채널에 맞게 encoding을 지정하세요: BGR이면 "bgr8", Gray면 "mono8"
        cv_out.encoding = (image_with_lidar_projection_.channels() == 1) ? "mono8" : "bgr8";
        cv_out.image = image_with_lidar_projection_;

        fusion_pub_.publish(cv_out.toImageMsg()); // => /calib/fusion (raw)
                                                  // => /calib/fusion/compressed (자동)
    }

    void runprojection()
    {

        // 여기에 이미지, pcd 경로 작성
        std::string pcd_path = "test";
        std::string image_path = "test";
        pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *current_cloud_);
        current_frame_ = cv::imread(image_path, cv::IMREAD_COLOR);

        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        transformed_cloud->points.reserve(current_cloud_->points.size());

        for (const auto &pt_lidar : current_cloud_->points)
        {
            cv::Mat pt_mat = (cv::Mat_<double>(3, 1) << pt_lidar.x, pt_lidar.y, pt_lidar.z);
            cv::Mat pt_transformed = lidar2cam_R_ * pt_mat + lidar2cam_t_;

            pcl::PointXYZI p_transformed;
            p_transformed.x = pt_transformed.at<double>(0);
            p_transformed.y = pt_transformed.at<double>(1);
            p_transformed.z = pt_transformed.at<double>(2);
            p_transformed.intensity = pt_lidar.intensity;
            transformed_cloud->points.push_back(p_transformed);
        }

        projectLidarToImage(transformed_cloud, current_frame_, image_with_lidar_projection_);

        // 이미지 출력 코드 추가 필요
        // image_with_lidar_projection_ 해당 cv::Mat 사용
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    std::vector<std::vector<double>> intrinsic = {{2.3804861502329672e+03, 0., 1.0367861314361255e+03},
                                                  {0.0, 2.3812829051328949e+03, 7.6724834534040474e+02},
                                                  {0.0, 0.0, 1.0}};

    std::vector<double> distortion = {-9.5762145770576887e-02, 9.0701282160675162e-02, -1.6180890835518554e-03, -9.7839810350063222e-04, -4.2237016474371038e-02};

    std::array<std::array<double, 3>, 3> rotation = {{{0.007025, -0.999968, 0.003872},
                                                      {0.011135, -0.003793, -0.999931},
                                                      {0.999913, 0.007067, 0.011108}}};

    std::array<double, 3> translation = {-0.008050, -0.065787, -0.068252};

    std::vector<std::vector<double>> extrinsic = {{rotation[0][0], rotation[0][1], rotation[0][2], translation[0]},
                                                  {rotation[1][0], rotation[1][1], rotation[1][2], translation[1]},
                                                  {rotation[2][0], rotation[2][1], rotation[2][2], translation[2]},
                                                  {0.0, 0.0, 0.0, 1.0}};

    std::cout << "(cam)T(lid) : " << std::endl;
    for (size_t i = 0; i < extrinsic.size(); i++)
    {
        for (size_t j = 0; j < extrinsic.size(); j++)
        {
            std::cout << extrinsic[i][j] << " ";
        }
        std::cout << "\n";
    }

    auto node = std::make_shared<CamLidarFusionNode>(intrinsic, distortion, extrinsic);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}