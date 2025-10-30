#include "rclcpp/rclcpp.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include <Eigen/Dense>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "jmh_utils/jmh_utils.hpp"

using namespace std::chrono_literals;
namespace fs = std::filesystem;
using std::placeholders::_1;
using std::placeholders::_2;

class SyncNode : public rclcpp::Node
{
public:
    SyncNode() : Node("z_time_synchronizer")
    {
        auto qos_r = rclcpp::QoS(rclcpp::SensorDataQoS()).reliable();
        auto qos_b = rclcpp::QoS(rclcpp::SensorDataQoS()).best_effort();

        std::string what = "hard";
        if (what == "hard")
        {
        imu_sub__ = this -> create_subscription<sensor_msgs::msg::Imu>("/imu/data", 
            rclcpp::SensorDataQoS(), 
            std::bind(&SyncNode::imuCallback, this, _1));
        lidar_sub__ = this -> create_subscription<sensor_msgs::msg::PointCloud2>("/ouster/points",
             rclcpp::SensorDataQoS(), 
             std::bind(&SyncNode::lidCallback, this, _1));
        }
        else if (what == "filter")
        {
            imu_sub_.subscribe(this, "/imu/data", qos_r.get_rmw_qos_profile());
            lidar_sub_.subscribe(this, "/ouster/points", qos_b.get_rmw_qos_profile());

            imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("sync_imu", 10);
            lidar_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("sync_lidar", 10);

            uint32_t queue_size = 30;
            sync_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::
                                                                       ApproximateTime<sensor_msgs::msg::Imu, sensor_msgs::msg::PointCloud2>>>(
                message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Imu,
                                                                sensor_msgs::msg::PointCloud2>(queue_size),
                imu_sub_, lidar_sub_);
            // sync_ -> setAgePenalty(0.1);
            // sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.05));
            sync_->registerCallback(std::bind(&SyncNode::syncCallback, this, _1, _2));
        }
    }

private:
    void lidCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lid)
    {
        bool save = true;
        imuRotation(imu_, save);
        lidarRotation(lid, save);
    }

    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &imu)
    {
        imu_ = imu;
    }
    void syncCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &imu,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lidar)
    {

        RCLCPP_INFO(this->get_logger(), "Synced messages with timestamps: IMU=%.6f, LiDAR=%.6f",
                    rclcpp::Time(imu->header.stamp).seconds(),
                    rclcpp::Time(lidar->header.stamp).seconds());

        imu_pub_->publish(*imu);
        lidar_pub_->publish(*lidar);
        bool save = true;
        imuRotation(imu, save);
        lidarRotation(lidar, save);
    }

    void imuRotation(const sensor_msgs::msg::Imu::ConstSharedPtr &imu, const bool &data_save = false)
    {
        Eigen::Quaterniond rot_quarter(imu->orientation.x, imu->orientation.y, imu->orientation.z, imu->orientation.w);
        static bool step = false;

        if (!prev_imu_)
        {
            prev_imu_ = std::make_shared<const geometry_msgs::msg::Quaternion>(imu->orientation);
            prev_rot_ = rot_quarter;
            return;
        }

        Eigen::Quaterniond curr_rot = prev_rot_.conjugate() * rot_quarter;
        Eigen::Matrix3d rot_mat = curr_rot.toRotationMatrix();
        double y_rad = std::asin(-rot_mat(2, 0));
        double z_rad = std::atan2(rot_mat(1, 0), rot_mat(0, 0));
        double x_rad = std::atan2(rot_mat(2, 1), rot_mat(2, 2));

        Eigen::Vector3f xyz_rad(x_rad, y_rad, z_rad);

        if (data_save == true)
        {
            std::string home_dir = std::getenv("HOME");
            std::string file_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_test/data";
            fs::path file_path = file_dir + "/imu_rotation.csv";
            if (fs::exists(file_dir) == false)
            {
                fs::create_directories(file_dir);
            }

            if (step == false)
            {
                std::ofstream init_output(file_path, std::ios::out | std::ios::trunc);
                init_output.close();
                step = true;
            }

            std::ofstream output(file_path, std::ios::out | std::ios::app);
            output << std::fixed << std::setprecision(6) << xyz_rad[0] << "," << xyz_rad[1] << "," << xyz_rad[2] << std::endl;
        }
        prev_rot_ = rot_quarter;
    }

    void lidarRotation(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lidar, const bool &data_save = false)
    {
        auto cloud_raw = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(*lidar, *cloud_raw);
        static bool step = false;

        if (!prev_cloud_)
        {
            prev_cloud_ = cloud_raw;
            return;
        }

        Eigen::VectorXf ndt_param;
        ndt_param.resize(4);

        ndt_param[0] = 1.0;
        ndt_param[1] = 0.1;
        ndt_param[2] = 0.01;
        ndt_param[3] = static_cast<float>(35);
        std::string res_type = "radian";

        auto lid_xyz = jmh_utils::ndtRotation(cloud_raw, prev_cloud_, 0.6, ndt_param, res_type);

        if (data_save == true)
        {
            std::string home_dir = std::getenv("HOME");
            std::string file_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_test/data";
            fs::path file_path = file_dir + "/lidar_rotation.csv";
            if (fs::exists(file_dir) == false)
            {
                fs::create_directories(file_dir);
            }

            if (step == false)
            {
                std::ofstream init_output(file_path, std::ios::out | std::ios::trunc);
                init_output.close();
                step = true;
            }

            std::ofstream output(file_path, std::ios::out | std::ios::app);
            output << std::fixed << std::setprecision(6) << lid_xyz[0] << "," << lid_xyz[1] << "," << lid_xyz[2] << std::endl;
        }
        prev_cloud_ = cloud_raw;
    }

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub__;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub__;

    message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Imu, sensor_msgs::msg::PointCloud2>>>
        sync_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_pub_;
    Eigen::Quaterniond prev_rot_;
    geometry_msgs::msg::Quaternion::ConstSharedPtr prev_imu_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud_;
    sensor_msgs::msg::Imu::ConstSharedPtr imu_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SyncNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
