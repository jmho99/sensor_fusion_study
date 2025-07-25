#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <string>
#include <sys/select.h>
#include <unistd.h>
#include <filesystem>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace fs = std::filesystem;

class MultiLidarCalibNode : public rclcpp::Node
{
public:
    MultiLidarCalibNode()
        : Node("multi_lidar_calib")
    {
        RCLCPP_INFO(this->get_logger(), "----------------------------");
        RCLCPP_INFO(this->get_logger(), "Start multi lidar calib node");
        RCLCPP_INFO(this->get_logger(), "----------------------------");
        filePath();
        left_lidar_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/left/points", rclcpp::SensorDataQoS(),
                                                                                            std::bind(&MultiLidarCalibNode::leftPcdCallback,
                                                                                                      this, std::placeholders::_1));

        right_lidar_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/right/points", rclcpp::SensorDataQoS(),
                                                                                             std::bind(&MultiLidarCalibNode::rightPcdCallback,
                                                                                                       this, std::placeholders::_1));
        keyboard_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&MultiLidarCalibNode::keyboardCallback, this));
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr left_lidar_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr right_lidar_subscription_;
    rclcpp::TimerBase::SharedPtr keyboard_timer_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr left_cloud_{new pcl::PointCloud<pcl::PointXYZRGB>};
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr right_cloud_{new pcl::PointCloud<pcl::PointXYZRGB>};
    std::string raw_left_path_, raw_right_path_;

    void leftPcdCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::fromROSMsg(*msg, *left_cloud_);
    }

    void rightPcdCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::fromROSMsg(*msg, *right_cloud_);
    }

    void keyboardCallback()
    {
        if (keyboardAvailable())
        {
            std::string input;
            std::getline(std::cin, input);

            if (input == "s")
            {
                savePointCloud();
            }
        }
    }

    bool keyboardAvailable()
    {
        struct timeval tv{0L, 0L};
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        return select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) > 0;
    }

    void filePath()
    {
        std::string home_dir = std::getenv("HOME");
        RCLCPP_INFO(this->get_logger(), "Home directory : %s", home_dir);
        std::string data_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data";

        raw_left_path_ = data_dir + "/raw_left_pcd/";
        raw_right_path_ = data_dir + "/raw_right_pcd/";

        fs::create_directories(raw_left_path_);
        fs::create_directories(raw_right_path_);
    }

    void savePointCloud()
    {
    }
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MultiLidarCalibNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
