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
        RCLCPP_INFO(this->get_logger(), "-----------\nStart multi lidar calib node\n----------");
        filePath();
        keyboard_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&MultiLidarCalibNode::keyboardCallback, this));
    }

private:
    std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subscription_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds_;
    rclcpp::TimerBase::SharedPtr keyboard_timer_;
    
    int number_lidars_ = 0;
    int frame_counter_ = 0;
    std::string origin_path_;

    void keyboardCallback()
    {
        if (keyboardAvailable())
        {
            std::string input;
            std::getline(std::cin, input);

            if (!input.empty() && std::all_of(input.begin(), input.end(), ::isdigit))
            {
                number_lidars_ = std::stoi(input);
                clouds_.resize(number_lidars_);
                RCLCPP_INFO(this->get_logger(), "Start subscribe %d lidars", number_lidars_);

                for (int i = 0; i < number_lidars_; i++)
                {
                    std::string topic_name = "/lidar" + std::to_string(i) + "/points";
                    clouds_[i] = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                    auto sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_name, rclcpp::SensorDataQoS(),
                                                                                        [this, i, topic_name](const sensor_msgs::msg::PointCloud2::SharedPtr msg)
                                                                                        { pcdCallback(msg, i); });
                    subscription_.push_back(sub);
                }
            }
            else if (input == "s")
            {
                savePointCloud(number_lidars_, frame_counter_);
                frame_counter_++;
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
        RCLCPP_INFO(this->get_logger(), "Home directory : %s", home_dir.c_str());
        std::string data_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/multi_lidar_calib";
        origin_path_ = data_dir + "/origin_pointclouds/";
        fs::create_directories(origin_path_);
    }

    void pcdCallback(const sensor_msgs::msg::PointCloud2::SharedPtr message, int index)
    {
        pcl::fromROSMsg(*message, *clouds_[index]);
    }

    void savePointCloud(int num, int frame_counter)
    {
        RCLCPP_INFO(this->get_logger(), "Save current Point Cloud");
        if (clouds_.empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("savePointCloud"), "Lidar is Not Working!!!");
        }
        for (int i = 0; i < num; i++)
        {
            std::string filename = origin_path_ + "lidar" + std::to_string(i) + "_" + std::to_string(frame_counter) + ".pcd";
            pcl::io::savePCDFile(filename, *clouds_[i]);
        }
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
