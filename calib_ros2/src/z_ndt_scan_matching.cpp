// 기본 흐름 설명용 C++ 기반 ROS 2 NDT Mapping 노드 구조

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <filesystem>

class NDTMapper : public rclcpp::Node {
public:
    NDTMapper() : Node("ndt_mapper") {
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ouster/points", 10,
            std::bind(&NDTMapper::pointCloudCallback, this, std::placeholders::_1));

        map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ndt_map", 10);

        ndt_.setTransformationEpsilon(0.01);
        ndt_.setStepSize(0.1);
        ndt_.setResolution(1.0);
        ndt_.setMaximumIterations(35);

        map_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        is_initialized_ = false;
    }

    ~NDTMapper() {
        saveMapToPCD();
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(0.5f, 0.5f, 0.5f);
        voxel.filter(*filtered);

        if (!is_initialized_) {
            *map_ += *filtered;
            is_initialized_ = true;
            prev_cloud_ = filtered;
            return;
        }

        ndt_.setInputSource(filtered);
        ndt_.setInputTarget(map_);

        Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
        pcl::PointCloud<pcl::PointXYZ> aligned;
        ndt_.align(aligned, init_guess);

        Eigen::Matrix4f T = ndt_.getFinalTransformation();
        pcl::transformPointCloud(*filtered, aligned, T);
        *map_ += aligned;

        sensor_msgs::msg::PointCloud2 map_msg;
        pcl::toROSMsg(*map_, map_msg);
        map_msg.header = msg->header;
        map_pub_->publish(map_msg);

        prev_cloud_ = filtered;
    }

    void saveMapToPCD() {
        std::string filename = "ndt_map.pcd";
        if (map_ && !map_->empty()) {
            pcl::io::savePCDFile(filename, *map_);
            RCLCPP_INFO(this->get_logger(), "Saved map to PCD: %s", filename.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "Map is empty, not saving.");
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;

    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud_;
    bool is_initialized_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NDTMapper>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
