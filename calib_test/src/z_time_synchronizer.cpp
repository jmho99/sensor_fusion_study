#include "rclcpp/rclcpp.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

using namespace std::chrono_literals;

using std::placeholders::_1;
using std::placeholders::_2;

class SyncNode : public rclcpp::Node 
{
public:
  SyncNode() : Node("z_time_synchronizer") 
  {
    auto qos_r = rclcpp::QoS(rclcpp::SensorDataQoS()).reliable();
    auto qos_b = rclcpp::QoS(rclcpp::SensorDataQoS()).best_effort();
    imu_sub_.subscribe(this, "/imu/data", qos_r.get_rmw_qos_profile());
    lidar_sub_.subscribe(this, "/ouster/points", qos_b.get_rmw_qos_profile());
    
    imu_pub_ = this -> create_publisher<sensor_msgs::msg::Imu>("sync_imu", 10);
    lidar_pub_ = this -> create_publisher<sensor_msgs::msg::PointCloud2>("sync_lidar", 10);

    uint32_t queue_size = 30;
    sync_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::
    		ApproximateTime<sensor_msgs::msg::Imu, sensor_msgs::msg::PointCloud2>>>(
    		message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Imu,
    		sensor_msgs::msg::PointCloud2>(queue_size), imu_sub_, lidar_sub_);
    //sync_ -> setAgePenalty(0.1);
    //sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.05)); 
    sync_->registerCallback(std::bind(&SyncNode::syncCallback, this, _1, _2));
  }

private:
  void syncCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &imu,
                const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lidar) 
 {

    RCLCPP_INFO(this->get_logger(), "Synced messages with timestamps: IMU=%.6f, LiDAR=%.6f",
            rclcpp::Time(imu->header.stamp).seconds(),
            rclcpp::Time(lidar->header.stamp).seconds());

   imu_pub_ -> publish(*imu);
   lidar_pub_ -> publish(*lidar);
  }

  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
  std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Imu, sensor_msgs::msg::PointCloud2>>> sync_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_pub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SyncNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
