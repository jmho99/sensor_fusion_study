#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include <fstream>
#include <vector>

class SyncLogger : public rclcpp::Node
{
public:
  SyncLogger()
      : Node("z_time_synchronizer")
  {
    imu_sub_.subscribe(this, "/imu/data", rmw_qos_profile_sensor_data);
    lidar_sub_.subscribe(this, "/ouster/points", rmw_qos_profile_sensor_data);

    sync_ = std::make_shared<Sync>(
        SyncPolicy(10), imu_sub_, lidar_sub_);
    sync_->registerCallback(
        std::bind(&SyncLogger::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

    // 1분 타이머
    timer_ = this->create_wall_timer(
        std::chrono::seconds(60), std::bind(&SyncLogger::save_and_shutdown, this));

    RCLCPP_INFO(this->get_logger(), "SyncLogger started. Recording for 60 seconds...");
  }

private:
  void sync_callback(
      const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr lidar_msg)
  {
    imu_data_.push_back(*imu_msg);

    // LiDAR는 rosbag으로 수집하므로 따로 저장 X
    RCLCPP_INFO(this->get_logger(), "Synced: IMU %.3f | LiDAR %.3f",
                rclcpp::Time(imu_msg->header.stamp).seconds(),
                rclcpp::Time(lidar_msg->header.stamp).seconds());
  }

  void save_and_shutdown()
  {
    RCLCPP_INFO(this->get_logger(), "Saving IMU data...");
    std::ofstream imu_file("synced_imu_data.csv");
    imu_file << "timestamp,orientation_x,orientation_y,orientation_z,orientation_w,"
                "angular_velocity_x,angular_velocity_y,angular_velocity_z,"
                "linear_accel_x,linear_accel_y,linear_accel_z\n";

    for (const auto &imu : imu_data_)
    {
      double ts = rclcpp::Time(imu.header.stamp).seconds();
      imu_file << ts << ","
               << imu.orientation.x << "," << imu.orientation.y << "," << imu.orientation.z << "," << imu.orientation.w << ","
               << imu.angular_velocity.x << "," << imu.angular_velocity.y << "," << imu.angular_velocity.z << ","
               << imu.linear_acceleration.x << "," << imu.linear_acceleration.y << "," << imu.linear_acceleration.z << "\n";
    }

    imu_file.close();
    RCLCPP_INFO(this->get_logger(), "IMU data saved. Shutting down...");
    rclcpp::shutdown();
  }

  // message_filters
  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Imu, sensor_msgs::msg::PointCloud2>;
  using Sync = message_filters::Synchronizer<SyncPolicy>;
  std::shared_ptr<Sync> sync_;

  rclcpp::TimerBase::SharedPtr timer_;
  std::vector<sensor_msgs::msg::Imu> imu_data_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SyncLogger>());
  return 0;
}
