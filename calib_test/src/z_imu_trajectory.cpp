#include <chrono>
#include <deque>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "std_srvs/srv/empty.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Vector3.h"

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"

using std::placeholders::_1;

class ImuTrajectoryNode : public rclcpp::Node
{
public:
    ImuTrajectoryNode()
        : Node("z_imu_trajectory"),
          imu_topic_("/sync_imu"),
          fixed_frame_("map"),
          child_frame_("imu_link"),
          remove_gravity_(true),
          g_(9.80665),
          zupt_enable_(true),
          zupt_acc_thresh_(0.35),
          zupt_gyro_thresh_(0.05),
          bias_init_duration_(3.0),
          prev_t_(-1.0), initialized_(false),
          bias_initialized_(false)
    {
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/imu_path", 10);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom_imu", 10);
        path_msg_.header.frame_id = fixed_frame_;
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        reset_srv_ = this->create_service<std_srvs::srv::Empty>(
            "reset_imu_integration",
            std::bind(&ImuTrajectoryNode::handleReset, this, std::placeholders::_1, std::placeholders::_2));

        rclcpp::QoS qos(rclcpp::KeepLast(200));
        qos.best_effort();
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos, std::bind(&ImuTrajectoryNode::imuCallback, this, _1));

        RCLCPP_INFO(get_logger(), "IMU trajectory node started.");
    }

private:
    static tf2::Quaternion toQuat(const geometry_msgs::msg::Quaternion &q)
    {
        return {q.x, q.y, q.z, q.w};
    }
    static geometry_msgs::msg::Quaternion toMsg(const tf2::Quaternion &q)
    {
        geometry_msgs::msg::Quaternion m;
        m.x = q.x();
        m.y = q.y();
        m.z = q.z();
        m.w = q.w();
        return m;
    }
    static tf2::Vector3 rotate(const tf2::Quaternion &q, const tf2::Vector3 &v)
    {
        tf2::Matrix3x3 R(q);
        return R * v;
    }
    static double norm3(const tf2::Vector3 &v)
    {
        return std::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
    }

    void handleReset(const std::shared_ptr<std_srvs::srv::Empty::Request>,
                     std::shared_ptr<std_srvs::srv::Empty::Response>)
    {
        p_ = tf2::Vector3(0, 0, 0);
        v_ = tf2::Vector3(0, 0, 0);
        prev_t_ = -1.0;
        path_msg_ = nav_msgs::msg::Path();
        path_msg_.header.frame_id = fixed_frame_;
        initialized_ = false;
        bias_initialized_ = false;
        acc_bias_ = tf2::Vector3(0, 0, 0);
        gyro_bias_ = tf2::Vector3(0, 0, 0);
        bias_window_.clear();
        RCLCPP_INFO(get_logger(), "IMU integration reset.");
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        const double t = msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec;

        if (prev_t_ < 0.0)
        {
            prev_t_ = t;
            last_q_ = toQuat(msg->orientation);
            initialized_ = true;
            start_time_ = t;
            publishOutputs(msg->header.stamp);
            return;
        }

        double dt = t - prev_t_;
        if (dt <= 0.0 || dt > 0.2)
        {
            prev_t_ = t;
            return;
        }
        prev_t_ = t;

        tf2::Quaternion q_meas = toQuat(msg->orientation);
        tf2::Vector3 acc_body(msg->linear_acceleration.x,
                              msg->linear_acceleration.y,
                              msg->linear_acceleration.z);
        tf2::Vector3 gyro_body(msg->angular_velocity.x,
                               msg->angular_velocity.y,
                               msg->angular_velocity.z);

        last_q_ = q_meas;

        // 초기 바이어스 추정
        if (!bias_initialized_)
        {
            bias_window_.push_back({t, acc_body, gyro_body});
            if ((t - start_time_) >= bias_init_duration_)
            {
                tf2::Vector3 acc_sum(0, 0, 0), gyro_sum(0, 0, 0);
                for (auto &s : bias_window_)
                {
                    acc_sum += s.acc;
                    gyro_sum += s.gyr;
                }
                const double n = static_cast<double>(bias_window_.size());
                tf2::Vector3 acc_mean = (n > 0) ? (acc_sum * (1.0 / n)) : tf2::Vector3(0, 0, 0);
                tf2::Vector3 gyro_mean = (n > 0) ? (gyro_sum * (1.0 / n)) : tf2::Vector3(0, 0, 0);

                acc_bias_ = acc_mean - tf2::Vector3(0, 0, g_);
                gyro_bias_ = gyro_mean;

                bias_initialized_ = true;
                RCLCPP_INFO(get_logger(), "Bias initialized.");
                bias_window_.clear();
            }
        }

        tf2::Vector3 acc_body_corr = acc_body - acc_bias_;
        tf2::Vector3 gyro_body_corr = gyro_body - gyro_bias_;
        tf2::Vector3 acc_world = rotate(q_meas, acc_body_corr);

        if (remove_gravity_)
        {
            acc_world -= tf2::Vector3(0, 0, g_);
        }

        if (zupt_enable_)
        {
            const double acc_norm = std::fabs(norm3(acc_world));
            const double gyro_norm = std::fabs(norm3(gyro_body_corr));
            if (acc_norm < zupt_acc_thresh_ && gyro_norm < zupt_gyro_thresh_)
            {
                v_.setX(0.0);
                v_.setY(0.0);
                v_.setZ(0.0);
            }
        }

        v_ += acc_world * dt;
        p_ += v_ * dt;

        publishOutputs(msg->header.stamp);
    }

    void publishOutputs(const rclcpp::Time &stamp)
    {
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = stamp;
        ps.header.frame_id = fixed_frame_;
        ps.pose.position.x = p_.x();
        ps.pose.position.y = p_.y();
        ps.pose.position.z = p_.z();
        ps.pose.orientation = toMsg(last_q_);

        path_msg_.header.stamp = stamp;
        path_msg_.poses.push_back(ps);
        if (path_msg_.poses.size() > 5000)
        {
            path_msg_.poses.erase(path_msg_.poses.begin(),
                                  path_msg_.poses.begin() + (path_msg_.poses.size() - 5000));
        }
        path_pub_->publish(path_msg_);

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = fixed_frame_;
        odom.child_frame_id = child_frame_;
        odom.pose.pose = ps.pose;
        odom.twist.twist.linear.x = v_.x();
        odom.twist.twist.linear.y = v_.y();
        odom.twist.twist.linear.z = v_.z();
        odom_pub_->publish(odom);

        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = stamp;
        tf.header.frame_id = fixed_frame_; // "map" 또는 "odom"
        tf.child_frame_id = child_frame_;  // "imu_link"
        tf.transform.translation.x = p_.x();
        tf.transform.translation.y = p_.y();
        tf.transform.translation.z = p_.z();
        tf.transform.rotation = toMsg(last_q_);
        tf_broadcaster_->sendTransform(tf);
    }

    // --- Members ---
    std::string imu_topic_, fixed_frame_, child_frame_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;

    nav_msgs::msg::Path path_msg_;

    tf2::Vector3 p_{0, 0, 0};
    tf2::Vector3 v_{0, 0, 0};
    tf2::Quaternion last_q_{0, 0, 0, 1};

    double prev_t_;
    bool initialized_;

    bool remove_gravity_;
    double g_;
    bool zupt_enable_;
    double zupt_acc_thresh_;
    double zupt_gyro_thresh_;

    double bias_init_duration_;
    bool bias_initialized_;
    double start_time_;
    tf2::Vector3 acc_bias_{0, 0, 0};
    tf2::Vector3 gyro_bias_{0, 0, 0};

    struct BiasSample
    {
        double t;
        tf2::Vector3 acc;
        tf2::Vector3 gyr;
    };
    std::deque<BiasSample> bias_window_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuTrajectoryNode>());
    rclcpp::shutdown();
    return 0;
}
