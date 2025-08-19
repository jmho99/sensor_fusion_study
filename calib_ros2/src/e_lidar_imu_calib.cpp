#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_storage/storage_options.hpp>

// NDT를 위한 포인트 타입 정의
using PointType = pcl::PointXYZ;

// Levenberg-Marquardt 최적화를 위한 3D 회전 지수 맵(exponential map) 함수
Eigen::Matrix3d exp_map(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d u = omega / theta;
    Eigen::Matrix3d u_skew;
    u_skew << 0, -u(2), u(1),
              u(2), 0, -u(0),
              -u(1), u(0), 0;
    return Eigen::Matrix3d::Identity() + sin(theta) * u_skew + (1.0 - cos(theta)) * u_skew * u_skew;
}

class BagFileCalibratorLM : public rclcpp::Node {
public:
    BagFileCalibratorLM() : Node("e_lidar_imu_calib") {
        this->declare_parameter<std::string>("bag_file", "path/to/your/bagfile.bag");

        ndt_.setTransformationEpsilon(0.01);
        ndt_.setResolution(1.0);
        ndt_.setMaximumIterations(30);

        R_IL_ = Eigen::Quaterniond::Identity();
        prev_lidar_cloud_ = nullptr;
    }

    void process_bag_file() {
        std::string bag_file_path = this->get_parameter("bag_file").as_string();
        RCLCPP_INFO_STREAM(this->get_logger(), "Processing bag file: " << bag_file_path);

        rosbag2_cpp::Reader reader;
        try {
            reader.open(bag_file_path);
        } catch (const std::exception& e) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "Failed to open bag file: " << e.what());
            return;
        }

        const auto topics = reader.get_all_topics_and_types();
        std::string imu_topic_name = "";
        std::string lidar_topic_name = "";

        for (const auto& topic : topics) {
            RCLCPP_INFO_STREAM(this->get_logger(), "Found topic: " << topic.name << " with type: " << topic.type);
            if (topic.type == "sensor_msgs/msg/Imu") {
                imu_topic_name = topic.name;
            } else if (topic.type == "sensor_msgs/msg/PointCloud2") {
                lidar_topic_name = topic.name;
            }
        }
        
        if (imu_topic_name.empty() || lidar_topic_name.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Required topics (sensor_msgs/msg/Imu and sensor_msgs/msg/PointCloud2) not found in bag file.");
            return;
        }

        while (reader.has_next()) {
            auto bag_message = reader.read_next();
            
            if (bag_message->topic_name == lidar_topic_name) {
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                auto lidar_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
                rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serializer;
                serializer.deserialize_message(&serialized_msg, lidar_msg.get());
                lidar_msgs_.push_back(lidar_msg);
            }
            else if (bag_message->topic_name == imu_topic_name) {
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
                rclcpp::Serialization<sensor_msgs::msg::Imu> serializer;
                serializer.deserialize_message(&serialized_msg, imu_msg.get());
                imu_msgs_.push_back(imu_msg);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Finished reading bag file. Total IMU messages: %zu, Total Lidar messages: %zu", imu_msgs_.size(), lidar_msgs_.size());
        
        // 전체 데이터를 먼저 불러온 후, 시간 순서대로 데이터를 처리합니다.
        process_frames_sequentially();

        rclcpp::shutdown();
    }

private:
    void process_frames_sequentially() {
        RCLCPP_INFO(this->get_logger(), "Processing frames sequentially...");
        
        // Lidar와 IMU 메시지를 순서대로 처리하는 로직을 추가합니다.
        // 현재는 시간 동기화 로직이 없으므로, 메시지 인덱스를 기준으로 처리합니다.
        size_t min_data_size = std::min(imu_msgs_.size(), lidar_msgs_.size());

        for(size_t i = 0; i < min_data_size; ++i) {
            process_lidar_message(lidar_msgs_[i]);
            process_imu_message(imu_msgs_[i]);
        }

        if (min_data_size > 10) { // 최소 10개의 데이터 쌍이 있을 때만 최적화 시도
            optimize_initial_rotation(min_data_size);
        } else {
            RCLCPP_WARN(this->get_logger(), "Not enough synchronized data for optimization. Need at least 10 pairs.");
        }
    }

    void process_lidar_message(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
        pcl::PointCloud<PointType>::Ptr current_cloud_raw(new pcl::PointCloud<PointType>);
        pcl::fromROSMsg(*msg, *current_cloud_raw);

        pcl::PointCloud<PointType>::Ptr current_cloud(new pcl::PointCloud<PointType>);
        pcl::PassThrough<PointType> pass;
        pass.setInputCloud(current_cloud_raw);
        pass.filter(*current_cloud);

        if (!prev_lidar_cloud_) {
            if (current_cloud->points.size() < 100) {
                 RCLCPP_WARN(this->get_logger(), "Initial cloud has too few points. Skipping this frame.");
                 prev_lidar_cloud_ = nullptr;
            } else {
                 prev_lidar_cloud_ = current_cloud;
                 RCLCPP_INFO(this->get_logger(), "Initial lidar cloud saved. Waiting for next frame.");
            }
            return;
        }

        if (current_cloud->points.size() < 100 || prev_lidar_cloud_->points.size() < 100) {
            RCLCPP_WARN(this->get_logger(), "Not enough points in clouds for NDT. Skipping.");
            prev_lidar_cloud_ = current_cloud;
            return;
        }

        ndt_.setInputSource(current_cloud);
        ndt_.setInputTarget(prev_lidar_cloud_);

        pcl::PointCloud<PointType> output_cloud;
        ndt_.align(output_cloud);

        if (ndt_.hasConverged()) {
            Eigen::Matrix4f transform = ndt_.getFinalTransformation();
            Eigen::Matrix3d R_L_eigen = transform.block<3, 3>(0, 0).cast<double>();
            Eigen::Quaterniond Q_L(R_L_eigen);
            
            lidar_rotations_.push_back(Q_L);
            RCLCPP_INFO(this->get_logger(), "NDT converged. Lidar rotation added.");
        } else {
            RCLCPP_WARN(this->get_logger(), "NDT did not converge for this frame pair.");
        }
        
        prev_lidar_cloud_ = current_cloud;
    }

    void process_imu_message(const sensor_msgs::msg::Imu::SharedPtr& msg) {
        Eigen::Vector3d omega_I(msg->angular_velocity.x,
                                msg->angular_velocity.y,
                                msg->angular_velocity.z);
        imu_omegas_.push_back(omega_I);
        RCLCPP_INFO(this->get_logger(), "IMU omega added.");
    }

    void optimize_initial_rotation(size_t data_size) {
        RCLCPP_INFO(this->get_logger(), "Starting Levenberg-Marquardt optimization with %zu data pairs.", data_size);
        
        double lambda = 1e-3;
        const int max_iterations = 100;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            Eigen::VectorXd residuals(data_size * 3);
            Eigen::MatrixXd jacobian(data_size * 3, 3);
            
            for (size_t i = 0; i < data_size; ++i) {
                double dt = 1.0; 
                Eigen::Vector3d omega_I_dt = imu_omegas_[i] * dt;
                Eigen::Matrix3d R_I = exp_map(omega_I_dt);
                
                Eigen::Quaterniond Q_L_i = lidar_rotations_[i];
                Eigen::Matrix3d R_L_i = Q_L_i.toRotationMatrix();

                Eigen::Quaterniond Q_I(R_I);
                Eigen::Quaterniond error_quat = Q_L_i * R_IL_ * (R_IL_ * Q_I).inverse();
                Eigen::Vector3d error_vec = 2.0 * error_quat.vec(); 
                residuals.segment<3>(i * 3) = error_vec;

                Eigen::Matrix3d R_IL_mat = R_IL_.toRotationMatrix();
                Eigen::Matrix3d jacobian_block = -R_L_i * R_IL_mat.transpose() + R_IL_mat * R_I.transpose();
                jacobian.block<3, 3>(i * 3, 0) = jacobian_block;
            }
            
            Eigen::MatrixXd hessian = jacobian.transpose() * jacobian;
            Eigen::VectorXd gradient = -jacobian.transpose() * residuals;
            
            Eigen::MatrixXd hessian_lm = hessian + lambda * Eigen::MatrixXd::Identity(3, 3);
            Eigen::Vector3d delta_theta = hessian_lm.ldlt().solve(gradient);
            
            Eigen::Quaterniond update_quat(1.0, 0.5 * delta_theta.x(), 0.5 * delta_theta.y(), 0.5 * delta_theta.z());
            update_quat.normalize();
            Eigen::Quaterniond new_R_IL = update_quat * R_IL_;
            new_R_IL.normalize();
            
            R_IL_ = new_R_IL;
            
            RCLCPP_INFO(this->get_logger(), "Iteration %d: Residual Norm = %f", iter, residuals.norm());
            
            if (residuals.norm() < 1e-6) {
                break;
            }
        }
        
        RCLCPP_INFO_STREAM(this->get_logger(), "Optimization complete. Final R_IL (Quaternion): " << R_IL_.coeffs().transpose());
        Eigen::Vector3d rpy = R_IL_.toRotationMatrix().eulerAngles(2, 1, 0); 
        RCLCPP_INFO_STREAM(this->get_logger(), "Final R_IL (Roll, Pitch, Yaw): " << rpy.transpose() * 180.0 / M_PI << " degrees");
    }

    pcl::NormalDistributionsTransform<PointType, PointType> ndt_;
    pcl::PointCloud<PointType>::Ptr prev_lidar_cloud_;

    // 전체 데이터를 저장하기 위한 벡터
    std::vector<sensor_msgs::msg::PointCloud2::SharedPtr> lidar_msgs_;
    std::vector<sensor_msgs::msg::Imu::SharedPtr> imu_msgs_;

    // 최적화를 위한 데이터 벡터
    std::vector<Eigen::Vector3d> imu_omegas_;
    std::vector<Eigen::Quaterniond> lidar_rotations_;

    Eigen::Quaterniond R_IL_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BagFileCalibratorLM>();
    node->process_bag_file();
    return 0;
}
