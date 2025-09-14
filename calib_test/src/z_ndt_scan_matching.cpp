#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h> // removeNaNFromPointCloud

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>

#include "jmh_utils/jmh_utils.hpp"

namespace fs = std::filesystem;
using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

class NdtRotationNode : public rclcpp::Node
{
public:
    NdtRotationNode()
        : Node("ndt_rotation_node"),
          cloud_topic_("/sync_lidar"),
          voxel_leaf_(0.4f),
          ndt_trans_eps_(0.01),
          ndt_step_size_(0.1),
          ndt_resolution_(1.0),
          ndt_max_iter_(35),
          publish_tf_(true),
          fixed_frame_("map"),
          child_frame_("lidar_curr")
    {

        // QoS: LiDAR는 보통 Best Effort가 무난
        rclcpp::QoS qos(rclcpp::KeepLast(10));
        qos.reliable();

        sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, qos,
            std::bind(&NdtRotationNode::onCloud, this, std::placeholders::_1));

        pub_quat_ = create_publisher<geometry_msgs::msg::QuaternionStamped>("/ndt_rotation/quat", 10);
        pub_ypr_deg_ = create_publisher<geometry_msgs::msg::Vector3Stamped>("/ndt_rotation/ypr_deg", 10);
        pub_prev_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ndt/prev", 10);
        pub_cur_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ndt/cur", 10);
        pub_align_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ndt/align", 10);

        if (publish_tf_)
            tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        pcd_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30), // 30ms 주기로 퍼블리시
            std::bind(&NdtRotationNode::pcdTimerCallback, this));

        // example_ndt();
        RCLCPP_INFO(get_logger(), "NDT rotation node started. topic=%s", cloud_topic_.c_str());
    }

private:
    void onCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::string home_dir = std::getenv("HOME");
        std::string file_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_test/data";
        fs::path file_path = "/rotation.yaml";
        fs::create_directories(file_path.parent_path());

        auto cloud_raw = std::make_shared<CloudT>();
        pcl::fromROSMsg(*msg, *cloud_raw);
        if (cloud_raw->empty())
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Empty cloud, skip.");
            return;
        }

        // 첫 프레임이면 저장만
        if (!prev_cloud_)
        {
            prev_cloud_ = cloud_raw;
            prev_stamp_ = msg->header.stamp;
            std::ofstream init_output(file_path, std::ios::out | std::ios::trunc);
            init_output.close();
            return;
        }

        Eigen::VectorXf ndt_param;
        ndt_param.resize(4);

        ndt_param[0] = ndt_resolution_;
        ndt_param[1] = ndt_step_size_;
        ndt_param[2] = ndt_trans_eps_;
        ndt_param[3] = static_cast<float>(ndt_max_iter_);
        std::string res_type = "degree";

        auto res_xyz = jmh_utils::ndtRotation(cloud_raw, prev_cloud_, voxel_leaf_, ndt_param, res_type);

        RCLCPP_INFO(this->get_logger(), "NDT func rotation(%s): [%.2f, %.2f, %.2f]",
                    res_type.c_str(), res_xyz[0], res_xyz[1], res_xyz[2]);

        std::ofstream output(file_path, std::ios::out | std::ios::app);
        output << std::fixed << std::setprecision(6) << res_xyz.transpose() << std::endl;

        if (cloud_raw->empty())
        {
            output.close();
            RCLCPP_INFO(this->get_logger(), "save rotation");
        }

        // 다음을 위해 현재를 prev로 보관
        prev_cloud_ = cloud_raw;
        prev_stamp_ = msg->header.stamp;
    }

    static CloudT::Ptr voxelDown(CloudT::Ptr in, float leaf)
    {
        if (leaf <= 0.f)
            return in;
        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud(in);
        vg.setLeafSize(leaf, leaf, leaf);
        auto out = std::make_shared<CloudT>();
        vg.filter(*out);
        return out;
    }

    static CloudT::Ptr centerToCentroid(CloudT::Ptr in, Eigen::Vector4f &centroid_out)
    {
        pcl::compute3DCentroid(*in, centroid_out);
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 1>(0, 3) = -centroid_out.head<3>();
        auto out = std::make_shared<CloudT>();
        pcl::transformPointCloud(*in, *out, T);
        return out;
    }

    static Eigen::Matrix3f orthonormalize(const Eigen::Matrix3f &R)
    {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V = svd.matrixV();
        Eigen::Matrix3f Rn = U * V.transpose();
        // 보정 후 det가 -1이면 마지막 축 반전
        if (Rn.determinant() < 0)
        {
            U.col(2) *= -1.0f;
            Rn = U * V.transpose();
        }
        return Rn;
    }

    Eigen::VectorXf solveRotation(CloudT::Ptr source, CloudT::Ptr target, std::string result = "d")
    {
        Eigen::VectorXf res_xyz;
        std::string rot_type;
        // NDT 세팅
        pcl::NormalDistributionsTransform<PointT, PointT> ndt;
        ndt.setResolution(ndt_resolution_);
        ndt.setStepSize(ndt_step_size_);
        ndt.setTransformationEpsilon(ndt_trans_eps_);
        ndt.setMaximumIterations(ndt_max_iter_);
        ndt.setInputSource(source); // cur
        ndt.setInputTarget(target); // prev

        Eigen::AngleAxisf init_rotation(0.0, Eigen::Vector3f::UnitZ());
        Eigen::Translation3f init_translation(0.0, 0.0, 0.0);
        Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

        CloudT aligned;
        ndt.align(aligned, init_guess);

        Eigen::Matrix4f T = ndt.getFinalTransformation();
        Eigen::Matrix3f R = T.block<3, 3>(0, 0);
        R = orthonormalize(R);

        if (result == "q")
        {
            rot_type = "quarternion";
            // 회전(Qurternion)
            Eigen::Quaternionf q(R);
            geometry_msgs::msg::QuaternionStamped qs;
            // qs.header = msg->header; // stamp, frame_id 그대로
            qs.quaternion.x = q.x();
            qs.quaternion.y = q.y();
            qs.quaternion.z = q.z();
            qs.quaternion.w = q.w();
            pub_quat_->publish(qs);

            res_xyz.resize(4);
            res_xyz[0] = q.x();
            res_xyz[1] = q.y();
            res_xyz[2] = q.z();
            res_xyz[3] = q.w();
        }

        else if (result == "r")
        {
            rot_type = "radian";
            // 회전(Radian)
            double y_rad = std::asin(-R(2, 0));
            double z_rad = std::atan2(R(1, 0), R(0, 0));
            double x_rad = std::atan2(R(2, 1), R(2, 2));

            Eigen::Vector3f xyz_rad;
            xyz_rad[0] = x_rad;
            xyz_rad[1] = y_rad;
            xyz_rad[2] = z_rad;

            geometry_msgs::msg::Vector3Stamped xyz_msg;
            // ypr_msg.header = msg->header;
            xyz_msg.vector.x = xyz_rad[2];
            xyz_msg.vector.y = xyz_rad[1];
            xyz_msg.vector.z = xyz_rad[0];
            pub_ypr_deg_->publish(xyz_msg);

            res_xyz.resize(3);
            res_xyz[0] = xyz_rad[2];
            res_xyz[1] = xyz_rad[1];
            res_xyz[2] = xyz_rad[0];
        }

        else if (result == "d")
        {
            rot_type = "degree";
            // 회전(Degree)
            double y_rad = std::asin(-R(2, 0));
            double z_rad = std::atan2(R(1, 0), R(0, 0));
            double x_rad = std::atan2(R(2, 1), R(2, 2));

            Eigen::Vector3f xyz_rad;
            xyz_rad[0] = x_rad;
            xyz_rad[1] = y_rad;
            xyz_rad[2] = z_rad;

            double PI = 3.14159265358979;
            Eigen::Vector3f xyz_deg = xyz_rad * (180.0 / PI);

            geometry_msgs::msg::Vector3Stamped xyz_msg;
            // ypr_msg.header = msg->header;
            xyz_msg.vector.x = xyz_deg[0];
            xyz_msg.vector.y = xyz_deg[1];
            xyz_msg.vector.z = xyz_deg[2];
            pub_ypr_deg_->publish(xyz_msg);

            res_xyz.resize(3);
            res_xyz[0] = xyz_deg[0];
            res_xyz[1] = xyz_deg[1];
            res_xyz[2] = xyz_deg[2];
        }

        switch (res_xyz.size())
        {
        case 3:
            RCLCPP_INFO(this->get_logger(), "NDT func rotation(%s): [%.2f, %.2f, %.2f]",
                        rot_type.c_str(), res_xyz[0], res_xyz[1], res_xyz[2]);
            break;
        case 4:
            RCLCPP_INFO(this->get_logger(), "NDT func rotation(%s): [%.2f, %.2f, %.2f, %.2f]",
                        rot_type.c_str(), res_xyz[0], res_xyz[1], res_xyz[2], res_xyz[3]);
            break;
        }

        return res_xyz;
    }

    void example_ndt()
    {
        std::string pcd_name1 = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/e_lidar_imu_calib/room_scan1.pcd";
        std::string pcd_name2 = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/e_lidar_imu_calib/room_scan2.pcd";
        auto cloud1 = std::make_shared<CloudT>();
        auto cloud2 = std::make_shared<CloudT>();
        pcl::io::loadPCDFile(pcd_name1, *cloud1);
        pcl::io::loadPCDFile(pcd_name2, *cloud2);

        RCLCPP_INFO(this->get_logger(), "Loaded lidar point clouds");

        auto cloud_ds1 = voxelDown(cloud1, voxel_leaf_);
        auto cloud_ds = voxelDown(cloud2, voxel_leaf_);

        pcl::toROSMsg(*cloud_ds1, msg_prev_);
        msg_prev_.header.frame_id = "map";
        pcl::toROSMsg(*cloud_ds, msg_cur_);
        msg_cur_.header.frame_id = "map";

        prev_cloud_ = cloud_ds1;

        // 1) 각 프레임을 센트로이드로 평행이동 → translation 영향 억제
        Eigen::Vector4f c_prev, c_cur;
        auto prev_cc = centerToCentroid(prev_cloud_, c_prev);
        auto cur_cc = centerToCentroid(cloud_ds, c_cur);
        auto prev_c = prev_cloud_;
        auto cur_c = cloud_ds;

        Eigen::Vector3f cprev = c_prev.head<3>(); // ✅ 3x1 벡터
        Eigen::Vector3f ccur = c_cur.head<3>();

        // 2) NDT 세팅 (cur as source, prev as target)
        pcl::NormalDistributionsTransform<PointT, PointT> ndt;
        ndt.setTransformationEpsilon(ndt_trans_eps_);
        ndt.setStepSize(ndt_step_size_);
        ndt.setResolution(ndt_resolution_);
        ndt.setMaximumIterations(ndt_max_iter_);
        ndt.setInputSource(cur_c);
        ndt.setInputTarget(prev_c);

        Eigen::AngleAxisf init_rotation(0.6931, Eigen::Vector3f::UnitZ());
        Eigen::Translation3f init_translation(1.79387, 0.720047, 0);
        Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

        Eigen::Matrix4f init = Eigen::Matrix4f::Identity();

        CloudT aligned;
        ndt.align(aligned, init_guess);

        /*
                if (!ndt.hasConverged())
                {
                    RCLCPP_WARN(get_logger(), "NDT did not converge.");
                    // 그래도 다음 프레임 대비 업데이트
                    prev_cloud_ = cloud_ds;
                    prev_stamp_ = msg->header.stamp;
                    return;
                }
        */
        Eigen::Matrix4f T = ndt.getFinalTransformation();
        Eigen::Matrix3f R = T.block<3, 3>(0, 0);

        Eigen::Vector3f t = cprev - R * ccur;

        Eigen::Matrix4f T_full = Eigen::Matrix4f::Identity();
        T_full.block<3, 3>(0, 0) = R;
        T_full.block<3, 1>(0, 3) = t;

        CloudT align;
        pcl::transformPointCloud(*cloud2, align, T_full);
        pcl::toROSMsg(align, msg_align_);
        msg_align_.header.frame_id = "map";
        R = orthonormalize(R);
        std::cout << "transformation : " << std::endl;
        std::cout << T << std::endl;

        std::cout << "orthono rotation : " << std::endl;
        std::cout << R << std::endl;

        double y_rad = std::asin(-R(2, 0));
        double z_rad = std::atan2(R(1, 0), R(0, 0));
        double x_rad = std::atan2(R(2, 1), R(2, 2));

        Eigen::Vector3f xyz_rad;
        xyz_rad[0] = x_rad;
        xyz_rad[1] = y_rad;
        xyz_rad[2] = z_rad;
        // xyz_rad.(x_rad);
        // xyz_rad.push_back(y_rad);
        // xyz_rad.push_back(z_rad);

        std::cout << "xyz_rad : " << std::endl;
        std::cout << xyz_rad.transpose() << std::endl;

        double PI = 3.14159265358979;
        Eigen::Vector3f xyz_deg = xyz_rad * (180.0 / PI);
        std::cout << "xyz_deg : " << std::endl;
        std::cout << xyz_deg.transpose() << std::endl;

        // 회전 → 쿼터니언/롤피치야우
        Eigen::Quaternionf q(R);
        Eigen::Vector3f ypr = R.eulerAngles(2, 1, 0); // [yaw, pitch, roll] in rad //output: z,y,x

        Eigen::Vector3f ypr_deg = ypr * (180.0f / static_cast<float>(M_PI));
        std::cout << "ypr : " << std::endl;
        std::cout << ypr.transpose() << std::endl;

        rclcpp::Time stamp = this->get_clock()->now();
        // 퍼블리시: Quaternion
        geometry_msgs::msg::QuaternionStamped qs;
        qs.header = std_msgs::msg::Header();
        qs.header.stamp = stamp;
        qs.header.frame_id = fixed_frame_;
        qs.quaternion.x = q.x();
        qs.quaternion.y = q.y();
        qs.quaternion.z = q.z();
        qs.quaternion.w = q.w();
        pub_quat_->publish(qs);

        // 퍼블리시: YPR (deg)
        geometry_msgs::msg::Vector3Stamped ypr_msg;
        ypr_msg.header = std_msgs::msg::Header();
        ypr_msg.header.stamp = stamp;
        ypr_msg.header.frame_id = fixed_frame_;
        ypr_msg.vector.x = ypr_deg[2]; // roll
        ypr_msg.vector.y = ypr_deg[1]; // pitch
        ypr_msg.vector.z = ypr_deg[0]; // yaw
        pub_ypr_deg_->publish(ypr_msg);

        // (옵션) TF: lidar_prev -> lidar_curr (translation 0, rotation R)
        if (publish_tf_)
        {
            geometry_msgs::msg::TransformStamped tf;
            tf.header.stamp = stamp;
            tf.header.frame_id = fixed_frame_;
            tf.child_frame_id = child_frame_;
            tf.transform.translation.x = 0.0;
            tf.transform.translation.y = 0.0;
            tf.transform.translation.z = 0.0;
            tf.transform.rotation = qs.quaternion;
            tf_broadcaster_->sendTransform(tf);
        }

        // 다음을 위해 현재를 prev로 보관
        // prev_cloud_ = cloud_ds;
        // prev_stamp_ = msg->header.stamp;

        // 로그
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                             "NDT rot ypr(deg): [%.2f, %.2f, %.2f]", ypr_deg[2], ypr_deg[1], ypr_deg[0]); // RPY order in log: roll, pitch, yaw
        auto xyz_res = solveRotation(cur_c, prev_c);
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                             "NDT func xyz(deg): [%.2f, %.2f, %.2f]", xyz_res[0], xyz_res[1], xyz_res[2]); // RPY order in log: roll, pitch, yaw
    }

    void pcdTimerCallback()
    {
        // RCLCPP_INFO(this->get_logger(), "pcdTimerCallback is actively running."); // 디버깅용 로그
        pub_prev_->publish(msg_prev_);
        pub_cur_->publish(msg_cur_);
        pub_align_->publish(msg_align_);
    }

    // 고정 설정(파라미터 사용 안함)
    const std::string cloud_topic_;
    const float voxel_leaf_;
    const double ndt_resolution_;
    const double ndt_step_size_;
    const double ndt_trans_eps_;
    const int ndt_max_iter_;
    const bool publish_tf_;
    const std::string fixed_frame_; // TF용 가상 프레임 이름
    const std::string child_frame_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Publisher<geometry_msgs::msg::QuaternionStamped>::SharedPtr pub_quat_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_ypr_deg_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_prev_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cur_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_align_;
    rclcpp::TimerBase::SharedPtr pcd_timer_;

    sensor_msgs::msg::PointCloud2 msg_prev_;
    sensor_msgs::msg::PointCloud2 msg_cur_;
    sensor_msgs::msg::PointCloud2 msg_align_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    CloudT::Ptr prev_cloud_;
    rclcpp::Time prev_stamp_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NdtRotationNode>());
    rclcpp::shutdown();
    return 0;
}
