#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

#include <Eigen/Dense>

class PcdPublisher : public rclcpp::Node
{
public:
    PcdPublisher() : Node("pcd_publisher")
    {
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pcd_cloud", 10);
        pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_points", 10);

        std::string where = "company";
        read_write_path(where);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        std::string filename = pcd_path_ + "/lidar0_8.pcd";

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read file: %s", filename.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %zu points from %s", cloud->size(), filename.c_str());

        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud);
        crop.setMin(Eigen::Vector4f(-4.0, -2.0, -0.7, 1.0));
        crop.setMax(Eigen::Vector4f(-2.0, 1.0, 2.0, 1.0));
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZ>);
        crop.filter(*cloud_roi);

        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliners(new pcl::PointIndices);
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.001);
        seg.setInputCloud(cloud_roi);
        seg.segment(*inliners, *coefficients);

        auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        RCLCPP_INFO(this->get_logger(), "2-3. Post-processing...");
        for (const auto &pt : cloud_roi->points)
        {
            float distance = coefficients->values[0] * pt.x +
                             coefficients->values[1] * pt.y +
                             coefficients->values[2] * pt.z +
                             coefficients->values[3];

            if (std::abs(distance) < 0.05)
            {
                filtered_cloud->points.push_back(pt);
            }
        }

        filtered_cloud->width = filtered_cloud->points.size();
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = true;

        
        // KD-Tree for neighbor search
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud_roi);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

        const float flatness_threshold = 0.15f;
        const float search_radius = 0.1f;
        const int min_neighbors = 10;

        for (const auto &pt : filtered_cloud->points)
        {
            std::vector<int> indices;
            std::vector<float> dists;
            if (kdtree.radiusSearch(pt, search_radius, indices, dists) >= min_neighbors)
            {
                // neighbors 점들만 따로 추출
                pcl::PointCloud<pcl::PointXYZ> neighbors;
                for (int idx : indices)
                {
                    neighbors.points.push_back(filtered_cloud->points[idx]);
                }
                neighbors.width = neighbors.points.size();
                neighbors.height = 1;
                neighbors.is_dense = true;

                // 공분산, 중심 계산은 neighbors로
                Eigen::Vector4f centroid;
                Eigen::Matrix3f covariance;
                pcl::computeMeanAndCovarianceMatrix(neighbors, covariance, centroid);

                // 고유값 계산
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
                Eigen::Vector3f eigenvalues = solver.eigenvalues();

                float flatness = eigenvalues[0] / eigenvalues.sum();

                if (flatness < flatness_threshold) // threshold는 0.01~0.05 정도가 적당함
                {
                    pcl::PointXYZRGB pt_rgb;
                    pt_rgb.x = pt.x;
                    pt_rgb.y = pt.y;
                    pt_rgb.z = pt.z;
                    pt_rgb.r = 0;
                    pt_rgb.g = 255;
                    pt_rgb.b = 0;
                    filtered->points.push_back(pt_rgb);
                }
            }
        }

        filtered->width = filtered->size();
        filtered->height = 1;
        filtered->is_dense = true;

        RCLCPP_INFO(this->get_logger(), "2-4. Extract plane points...");
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(filtered);
        extract.setIndices(inliners);
        extract.setNegative(true);
        extract.filter(*filtered);


        pcl::toROSMsg(*cloud, cloud_msg_);
        cloud_msg_.header.frame_id = "map";

        pcl::toROSMsg(*filtered, plane_msg_);
        plane_msg_.header.frame_id = "map";

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&PcdPublisher::timerCallback, this));
    }

private:
    void timerCallback()
    {
        cloud_msg_.header.stamp = this->now();
        plane_msg_.header.stamp = this->now();
        publisher_->publish(cloud_msg_);
        pub_plane_->publish(plane_msg_);
    }

    void read_write_path(std::string where)
    {
        std::string change_path = (where == "company") ? "/antlab/sensor_fusion_study_ws"
                                                       : "/icrs/sensor_fusion_study_ws";
        std::string absolute_path = "/home" + change_path + "/src/sensor_fusion_study/calib_data/multi_lidar_calib";
        pcd_path_ = absolute_path + "/origin_pointclouds";
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_plane_;
    sensor_msgs::msg::PointCloud2 cloud_msg_;
    sensor_msgs::msg::PointCloud2 plane_msg_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::string pcd_path_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PcdPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
