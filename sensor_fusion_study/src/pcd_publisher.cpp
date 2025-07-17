#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/crop_box.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>

#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>

class PcdPublisher : public rclcpp::Node
{
public:
    PcdPublisher()
        : Node("pcd_publisher")
    {
        // 퍼블리셔 생성
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pcd_cloud", 10);
        pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_points", 10);

        std::string where = "company";
        read_write_path(where);

        // PCD 파일 로드
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        std::string filename = pcd_path_ + "/pcd_10.pcd";

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read PCD file: %s", filename.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %zu points from %s", filename.c_str());

        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud);
        crop.setMin(Eigen::Vector4f(-3.0, -0.8, -0.63, 1.0)); // ROI 최소 x,y,z
        crop.setMax(Eigen::Vector4f(0.0, 0.5, 3.0, 1.0));   // ROI 최대 x,y,z

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZ>);
        crop.filter(*cloud_roi);

        // 평면 분할
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.02);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        seg.setInputCloud(cloud_roi);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No planar model found.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Plane coefficients: %f %f %f %f",
                    coefficients->values[0],
                    coefficients->values[1],
                    coefficients->values[2],
                    coefficients->values[3]);

        // 평면 점 추출
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_roi);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_roi);

        RCLCPP_INFO(this->get_logger(), "Plane inliers: %zu", inliers->indices.size());
#if 0
        float distance_threshold = 0.005; // 5 mm

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);
for (const auto& pt : cloud_roi->points) {
    float dist = std::abs(
        coefficients->values[0] * pt.x +
        coefficients->values[1] * pt.y +
        coefficients->values[2] * pt.z +
        coefficients->values[3]
    ) / std::sqrt(
        coefficients->values[0]*coefficients->values[0] +
        coefficients->values[1]*coefficients->values[1] +
        coefficients->values[2]*coefficients->values[2]
    );

    if (dist < distance_threshold) {
        cloud_plane_filtered->points.push_back(pt);
    }
}

cloud_plane_filtered->width = cloud_plane_filtered->points.size();
cloud_plane_filtered->height = 1;
cloud_plane_filtered->is_dense = true;

RCLCPP_INFO(this->get_logger(), "Points after distance filtering: %zu", cloud_plane_filtered->points.size());

pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

ne.setSearchMethod(tree);
ne.setInputCloud(cloud_plane_filtered);
ne.setKSearch(30);
ne.compute(*normals);

pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
reg.setMinClusterSize(30);
reg.setMaxClusterSize(10000);
reg.setSearchMethod(tree);
reg.setNumberOfNeighbours(30);
reg.setInputCloud(cloud_plane_filtered);
reg.setInputNormals(normals);
reg.setSmoothnessThreshold(5.0 / 180.0 * M_PI); // 약 5도
reg.setCurvatureThreshold(0.05);

std::vector<pcl::PointIndices> clusters;
reg.extract(clusters);

// Region Growing 결과 중 가장 큰 cluster만 사용
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_cluster(new pcl::PointCloud<pcl::PointXYZ>);
if (!clusters.empty())
{
    size_t max_size = 0;
    int max_index = -1;
    for (size_t i = 0; i < clusters.size(); ++i)
    {
        if (clusters[i].indices.size() > max_size)
        {
            max_size = clusters[i].indices.size();
            max_index = i;
        }
    }

    for (auto idx : clusters[max_index].indices)
    {
        cloud_plane_cluster->points.push_back(cloud_roi->points[idx]);
    }

    cloud_plane_cluster->width = cloud_plane_cluster->points.size();
    cloud_plane_cluster->height = 1;
    cloud_plane_cluster->is_dense = true;

    RCLCPP_INFO(this->get_logger(), "Region Growing cluster points: %zu", cloud_plane_cluster->points.size());
}
else
{
    RCLCPP_WARN(this->get_logger(), "No Region Growing clusters found!");
}

#endif
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (const auto &pt : cloud_roi->points)
        {
            pcl::PointXYZRGB pt_rgb;
            pt_rgb.x = pt.x;
            pt_rgb.y = pt.y;
            pt_rgb.z = pt.z;

            pt_rgb.r = 0;
            pt_rgb.g = 0;
            pt_rgb.b = 255;

            plane_cloud->points.push_back(pt_rgb);
        }

        plane_cloud->width = plane_cloud->points.size();
        plane_cloud->height = 1;
        plane_cloud->is_dense = true;

        // PCL → ROS2 메시지 변환
        pcl::toROSMsg(*cloud, cloud_msg_);
        cloud_msg_.header.frame_id = "map"; // RViz에서 사용하는 좌표계 설정

        pcl::toROSMsg(*plane_cloud, plane_msg_);
        plane_msg_.header.frame_id = "map"; // RViz에서 사용하는 좌표계 설정

        // 타이머로 주기적 퍼블리시
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
        std::string change_path;
        if (where == "company")
        {
            change_path = "/antlab/sensor_fusion_study_ws";
        }
        else if (where == "home")
        {
            change_path = "/icrs/sensor_fusion_study_ws";
        }

        std::string absolute_path = "/home" + change_path + "/src/sensor_fusion_study/cam_lidar_calib";
        pcd_path_ = absolute_path + "/pointclouds";
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
