#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/convex_hull.h>

#include <Eigen/Dense>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>

class PcdPublisher : public rclcpp::Node
{
public:
    PcdPublisher() : Node("pcd_publisher")
    {
        filePath();
        keyboard_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&PcdPublisher::keyboardCallback, this));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pcd_cloud", 10);
        pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_points", 10);
        corner_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("corner_markers", 10);
        plane_normal_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("plane_normal_marker", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&PcdPublisher::timerCallback, this));
    }

private:
    void keyboardCallback()
    {
        if (keyboardAvailable())
        {
            std::string input;
            std::getline(std::cin, input);

            if (!input.empty() && std::all_of(input.begin(), input.end(), ::isdigit))
            {
                int frame = std::stoi(input);
                detectPlane(0, frame);
            }
            else if (input == "exit" || input == "q")
            {
                rclcpp::shutdown();
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

    void timerCallback()
    {
        cloud_msg_.header.stamp = this->now();
        plane_msg_.header.stamp = this->now();
        publisher_->publish(cloud_msg_);
        pub_plane_->publish(plane_msg_);
        corner_pub_->publish(corner_markers_);
        // 평면 법선 마커는 detectPlane에서 바로 publish 합니다.
    }

    void filePath()
    {
        std::string home_dir = std::getenv("HOME");
        RCLCPP_INFO(this->get_logger(), "Home directory : %s", home_dir.c_str());
        std::string data_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/multi_lidar_calib";
        pcd_path_ = data_dir + "/origin_pointclouds";
    }

    void publishPlaneNormalMarker(const Eigen::Vector3f &centroid,
                                  const Eigen::Vector3f &normal,
                                  int id)
    {
        visualization_msgs::msg::Marker arrow;
        arrow.header.frame_id = "map";
        arrow.header.stamp = this->now();
        arrow.ns = "plane_normals";
        arrow.id = id;
        arrow.type = visualization_msgs::msg::Marker::ARROW;
        arrow.action = visualization_msgs::msg::Marker::ADD;

        geometry_msgs::msg::Point start_point;
        start_point.x = centroid.x();
        start_point.y = centroid.y();
        start_point.z = centroid.z();

        geometry_msgs::msg::Point end_point;
        end_point.x = centroid.x() + normal.x() * 0.5f;
        end_point.y = centroid.y() + normal.y() * 0.5f;
        end_point.z = centroid.z() + normal.z() * 0.5f;

        arrow.points.push_back(start_point);
        arrow.points.push_back(end_point);

        arrow.scale.x = 0.05; // 화살표 두께
        arrow.scale.y = 0.1;  // 화살촉 너비

        arrow.color.r = 1.0f;
        arrow.color.g = 0.0f;
        arrow.color.b = 0.0f;
        arrow.color.a = 1.0f;

        plane_normal_marker_pub_->publish(arrow);
    }

    void detectPlane(int num, int frame)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        std::string filename = pcd_path_ + "/lidar" + std::to_string(num) + "_" + std::to_string(frame) + ".pcd";

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read file: %s", filename.c_str());
            return;
        }

        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud);
        crop.setMin(Eigen::Vector4f(-4.0, -2.0, -0.85, 1.0));
        crop.setMax(Eigen::Vector4f(-2.0, 0.8, 2.0, 1.0));
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZ>);
        crop.filter(*cloud_roi);

        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.001);
        seg.setInputCloud(cloud_roi);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No plane found in the ROI.");
            return;
        }

        float A = coefficients->values[0];
        float B = coefficients->values[1];
        float C = coefficients->values[2];
        float D = coefficients->values[3];
        float norm = std::sqrt(A * A + B * B + C * C);

        Eigen::Vector3f normal(A, B, C);
        normal.normalize();

        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_points(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract_inliers;
        extract_inliers.setInputCloud(cloud_roi);
        extract_inliers.setIndices(inliers);
        extract_inliers.setNegative(false);
        extract_inliers.filter(*plane_points);

        Eigen::Vector4f centroid_4f;
        pcl::compute3DCentroid(*plane_points, centroid_4f);
        Eigen::Vector3f centroid(centroid_4f.x(), centroid_4f.y(), centroid_4f.z());

        publishPlaneNormalMarker(centroid, normal, 0);

        // 평면에서 먼 점 제거 (0.05m 이상)
        pcl::PointIndices::Ptr to_remove(new pcl::PointIndices);
        for (size_t i = 0; i < cloud_roi->points.size(); ++i)
        {
            const auto &pt = cloud_roi->points[i];
            float dist_to_plane = std::fabs(A * pt.x + B * pt.y + C * pt.z + D) / norm;
            if (dist_to_plane < 0.05)
            {
                to_remove->indices.push_back(i);
            }
        }
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cleaned(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(cloud_roi);
        extract.setIndices(to_remove);
        extract.setNegative(true);
        extract.filter(*cleaned);

        // 보드 크기 및 조건 변수
        const float TARGET_BOARD_WIDTH_MIN = 0.40;
        const float TARGET_BOARD_WIDTH_MAX = 0.70;
        const float TARGET_BOARD_HEIGHT_MIN = 0.70;
        const float TARGET_BOARD_HEIGHT_MAX = 1.00;
        const float ASPECT_RATIO_TOLERANCE = 1.0;
        const float MIN_PLANE_POINTS = 100;
        const float MAX_BOARD_THICKNESS = 0.2;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_board_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        if (inliers->indices.size() >= MIN_PLANE_POINTS)
        {
            pcl::VoxelGrid<pcl::PointXYZ> voxel;
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_plane(new pcl::PointCloud<pcl::PointXYZ>);
            voxel.setInputCloud(cleaned);
            voxel.setLeafSize(0.05f, 0.05f, 0.05f);
            voxel.filter(*filtered_plane);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(filtered_plane);

            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(0.06);
            ec.setMinClusterSize(10);
            ec.setMaxClusterSize(10000);
            ec.setSearchMethod(tree);
            ec.setInputCloud(filtered_plane);
            ec.extract(cluster_indices);

            if (!cluster_indices.empty())
            {
                auto largest_cluster = std::max_element(cluster_indices.begin(), cluster_indices.end(),
                                                        [](const pcl::PointIndices &a, const pcl::PointIndices &b)
                                                        {
                                                            return a.indices.size() < b.indices.size();
                                                        });

                pcl::PointCloud<pcl::PointXYZ>::Ptr board_candidate(new pcl::PointCloud<pcl::PointXYZ>);
                for (int idx : largest_cluster->indices)
                {
                    board_candidate->points.push_back(filtered_plane->points[idx]);
                }

                pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
                feature_extractor.setInputCloud(board_candidate);
                feature_extractor.compute();

                pcl::PointXYZ min_point_OBB, max_point_OBB, position_OBB;
                Eigen::Matrix3f rotational_matrix_OBB;
                feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

                float obb_len_x = max_point_OBB.x - min_point_OBB.x;
                float obb_len_y = max_point_OBB.y - min_point_OBB.y;
                float obb_len_z = max_point_OBB.z - min_point_OBB.z;

                float board_dim1 = std::max(obb_len_x, obb_len_y);
                float board_dim2 = std::min(obb_len_x, obb_len_y);
                float board_thickness = obb_len_z;

                bool is_correct_size = (board_dim1 >= TARGET_BOARD_WIDTH_MIN && board_dim1 <= TARGET_BOARD_WIDTH_MAX &&
                                        board_dim2 >= TARGET_BOARD_HEIGHT_MIN && board_dim2 <= TARGET_BOARD_HEIGHT_MAX) ||
                                       (board_dim2 >= TARGET_BOARD_WIDTH_MIN && board_dim2 <= TARGET_BOARD_WIDTH_MAX &&
                                        board_dim1 >= TARGET_BOARD_HEIGHT_MIN && board_dim1 <= TARGET_BOARD_HEIGHT_MAX);

                float aspect_ratio = board_dim1 / board_dim2;
                bool is_correct_aspect_ratio = (aspect_ratio >= (1.0 - ASPECT_RATIO_TOLERANCE) &&
                                                aspect_ratio <= (1.0 + ASPECT_RATIO_TOLERANCE));

                bool is_thin_enough = (board_thickness < MAX_BOARD_THICKNESS);

                RCLCPP_INFO(this->get_logger(), "Filtered Cluster OBB Dims: %.3f x %.3f x %.3f (L, W, T)",
                            board_dim1, board_dim2, board_thickness);
                RCLCPP_INFO(this->get_logger(), "Cluster Filter Check: Size=%d, Aspect=%d, Thin=%d",
                            is_correct_size, is_correct_aspect_ratio, is_thin_enough);

                if (is_correct_size && is_correct_aspect_ratio && is_thin_enough)
                {
                    for (const auto &p : board_candidate->points)
                    {
                        pcl::PointXYZRGB pt_rgb;
                        pt_rgb.x = p.x;
                        pt_rgb.y = p.y;
                        pt_rgb.z = p.z;
                        pt_rgb.r = 0;
                        pt_rgb.g = 255;
                        pt_rgb.b = 0;
                        final_board_cloud->points.push_back(pt_rgb);

                        // 여기에 추가, 위로는 건드리지 마시오.
                        // Calculate the 4 corners of the detected board using OBB
                        // We take the 4 corners corresponding to one of the main faces (e.g., min_z in OBB local frame)
                        std::vector<Eigen::Vector3f> local_board_corners;
                        local_board_corners.push_back(Eigen::Vector3f(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z));
                        local_board_corners.push_back(Eigen::Vector3f(max_point_OBB.x, min_point_OBB.y, min_point_OBB.z));
                        local_board_corners.push_back(Eigen::Vector3f(min_point_OBB.x, max_point_OBB.y, min_point_OBB.z));
                        local_board_corners.push_back(Eigen::Vector3f(max_point_OBB.x, max_point_OBB.y, min_point_OBB.z));

                        // Transform these local corners to the global coordinate system
                        corner_markers_.markers.clear(); // Clear previous markers
                        int marker_id = 1;               // Start ID from 1, 0 is for plane normal

                        for (const auto &local_corner : local_board_corners)
                        {
                            Eigen::Vector3f global_corner = rotational_matrix_OBB * local_corner + Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z);

                            visualization_msgs::msg::Marker corner_marker;
                            corner_marker.header.frame_id = "map";
                            corner_marker.header.stamp = this->now();
                            corner_marker.ns = "board_corners";
                            corner_marker.id = marker_id++;
                            corner_marker.type = visualization_msgs::msg::Marker::SPHERE;
                            corner_marker.action = visualization_msgs::msg::Marker::ADD;
                            corner_marker.pose.position.x = global_corner.x();
                            corner_marker.pose.position.y = global_corner.y();
                            corner_marker.pose.position.z = global_corner.z();
                            corner_marker.pose.orientation.w = 1.0; // No rotation for a sphere
                            corner_marker.scale.x = 0.1;            // Size of the sphere
                            corner_marker.scale.y = 0.1;
                            corner_marker.scale.z = 0.1;
                            corner_marker.color.r = 0.0f;
                            corner_marker.color.g = 0.0f;
                            corner_marker.color.b = 1.0f; // Blue color for corners
                            corner_marker.color.a = 1.0f;
                            corner_markers_.markers.push_back(corner_marker);
                        }
                    }
                }
            }
        }

        // 최종 메시지 발행
        pcl::toROSMsg(*cloud, cloud_msg_);
        cloud_msg_.header.frame_id = "map";

        pcl::toROSMsg(*final_board_cloud, plane_msg_);
        plane_msg_.header.frame_id = "map";
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_plane_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr corner_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_normal_marker_pub_;

    sensor_msgs::msg::PointCloud2 cloud_msg_;
    sensor_msgs::msg::PointCloud2 plane_msg_;
    visualization_msgs::msg::MarkerArray corner_markers_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr keyboard_timer_;
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
