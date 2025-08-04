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
#include <algorithm> // for std::sort
#include <cmath>     // for std::atan2, std::fabs

class PcdPublisher : public rclcpp::Node
{
public:
    PcdPublisher() : Node("z_pcd_publisher")
    {
        filePath();
        keyboard_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&PcdPublisher::keyboardCallback, this));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pcd_cloud", 10);
        pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_points", 10);
        corner_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("corner_markers", 10);
        plane_normal_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("plane_normal_marker", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&PcdPublisher::timerCallback, this));
    }

private:
    int frame_ = 0;
    void keyboardCallback()
    {

        if (keyboardAvailable())
        {
            std::string input;
            std::getline(std::cin, input);
            if (!input.empty() && std::all_of(input.begin(), input.end(), ::isdigit))
            {
                frame_ = std::stoi(input);
            }

            else if (input == "a")
            {
                detectPlane(0, frame_); // Lidar 0에 대해 평면 검출 및 시각화
            }

            else if (input == "s")
            {
                detectPlane(1, frame_); // Lidar 0에 대해 평면 검출 및 시각화
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
        corner_pub_->publish(corner_markers_); // OBB, 코너 스피어, 라인 마커 발행
        // 평면 법선 마커는 detectPlane에서 바로 publish 합니다.
    }

    void filePath()
    {
        std::string home_dir = std::getenv("HOME");
        RCLCPP_INFO(this->get_logger(), "Home directory : %s", home_dir.c_str());
        std::string data_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/d_multi_lidar_calib";
        pcd_path_ = data_dir + "/origin_pointclouds";
    }

    void publishPlaneNormalMarker(const pcl::PointXYZ &obb_position,
                                  const Eigen::Matrix3f &obb_rotation_matrix)
    {
        visualization_msgs::msg::MarkerArray axis_marker_array;

        for (auto &marker : axis_marker_array.markers)
        {
            marker.header.frame_id = "map";
            marker.header.stamp = this->get_clock()->now();
        }
        int axis_id_counter = 0;

        Eigen::Vector3f origin(obb_position.x, obb_position.y, obb_position.z);
        auto make_axis_marker = [&](const Eigen::Vector3f &direction,
                                    const std::string &ns,
                                    int id,
                                    float r, float g, float b)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->get_clock()->now();
            marker.ns = ns;
            marker.id = id;
            marker.type = visualization_msgs::msg::Marker::ARROW;
            marker.action = visualization_msgs::msg::Marker::ADD;

            geometry_msgs::msg::Point p_start, p_end;
            p_start.x = origin.x();
            p_start.y = origin.y();
            p_start.z = origin.z();

            Eigen::Vector3f endpoint = origin + direction.normalized() * 0.5f; // 방향벡터를 길이 0.5로

            p_end.x = endpoint.x();
            p_end.y = endpoint.y();
            p_end.z = endpoint.z();

            marker.points.push_back(p_start);
            marker.points.push_back(p_end);

            marker.scale.x = 0.03; // shaft diameter
            marker.scale.y = 0.06; // head diameter
            marker.scale.z = 0.1;  // head length

            marker.color.r = r;
            marker.color.g = g;
            marker.color.b = b;
            marker.color.a = 1.0;

            marker.lifetime = rclcpp::Duration::from_seconds(0.0);

            return marker;
        };

        // obb_rotation_matrix 각 열 벡터를 각 축으로 사용
        axis_marker_array.markers.push_back(make_axis_marker(obb_rotation_matrix.col(0), "board_axes_x", axis_id_counter++, 1.0, 0.0, 0.0)); // X: 빨강
        axis_marker_array.markers.push_back(make_axis_marker(obb_rotation_matrix.col(1), "board_axes_y", axis_id_counter++, 0.0, 1.0, 0.0)); // Y: 초록
        axis_marker_array.markers.push_back(make_axis_marker(obb_rotation_matrix.col(2), "board_axes_z", axis_id_counter++, 0.0, 0.0, 1.0)); // Z: 파랑

        plane_normal_marker_pub_->publish(axis_marker_array);
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
                    }

                    const double GEOMETRY_EPSILON = 1e-4;                         // 거리 및 좌표 유사성 판단 임계값
                    const Eigen::Vector3f lidar_origin = Eigen::Vector3f::Zero(); // 라이다 원점

                    // OBB의 원래 축 (글로벌 좌표계)
                    Eigen::Vector3f obb_global_x_axis = rotational_matrix_OBB.col(0);
                    Eigen::Vector3f obb_global_y_axis = rotational_matrix_OBB.col(1);
                    Eigen::Vector3f obb_global_z_axis = rotational_matrix_OBB.col(2);

                    Eigen::Vector3f new_x_axis, new_y_axis, new_z_axis;

                    // 1. new_z_axis (법선) 결정: obb_global_z_axis를 기반으로, 라이다 원점(0,0,0)을 향하도록 정렬
                    Eigen::Vector3f vec_center_to_lidar = lidar_origin - Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z);
                    new_z_axis = obb_global_z_axis;
                    if (new_z_axis.dot(vec_center_to_lidar) < 0)
                    { // 법선이 라이다 반대 방향이면 뒤집기
                        new_z_axis *= -1.0f;
                    }
                    new_z_axis.normalize();

                    // 2. new_y_axis (짧은 변) 결정: obb_global_x_axis 또는 obb_global_y_axis 중 짧은 것을 후보로
                    Eigen::Vector3f candidate_short_axis;
                    if (obb_len_x < obb_len_y)
                    { // obb_len_x가 짧은 변에 해당
                        candidate_short_axis = obb_global_x_axis;
                    }
                    else
                    { // obb_len_y가 짧은 변에 해당
                        candidate_short_axis = obb_global_y_axis;
                    }

                    // new_z_axis에 수직이 되도록 투영 및 정규화
                    new_y_axis = (candidate_short_axis - candidate_short_axis.dot(new_z_axis) * new_z_axis);
                    if (new_y_axis.norm() < GEOMETRY_EPSILON)
                    {                                                            // 퇴화된 경우, 대체 축 사용
                        new_y_axis = new_z_axis.cross(Eigen::Vector3f::UnitX()); // Z x X = Y
                        if (new_y_axis.norm() < GEOMETRY_EPSILON)
                        {
                            new_y_axis = new_z_axis.cross(Eigen::Vector3f::UnitY()); // Z x Y = -X
                        }
                    }
                    new_y_axis.normalize();

                    // new_y_axis의 방향을 전역 Y축과 일관되게 정렬 (선택적, 보드의 '상단' 방향이 일관될 때 유용)
                    // 이 조건은 보드가 주로 수직으로 세워져 있고, '위' 방향이 대략 전역 Y축과 일치할 때 효과적입니다.
                    if (new_y_axis.dot(Eigen::Vector3f::UnitY()) < 0)
                    {
                        new_y_axis *= -1.0f;
                    }
                    new_y_axis.normalize(); // 재정규화

                    // 3. new_x_axis (긴 변) 결정: 오른손 법칙 (new_x_axis = new_y_axis x new_z_axis)
                    new_x_axis = new_y_axis.cross(new_z_axis);
                    new_x_axis.normalize();

                    // 최종 회전 행렬 구성
                    Eigen::Matrix3f final_rotational_matrix_OBB;
                    final_rotational_matrix_OBB.col(0) = new_x_axis;
                    final_rotational_matrix_OBB.col(1) = new_y_axis;
                    final_rotational_matrix_OBB.col(2) = new_z_axis;

                    // OBB 축 정렬이 일관적이라면, 이 코너점들도 일관된 순서로 나올 것입니다.
                    float actual_half_len_long = std::max(obb_len_x, obb_len_y) / 2.0f;
                    float actual_half_len_short = std::min(obb_len_x, obb_len_y) / 2.0f;

                    // 보드 법선(new_z_axis)이 라이다를 향하고 있으므로, OBB의 Z축 방향을 따라 가장 먼 면을 선택
                    // chosen_z_local은 OBB 로컬 Z축에서 라이다를 향하는 면의 Z값입니다.
                    // 이 값은 OBB의 min_point_OBB.z 또는 max_point_OBB.z 중 하나가 됩니다.
                    float chosen_z_local = (new_z_axis.dot(Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z) - lidar_origin) > 0) ? max_point_OBB.z : min_point_OBB.z;

                    // Define 4 corner points in the NEW OBB local coordinates (aligned with new_x_axis, new_y_axis, new_z_axis)
                    std::vector<Eigen::Vector3f> new_local_face_corners;
                    // Top-Right, Bottom-Right, Top-Left
                    new_local_face_corners.push_back(Eigen::Vector3f(actual_half_len_long, actual_half_len_short, chosen_z_local));  // Corner 1 (Top-Right)
                    new_local_face_corners.push_back(Eigen::Vector3f(actual_half_len_long, -actual_half_len_short, chosen_z_local)); // Corner 2 (Bottom-Right)
                    new_local_face_corners.push_back(Eigen::Vector3f(-actual_half_len_long, actual_half_len_short, chosen_z_local)); // Corner 3 (Top-Left)

                    // Now, transform local corners to global coordinates
                    std::vector<Eigen::Vector3f> final_ordered_corners_global;
                    for (const auto &local_corner : new_local_face_corners)
                    {
                        Eigen::Vector3f global_corner = final_rotational_matrix_OBB * local_corner + Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z);
                        final_ordered_corners_global.push_back(global_corner);
                    }
                    // --- 시각화 마커 생성 ---

                    int marker_id_counter = 0;

                    // 1. OBB 시각화 (Cube)
                    visualization_msgs::msg::Marker obb_marker;
                    obb_marker.header.frame_id = "map";
                    obb_marker.header.stamp = this->now();
                    obb_marker.ns = "obb_boxes";
                    obb_marker.id = marker_id_counter++;
                    obb_marker.type = visualization_msgs::msg::Marker::CUBE;
                    obb_marker.action = visualization_msgs::msg::Marker::ADD;

                    obb_marker.pose.position.x = position_OBB.x;
                    obb_marker.pose.position.y = position_OBB.y;
                    obb_marker.pose.position.z = position_OBB.z;

                    Eigen::Quaternionf q(final_rotational_matrix_OBB); // 조정된 회전 행렬 사용
                    obb_marker.pose.orientation.x = q.x();
                    obb_marker.pose.orientation.y = q.y();
                    obb_marker.pose.orientation.z = q.z();
                    obb_marker.pose.orientation.w = q.w();

                    obb_marker.scale.x = obb_len_x;
                    obb_marker.scale.y = obb_len_y;
                    obb_marker.scale.z = obb_len_z;

                    obb_marker.color.r = 0.8f; // 연한 회색
                    obb_marker.color.g = 0.8f;
                    obb_marker.color.b = 0.8f;
                    obb_marker.color.a = 0.2f; // 투명도
                    corner_markers_.markers.push_back(obb_marker);

                    // 2. 3개 코너점 시각화 (Sphere) - 순서별 다른 색상
                    std::vector<std_msgs::msg::ColorRGBA> corner_colors = {
                        std_msgs::msg::ColorRGBA(), // Red (1번 코너)
                        std_msgs::msg::ColorRGBA(), // Green (2번 코너)
                        std_msgs::msg::ColorRGBA()  // Blue (3번 코너)
                    };
                    corner_colors[0].r = 1.0f;
                    corner_colors[0].a = 1.0f; // Red
                    corner_colors[1].g = 1.0f;
                    corner_colors[1].a = 1.0f; // Green
                    corner_colors[2].b = 1.0f;
                    corner_colors[2].a = 1.0f; // Blue

                    for (size_t i = 0; i < final_ordered_corners_global.size(); ++i)
                    {
                        visualization_msgs::msg::Marker corner_sphere;
                        corner_sphere.header.frame_id = "map";
                        corner_sphere.header.stamp = this->now();
                        corner_sphere.ns = "detected_corners_spheres";
                        corner_sphere.id = marker_id_counter++;
                        corner_sphere.type = visualization_msgs::msg::Marker::SPHERE;
                        corner_sphere.action = visualization_msgs::msg::Marker::ADD;
                        corner_sphere.pose.position.x = final_ordered_corners_global[i].x();
                        corner_sphere.pose.position.y = final_ordered_corners_global[i].y();
                        corner_sphere.pose.position.z = final_ordered_corners_global[i].z();
                        corner_sphere.pose.orientation.w = 1.0;
                        corner_sphere.scale.x = 0.08;
                        corner_sphere.scale.y = 0.08;
                        corner_sphere.scale.z = 0.08;
                        corner_sphere.color = corner_colors[i];
                        corner_markers_.markers.push_back(corner_sphere);
                    }

                    // 3. 코너점 연결 라인 시각화 (Line Strip) - 3개 코너 연결
                    visualization_msgs::msg::Marker corner_line_strip;
                    corner_line_strip.header.frame_id = "map";
                    corner_line_strip.header.stamp = this->now();
                    corner_line_strip.ns = "detected_corners_lines";
                    corner_line_strip.id = marker_id_counter++;
                    corner_line_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
                    corner_line_strip.action = visualization_msgs::msg::Marker::ADD;
                    corner_line_strip.scale.x = 0.01; // 라인 두께
                    corner_line_strip.color.r = 1.0f; // 흰색
                    corner_line_strip.color.g = 1.0f;
                    corner_line_strip.color.b = 1.0f;
                    corner_line_strip.color.a = 1.0f;

                    for (const auto &corner : final_ordered_corners_global)
                    {
                        geometry_msgs::msg::Point p;
                        p.x = corner.x();
                        p.y = corner.y();
                        p.z = corner.z();
                        corner_line_strip.points.push_back(p);
                    }
                    // 마지막 점과 첫 점 연결하여 삼각형 닫기
                    if (!final_ordered_corners_global.empty())
                    {
                        geometry_msgs::msg::Point p;
                        p.x = final_ordered_corners_global[0].x();
                        p.y = final_ordered_corners_global[0].y();
                        p.z = final_ordered_corners_global[0].z();
                        corner_line_strip.points.push_back(p);
                    }
                    corner_markers_.markers.push_back(corner_line_strip);

                    publishPlaneNormalMarker(position_OBB, final_rotational_matrix_OBB);
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
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr plane_normal_marker_pub_;

    sensor_msgs::msg::PointCloud2 cloud_msg_;
    sensor_msgs::msg::PointCloud2 plane_msg_;
    visualization_msgs::msg::MarkerArray corner_markers_; // OBB, 코너 스피어, 라인 마커를 모두 담을 MarkerArray

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
