#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <string>
#include <sys/select.h>
#include <unistd.h>
#include <filesystem>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/transforms.h>

namespace fs = std::filesystem;
// #define LOOK_DEBUG

class MultiLidarCalibNode : public rclcpp::Node
{
public:
    MultiLidarCalibNode()
        : Node("d_multi_lidar_calib")
    {
        RCLCPP_INFO(this->get_logger(), "-----------\nStart multi lidar calib node\n----------");
        filePath();
        keyboard_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&MultiLidarCalibNode::keyboardCallback, this));
    }

private:
    std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subscription_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds_;
    rclcpp::TimerBase::SharedPtr keyboard_timer_;
    int number_lidars_ = 0;
    int frame_counter_ = 0;
    std::string origin_path_;

    void keyboardCallback()
    {
        if (keyboardAvailable())
        {
            std::string input;
            std::getline(std::cin, input);

            if (!input.empty() && std::all_of(input.begin(), input.end(), ::isdigit))
            {
                number_lidars_ = std::stoi(input);
                clouds_.resize(number_lidars_);
                RCLCPP_INFO(this->get_logger(), "Start subscribe %d lidars", number_lidars_);

                for (int i = 0; i < number_lidars_; i++)
                {
                    std::string topic_name = "/lidar" + std::to_string(i) + "/points";
                    clouds_[i] = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                    auto sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_name, rclcpp::SensorDataQoS(),
                                                                                        [this, i, topic_name](const sensor_msgs::msg::PointCloud2::SharedPtr msg)
                                                                                        { pcdCallback(msg, i); });
                    subscription_.push_back(sub);
                }
            }
            else if (input == "s")
            {
                savePointCloud(number_lidars_, frame_counter_);
                frame_counter_++;
            }
            else if (input == "c")
            {
                runMultiLidarCalibrate(number_lidars_);
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

    void filePath()
    {
        std::string home_dir = std::getenv("HOME");
        RCLCPP_INFO(this->get_logger(), "Home directory : %s", home_dir.c_str());
        std::string data_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/d_multi_lidar_calib";
        origin_path_ = data_dir + "/origin_pointclouds/";
        if (!fs::exists(origin_path_))
        {
            fs::create_directories(origin_path_);
        }
    }

    void pcdCallback(const sensor_msgs::msg::PointCloud2::SharedPtr message, int index)
    {
        pcl::fromROSMsg(*message, *clouds_[index]);
    }

    void savePointCloud(int num, int frame_counter)
    {
        RCLCPP_INFO(this->get_logger(), "Save current Point Cloud");
        if (clouds_.empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("savePointCloud"), "Lidar is Not Working!!!");
        }
        for (int i = 0; i < num; i++)
        {
            std::string filename = origin_path_ + "lidar" + std::to_string(i) + "_" + std::to_string(frame_counter) + ".pcd";
            pcl::io::savePCDFile(filename, *clouds_[i]);
        }
    }

    void runMultiLidarCalibrate(int number)
    {
        RCLCPP_INFO(this->get_logger(), "Start Calibration...");

        RCLCPP_INFO(this->get_logger(), "1. Load .pcd files...");
        int total_pcd_files = 0;
        for (const auto &entry : fs::directory_iterator(origin_path_))
        {
            if (entry.path().extension() == ".pcd")
                total_pcd_files++;
        }

        int frame_count = total_pcd_files / number;
        RCLCPP_INFO(this->get_logger(), "Found %d .pcd files, %d lidars ,%d frames", total_pcd_files, number, frame_count);

        std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> loaded_clouds;

        for (int frame_idx = 0; frame_idx < frame_count; frame_idx++)
        {
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> frame_clouds;
            for (int lidar_idx = 0; lidar_idx < number; lidar_idx++)
            {
                std::string pcd_name = origin_path_ + "lidar" + std::to_string(lidar_idx) + "_" + std::to_string(frame_idx) + ".pcd";
                auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                pcl::io::loadPCDFile(pcd_name, *cloud);
                frame_clouds.push_back(cloud);
            }
            loaded_clouds.push_back(frame_clouds);
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %lu frames of multi-lidar point clouds", loaded_clouds.size());

        std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> detected_plane_clouds;
        std::vector<std::vector<std::vector<Eigen::Vector3f>>> all_corners;
        RCLCPP_INFO(this->get_logger(), "2. Process each frame and find planes...");
        int just_using_log_count = 0;
        for (int frame_index = 0; frame_index < frame_count; frame_index++)
        {
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> frame_plane_clouds;
            std::vector<std::vector<Eigen::Vector3f>> current_frame_corners;
            for (int lidar_index = 0; lidar_index < number; lidar_index++)
            {
#ifdef LOOK_DEBUG
                RCLCPP_INFO(this->get_logger(), "Processing data from LIDAR %d (%d frame)", lidar_index, frame_index);
#endif
                pcl::PointCloud<pcl::PointXYZ>::Ptr frame_cloud = loaded_clouds[frame_index][lidar_index];

                if (just_using_log_count == 0)
                {
                    RCLCPP_INFO(this->get_logger(), "2-1. Set cloud ROI...");
                }
                pcl::CropBox<pcl::PointXYZ> crop;
                crop.setInputCloud(frame_cloud);
                crop.setMin(Eigen::Vector4f(-4.0, -2.0, -0.85, 1.0)); // X, Y, Z, 1.0
                crop.setMax(Eigen::Vector4f(-2.0, 0.8, 2.0, 1.0));
                crop.filter(*frame_cloud);
                if (just_using_log_count == 0)
                {
                    RCLCPP_INFO(this->get_logger(), "2-2. Delete floor...");
                }
                pcl::SACSegmentation<pcl::PointXYZ> ransac_seg;
                pcl::ModelCoefficients::Ptr ransac_coeff(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr ransac_inliers(new pcl::PointIndices);
                ransac_seg.setOptimizeCoefficients(true);
                ransac_seg.setModelType(pcl::SACMODEL_PLANE);
                ransac_seg.setMethodType(pcl::SAC_RANSAC);
                ransac_seg.setDistanceThreshold(0.001);
                ransac_seg.setInputCloud(frame_cloud);
                ransac_seg.segment(*ransac_inliers, *ransac_coeff);

                float A = ransac_coeff->values[0];
                float B = ransac_coeff->values[1];
                float C = ransac_coeff->values[2];
                float D = ransac_coeff->values[3];
                float norm = std::sqrt(A * A + B * B + C * C);

                pcl::PointIndices::Ptr plane_remove(new pcl::PointIndices);

                for (size_t i = 0; i < frame_cloud->points.size(); ++i)
                {
                    const auto &pt = frame_cloud->points[i];

                    float dist_to_plane = std::fabs(A * pt.x + B * pt.y + C * pt.z + D) / norm;

                    float ground_z = 0.0;
                    for (const auto &idx : ransac_inliers->indices)
                        ground_z += frame_cloud->points[idx].z;
                    ground_z /= static_cast<float>(ransac_inliers->indices.size());

                    float dx = pt.x - frame_cloud->points[ransac_inliers->indices[0]].x;
                    float dy = pt.y - frame_cloud->points[ransac_inliers->indices[0]].y;
                    float dist_xy = std::sqrt(dx * dx + dy * dy);

                    if (dist_to_plane < 0.05)
                    {
                        plane_remove->indices.push_back(i);
                    }
                }

                pcl::ExtractIndices<pcl::PointXYZ> plane_extract;
                pcl::PointCloud<pcl::PointXYZ>::Ptr cleaned(new pcl::PointCloud<pcl::PointXYZ>);
                plane_extract.setInputCloud(frame_cloud);
                plane_extract.setIndices(plane_remove);
                plane_extract.setNegative(true);
                plane_extract.filter(*frame_cloud);
                if (just_using_log_count == 0)
                {
                    RCLCPP_INFO(this->get_logger(), "2-3. Find plane using voxel...");
                }
                const float TARGET_BOARD_WIDTH_MIN = 0.40;
                const float TARGET_BOARD_WIDTH_MAX = 0.70;
                const float TARGET_BOARD_HEIGHT_MIN = 0.70;
                const float TARGET_BOARD_HEIGHT_MAX = 1.00;
                const float ASPECT_RATIO_TOLERANCE = 1.0;
                const float MIN_PLANE_POINTS = 100;
                const float MAX_BOARD_THICKNESS = 0.2;

                auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

                if (ransac_inliers->indices.size() >= MIN_PLANE_POINTS)
                {
                    pcl::VoxelGrid<pcl::PointXYZ> voxel;
                    voxel.setInputCloud(frame_cloud);
                    voxel.setLeafSize(0.05f, 0.05f, 0.05f);
                    voxel.filter(*filtered_cloud);

                    std::vector<pcl::PointIndices> cluster_indices;
                    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
                    tree->setInputCloud(filtered_cloud);

                    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
                    ec.setClusterTolerance(0.06);
                    ec.setMinClusterSize(10);
                    ec.setMaxClusterSize(10000);
                    ec.setSearchMethod(tree);
                    ec.setInputCloud(filtered_cloud);
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
                            board_candidate->points.push_back(filtered_cloud->points[idx]);
                        }

                        pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
                        feature_extractor.setInputCloud(board_candidate);
                        feature_extractor.compute();

                        if (just_using_log_count == 0)
                        {
                            RCLCPP_INFO(this->get_logger(), "2-4. Find corner using OBB...");
                        }
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
#ifdef LOOK_DEBUG
                        RCLCPP_INFO(this->get_logger(), "Filtered Cluster OBB Dims: %.3f x %.3f x %.3f (L, W, T)",
                                    board_dim1, board_dim2, board_thickness);
                        RCLCPP_INFO(this->get_logger(), "Cluster Filter Check: Size=%d, Aspect=%d, Thin=%d",
                                    is_correct_size, is_correct_aspect_ratio, is_thin_enough);
#endif

                        if (is_correct_size && is_correct_aspect_ratio && is_thin_enough)
                        {
                            for (const auto &p : board_candidate->points)
                            {
                                pcl::PointXYZ pt_rgb;
                                pt_rgb.x = p.x;
                                pt_rgb.y = p.y;
                                pt_rgb.z = p.z;
                                filtered_cloud->points.push_back(pt_rgb);
                            }
                            const double GEOMETRY_EPSILON = 1e-4;
                            const Eigen::Vector3f lidar_origin = Eigen::Vector3f::Zero();

                            Eigen::Vector3f original_obb_x_axis = rotational_matrix_OBB.col(0);
                            Eigen::Vector3f original_obb_y_axis = rotational_matrix_OBB.col(1);
                            Eigen::Vector3f original_obb_z_axis = rotational_matrix_OBB.col(2);

                            Eigen::Vector3f vec_center_to_lidar = lidar_origin - Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z);

                            Eigen::Vector3f new_z_axis = original_obb_z_axis;
                            if (new_z_axis.dot(vec_center_to_lidar) < 0)
                            {
                                new_z_axis *= -1.0f;
                            }
                            new_z_axis.normalize();

                            Eigen::Vector3f candidate_short_axis;

                            if (obb_len_x < obb_len_y)
                            {
                                candidate_short_axis = original_obb_x_axis;
                            }
                            else
                            {
                                candidate_short_axis = original_obb_y_axis;
                            }

                            Eigen::Vector3f new_y_axis = candidate_short_axis - candidate_short_axis.dot(new_z_axis) * new_z_axis;
                            if (new_y_axis.norm() < GEOMETRY_EPSILON)
                            {
                                new_y_axis = new_z_axis.cross(Eigen::Vector3f::UnitY());
                                if (new_y_axis.norm() < GEOMETRY_EPSILON)
                                {
                                    new_y_axis = new_z_axis.cross(Eigen::Vector3f::UnitX());
                                }
                            }
                            new_y_axis.normalize();

                            if (new_y_axis.dot(Eigen::Vector3f::UnitX()) < 0)
                            {
                                new_y_axis *= -1.0f;
                            }
                            new_y_axis.normalize();

                            Eigen::Vector3f new_x_axis = new_y_axis.cross(new_z_axis);
                            new_x_axis.normalize();

                            Eigen::Matrix3f final_rotational_matrix_OBB;
                            final_rotational_matrix_OBB.col(0) = new_x_axis;
                            final_rotational_matrix_OBB.col(1) = new_y_axis;
                            final_rotational_matrix_OBB.col(2) = new_z_axis;

                            float actual_half_len_long = std::max(obb_len_x, obb_len_y) / 2.0f;
                            float actual_half_len_short = std::min(obb_len_x, obb_len_y) / 2.0f;
                            float chosen_z_local = (new_z_axis.dot(Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z) - lidar_origin) > 0) ? max_point_OBB.z : min_point_OBB.z;

                            std::vector<Eigen::Vector3f> local_face_corners;
                            local_face_corners.push_back(Eigen::Vector3f(actual_half_len_long, actual_half_len_short, chosen_z_local));
                            local_face_corners.push_back(Eigen::Vector3f(actual_half_len_long, -actual_half_len_short, chosen_z_local));
                            local_face_corners.push_back(Eigen::Vector3f(-actual_half_len_long, actual_half_len_short, chosen_z_local));

                            std::vector<Eigen::Vector3f> final_ordered_corners_global;
                            for (const auto &local_corner : local_face_corners)
                            {
                                Eigen::Vector3f global_corner = final_rotational_matrix_OBB * local_corner + Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z);
                                final_ordered_corners_global.push_back(global_corner);
                            }

                            current_frame_corners.push_back(final_ordered_corners_global);
                        }
                    }
                }
                frame_plane_clouds.push_back(filtered_cloud);
                just_using_log_count++;
            }
            detected_plane_clouds.push_back(frame_plane_clouds);
            all_corners.push_back(current_frame_corners);
        }
#ifdef LOOK_DEBUG
        RCLCPP_INFO(this->get_logger(), "--- All Corners Data ---");
        for (size_t frame_i = 0; frame_i < all_corners.size(); ++frame_i)
        {
            RCLCPP_INFO(this->get_logger(), "Frame %lu:", frame_i);
            for (size_t lidar_i = 0; lidar_i < all_corners[frame_i].size(); ++lidar_i)
            {
                RCLCPP_INFO(this->get_logger(), "  Lidar %lu:", lidar_i);
                if (all_corners[frame_i][lidar_i].empty())
                {
                    RCLCPP_INFO(this->get_logger(), "    No corners detected for this lidar.");
                }
                else
                {
                    for (size_t corner_idx = 0; corner_idx < all_corners[frame_i][lidar_i].size(); ++corner_idx)
                    {
                        const auto &corner = all_corners[frame_i][lidar_i][corner_idx];
                        RCLCPP_INFO(this->get_logger(), "    Corner %lu: (%.3f, %.3f, %.3f)",
                                    corner_idx, corner.x(), corner.y(), corner.z());
                    }
                }
            }
        }
        RCLCPP_INFO(this->get_logger(), "--- End All Corners Data ---");
#endif

        RCLCPP_INFO(this->get_logger(), "3. Run calibrate using SVD...");

        std::vector<Eigen::Matrix3f> rotations_frame;
        std::vector<Eigen::Vector3f> translations_frame;

        for (int frame_i = 0; frame_i < frame_count; frame_i++)
        {
            if (all_corners[frame_i].size() >= 2 && !all_corners[frame_i][0].empty() && !all_corners[frame_i][1].empty())
            {
                const std::vector<Eigen::Vector3f> &ref_corner = all_corners[frame_i][0];
                const std::vector<Eigen::Vector3f> &src_corner = all_corners[frame_i][1];

                Eigen::Vector3f src_mean = Eigen::Vector3f::Zero();
                Eigen::Vector3f ref_mean = Eigen::Vector3f::Zero();

                for (int i = 0; i < src_corner.size(); i++)
                {
                    src_mean += src_corner[i];
                    ref_mean += ref_corner[i];
                }
                src_mean /= src_corner.size();
                ref_mean /= ref_corner.size();

                Eigen::Matrix3f Homogeneous = Eigen::Matrix3f::Zero();
                for (int i = 0; i < src_corner.size(); i++)
                {
                    Homogeneous += (src_corner[i] - src_mean) * (ref_corner[i] - ref_mean).transpose();
                }

                Eigen::JacobiSVD<Eigen::Matrix3f> svd_corner(Homogeneous, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3f U = svd_corner.matrixU();
                Eigen::Matrix3f V = svd_corner.matrixV();
                Eigen::Matrix3f R = V * U.transpose();
                if (R.determinant() < 0)
                {
                    V.col(2) *= -1;
                    R = V * U.transpose();
                }
                Eigen::Vector3f t = ref_mean - R * src_mean;

                rotations_frame.push_back(R);
                translations_frame.push_back(t);

#ifdef LOOK_DEBUG
                std::stringstream ss_r, ss_t;
                ss_r << "\n[Frame" << frame_i << "] Rotation matrix : \n"
                     << R;
                ss_t << "\nTranslation vector : \n"
                     << t.transpose();
                RCLCPP_INFO(this->get_logger(), "%s", ss_r.str().c_str());
                RCLCPP_INFO(this->get_logger(), "%s", ss_t.str().c_str());
#endif
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "[Frame %d] Not enough valid corners detected for SVD calibration.", frame_i);
            }

#ifdef LOOK_DEBUG
            int img_size = 500;
            int margin = 50;
            cv::Mat image(img_size + 2 * margin, img_size + 2 * margin, CV_8UC3, cv::Scalar(255, 255, 255));

            std::vector<Eigen::Vector3f> all_points_for_vis;
            for (int i = 0; i < number; ++i)
            {
                if (!detected_plane_clouds[frame_i][i]->points.empty())
                {
                    for (const auto &pt : detected_plane_clouds[frame_i][i]->points)
                    {
                        all_points_for_vis.emplace_back(Eigen::Vector3f(pt.x, pt.y, pt.z));
                    }
                }
            }
            float min_x = std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_x = std::numeric_limits<float>::lowest();
            float max_y = std::numeric_limits<float>::lowest();

            if (!all_points_for_vis.empty())
            {
                for (const auto &pt : all_points_for_vis)
                {
                    min_x = std::min(min_x, pt.x());
                    max_x = std::max(max_x, pt.x());
                    min_y = std::min(min_y, pt.y());
                    max_y = std::max(max_y, pt.y());
                }
            }
            else
            {
                min_x = -1.0;
                max_x = 1.0;
                min_y = -1.0;
                max_y = 1.0;
            }

            float range_x = max_x - min_x;
            float range_y = max_y - min_y;
            float scale = img_size / std::max(range_x, range_y);
            if (scale == 0 || std::isinf(scale))
                scale = img_size;

            for (int lidar_i = 0; lidar_i < number; ++lidar_i)
            {
                const auto &plane = detected_plane_clouds[frame_i][lidar_i];
                cv::Scalar color = (lidar_i == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);

                for (const auto &pt : plane->points)
                {
                    int x = static_cast<int>((pt.x - min_x) * scale + margin);
                    int y = image.rows - static_cast<int>((pt.y - min_y) * scale + margin);
                    if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
                    {
                        image.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0], color[1], color[2]);
                    }
                }

                if (lidar_i < all_corners[frame_i].size() && !all_corners[frame_i][lidar_i].empty())
                {
                    const auto &corners = all_corners[frame_i][lidar_i];
                    cv::Scalar corner_color;
                    switch (lidar_i)
                    {
                    case 0:
                        corner_color = cv::Scalar(0, 0, 255);
                        break;
                    case 1:
                        corner_color = cv::Scalar(255, 0, 255);
                        break;
                    case 2:
                        corner_color = cv::Scalar(255, 255, 0);
                        break;
                    default:
                        corner_color = cv::Scalar(0, 255, 255);
                        break;
                    }

                    const auto &pt = corners[0];
                    int x = static_cast<int>((pt.x() - min_x) * scale + margin);
                    int y = image.rows - static_cast<int>((pt.y() - min_y) * scale + margin);
                    cv::circle(image, cv::Point(x, y), 5, corner_color, -1);

                    const auto &next_pt = corners[1];
                    int nx = static_cast<int>((next_pt.x() - min_x) * scale + margin);
                    int ny = image.rows - static_cast<int>((next_pt.y() - min_y) * scale + margin);
                    cv::line(image, cv::Point(x, y), cv::Point(nx, ny), corner_color, 2);
                }
            }

            std::string save_all_path = origin_path_ + "visual_frame_" + std::to_string(frame_i) + ".png";
            cv::imwrite(save_all_path, image);
            RCLCPP_INFO(this->get_logger(), "Saved visualization image to: %s", save_all_path.c_str());
#endif
        }

        RCLCPP_INFO(this->get_logger(), "4. Global Calibration Optimization (Levenberg-Marquardt)");

        Eigen::Matrix3d R_optimized = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_optimized = Eigen::Vector3d::Zero();

        if (rotations_frame.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No valid frames found for global calibration. Optimization skipped.");
        }
        else
        {
            for (const auto &t_val : translations_frame)
            {
                t_optimized += t_val.cast<double>();
            }
            t_optimized /= static_cast<double>(translations_frame.size());

            Eigen::Quaterniond q_init(0, 0, 0, 0);
            for (const auto &R_val : rotations_frame)
            {
                Eigen::Quaterniond q(R_val.cast<double>()); // Cast to double
                if (q.dot(q_init) < 0)
                {
                    q.coeffs() *= -1.0;
                }
                q_init.coeffs() += q.coeffs();
            }
            q_init.normalize();
            R_optimized = q_init.toRotationMatrix();

#ifdef LOOK_DEBUG
            std::stringstream ss_initial_r, ss_initial_t;
            ss_initial_r << R_optimized.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", ""));
            ss_initial_t << t_optimized.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", ""));

            RCLCPP_INFO(this->get_logger(), "Initial R (from averaging SVD): \n%s", ss_initial_r.str().c_str());
            RCLCPP_INFO(this->get_logger(), "Initial t (from averaging SVD): \n%s", ss_initial_t.str().c_str());
#endif

            // Nonlinear Optimization (Levenberg-Marquardt)
            const int max_iterations = 100;
            const double convergence_threshold = 1e-6; // Threshold for parameter change
            double lambda = 1e-3;                      // Initial damping parameter
            double nu = 2.0;                           // Factor for increasing lambda

            Eigen::Matrix3d best_R = R_optimized; // Changed to double
            Eigen::Vector3d best_t = t_optimized; // Changed to double
            double best_error = std::numeric_limits<double>::max();

            for (int iter = 0; iter < max_iterations; ++iter)
            {
                Eigen::MatrixXd J(0, 6); // Jacobian matrix
                Eigen::VectorXd r(0);    // Residual vector

                // Calculate total number of residuals for dynamic resizing
                int total_residuals = 0;
                for (int frame_i = 0; frame_i < frame_count; ++frame_i)
                {
                    if (all_corners[frame_i].size() >= 2 && all_corners[frame_i][0].size() == 3 && all_corners[frame_i][1].size() == 3)
                    {
                        total_residuals += 3 * 3; // 3 corners * 3 dimensions
                    }
                }
                J.resize(total_residuals, 6);
                r.resize(total_residuals);

                int row_idx = 0;
                for (int frame_i = 0; frame_i < frame_count; ++frame_i)
                {
                    if (all_corners[frame_i].size() >= 2 && all_corners[frame_i][0].size() == 3 && all_corners[frame_i][1].size() == 3)
                    {
                        const std::vector<Eigen::Vector3f> &ref_corners_frame = all_corners[frame_i][0]; // Lidar 0 corners
                        const std::vector<Eigen::Vector3f> &src_corners_frame = all_corners[frame_i][1]; // Lidar 1 corners

                        for (int i = 0; i < 3; ++i)
                        {                                                                      // Iterate over 3 corners
                            const Eigen::Vector3d P_ref = ref_corners_frame[i].cast<double>(); // Cast to double
                            const Eigen::Vector3d P_src = src_corners_frame[i].cast<double>(); // Cast to double

                            // Current transformed point
                            Eigen::Vector3d P_transformed = R_optimized * P_src + t_optimized;

                            // Residual vector for this point
                            Eigen::Vector3d residual_pt = P_ref - P_transformed;
                            r.segment<3>(row_idx) = residual_pt;

                            // Jacobian for this point (3x6 matrix)
                            Eigen::Matrix3d skew_P_transformed_src = Eigen::Matrix3d::Zero();
                            skew_P_transformed_src << 0, -P_transformed.z(), P_transformed.y(),
                                P_transformed.z(), 0, -P_transformed.x(),
                                -P_transformed.y(), P_transformed.x(), 0;

                            // Jacobian block for rotation (3x3)
                            J.block<3, 3>(row_idx, 0) = -skew_P_transformed_src;
                            // Jacobian block for translation (3x3)
                            J.block<3, 3>(row_idx, 3) = -Eigen::Matrix3d::Identity();

                            row_idx += 3;
                        }
                    }
                }

                // Calculate current error before update
                double current_total_error = r.norm();
                if (iter == 0)
                {
                    best_error = current_total_error;
                }

                // Solve the normal equations: (J^T * J + lambda * I) * delta_params = J^T * r
                Eigen::Matrix<double, 6, 6> Hessian = J.transpose() * J;
                Eigen::Matrix<double, 6, 1> gradient = J.transpose() * r;

                Eigen::Matrix<double, 6, 1> delta_params;
                Eigen::Matrix<double, 6, 6> damped_Hessian = Hessian + lambda * Eigen::Matrix<double, 6, 6>::Identity();
                delta_params = damped_Hessian.ldlt().solve(gradient); // Solve using LDLT decomposition

                Eigen::Vector3d delta_rotation_vec = delta_params.head<3>();    // Already double
                Eigen::Vector3d delta_translation_vec = delta_params.tail<3>(); // Already double

                // Evaluate the new parameters
                Eigen::Matrix3d R_new = Eigen::AngleAxisd(delta_rotation_vec.norm(), delta_rotation_vec.normalized()).toRotationMatrix() * R_optimized;
                Eigen::Vector3d t_new = t_optimized + delta_translation_vec;

                // Calculate error with new parameters
                Eigen::VectorXd r_new(total_residuals);
                int new_row_idx = 0;
                for (int frame_i = 0; frame_i < frame_count; ++frame_i)
                {
                    if (all_corners[frame_i].size() >= 2 && all_corners[frame_i][0].size() == 3 && all_corners[frame_i][1].size() == 3)
                    {
                        const std::vector<Eigen::Vector3f> &ref_corners_frame = all_corners[frame_i][0];
                        const std::vector<Eigen::Vector3f> &src_corners_frame = all_corners[frame_i][1];
                        for (int i = 0; i < 3; ++i)
                        {
                            const Eigen::Vector3d P_ref = ref_corners_frame[i].cast<double>();
                            const Eigen::Vector3d P_src = src_corners_frame[i].cast<double>();
                            Eigen::Vector3d P_transformed_new = R_new * P_src + t_new;
                            r_new.segment<3>(new_row_idx) = P_ref - P_transformed_new;
                            new_row_idx += 3;
                        }
                    }
                }
                double new_total_error = r_new.norm();

                // Levenberg-Marquardt damping update
                // Calculate actual reduction vs. expected reduction
                double actual_reduction = current_total_error * current_total_error - new_total_error * new_total_error;
                double expected_reduction = (gradient.transpose() * delta_params)(0, 0) - 0.5 * (delta_params.transpose() * Hessian * delta_params)(0, 0);

                double gain_ratio = 0.0;
                if (expected_reduction > 1e-9)
                { // Avoid division by zero or very small expected reduction
                    gain_ratio = actual_reduction / expected_reduction;
                }
                else
                {
                    gain_ratio = (actual_reduction > 0) ? 1.0 : -1.0; // If expected is zero, check if actual improved
                }

#ifdef LOOK_DEBUG
                RCLCPP_INFO(this->get_logger(), "Iteration %d: Current Error = %.6f, New Error = %.6f, Gain Ratio = %.6f, Lambda = %.6f",
                            iter, current_total_error, new_total_error, gain_ratio, lambda);
#endif

                if (gain_ratio > 0)
                { // Actual reduction is positive, step is good
                    R_optimized = R_new;
                    t_optimized = t_new;
                    lambda = std::max(lambda * 0.1, 1e-7); // Decrease lambda
                    nu = 2.0;
                    if (new_total_error < best_error)
                    {
                        best_error = new_total_error;
                        best_R = R_optimized;
                        best_t = t_optimized;
                    }
                }
                else
                {                 // Actual reduction is zero or negative, step is bad
                    lambda *= nu; // Increase lambda
                    nu *= 2.0;
                }

                if (delta_params.norm() < convergence_threshold || lambda > 1e10)
                { // Also add a max lambda to prevent explosion
                    RCLCPP_INFO(this->get_logger(), "Step rejected. Increasing lambda to %.6f", lambda);
                    RCLCPP_INFO(this->get_logger(), "Optimization converged or lambda exploded.");
                    break;
                }
            }
            R_optimized = best_R; // Use the best parameters found
            t_optimized = best_t;

            std::stringstream ss_final_r, ss_final_t;
            ss_final_r << R_optimized.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", ""));
            ss_final_t << t_optimized.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", ""));
            RCLCPP_INFO(this->get_logger(), "\n-----------\nGlobal Calibrated Rotation Matrix (Lidar1 to Lidar0) - Optimized:\n%s", ss_final_r.str().c_str());
            RCLCPP_INFO(this->get_logger(), "Global Calibrated Translation Vector (Lidar1 to Lidar0) - Optimized:\n%s", ss_final_t.str().c_str());
            RCLCPP_INFO(this->get_logger(), "-----------\n");
        }
    }
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MultiLidarCalibNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
