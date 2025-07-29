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

namespace fs = std::filesystem;

class MultiLidarCalibNode : public rclcpp::Node
{
public:
    MultiLidarCalibNode()
        : Node("multi_lidar_calib")
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
        std::string data_dir = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/multi_lidar_calib";
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

        // 1. Load File
        RCLCPP_INFO(this->get_logger(), "1-1. Load .pcd files...");
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
        // 2. Detect Plane
        for (int frame_index = 0; frame_index < frame_count; frame_index++)
        {
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> frame_plane_clouds;
            for (int lidar_index = 0; lidar_index < number; lidar_index++)
            {
                RCLCPP_INFO(this->get_logger(), "Processing data from LIDAR %d (%d frame)", lidar_index, frame_index);
                pcl::PointCloud<pcl::PointXYZ>::Ptr frame_cloud = loaded_clouds[frame_index][lidar_index];

                RCLCPP_INFO(this->get_logger(), "2-1. Set cloud ROI...");
                pcl::CropBox<pcl::PointXYZ> crop;
                crop.setInputCloud(frame_cloud);
                crop.setMin(Eigen::Vector4f(-4.0, -2.0, -0.7, 1.0)); // X, Y, Z, 1.0
                crop.setMax(Eigen::Vector4f(-2.0, 1.0, 2.0, 1.0));
                crop.filter(*frame_cloud);

                RCLCPP_INFO(this->get_logger(), "2-2. Set segmentation model...");
                pcl::SACSegmentation<pcl::PointXYZ> seg;
                pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr inliners(new pcl::PointIndices);
                seg.setOptimizeCoefficients(true);
                seg.setModelType(pcl::SACMODEL_PLANE);
                seg.setMethodType(pcl::SAC_RANSAC);
                seg.setDistanceThreshold(0.001);
                seg.setInputCloud(frame_cloud);
                seg.segment(*inliners, *coefficients);

                auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                RCLCPP_INFO(this->get_logger(), "2-3. Post-processing...");
                for (const auto &pt : frame_cloud->points)
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

                pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
                kdtree.setInputCloud(frame_cloud);

                pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);

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
                            pcl::PointXYZ pt_rgb;
                            pt_rgb.x = pt.x;
                            pt_rgb.y = pt.y;
                            pt_rgb.z = pt.z;

                            filtered->points.push_back(pt_rgb);
                        }
                    }
                }

                filtered->width = filtered->size();
                filtered->height = 1;
                filtered->is_dense = true;

                RCLCPP_INFO(this->get_logger(), "2-4. Extract plane points...");
                pcl::ExtractIndices<pcl::PointXYZ> extract;
                extract.setInputCloud(filtered);
                extract.setIndices(inliners);
                extract.setNegative(true);
                extract.filter(*filtered);

                frame_plane_clouds.push_back(filtered);
            }
            detected_plane_clouds.push_back(frame_plane_clouds);
        }

        RCLCPP_INFO(this->get_logger(), "3-1. Run calibrate...");
        for (int frame_i = 0; frame_i < frame_count; frame_i++)
        {
            std::vector<std::vector<Eigen::Vector3f>> find_corners;
            for (int lidar_i = 0; lidar_i < number; lidar_i++)
            {
                auto plane_cloud = detected_plane_clouds[frame_i][lidar_i];

                Eigen::Vector4f centroid4f;
                pcl::compute3DCentroid(*plane_cloud, centroid4f);
                Eigen::Vector3f centroid = centroid4f.head<3>();

                Eigen::Matrix3f covariance;
                pcl::computeCovarianceMatrixNormalized(*plane_cloud, centroid4f, covariance);

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig_solver(covariance);
                Eigen::Matrix3f eig_vecs = eig_solver.eigenvectors();

                if (eig_vecs.col(0).dot(Eigen::Vector3f::UnitX()) < 0)
                {
                    eig_vecs.col(0) = -eig_vecs.col(0);
                }

                eig_vecs.col(1) = eig_vecs.col(2).cross(eig_vecs.col(0)).normalized();
                Eigen::Matrix3f R_pca = eig_vecs;

                std::vector<Eigen::Vector3f> projected_pca;
                for (const auto &pts : plane_cloud->points)
                {
                    Eigen::Vector3f p(pts.x, pts.y, pts.z);
                    projected_pca.push_back(R_pca.transpose() * (p - centroid));
                }

                std::vector<cv::Point2f> points_2d;
                for (const auto &pt : projected_pca)
                {
                    points_2d.emplace_back(pt.y(), pt.z());
                }

                cv::RotatedRect rectangle = cv::minAreaRect(points_2d);
                cv::Point2f rect_pts[4];
                rectangle.points(rect_pts);

                std::vector<Eigen::Vector3f> rect_corner;
                for (int i = 0; i < 4; i++)
                {
                    Eigen::Vector3f point_local(0.0f, rect_pts[i].x, rect_pts[i].y);
                    Eigen::Vector3f point_global = R_pca * point_local + centroid;
                    rect_corner.push_back(point_global);
                }

                find_corners.push_back(rect_corner);

                // 2D 점과 꼭짓점 그리기 위한 이미지 준비
                int img_size = 500;
                cv::Mat img = cv::Mat::zeros(img_size, img_size, CV_8UC3);

                // pca 투영 점을 이미지 좌표로 변환 함수
                auto toImgCoord = [&](const Eigen::Vector3f &pt) -> cv::Point
                {
                    float scale = 100.0; // 임의 scale 조절
                    int y = static_cast<int>(pt.y() * scale + img_size / 2);
                    int z = static_cast<int>(pt.z() * scale + img_size / 2);
                    return cv::Point(y, img_size - z); // OpenCV는 좌상단이 (0,0)
                };

                // 점들 그리기
                for (const auto &pt : projected_pca)
                {
                    cv::circle(img, toImgCoord(pt), 2, cv::Scalar(255, 255, 255), -1);
                }

                // 사각형 꼭짓점 그리기
                for (int i = 0; i < 4; i++)
                {
                    cv::line(img, toImgCoord(Eigen::Vector3f(0, rect_pts[i].x, rect_pts[i].y)), toImgCoord(Eigen::Vector3f(0, rect_pts[(i + 1) % 4].x, rect_pts[(i + 1) % 4].y)), cv::Scalar(0, 255, 0), 2);
                }

                // 이미지 윈도우 표시
                cv::imshow("PCA Plane Points", img);
                std::string img_filename = origin_path_ + "img_" + std::to_string(lidar_i) + "_" + std::to_string(frame_i) + ".png";
                cv::imwrite(img_filename, img);
                cv::waitKey(1); // 짧게 대기해서 계속 띄움
            }

            if (find_corners.size() >= 2)
            {
                const std::vector<Eigen::Vector3f> &ref_corner = find_corners[0];
                const std::vector<Eigen::Vector3f> &src_corner = find_corners[1];

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
                Eigen::Matrix3f R = svd_corner.matrixV() * svd_corner.matrixU().transpose();
                Eigen::Vector3f t = ref_mean - R * src_mean;

                std::stringstream ss_r, ss_t;
                ss_r << "\n[Frame" << frame_i << "] Rotation matrix : \n"
                     << R;
                ss_t << "\nTranslation vector : \n"
                     << t.transpose();
                RCLCPP_INFO(this->get_logger(), "%s", ss_r.str().c_str());
                RCLCPP_INFO(this->get_logger(), "%s", ss_t.str().c_str());
            }
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
