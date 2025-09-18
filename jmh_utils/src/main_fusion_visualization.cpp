#include "jmh_utils/main_fusion_visualization.hpp"

#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace jmh_utils
{
    cv::Mat resultCamLidarFusion(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_in_cam,
                                 const cv::Mat &image_in, cv::Mat &image_out,
                                 cv::Mat intrinsic, cv::Mat distortion,
                                 const bool use_undistorte_image)
    {
        image_out = image_in.clone();

        if (use_undistorte_image == true)
        {
            intrinsic = cv::getOptimalNewCameraMatrix(intrinsic, distortion,
                                                      image_out.size(), 0 /*alpha*/, image_out.size());
        }

        const cv::Mat &K = intrinsic;
        double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);

        float min_horizontal_dist = std::numeric_limits<float>::max();
        float max_horizontal_dist = std::numeric_limits<float>::min();

        for (const auto &pt : cloud_in_cam->points)
        {
            if (pt.z <= 0.0 || pt.z > 30.0f)
                continue;

            float current_horizontal_dist = std::sqrt(pt.x * pt.x + pt.y * pt.y);
            min_horizontal_dist = std::min(min_horizontal_dist, current_horizontal_dist);
            max_horizontal_dist = std::max(max_horizontal_dist, current_horizontal_dist);
        }

        if (max_horizontal_dist - min_horizontal_dist < 1e-6f)
        {
            min_horizontal_dist = 0.0f;
            max_horizontal_dist = 120.0f;
        }

        for (const auto &pt : cloud_in_cam->points)
        {
            if (pt.z <= 0.0 || pt.z > 12.0f)
                continue;

            double x = pt.x;
            double y = pt.y;
            double z = pt.z;

            int u = static_cast<int>((fx * x / z) + cx);
            int v = static_cast<int>((fy * y / z) + cy);

            if (u >= 0 && u < image_out.cols && v >= 0 && v < image_out.rows)
            {
                float horizontal_dist = std::sqrt(x * x + z * z);
                float t = (horizontal_dist - min_horizontal_dist) / (max_horizontal_dist - min_horizontal_dist);
                t = std::clamp(t, 0.0f, 1.0f);

                float h = t * 240.0f;
                float s = 1.0f, v_val = 1.0f;
                float c = v_val * s;
                float x_hsv = c * (1 - std::fabs(fmod(h / 60.0f, 2) - 1));

                float m = v_val - c;

                float r = 0, g = 0, b = 0;
                if (h < 60)
                {
                    r = c;
                    g = x_hsv;
                    b = 0;
                }
                else if (h < 120)
                {
                    r = x_hsv;
                    g = c;
                    b = 0;
                }
                else if (h < 180)
                {
                    r = 0;
                    g = c;
                    b = x_hsv;
                }
                else if (h < 240)
                {
                    r = 0;
                    g = x_hsv;
                    b = c;
                }
                else
                {
                    r = 0, g = 0, b = 0;
                }

                uint8_t R = static_cast<uint8_t>((r + m) * 255);
                uint8_t G = static_cast<uint8_t>((g + m) * 255);
                uint8_t B = static_cast<uint8_t>((b + m) * 255);

                cv::Scalar color = cv::Scalar(B, G, R);
                cv::circle(image_out, cv::Point(u, v), 5, color, -1);
            }
        }

        return image_out;
    }
}