#include "jmh_utils/calc_convert.hpp"

#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace jmh_utils
{
    cv::Mat vector2Mat(const std::vector<std::vector<double>> &input)
    {
        std::cout << "convert vector<vector> to cv::Mat" << std::endl;
        std::size_t columns = input.front().size();
        std::size_t rows = input.size();

        cv::Mat output(rows, columns, CV_64F);
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < columns; j++)
            {
                output.at<double>(i, j) = input[i][j];
            }
        }
        return output;
    }

    cv::Mat vector2Mat(const std::vector<double> &input)
    {
        std::cout << "convert vector to cv::Mat" << std::endl;
        std::size_t nums = input.size();

        cv::Mat output(nums, 1, CV_64F);

        for (size_t j = 0; j < nums; j++)
        {
            output.at<double>(j, 0) = input[j];
        }

        return output;
    }

    std::vector<Eigen::Vector4d> convertPcl2Vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pointcloud)
    {
        std::vector<Eigen::Vector4d> point_vector;
        point_vector.reserve(pointcloud->points.size());
        for (const auto &p : pointcloud->points)
        {
            point_vector.push_back(Eigen::Vector4d(p.x, p.y, p.z, p.intensity));
        }
        return point_vector;
    }
}