#ifndef SAVE_HPP
#define SAVE_HPP

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace calib_utils
{
    template <typename T>
    void saveFile(const std::string extension,
                  const std::string &directory,
                  const std::string &filename,
                  const T &data);

    void saveImageFile(const std::string extension,
                       const std::string &directory,
                       const int counter,
                       const cv::Mat &data);

    void savePcdFile(const std::string extension,
                     const std::string &directory,
                     const int counter,
                     const pcl::PointCloud<pcl::PointXYZI>::Ptr &data);

    void savePcdFile(const std::string extension,
                     const std::string &directory,
                     const int number,
                     const int counter,
                     const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &data);
}

#endif