#include "save.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

template <typename T>
void saveFile(const std::string extension,
              const std::string &directory,
              const std::string &filename,
              const T &data)
{
    std::string file_path = directory + filename + "." + extension;
    std::ofstream ofs(file_path);

    if (!ofs.is_open())
    {
        std::cout << "[WARN] No current camera frame! Cannot save file." << std::endl;
    }

    ofs << data << std::endl;
    ofs.close();
    std::cout << "[INFO] Successfully saved the file." << std::endl;
}

template void saveFile<std::string>(const std::string,
                                    const std::string &,
                                    const std::string &,
                                    const std::string &);

void saveImageFile(const std::string extension,
                   const std::string &directory,
                   const int counter,
                   const cv::Mat &data)
{
    if (data.empty())
    {
        std::cout << "[WARN] No current camera frame! Cannot save image." << std::endl;
        return;
    }

    std::string img_path = directory + "img_" + std::to_string(counter) + "." + extension;
    cv::imwrite(img_path, data);
    std::cout << "[INFO] Successfully saved the image." << std::endl;
}

void savePcdFile(const std::string extension,
                 const std::string &directory,
                 const int counter,
                 const pcl::PointCloud<pcl::PointXYZI>::Ptr &data)
{
    if (data->empty())
    {
        std::cout << "[WARN] No current lidar point cloud! Cannot save pointcloud." << std::endl;
        return;
    }
    std::string pcd_path = directory + "pcd_" + std::to_string(counter) + "." + extension;
    pcl::io::savePCDFile(pcd_path, *data);
    std::cout << "[INFO] Successfully saved the pointcloud." << std::endl;
}

void savePcdFile(const std::string extension,
                 const std::string &directory,
                 const int number,
                 const int counter,
                 const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &data)
{
    if (data.empty())
    {
        std::cout << "[WARN] No current lidar point cloud! Cannot save pointcloud." << std::endl;
        return;
    }

    for (int i = 0; i < number; i++)
    {
        std::string pcd_path = directory + "lidar" + std::to_string(i) + "_" + std::to_string(counter) + +"." + extension;
        pcl::io::savePCDFile(pcd_path, *data[i]);
    }
    std::cout << "[INFO] Successfully saved the pointcloud." << std::endl;
}