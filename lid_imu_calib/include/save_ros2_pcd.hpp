#ifndef SAVE_ROS2_PCD_HPP
#define SAVE_ROS2_PCD_HPP

#include "sensor_msgs/msg/point_cloud2.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

namespace jmh_utils
{
    void save_pcd_from_msg(
        const sensor_msgs::msg::PointCloud2 &msg,
        const std::string &file_path,
        int lid_res,
        std::vector<std::string> wanted_fields,
        std::string data_type);
}
#endif