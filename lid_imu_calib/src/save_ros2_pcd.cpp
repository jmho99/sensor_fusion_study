#include "save_ros2_pcd.hpp"
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
        int lid_res = 128,
        std::vector<std::string> wanted_fields = {"x", "y", "z", "intensity", "t"},
        std::string data_type = "ascii")
    {

        // =========================
        // No.1 파일 확인
        // =========================
        if (std::filesystem::exists(file_path) == false)
        {
            throw std::runtime_error("Not exist file path, make file");
        }

        std::ofstream init_output(file_path, std::ios::out | std::ios::trunc);
        if (!init_output.is_open())
        {
            throw std::runtime_error("Cannot open output PCD file.");
        }

        // =========================
        // No.2 필드 추출 및 체크
        // =========================
        std::vector<std::string> field_names;
        field_names.reserve(msg.fields.size());

        for (const auto &field : msg.fields)
        {
            field_names.push_back(field.name);
        }

        for (const auto &check : wanted_fields)
        {
            if (std::find(field_names.start(), field_names.end(), check) == field_names.end())
            {
                throw std::runtime_error("Not found '" + check + "' in field");
            }
        }

        // =========================
        // No.3 field offset 찾기
        // =========================
        auto get_offset = [&](const string &name) -> int
        {
            for (const auto &f_info : msg.fields)
                if (f_info.name == name)
                    return f_info.offset;
            return -1;
        };
        (void)lid_res;
        (void)data_type;
        /*
                 for key in wanted_fields:
                     if key not in field_names:
                         raise ValueError(f"Required field '{key}' not found in msg.")

                point_iter = pc2.read_points(msg, field_names=wanted_fields, skip_nans=False)
                points = list(point_iter)
                n_points = len(points)

                with open(file_path, "w") as f:
                    enter = "\n"

                    fields_str = []
                    size_str = []
                    type_str = []
                    count_str = []

                    for name in ["x", "y", "z"]:
                        fields_str.append(name)
                        size_str.append("4")
                        type_str.append("F")
                        count_str.append("1")

                    if "intensity" in wanted_fields:
                        fields_str.append("intensity")
                        size_str.append("4")
                        type_str.append("F")
                        count_str.append("1")

                    if "t" in wanted_fields:
                        fields_str.append("t")
                        size_str.append("4")
                        type_str.append("U")
                        count_str.append("1")

                    fields_str.append("sec")
                    size_str.append("4")
                    type_str.append("U")
                    count_str.append("1")

                    fields_str.append("nanosec")
                    size_str.append("4")
                    type_str.append("U")
                    count_str.append("1")

                    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                    f.write("VERSION 0.7\n")
                    f.write("FIELDS " + " ".join(fields_str) + enter)
                    f.write("SIZE " + " ".join(size_str) + enter)
                    f.write("TYPE " + " ".join(type_str) + enter)
                    f.write("COUNT " + " ".join(count_str) + enter)
                    f.write(f"WIDTH {int(n_points/lid_res)}\n")
                    f.write(f"HEIGHT {lid_res}\n")
                    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                    f.write(f"POINTS {n_points}\n")
                    f.write(f"DATA {data_type}\n")

        #-- -- -- -- DATA -- -- -- -- -
                    for p in points:
                        x = p[0]
                        y = p[1]
                        z = p[2]
                        lin_vals = [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]

                        idx = 3

                        if "intensity" in wanted_fields:
                            intensity = p[idx]
                            lin_vals.append(f"{int(intensity)}")
                            idx += 1

                        if "t" in wanted_fields:
                            t_offset = p[idx]
                            lin_vals.append(f"{int(t_offset)}")
                        lin_vals.append(f"{int(self.t_sec)}")
                        lin_vals.append(f"{int(self.t_nsec)}")
                        f.write(" ".join(lin_vals) + enter)
                self.get_logger().info(f"Saved: {file_path}")
        */
    }
}