#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import os
import numpy as np


class SavePCD(Node):
    def __init__(self):
        super().__init__("ouster_point_time_logger")

        self.declare_parameter("pointcloud_topic", "/ouster/points")

        topic = (
            self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        )
        self.frame_idx = 0
        self.output_dir = "/home/antlab/ROS2/calib_ws/pcd_files/"
        self.subscription = self.create_subscription(
            PointCloud2, topic, self.cloud_callback, qos_profile_sensor_data
        )

    def cloud_callback(self, msg: PointCloud2):
        self.t_sec = msg.header.stamp.sec
        self.t_nsec = msg.header.stamp.nanosec
        save_pcd_path = self.make_pcd_filename()
        self.save_pcd_from_msg(msg, save_pcd_path)
        self.frame_idx += 1

    def make_pcd_filename(self):

        pcd_name = f"frame_{self.frame_idx:04d}.pcd"
        pcd_path = self.output_dir + pcd_name

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(pcd_path):
            return pcd_path
        counter = 1
        new_pcd_path = self.output_dir + f"frame_{self.frame_idx:04d}({counter}).pcd"
        while os.path.exists(new_pcd_path):
            counter += 1
            new_pcd_path = (
                self.output_dir + f"frame_{self.frame_idx:04d}({counter}).pcd"
            )

        return new_pcd_path

    def save_pcd_from_msg(
        self,
        msg: PointCloud2,
        file_path: str,
        wanted_fields=["x", "y", "z", "intensity", "t"],
        lid_res=128,
        data_type="ascii",
    ):
        field_names = [f.name for f in msg.fields]
        for key in wanted_fields:
            if key not in field_names:
                raise ValueError(f"Required field '{key}' not found in msg.")

        point_iter = pc2.read_points(msg, field_names=wanted_fields, skip_nans=False)
        points = list(point_iter)
        n_points = len(points)

        with open(file_path, "w") as f:
            enter = "\n"
            # --------HEADER---------
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

            # --------DATA---------
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


def main(args=None):
    rclpy.init(args=args)
    node = SavePCD()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
