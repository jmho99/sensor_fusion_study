#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import csv
import os


class OusterPointTimeLogger(Node):
    def __init__(self):
        super().__init__("ouster_point_time_logger")

        # 파라미터: pointcloud 토픽 이름
        self.declare_parameter("pointcloud_topic", "/ouster/points")
        topic = (
            self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            topic,
            self.cloud_callback,
            qos_profile_sensor_data,  # QoS depth
        )

        self.get_logger().info(f"Subscribing to PointCloud2 topic: {topic}")

        # t 필드 이름 저장 (처음 한 번만 찾음)
        self.time_field_name = None

        self.csv_path = "/home/antlab/ROS2/calib_ws/point_per_time.csv"
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0

        self.csv_file = open(self.csv_path, mode="a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

    def cloud_callback(self, msg: PointCloud2):
        # 처음 들어온 메시지에서 fields 확인해서 t 필드 찾기
        if self.time_field_name is None:
            field_names = [f.name for f in msg.fields]
            self.get_logger().info(f"PointCloud2 fields: {field_names}")

            # 후보 이름들: 't', 'time', 'timestamp' 중 하나 있을 거라고 가정
            for cand in ["t", "time", "timestamp"]:
                if cand in field_names:
                    self.time_field_name = cand
                    break

            if self.time_field_name is None:
                self.get_logger().error(
                    "시간 필드(t, time, timestamp)를 찾지 못했습니다. "
                    "이 PointCloud2에는 포인트별 time 정보가 없는 것 같습니다."
                )
                return
            else:
                self.get_logger().info(
                    f'Per-point time field로 "{self.time_field_name}" 를 사용합니다.'
                )

        # 프레임 기준 시간 (scan_ts) [sec]
        t_frame = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 포인트 읽기: x, y, z, t_offset
        fields_to_read = ("x", "y", "z", self.time_field_name)

        points = pc2.read_points(msg, field_names=fields_to_read, skip_nans=False)

        self.get_logger().info("--- New PointCloud frame ---")

        for i, p in enumerate(points):

            x, y, z, t_offset = p

            # 여기서는 t_offset 단위를 ns로 가정 (Ouster ROS 기본)
            # 만약 초 단위라면 1e-9 곱하지 말고 그대로 더하면 됨.
            t_point = t_frame + float(t_offset) * 1e-9

            self.get_logger().info(
                f"pt[{i}]: "
                f"xyz=({x:.3f}, {y:.3f}, {z:.3f}), "
                f"frame_time={t_frame:.9f} s, "
                f"offset={float(t_offset):.0f} ns, "
                f"point_time={t_point:.9f} s"
            )

            self.csv_writer.writerow(
                [
                    self.frame_idx,
                    msg.header.stamp.sec,
                    msg.header.stamp.nanosec,
                    i,
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{z:.6f}",
                    int(t_offset),
                    f"{t_point:.9f}",
                ]
            )
        self.csv_file.flush()

        self.frame_idx += 1

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = OusterPointTimeLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()


if __name__ == "__main__":
    main()
