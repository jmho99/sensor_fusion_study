#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2

from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSPresetProfiles, QoSProfile, ReliabilityPolicy, HistoryPolicy

import math
from collections import deque


class TopicSyncTest(Node):
    t_imu_ = 0.0

    def __init__(self):
        super().__init__("z_topic_sync_test")

        qos_lid = QoSPresetProfiles.SENSOR_DATA.value

        qos_imu = QoSPresetProfiles.SENSOR_DATA.value

        self.lid_sub = self.create_subscription(
            PointCloud2, "/ouster/points", self.lid_callback, qos_lid
        )
        self.imu_sub = self.create_subscription(
            Imu, "/imu/data", self.imu_callback, qos_imu
        )

    def lid_callback(self, msg):
        t_lid = Time.from_msg(msg.header.stamp).nanoseconds *1e-9
        self.get_logger().info(f"t_lid = {t_lid:.6f}")
        self.get_logger().info(f"t_imu = {TopicSyncTest.t_imu_:.6f}")
        self.get_logger().info(f"dt = {TopicSyncTest.t_imu_ - t_lid:.6f}")

    def imu_callback(self, msg):
        t_imu = Time.from_msg(msg.header.stamp)
        TopicSyncTest.t_imu_ = t_imu.nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    topic_test = TopicSyncTest()
    rclpy.spin(topic_test)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
