#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data

# numpy>=1.24 호환 (구버전 모듈 대비)
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# 회전/쿼터니언 유틸
import tf_transformations as tft


def quat_mul(q1, q2):
    return tft.quaternion_multiply(q1, q2)


def quat_inv(q):
    return tft.quaternion_conjugate(q)


def quat_from_omega_dt(w, dt):
    th = float(np.linalg.norm(w) * dt)
    if th < 1e-12:
        return np.array([0, 0, 0, 1], dtype=float)
    ax = w / (np.linalg.norm(w) + 1e-12)
    return tft.quaternion_about_axis(th, ax)


def mat_from_quat(q):
    return tft.quaternion_matrix(q)[:3, :3]


# Open3D (필수)
try:
    import open3d as o3d
except Exception as e:
    o3d = None


class NodeLidarDeskewICP(Node):
    def __init__(self):
        super().__init__("node_lidar_deskew_icp_o3d")
        if o3d is None:
            self.get_logger().error(
                "Open3D가 설치되어 있지 않습니다. `pip install open3d`로 설치하세요."
            )
            raise SystemExit(1)

        # -------- Parameters --------
        self.declare_parameter("imu_time_offset", 0.0)  # LiDAR - IMU [s]
        self.declare_parameter("point_time_unit", "auto")  # 'auto'|'us'|'ns'
        self.declare_parameter("imu_buffer_seconds", 8.0)
        self.declare_parameter(
            "imu_to_lidar_so3", [-1, 0, 0, 0, 1, 0, 0, 0, -1]
        )  # row-major 3x3
        self.declare_parameter("pub_frame_fixed", "")  # ''면 입력 frame_id 사용
        # Open3D registration params
        self.declare_parameter("voxel_leaf", 0.60)  # [m] 다운샘플
        self.declare_parameter("icp_max_dist", 2.0)  # 대응 최대 거리
        self.declare_parameter("use_point_to_plane", True)
        self.declare_parameter("build_global_map", True)
        self.declare_parameter("keyframe_dist", 1.0)  # [m] 키프레임 간 거리 임계

        self.offset = float(self.get_parameter("imu_time_offset").value)
        self.unit = str(self.get_parameter("point_time_unit").value)
        self.imu_hz = float(self.get_parameter("imu_buffer_seconds").value)
        R_list = self.get_parameter("imu_to_lidar_so3").value
        self.pub_fixed = str(self.get_parameter("pub_frame_fixed").value)
        self.voxel_leaf = float(self.get_parameter("voxel_leaf").value)
        self.icp_max_dist = float(self.get_parameter("icp_max_dist").value)
        self.use_pt2pl = bool(self.get_parameter("use_point_to_plane").value)
        self.build_map = bool(self.get_parameter("build_global_map").value)
        self.keyframe_dist = float(self.get_parameter("keyframe_dist").value)

        self.R_IL = np.array(R_list, dtype=float).reshape(3, 3)
        # SO(3) 정규화
        U, _, Vt = np.linalg.svd(self.R_IL)
        self.R_IL = U @ Vt
        if np.linalg.det(self.R_IL) < 0:
            U[:, -1] *= -1
            self.R_IL = U @ Vt
        self.get_logger().info(f"IMU→LiDAR R:\n{self.R_IL}")

        # -------- IMU global orientation buffer (LiDAR frame) --------
        self.g_times = deque()  # times
        self.g_quats = deque()  # quats
        self.prev_imu = None

        # -------- Anchor / map pose --------
        self.anchor_set = False
        self.q_anchor = np.array([0, 0, 0, 1], dtype=float)
        self.anchor_time = None

        self.T_map = np.eye(4, dtype=float)  # 누적 포즈(map->lidar)
        self.last_kf_pose = np.eye(4, dtype=float)  # 키프레임 포즈

        # 누적 맵(키프레임) - Open3D 포인트클라우드로 관리
        self.keyframes = []  # 각 항목: o3d.geometry.PointCloud (map 좌표)
        self.map_cloud = o3d.geometry.PointCloud()  # 편의용(필요 시 생성)

        # -------- IO --------
        self.create_subscription(Imu, "/imu/data", self.cb_imu, qos_profile_sensor_data)
        self.create_subscription(
            PointCloud2, "/ouster/points", self.cb_lidar, qos_profile_sensor_data
        )
        self.pub_deskew = self.create_publisher(PointCloud2, "/deskewed_points", 10)
        self.pub_aligned = self.create_publisher(PointCloud2, "/aligned_points", 10)
        self.pub_odom = self.create_publisher(Odometry, "/lidar_odom", 10)
        self.tf_br = TransformBroadcaster(self)

        self.get_logger().info(
            "node_lidar_deskew_icp_o3d started (first scan = anchor)."
        )

    # ---------------- IMU handling ----------------
    def cb_imu(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        w_I = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=float,
        )
        w_L = w_I @ self.R_IL  # IMU→LiDAR

        if not self.g_times:
            self.g_times.append(t)
            self.g_quats.append(np.array([0, 0, 0, 1], dtype=float))
            self.prev_imu = (t, w_L)
            return

        t_prev, w_prev = self.prev_imu
        dt = max(1e-6, t - t_prev)
        w_avg = 0.5 * (w_prev + w_L)
        dq = quat_from_omega_dt(w_avg, dt)
        q_new = quat_mul(self.g_quats[-1], dq)
        q_new = q_new / np.linalg.norm(q_new)
        self.g_times.append(t)
        self.g_quats.append(q_new)
        self.prev_imu = (t, w_L)

        # trim
        now = t
        while self.g_times and (now - self.g_times[0] > self.imu_hz):
            self.g_times.popleft()
            self.g_quats.popleft()

    def rot_at_global(self, tq):
        if not self.g_times:
            return None
        ts, qs = self.g_times, self.g_quats
        if tq <= ts[0]:
            return qs[0]
        if tq >= ts[-1]:
            return qs[-1]
        i = int(np.searchsorted(ts, tq) - 1)
        t0, t1 = ts[i], ts[i + 1]
        q0, q1 = qs[i], qs[i + 1]
        u = (tq - t0) / (t1 - t0 + 1e-12)
        return tft.quaternion_slerp(q0, q1, float(u))

    # ---------------- LiDAR handling ----------------
    def cb_lidar(self, msg: PointCloud2):
        fields = [f.name for f in msg.fields]
        time_field = "t" if "t" in fields else ("time" if "time" in fields else None)
        if time_field is None:
            self.get_logger().warn("No per-point time field; skip.")
            return

        pts_iter = point_cloud2.read_points_list(
            msg, field_names=("x", "y", "z", time_field), skip_nans=True
        )
        arr = np.array([p for p in pts_iter], dtype=np.float64)
        if arr.size == 0:
            return
        xyz = arr[:, :3]
        tpc = arr[:, 3]

        # relative time (s)
        unit = self.unit
        if unit == "auto":
            span = float(np.percentile(tpc, 99) - np.percentile(tpc, 1))
            unit = "ns" if span > 5e8 else "us"
        t_rel = (tpc - np.min(tpc)) * (1e-9 if unit == "ns" else 1e-6)

        t_scan0 = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        t0_imu = t_scan0 + self.offset
        t_scan1 = t_scan0 + float(np.max(t_rel))

        # ---- Anchor: 첫 스캔은 기준으로 그대로 퍼블리시 ----
        fixed_frame = self.pub_fixed or msg.header.frame_id
        hdr = Header()
        hdr.stamp = msg.header.stamp
        hdr.frame_id = fixed_frame

        if not self.anchor_set:
            q0 = self.rot_at_global(t0_imu)
            self.q_anchor = (
                q0 if q0 is not None else np.array([0, 0, 0, 1], dtype=float)
            )
            self.anchor_time = t0_imu
            self.anchor_set = True

            # 첫 스캔: deskew X (기준)
            self.publish_cloud(self.pub_deskew, hdr, xyz)
            hdr_lidar = Header()
            hdr_lidar.stamp = msg.header.stamp
            hdr_lidar.frame_id = "map"
            self.publish_cloud(self.pub_aligned, hdr_lidar, xyz)

            # 맵/키프레임 시작
            self.add_keyframe_np(xyz)  # map 좌표(=anchor)로 저장
            self.last_kf_pose = self.T_map.copy()
            self.publish_pose_and_tf(msg.header.stamp, self.T_map, fixed_frame)
            self.get_logger().info("First scan published as anchor (no deskew).")
            return

        # IMU coverage check
        if (
            self.rot_at_global(t0_imu) is None
            or self.rot_at_global(t_scan1 + self.offset) is None
        ):
            self.get_logger().warn("IMU coverage insufficient; skip scan.")
            return

        # ---- 회전 deskew (anchor 기준) ----
        R_anchor = mat_from_quat(self.q_anchor)
        xyz_ds = xyz.copy()
        for i in range(xyz_ds.shape[0]):
            tq = t0_imu + float(t_rel[i])
            qg = self.rot_at_global(tq)
            if qg is None:
                qg = self.g_quats[0] if tq < self.g_times[0] else self.g_quats[-1]
            RgT = mat_from_quat(quat_inv(qg))
            xyz_ds[i] = (R_anchor @ RgT) @ xyz_ds[i]

        # deskew 결과 퍼블리시
        self.publish_cloud(self.pub_deskew, hdr, xyz_ds)

        # ---- Open3D ICP (scan-to-map) ----
        T_delta = self.align_icp_o3d(xyz_ds)
        self.T_map = self.T_map @ T_delta

        # 정합된 포인트 맵좌표로 퍼블리시
        xyz_aligned = self.transform_points(xyz_ds, self.T_map)
        hdr_lidar = Header()
        hdr_lidar.stamp = msg.header.stamp
        hdr_lidar.frame_id = "map"
        self.publish_cloud(self.pub_aligned, hdr_lidar, xyz_aligned)

        # 맵 업데이트(키프레임 기준)
        if (
            self.build_map
            and self.trans_norm(self.last_kf_pose, self.T_map) > self.keyframe_dist
        ):
            self.add_keyframe_np(xyz_aligned)
            self.last_kf_pose = self.T_map.copy()

        # 포즈/TF
        self.publish_pose_and_tf(msg.header.stamp, self.T_map, fixed_frame)

    # ---------- Open3D registration ----------
    def align_icp_o3d(self, xyz_scan):
        if len(self.keyframes) == 0:
            return np.eye(4)
        # 전역 맵 만들기(키프레임 합성)
        tgt = self.get_global_map_np(self.voxel_leaf)
        src = self.voxel_downsample_np(xyz_scan, self.voxel_leaf)

        src_o = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
        tgt_o = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))

        # 노말 추정 (point-to-plane 용)
        if self.use_pt2pl:
            src_o.estimate_normals()
            tgt_o.estimate_normals()

        init = np.eye(4)
        try:
            if self.use_pt2pl:
                reg = o3d.pipelines.registration.registration_icp(
                    src_o,
                    tgt_o,
                    self.icp_max_dist,
                    init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                )
            else:
                reg = o3d.pipelines.registration.registration_icp(
                    src_o,
                    tgt_o,
                    self.icp_max_dist,
                    init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )
        except Exception as e:
            self.get_logger().warn(f"ICP failed: {e}")
            return np.eye(4)

        if reg.fitness < 0.15:
            self.get_logger().warn(f"ICP low fitness: {reg.fitness:.3f}")
            return np.eye(4)
        return np.asarray(reg.transformation, dtype=float)

    # ---------- Map helpers ----------
    def add_keyframe_np(self, xyz_map):
        leaf = self.voxel_leaf
        pc = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(self.voxel_downsample_np(xyz_map, leaf))
        )
        self.keyframes.append(pc)

    def get_global_map_np(self, leaf):
        if len(self.keyframes) == 0:
            return np.zeros((0, 3), dtype=float)
        # 간단 합치기 + 다운샘플
        all_pts = np.concatenate(
            [np.asarray(kf.points) for kf in self.keyframes], axis=0
        )
        return self.voxel_downsample_np(all_pts, leaf)

    # ---------- Utils & publishing ----------
    def voxel_downsample_np(self, pts, leaf):
        if leaf <= 1e-6 or pts.shape[0] == 0:
            return pts
        grid = np.floor(pts / leaf)
        _, idx = np.unique(grid, axis=0, return_index=True)
        return pts[np.sort(idx)]

    def transform_points(self, xyz, T):
        R = T[:3, :3]
        t = T[:3, 3]
        return (xyz @ R.T) + t

    def trans_norm(self, A, B):
        ta = A[:3, 3]
        tb = B[:3, 3]
        return float(np.linalg.norm(tb - ta))

    def publish_cloud(self, pub, header, xyz):
        msg = point_cloud2.create_cloud_xyz32(header, xyz.astype(np.float32).tolist())
        pub.publish(msg)

    def publish_pose_and_tf(self, stamp, T_map, fixed="map"):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = fixed
        odom.child_frame_id = "os_sensor"
        odom.pose.pose.position.x = float(T_map[0, 3])
        odom.pose.pose.position.y = float(T_map[1, 3])
        odom.pose.pose.position.z = float(T_map[2, 3])
        q = tft.quaternion_from_matrix(T_map)
        odom.pose.pose.orientation.x = float(q[0])
        odom.pose.pose.orientation.y = float(q[1])
        odom.pose.pose.orientation.z = float(q[2])
        odom.pose.pose.orientation.w = float(q[3])
        self.pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = fixed
        tf.child_frame_id = "os_sensor"
        tf.transform.translation.x = odom.pose.pose.position.x
        tf.transform.translation.y = odom.pose.pose.position.y
        tf.transform.translation.z = odom.pose.pose.position.z
        tf.transform.rotation.x = q[0]
        tf.transform.rotation.y = q[1]
        tf.transform.rotation.z = q[2]
        tf.transform.rotation.w = q[3]
        self.tf_br.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = NodeLidarDeskewICP()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
