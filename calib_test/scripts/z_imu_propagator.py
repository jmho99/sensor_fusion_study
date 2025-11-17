#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
from collections import deque
import open3d as o3d

from concurrent.futures import ThreadPoolExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading, queue

# --- NumPy compat shim (구버전 의존성 대비) ---
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object
# ------------------------------------------------

# 우리 파이프라인(이미 작성한 모듈)
from lid_imu_calib_pipeline import (
    IMUPropagator,
    Deskewer,
    NDTMatcher,
    ExtrinsicEKF,
    Lie,
)


class LIOCalibNode(Node):
    """
    ✅ LiDAR header time을 기준 시계로 고정
    ✅ IMU header time에 오프셋(EMA) 더해 보정: t_corr = t_imu_hdr + offset
    ✅ LiDAR 콜백에서: IMU추출 → propagate → deskew → NDT → EKF update
    """

    def __init__(self):
        super().__init__("lio_calib_node")

        # ---- 파라미터 ----
        self.declare_parameter(
            "init_R_deg", [0.0, 0.0, 0.0]
        )  # extrinsic 초기 R (roll,pitch,yaw deg)
        self.declare_parameter("lidar_hz", 10.0)  # LiDAR frame rate (e.g., 10 Hz)
        self.declare_parameter("offset_alpha", 0.2)  # EMA 계수
        self.declare_parameter("offset_init", -0.134)  # 초기 오프셋(초)
        self.declare_parameter("offset_clip", -0.025)  # EMA 업데이트 허용 한계(초)

        init_r_deg = self.get_parameter("init_R_deg").value
        lidar_hz = float(self.get_parameter("lidar_hz").value)
        self.alpha = float(self.get_parameter("offset_alpha").value)
        self.offset = float(self.get_parameter("offset_init").value)
        self.offclip = float(self.get_parameter("offset_clip").value)
        self.scan_period = 1.0 / max(1e-6, lidar_hz)

        # extrinsic 초기화
        init_r_rad = np.deg2rad(np.array(init_r_deg, dtype=float))
        self.R_IL = Lie.exp6(np.hstack([np.zeros(3), init_r_rad]))[0:3, 0:3]
        self.t_IL = np.zeros(3)

        # 파이프라인 객체
        self.prop = IMUPropagator()  # 15-state prop
        self.desk = Deskewer()
        self.ndt = NDTMatcher(voxel=0.3, max_iter=60)  # Open3D 기반(G-ICP 대체)
        self.ekf = ExtrinsicEKF()
        self.ekf.RiL = self.R_IL
        self.ekf.tiL = self.t_IL
        self.ekf.set_from_propagator(self.prop, self.R_IL, self.t_IL)

        # 버퍼: (t_hdr, w, a)
        self.imu_buf = deque(maxlen=5000)
        self.last_pts = None
        self.last_t_end_hdr = None  # 직전 프레임의 LiDAR header end time

        # 구독
        imu_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.cb_imu = ReentrantCallbackGroup()
        self.cb_lidar = ReentrantCallbackGroup()
        self.pool = ThreadPoolExecutor(max_workers=2)

        self.create_subscription(
            Imu,
            "/imu/data",
            self.imu_callback,
            qos_profile_sensor_data,
            callback_group=self.cb_imu,
        )
        self.create_subscription(
            PointCloud2,
            "/ouster/points",
            self.lidar_callback,
            qos_profile_sensor_data,
            callback_group=self.cb_lidar,
        )

        self.get_logger().info(
            f"[LIOCalibNode] Ready. LiDAR-time based. init_R(deg)={init_r_deg}, "
            f"lidar_hz={lidar_hz}, scan_period={self.scan_period:.3f}s"
        )

    def destroy_node(self):
        # 종료 시 워커 정리
        try:
            self.worker_alive = False
            self.job_q.put(None)
        except Exception:
            pass
        return super().destroy_node()

    # ---------------- IMU 콜백: header time만 사용 ----------------
    def imu_callback(self, msg: Imu):
        t_hdr = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        w = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=float,
        )
        a = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=float,
        )
        self.imu_buf.append((t_hdr, w, a))

    # ------------- LiDAR 콜백: LiDAR header 기준으로 전체 파이프라인 -------------

    def lidar_callback(self, msg: PointCloud2):
        # 1) LiDAR 기준 절대시각
        t_lid = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        t_start = t_lid - self.scan_period
        t_end = t_lid
        guard = 0.09

        # 2) IMU 버퍼 확인 및 오프셋 추정
        if len(self.imu_buf) < 98:
            self.get_logger().warn("Waiting for IMU buffer to fill...")
            return
        t_imu_latest = self.imu_buf[-1][0]
        offset = t_lid - t_imu_latest  # LiDAR - IMU
        # LiDAR 기준으로 재타임스탬프
        imu_aligned = []
        for t_hdr, g, a in self.imu_buf:
            t_new = t_hdr + offset
            g_np = np.asarray(g, dtype=np.float64).reshape(-1)
            a_np = np.asarray(a, dtype=np.float64).reshape(-1)
            if g_np.size != 3 or a_np.size != 3:
                continue
            imu_aligned.append((t_new, g_np, a_np))

        # 3) propagate 윈도우링
        win0, win1 = t_start - guard, t_end + guard
        imu_seg = [(t, g, a) for (t, g, a) in imu_aligned if win0 <= t <= win1]
        if len(imu_seg) < 2:
            self.get_logger().warn(f"Not enough IMU in window [{win0:.3f}..{win1:.3f}]")
            return

        #####################################################
        imu_traj = self.prop.propagate(imu_seg, win0, win1)
        #####################################################

        # 4) PointCloud2 → Open3D → numpy xyz
        try:
            pts = point_cloud2.read_points_list(msg, field_names=("x", "y", "z"))
            if not pts:
                raise ValueError("Empty LiDAR scan.")
            pts_np = np.asarray(pts)
            xyz = pts_np.astype(np.float64).reshape(-1, 3)

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(xyz)
        except Exception as e:
            self.get_logger().error(f"PointCloud2→Open3D failed: {e}")
            return
        N = xyz.shape[0]
        if N < 50:
            self.get_logger().warn(f"Too few points after downsample: N={N}")
            return

        # 5) 포인트별 절대시각 생성 (선형 분포)
        #  - 포인트 타임이 없으므로 스캔 기간을 균등 분배: [t_start .. t_end]
        t_rel = np.linspace(0.0, self.scan_period, N, dtype=np.float32)
        t_abs = t_lid - (self.scan_period - t_rel)

        # 6) deskew
        T_IL = self.ekf.T_IL()
        pts_deskew = self.desk.deskew_points(xyz, t_abs, imu_traj, T_IL, t_start, t_end)

        # 첫 스캔이면 레퍼런스로 저장 후 종료
        if self.last_pts is None:
            self.last_pts = pts_deskew
            self.last_t_end_hdr = t_end
            self.get_logger().info(f"[Sync] Offset(LiDAR-IMU) = {offset:+.3f} s (init)")
            return

        # 7) NDT/ICP (Open3D G-ICP를 내부에서 쓰는 NDTMatcher)
        try:
            Zk, Rk = self.ndt.align(self.last_pts, pts_deskew, np.eye(4))
        except Exception as e:
            self.get_logger().warn(f"NDT/G-ICP failed: {e}")
            self.last_pts = pts_deskew
            self.last_t_end_hdr = t_end
            return
        # 8) IMU 상대변환 (이전/현재 스캔 끝 시각에서)
        q0, p0 = self.desk.interpolate_pose(self.last_t_end_hdr, imu_traj)
        q1, p1 = self.desk.interpolate_pose(t_end, imu_traj)
        T_WI_0 = np.eye(4)
        T_WI_0[0:3, 0:3] = Lie.R_from_q(q0)
        T_WI_0[0:3, 3] = p0
        T_WI_1 = np.eye(4)
        T_WI_1[0:3, 0:3] = Lie.R_from_q(q1)
        T_WI_1[0:3, 3] = p1
        # 9) EKF 업데이트
        resid, K = self.ekf.update(Zk, Rk, T_WI_0, T_WI_1)
        R_est, t_est = self.ekf.get_state()
        print(
            "[EKF] resid_norm=%.3f, rpy(deg)=%s, t=%s"
            % (np.linalg.norm(resid), self.ekf.get_rpy_deg(), np.round(t_est, 4))
        )
        # 10) 다음 프레임 준비
        self.last_pts = pts_deskew
        self.last_t_end_hdr = t_end


def main(args=None):
    rclpy.init(args=args)
    node = LIOCalibNode()
    try:
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
