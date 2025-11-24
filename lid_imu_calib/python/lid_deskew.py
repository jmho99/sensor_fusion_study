#!/usr/bin/env python3
import math
import numpy as np
import argparse
import os

from dataclasses import dataclass


@dataclass
class LidarDeskewParams:
    xyz: np.ndarray  # lidar points, shape (N,3)
    t_lid_pts: np.ndarray  # lidar point absolute timestamps
    t_imu: np.ndarray  # imu timestamps
    q_IG: np.ndarray  # quaternion [w,x,y,z], I^G q (IMU orientation w.r.t. G)
    p_IG: np.ndarray  # position, shape (N,3)
    q_IL: np.ndarray  # quaternion [w,x,y,z], I^L q
    p_IL: np.ndarray  # translation, shape (3,)


class LidarDeskew:
    def __init__(self, imu_csv="", pcd_in="", pcd_out="", params=LidarDeskewParams):

        if imu_csv != "" and pcd_in != "" and pcd_out != "":
            self.deskew_pcd_with_imu(imu_csv, pcd_in, pcd_out)
        else:
            self.deskew_params = params

    # ---------- Deskew 오프라인 파이프라인 ----------

    def deskew_pcd_with_imu(
        self,
        imu_csv,
        pcd_in,
        pcd_out,
        imu_time_offset=0.0,
        R_IL: np.ndarray = np.zeros((3, 3)),
    ):

        R_IL = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        print(f"[INFO] Loading IMU CSV: {imu_csv}")
        t_imu, q_imu = self.imu_state(imu_csv)
        print(
            f"[INFO] IMU samples: {t_imu.shape[0]}, t range = [{t_imu[0]:.6f}, {t_imu[-1]:.6f}]"
        )

        print(f"[INFO] Loading PCD: {pcd_in}")
        header_lines, fields, xyz_t = self.load_pcd_ascii_with_t(pcd_in)
        print(f"[INFO] Loaded {xyz_t.shape[0]} points from PCD.")

        x = xyz_t[:, 0]
        y = xyz_t[:, 1]
        z = xyz_t[:, 2]
        intensity = xyz_t[:, 3]
        t_off = xyz_t[:, 4]  # ns
        t_sec = xyz_t[:, 5]
        t_nsec = xyz_t[:, 6]
        t_points = t_sec + t_nsec * 1e-9 + t_off * 1e-9

        t_lid_imu_off = t_points[0] - t_imu[0]
        t_points = t_points - t_lid_imu_off

        print(f"[INFO] Point time range: [{t_points.min():.6f}, {t_points.max():.6f}]")

        deskewed_xyz_t = np.zeros_like(xyz_t)

        t0 = t_points.min()
        q_0 = self.get_q_at_time(t_imu, q_imu, t0)
        r_0 = self.quat_to_R(q_0)

        for i in range(xyz_t.shape[0]):
            p = np.array([x[i], y[i], z[i]], dtype=float)
            t_p = t_points[i]

            q_p = self.get_q_at_time(t_imu, q_imu, t_p)
            r_p = self.quat_to_R(q_p)
            # q_p: 기준(시작) -> t_p 회전
            # 스캔 시작 좌표계로 돌리기 위해 역회전 적용
            R = R_IL.T @ r_0.T @ r_p @ R_IL
            # print("R:", t_imu, q_imu, t_p, q_p, R)
            p_deskew = (R @ p.T).T

            deskewed_xyz_t[i, 0:3] = p_deskew
            deskewed_xyz_t[i, 3] = intensity[i]
            deskewed_xyz_t[i, 4] = t_off[i]  # t 값은 그대로 유지
            deskewed_xyz_t[i, 5] = t_sec[i]
            deskewed_xyz_t[i, 6] = t_nsec[i]

        print(f"[INFO] Saving deskewed PCD to: {pcd_out}")
        self.save_pcd_ascii(pcd_out, header_lines, fields, deskewed_xyz_t)
        print("[INFO] Done.")

    # ---------- Deskew 오프라인 파이프라인 ----------

    def deskew_pcd_with_imu(
        self,
        pcd_in,
        pcd_out,
        imu_time_offset=0.0,
    ):
        params = self.deskew_params
        R_IL = self.quat_to_R(params.q_IL[0])
        p_IL = params.p_IL[0]
        T_IL = self.so3_to_so4(R_IL, p_IL)

        print(f"[INFO] Loading IMU")
        t_imu, q_imu, p_imu = params.t_imu, params.q_IG, params.p_IG

        print(
            f"[INFO] IMU samples: {t_imu.shape[0]}, t range = [{t_imu[0]:.6f}, {t_imu[-1]:.6f}]"
        )

        print(f"[INFO] Loading PCD: {pcd_in}")
        header_lines, fields, xyz_t = self.load_pcd_ascii_with_t(pcd_in)
        print(f"[INFO] Loaded {xyz_t.shape[0]} points from PCD.")

        x = xyz_t[:, 0]
        y = xyz_t[:, 1]
        z = xyz_t[:, 2]
        intensity = xyz_t[:, 3]
        t_off = xyz_t[:, 4]  # ns
        t_sec = xyz_t[:, 5]
        t_nsec = xyz_t[:, 6]
        t_points = t_sec + t_nsec * 1e-9 + t_off * 1e-9

        idx = np.searchsorted(t_imu, t_points)
        t_lid_imu_off = t_points[0] - t_imu[idx]
        t_points = t_points - t_lid_imu_off

        print(f"[INFO] Point time range: [{t_points.min():.6f}, {t_points.max():.6f}]")

        deskewed_xyz_t = np.zeros_like(xyz_t)

        t0 = t_points.min()
        q_0 = self.get_q_at_time(t_imu, q_imu, t0)
        r_0 = self.quat_to_R(q_0)
        p_0 = self.get_p_at_time(t_imu, p_imu, t0)
        T_GI_0 = self.so3_to_so4(r_0, p_0)

        for i in range(xyz_t.shape[0]):
            p = np.array([x[i], y[i], z[i]], dtype=float)
            t_p = t_points[i]

            q_p = self.get_q_at_time(t_imu, q_imu, t_p)
            r_p = self.quat_to_R(q_p)
            p_p = self.get_p_at_time(t_imu, p_imu, t_p)
            T_GI_p = self.so3_to_so4(r_p, p_p)
            # q_p: 기준(시작) -> t_p 회전
            # 스캔 시작 좌표계로 돌리기 위해 역회전 적용
            T_LL = np.linalg.inv(T_IL) @ np.linalg.inv(T_GI_0) @ T_GI_p @ T_IL
            # print("R:", t_imu, q_imu, t_p, q_p, R)
            p_deskew = (T_LL[0:2, 0:2] @ p.T + T_LL[0:2, 3]).T

            deskewed_xyz_t[i, 0:3] = p_deskew
            deskewed_xyz_t[i, 3] = intensity[i]
            deskewed_xyz_t[i, 4] = t_off[i]  # t 값은 그대로 유지
            deskewed_xyz_t[i, 5] = t_sec[i]
            deskewed_xyz_t[i, 6] = t_nsec[i]

        print(f"[INFO] Saving deskewed PCD to: {pcd_out}")
        self.save_pcd_ascii(pcd_out, header_lines, fields, deskewed_xyz_t)
        print("[INFO] Done.")
        return T_GI_0, T_IL

    # ---------- SO3 / 쿼터니언 유틸 ----------

    def so3_to_so4(self, R, p):
        """SO(3) R(3,3), p(3,) -> SO(4) T(4,4)"""
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = p
        return T

    def exp_so3(self, w):
        """회전벡터 w(3,) -> 회전행렬 R(3,3)"""
        theta = np.linalg.norm(w)
        if theta < 1e-12:
            return np.eye(3)
        k = w / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        s = np.sin(theta)
        c = np.cos(theta)
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    def log_so3(self, R):
        """회전행렬 R(3,3) -> 회전벡터 w(3,)"""
        cos_theta = (np.trace(R) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta < 1e-12:
            return np.zeros(3)
        w_hat = (R - R.T) / (2.0 * np.sin(theta))
        return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])

    def vec_to_quat(self, delta_w):
        """작은 회전 벡터 delta_w (rad, 3,) -> 쿼터니언 [w, x, y, z]"""
        theta = np.linalg.norm(delta_w)
        if theta < 1e-8:
            # 작은 각: 1차 근사
            return np.array([1.0, 0.5 * delta_w[0], 0.5 * delta_w[1], 0.5 * delta_w[2]])
        axis = delta_w / theta
        half = 0.5 * theta
        s = math.sin(half)
        return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])

    def quat_to_R(self, q):
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def R_to_quat(self, R):
        """회전행렬 R(3,3) -> 쿼터니언 [w,x,y,z]"""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])

    def quat_mul(self, q1, q2):
        """쿼터니언 곱 q = q1 * q2 (w,x,y,z)"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def quat_conj(self, q):
        """켤레(역회전용)"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_rotate(self, q, v):
        """쿼터니언 q로 3D 벡터 v 회전 (q * v * q⁻¹)"""
        qv = np.array([0.0, v[0], v[1], v[2]])
        return self.quat_mul(self.quat_mul(q, qv), self.quat_conj(q))[1:]

    # ---------- IMU CSV 로드 & 회전 적분 ----------

    def imu_state(self, path):
        data = np.loadtxt(path, delimiter=",", dtype=str)
        ts = data[19:30, 0].astype(float)
        tn = data[19:30, 1].astype(float)
        t = ts + tn * 1e-9
        gx = data[19:30, 16].astype(float)
        gy = data[19:30, 17].astype(float)
        gz = data[19:30, 18].astype(float)
        w = np.stack([gx, gy, gz], axis=1)

        N = t.shape[0]
        q_list = np.zeros((N, 4), dtype=float)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        q_list[0] = q

        for i in range(1, N):
            dt = float(t[i] - t[i - 1])
            if i == 1:
                print(f"dt[{i}] = {dt:.9f} sec")
            if dt <= 0:
                q_list[i] = q_list[i - 1]
                continue
            rot_imu = w[i - 1] * dt
            delta_q = self.vec_to_quat(rot_imu)
            q = self.quat_mul(delta_q, q)
            q_list[i] = q

        return t, q_list

    def get_q_at_time(self, t_imu, q_imu, t_query):
        """
        t_query(스칼라)에 대해, t_imu, q_imu 테이블에서
        가장 가까운 시각의 쿼터니언을 가져오는 간단한 보간 함수.
        """
        idx = np.searchsorted(t_imu, t_query)
        if idx <= 0:
            return q_imu[0]
        elif idx >= t_imu.shape[0]:
            return q_imu[-1]
        else:
            t1, t2 = t_imu[idx - 1], t_imu[idx]
            s = (t_query - t1) / (t2 - t1)
            if s <= 0.0:
                return q_imu[idx - 1]
            elif s >= 1.0:
                return q_imu[idx]
            q_rel = self.quat_mul(self.quat_conj(q_imu[idx - 1]), q_imu[idx])
            R_rel = self.quat_to_R(q_rel)
            w = self.log_so3(R_rel)
            R_inc = self.exp_so3(s * w)
            R_last = self.quat_to_R(q_imu[idx - 1])
            R_interp = R_inc @ R_last
            q_interp = self.R_to_quat(R_interp)
            return q_interp

    def get_p_at_time(self, t_imu, p_imu, t_query):
        """
        t_query(스칼라)에 대해, t_imu, p_imu 테이블에서
        가장 가까운 시각의 위치를 가져오는 간단한 보간 함수.
        """
        idx = np.searchsorted(t_imu, t_query)
        if idx <= 0:
            return p_imu[0]
        elif idx >= t_imu.shape[0]:
            return p_imu[-1]
        else:
            t1, t2 = t_imu[idx - 1], t_imu[idx]
            s = (t_query - t1) / (t2 - t1)
            p1, p2 = p_imu[idx - 1], p_imu[idx]
            p_interp = (1 - s) * p1 + s * p2
            return p_interp

    # ---------- PCD 로드 / 저장 (ASCII) ----------

    def load_pcd_ascii_with_t(self, path):
        """
        ASCII PCD 파일을 읽어 header 정보와 데이터(x,y,z, t 포함)를 반환.
        가정: FIELDS 중에 x, y, z, 그리고 t(time)가 있음.
        """
        with open(path, "r") as f:
            lines = f.readlines()

        header_lines = []
        data_start_idx = None

        for i, line in enumerate(lines):
            if line.strip().startswith("DATA"):
                header_lines = lines[: i + 1]
                data_start_idx = i + 1
                break

        if data_start_idx is None:
            raise ValueError("PCD: DATA 라인을 찾지 못했습니다.")

        # 헤더에서 FIELDS 파싱
        fields_line = None
        for line in header_lines:
            if line.startswith("FIELDS"):
                fields_line = line
                break
        if fields_line is None:
            raise ValueError("PCD: FIELDS 라인을 찾지 못했습니다.")

        fields = fields_line.strip().split()[
            1:
        ]  # 'FIELDS x y z t' -> ['x','y','z','t']

        # x,y,z,t 필드 인덱스 찾기
        try:
            ix = fields.index("x")
            iy = fields.index("y")
            iz = fields.index("z")
            iintensity = fields.index("intensity")
            it = fields.index("t")
            isec = fields.index("sec")
            inanosec = fields.index("nanosec")
        except ValueError:
            raise ValueError(
                f"PCD: FIELDS에 포함되지 않은 인덱스를 포함하고 있습니다: {fields}"
            )

        # 데이터 부분 읽기
        data_str = lines[data_start_idx:]
        # 공백으로 구분된 float/정수들이므로 numpy로 파싱
        data = np.loadtxt(data_str, dtype=float)  # (N, num_fields)

        x = data[:, ix]
        y = data[:, iy]
        z = data[:, iz]
        intensity = data[:, iintensity]
        t_offset = data[:, it]  # ns 기준이라고 가정
        sec = data[:, isec]
        nanosec = data[:, inanosec]

        return (
            header_lines,
            fields,
            np.vstack([x, y, z, intensity, t_offset, sec, nanosec]).T,
        )

    def save_pcd_ascii(self, path, header_lines, fields, xyz_t):
        """
        header_lines: 기존 PCD 헤더 (DATA 줄까지)
        fields: FIELDS의 필드 리스트 (ex. ['x','y','z','intensity','t'])
        xyz_t: (N,4) [x,y,z,t] 형태 (t는 그대로 유지)
        """
        # fields 내에서 x,y,z, 그리고 t 의 위치만 쓸 거라고 가정 (간단한 버전)
        ix = fields.index("x")
        iy = fields.index("y")
        iz = fields.index("z")
        intensity = fields.index("intensity")
        sec = fields.index("sec")
        nanosec = fields.index("nanosec")
        it = None
        for cand in ["t", "time", "timestamp"]:
            if cand in fields:
                it = fields.index(cand)
                break

        if it is None:
            raise ValueError("save_pcd_ascii: 시간 필드를 찾지 못했습니다.")

        N = xyz_t.shape[0]

        with open(path, "w") as f:
            # 헤더 쓰기
            for line in header_lines:
                f.write(line)
            # DATA ascii 뒤부터 데이터 작성
            # (header_lines 안에 DATA ascii 줄이 포함되어 있어야 함)
            # xyz_t는 x,y,z,t 순서, fields 순서를 그대로 쓰려면 fields 길이만큼 라인 만들 수도 있음.
            for i in range(N):
                x, y, z, intensity, t_off, sec, nanosec = xyz_t[i]
                # 여기서는 x y z t만 쓰는 단순 버전
                # 다른 필드가 있었던 PCD였다면, 필요 시 확장해야 함.
                line_vals = [
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{z:.6f}",
                    str(int(intensity)),
                    str(int(t_off)),
                    str(int(sec)),
                    str(int(nanosec)),
                ]
                f.write(" ".join(line_vals) + "\n")


# ---------- main ----------


def main(args=None):

    run = LidarDeskew(
        imu_csv="/home/antlab/imu_test_trim_01.csv",
        pcd_in="/home/antlab/ROS2/calib_ws/pcd_files/frame_0001.pcd",
        pcd_out="/home/antlab/ROS2/calib_ws/pcd_files/frame_0001_deskewed.pcd",
    )


if __name__ == "__main__":
    main()
