#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

# 선택: Open3D for G-ICP (NDT 대체). 설치가 없다면 주석 처리하고 TODO 로 두세요.
try:
    import open3d as o3d

    HAS_O3D = True
except Exception:
    HAS_O3D = False


# =========================================================
# SO3 / SE3 helpers  (모두 클래스 내부에서만 사용하도록 구성)
# =========================================================
class Lie:
    @staticmethod
    def skew(w):
        wx, wy, wz = w
        return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)

    @staticmethod
    def qmul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    @staticmethod
    def qnorm(q):
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n

    @staticmethod
    def q_from_small(dtheta):
        th = float(np.linalg.norm(dtheta))
        if th < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = dtheta / th
        h = 0.5 * th
        return np.hstack([math.cos(h), axis * np.sin(h)])

    @staticmethod
    def R_from_q(q):
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    @staticmethod
    def q_from_R(R):
        # robust quaternion from rotation matrix
        K = (
            np.array(
                [
                    [
                        R[0, 0] - R[1, 1] - R[2, 2],
                        R[1, 0] + R[0, 1],
                        R[2, 0] + R[0, 2],
                        R[1, 2] - R[2, 1],
                    ],
                    [
                        R[1, 0] + R[0, 1],
                        -R[0, 0] + R[1, 1] - R[2, 2],
                        R[2, 1] + R[1, 2],
                        R[2, 0] - R[0, 2],
                    ],
                    [
                        R[2, 0] + R[0, 2],
                        R[2, 1] + R[1, 2],
                        -R[0, 0] - R[1, 1] + R[2, 2],
                        R[0, 1] - R[1, 0],
                    ],
                    [
                        R[1, 2] - R[2, 1],
                        R[2, 0] - R[0, 2],
                        R[0, 1] - R[1, 0],
                        R[0, 0] + R[1, 1] + R[2, 2],
                    ],
                ]
            )
            / 3.0
        )
        w, v = np.linalg.eigh(K)
        q = np.array([v[0, 3], v[1, 3], v[2, 3], v[3, 3]])
        if q[0] < 0:
            q = -q
        return q

    @staticmethod
    def hat6(xi):
        # xi = [rho(3), phi(3)]
        rho = xi[0:3]
        phi = xi[3:6]
        X = np.zeros((4, 4))
        X[0:3, 0:3] = Lie.skew(phi)
        X[0:3, 3] = rho
        return X

    @staticmethod
    def exp6(xi):
        rho = xi[0:3]
        phi = xi[3:6]
        th = np.linalg.norm(phi)
        I = np.eye(3)
        if th < 1e-8:
            R = I + Lie.skew(phi)
            V = I + 0.5 * Lie.skew(phi)
        else:
            a = math.sin(th) / th
            b = (1 - math.cos(th)) / (th * th)
            c = (1 - a) / (th * th)
            K = Lie.skew(phi)
            R = I + a * K + b * (K @ K)
            V = I + b * K + c * (K @ K)
        t = V @ rho
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        return T

    @staticmethod
    def log6(T):
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        I = np.eye(3)
        tr = np.trace(R)
        cos_th = (tr - 1) / 2
        cos_th = max(min(cos_th, 1.0), -1.0)
        th = math.acos(cos_th)
        if th < 1e-8:
            phi = np.array([0.0, 0.0, 0.0])
            V_inv = (
                I - 0.5 * Lie.skew(phi) + (1 / 12.0) * (Lie.skew(phi) @ Lie.skew(phi))
            )
        else:
            K = (R - R.T) / (2 * math.sin(th)) * th
            phi = np.array([K[2, 1], K[0, 2], K[1, 0]])
            A = math.sin(th) / th
            B = (1 - math.cos(th)) / (th * th)
            V_inv = I - 0.5 * K + (1 / (th * th)) * (1 - A / (2 * B)) * (K @ K)
        rho = V_inv @ t
        return np.hstack([rho, phi])

    @staticmethod
    def Ad(T):
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        Ad = np.zeros((6, 6))
        Ad[0:3, 0:3] = R
        Ad[3:6, 3:6] = R
        Ad[0:3, 3:6] = Lie.skew(t) @ R
        return Ad

    @staticmethod
    def inv(T):
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        Ti = np.eye(4)
        Ti[0:3, 0:3] = R.T
        Ti[0:3, 3] = -R.T @ t
        return Ti


# =========================================================
# 1) IMU Propagation (prediction only, 15x15 error-state)
# =========================================================
class IMUPropagator:
    """
    x = {q_IW(4), v(3), p(3), b_g(3), b_a(3)}
    δx = [δθ, δv, δp, δb_g, δb_a] (15)
    dot:
      Ṙ = R(ω_m - b_g - n_g)^
      v̇ = g + R(a_m - b_a - n_a)
      ṗ = v
      ḃ_g = n_wg
      ḃ_a = n_wa
    Qc = diag( n_g^2 I, n_a^2 I, n_wg^2 I, n_wa^2 I )
    """

    def __init__(
        self,
        gravity=np.array([0, 0, -9.81]),
        ng=4.24e-5,
        na=1.96e-4,
        nwg=1e-6,
        nwa=2e-4,
    ):
        self.q = np.array([1, 0, 0, 0], dtype=float)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.P = np.eye(15) * 1e-3
        self.g = gravity.astype(float)
        self.ng, self.na, self.nwg, self.nwa = (
            float(ng),
            float(na),
            float(nwg),
            float(nwa),
        )

    def state_tuple(self):
        return (
            self.q.copy(),
            self.v.copy(),
            self.p.copy(),
            self.bg.copy(),
            self.ba.copy(),
            self.P.copy(),
        )

    def set_state(self, q, v, p, bg, ba, P=None):
        self.q = Lie.qnorm(q)
        self.v = v.copy()
        self.p = p.copy()
        self.bg = bg.copy()
        self.ba = ba.copy()
        if P is not None:
            self.P = P.copy()

    def propagate(self, imu_data, t0, t1):
        """
        imu_data: list of (t, omega[3], accel[3]) with absolute times in seconds
        t0, t1: time window to integrate
        returns: traj = [(t_k, q_k, p_k), ...] including first time in window
        """
        # (1) 윈도우 필터 + 정렬
        seg = [s for s in imu_data if t0 <= s[0] <= t1]
        if len(seg) < 2:
            return []
        seg.sort(key=lambda s: s[0])

        # (2) 현재 내부 상태에서 시작
        q = self.q.copy()
        v = self.v.copy()
        p = self.p.copy()

        traj = [(seg[0][0], q.copy(), p.copy())]

        # (3) 샘플 단위 적분
        for i in range(1, len(seg)):
            t_i, omega_m, acc_m = seg[i]
            t_im1, _, _ = seg[i - 1]
            dt = max(1e-6, float(t_i - t_im1))

            omega_m = np.array(omega_m, dtype=float).reshape(3)
            acc_m = np.array(acc_m, dtype=float).reshape(3)

            # bias 보정
            omega = omega_m - self.bg
            acc = acc_m - self.ba

            # dq: θ = ω·dt  →  q_inc = [cos(θ/2), (θ/||θ||) sin(θ/2)]
            theta = omega * dt
            th = float(np.linalg.norm(theta))
            if th < 1e-12:
                dq = np.array(
                    [1.0, 0.5 * theta[0], 0.5 * theta[1], 0.5 * theta[2]], dtype=float
                )
            else:
                half = 0.5 * th
                s = math.sin(half) / th
                dq = np.array(
                    [math.cos(half), theta[0] * s, theta[1] * s, theta[2] * s],
                    dtype=float,
                )

            # q ← q ⊗ dq
            q = Lie.qnorm(Lie.qmul(q, dq))
            R = Lie.R_from_q(q)

            # 가속도 월드, 중력 self.g 사용
            a_world = R @ acc + self.g
            v = v + a_world * dt
            p = p + v * dt

            traj.append((t_i, q.copy(), p.copy()))

        # (4) 내부 상태 업데이트
        self.q, self.v, self.p = q, v, p
        return traj


# =========================================================
# 2) Deskew (IMU traj + extrinsic T_IL)
# =========================================================
class Deskewer:
    """
    포인트별 시간 ti에 대해:
      T_WL(t) = T_WI(t) * T_IL
      p_i_deskew = inv( T_WL(t_end) ) * T_WL(t_i) * p_i
    """

    def __init__(self):
        pass

    def interpolate_pose(self, t, traj):
        """
        traj: list of (t_i, q, p)
        slerp/lerp
        """
        ts = [a[0] for a in traj]
        idx = np.searchsorted(ts, t)
        if idx <= 0:
            return traj[0][1], traj[0][2]
        if idx >= len(traj):
            return traj[-1][1], traj[-1][2]
        t0, q0, p0 = traj[idx - 1]
        t1, q1, p1 = traj[idx]
        a = (t - t0) / max(1e-9, (t1 - t0))
        # slerp (small-angle approx by log/exp on SO(3))
        R0 = Lie.R_from_q(q0)
        R1 = Lie.R_from_q(q1)
        dR = R0.T @ R1
        xi = Lie.log6(
            np.block([[dR, np.zeros((3, 1))], [np.zeros((1, 3)), np.array([[1.0]])]])
        )[3:6]
        R = R0 @ Lie.exp6(np.hstack([np.zeros(3), a * xi]))[0:3, 0:3]
        q = Lie.q_from_R(R)
        p = p0 + a * (p1 - p0)
        return q, p

    def deskew_points(self, xyz, t_abs, imu_traj, T_IL, scan_start, scan_end):
        """
        xyz:   (N,3) float64
        t_abs: (N,)  float64
        imu_traj: [(t, q, p)] 연속 시계열
        T_IL: 4x4 extrinsic (IMU->LiDAR)
        """
        q_end, p_end = self.interpolate_pose(scan_end, imu_traj)
        R_end = Lie.R_from_q(q_end)
        T_WI_end = np.eye(4)
        T_WI_end[0:3, 0:3] = R_end
        T_WI_end[0:3, 3] = p_end
        T_WL_end = T_WI_end @ T_IL
        T_WL_end_inv = Lie.inv(T_WL_end)

        N = xyz.shape[0]
        out = np.zeros((N, 3))
        for i in range(N):
            q_i, p_i = self.interpolate_pose(float(t_abs[i]), imu_traj)
            R_i = Lie.R_from_q(q_i)
            T_WI_i = np.eye(4)
            T_WI_i[0:3, 0:3] = R_i
            T_WI_i[0:3, 3] = p_i
            T_WL_i = T_WI_i @ T_IL

            pt = np.array([xyz[i, 0], xyz[i, 1], xyz[i, 2], 1.0])
            out[i] = (T_WL_end_inv @ (T_WL_i @ pt))[0:3]
        return out


# =========================================================
# 3) NDTMatcher (여기선 Open3D G-ICP로 대체)
#    반환: Z_k (4x4), Rk(6x6) 측정 공분산 근사
# =========================================================
class NDTMatcher:
    """
    실제 NDT 대신 Open3D Generalized-ICP 사용.
    - 입력: source(Nx3), target(Nx3)
    - 출력: Z_k (SE3), R_k (6x6) 근사
    """

    def __init__(self, voxel=0.3, max_iter=50):
        self.voxel = voxel
        self.max_iter = max_iter
        if not HAS_O3D:
            print("[WARN] Open3D not found. NDTMatcher will raise if called.")

    def align(self, src_xyz, tgt_xyz, init_T=np.eye(4)):
        if not HAS_O3D:
            raise RuntimeError("Open3D not available. Install open3d to use G-ICP.")

        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(src_xyz)
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(tgt_xyz)

        if self.voxel and self.voxel > 0:
            src = src.voxel_down_sample(self.voxel)
            tgt = tgt.voxel_down_sample(self.voxel)

        reg = o3d.pipelines.registration.registration_generalized_icp(
            src,
            tgt,
            max_correspondence_distance=1.0,
            init=init_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iter
            ),
        )
        T = reg.transformation

        # 측정 공분산 근사: rmse/fitness 기반의 단순 가중(튜닝 포인트)
        rmse = max(1e-4, reg.inlier_rmse)
        sigma_t = (rmse**2) * 0.05
        sigma_r = (rmse**2) * 0.05
        Rk = np.diag([sigma_t] * 3 + [sigma_r] * 3)
        return T, Rk


# =========================================================
# 4) EKF Update (상태: [IMU 15] + [Extrinsic R_IL(3), t_IL(3)] = 21)
#    잔차 r = log( Z^-1 * (T_IL^-1 * T_Irel * T_IL) )
#    H는 수치 미분(안정/가독). 실제 구현 시 해석 Jacobian으로 최적화 가능.
# =========================================================
class ExtrinsicEKF:
    def __init__(self, P_init=None):
        # state containers
        self.q = np.array([1, 0, 0, 0], float)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.RiL = np.eye(3)
        self.tiL = np.zeros(3)

        # covariance
        self.P = np.eye(21) * 1e-2 if P_init is None else P_init.copy()

    def set_from_propagator(self, prop: IMUPropagator, R_iL=None, t_iL=None):
        q, v, p, bg, ba, P15 = prop.state_tuple()
        self.q, self.v, self.p, self.bg, self.ba = q, v, p, bg, ba
        if R_iL is not None:
            self.RiL = R_iL.copy()
        if t_iL is not None:
            self.tiL = t_iL.copy()
        # embed 15x15 into 21x21
        self.P = np.eye(21) * 1e-2
        self.P[0:15, 0:15] = P15

    def set_extrinsic(self, R_iL, t_iL, P_extrin_diag=None):
        """외부에서 초기 extrinsic 주입 + (선택) extrinsic 공분산 초기화"""
        self.RiL = R_iL.copy()
        self.tiL = t_iL.copy()
        if P_extrin_diag is not None:
            P_extrin_diag = np.asarray(P_extrin_diag, float).reshape(6)
            self.P[15:21, 15:21] = np.diag(P_extrin_diag)

    def freeze_extrinsic(self, freeze_rot=False, freeze_trans=False, big=1e-12):
        """특정부위를 사실상 고정(공분산을 극소로)"""
        if freeze_rot:
            self.P[15:18, 15:18] = np.eye(3) * big
        if freeze_trans:
            self.P[18:21, 18:21] = np.eye(3) * big

    def get_state(self):
        """최종 extrinsic 반환 (R_IL, t_IL)"""
        return self.RiL.copy(), self.tiL.copy()

    def get_rpy_deg(self):
        """R_IL의 roll/pitch/yaw(zyx) [deg]"""
        R = self.RiL
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return np.rad2deg([roll, pitch, yaw])

    def T_IL(self):
        T = np.eye(4)
        T[0:3, 0:3] = self.RiL
        T[0:3, 3] = self.tiL
        return T

    def T_WI(self):
        R = Lie.R_from_q(self.q)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = self.p
        return T

    def predict_rel_I(self, T_WI_prev, T_WI_cur):
        return Lie.inv(T_WI_prev) @ T_WI_cur

    def predict_rel_L(self, T_Irel):
        T_IL = self.T_IL()
        return Lie.inv(T_IL) @ T_Irel @ T_IL

    def residual(self, Z, T_hat):
        # r = log(Z^-1 T_hat)
        return Lie.log6(Lie.inv(Z) @ T_hat)

    def boxplus(self, dx):
        # dx: 21x1 [δθ, δv, δp, δbg, δba, δθ_IL, δt_IL]
        dth = dx[0:3]
        dv = dx[3:6]
        dp = dx[6:9]
        dbg = dx[9:12]
        dba = dx[12:15]
        dthIL = dx[15:18]
        dtIL = dx[18:21]

        # IMU pose
        self.q = Lie.qnorm(Lie.qmul(self.q, Lie.q_from_small(dth)))
        self.v = self.v + dv
        self.p = self.p + dp
        self.bg = self.bg + dbg
        self.ba = self.ba + dba

        # extrinsic
        self.RiL = self.RiL @ Lie.exp6(np.hstack([np.zeros(3), dthIL]))[0:3, 0:3]
        self.tiL = self.tiL + dtIL

    def numeric_H(self, Z, T_WI_prev, T_WI_cur, eps=1e-6):
        """
        수치 미분으로 H 계산 (안전/가독용 프로토타입)
        x = [q(3), v(3), p(3), bg(3), ba(3), R_IL(3), t_IL(3)] -> 21
        q는 오차상태 3축로 취급(좌곱 소각도)
        """
        H = np.zeros((6, 21))
        r0 = self._residual_given_state(Z, T_WI_prev, T_WI_cur)
        for j in range(21):
            dx = np.zeros(21)
            dx[j] = eps
            self.boxplus(dx)
            rj = self._residual_given_state(Z, T_WI_prev, T_WI_cur)
            self.boxplus(-dx)  # revert
            H[:, j] = (rj - r0) / eps
        return H

    def _residual_given_state(self, Z, T_WI_prev, T_WI_cur):
        T_Irel = self.predict_rel_I(T_WI_prev, T_WI_cur)
        T_hatL = self.predict_rel_L(T_Irel)
        return self.residual(Z, T_hatL)

    def update(self, Z, Rk, T_WI_prev, T_WI_cur):
        # residual
        r = self._residual_given_state(Z, T_WI_prev, T_WI_cur)
        # Jacobian
        H = self.numeric_H(Z, T_WI_prev, T_WI_cur)
        # Kalman gain
        S = H @ self.P @ H.T + Rk
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ r
        self.boxplus(dx)
        I = np.eye(21)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rk @ K.T
        # symmetrize
        self.P = 0.5 * (self.P + self.P.T)
        return r, K

    def summary(self):
        """간단 요약 dict (최신 상태 기준)"""
        rpy = self.get_rpy_deg()
        return {
            "R_IL": self.RiL.copy(),
            "t_IL": self.tiL.copy(),
            "rpy_deg": rpy,
            "P_extrin_diag": np.diag(self.P[15:21, 15:21]).copy(),
            "last_resid_norm": (
                self._hist["resid_norm"][-1] if self._hist["resid_norm"] else None
            ),
        }


# =========================================================
# 5) Pipeline (예: 두 스캔 연속 처리)
# =========================================================
class LIOCalibPipeline:
    """
    사용 흐름:
      - IMUPropagator 로 시간축 따라 적분 (traj 저장)
      - Deskewer 로 각 스캔 deskew
      - NDTMatcher 로 Z_k (L_{k-1}->L_k) 추정
      - ExtrinsicEKF.update 로 외부값/바이어스 보정
    """

    def __init__(self, ng=4.24e-5, na=1.96e-4, nwg=1e-6, nwa=2e-4):
        self.prop = IMUPropagator(ng=ng, na=na, nwg=nwg, nwa=nwa)
        self.desk = Deskewer()
        self.ndt = NDTMatcher(voxel=0.3, max_iter=60)
        self.ekf = ExtrinsicEKF()
        # extrinsic 초기값 (회전만 먼저 맞추고 병진 0 근처로 시작을 권장)
        self.ekf.RiL = np.eye(3)
        self.ekf.tiL = np.zeros(3)
        self.ekf.set_from_propagator(self.prop, self.ekf.RiL, self.ekf.tiL)

    def build_traj_buffer(self, imu_times, qs, ps):
        """외부에서 적분해둔 결과로 [(t,q,p)] 구성"""
        traj = []
        for t, q, p in zip(imu_times, qs, ps):
            traj.append((float(t), q.copy(), p.copy()))
        return traj

    def process_pair(
        self,
        scan_km1_xyz,
        scan_k_xyz,
        scan_km1_t,
        scan_k_t,
        scan_km1_start,
        scan_km1_end,
        scan_k_start,
        scan_k_end,
        imu_traj,
    ):
        """
        두 스캔(k-1, k)을 처리해 extrinsic과 상태를 한 번 업데이트
        """
        T_IL = self.ekf.T_IL()
        pts_km1 = self.desk.deskew_points(
            scan_km1_xyz, scan_km1_t, imu_traj, T_IL, scan_km1_start, scan_km1_end
        )
        pts_k = self.desk.deskew_points(
            scan_k_xyz, scan_k_t, imu_traj, T_IL, scan_k_start, scan_k_end
        )

        # NDT (G-ICP)로 상대변환
        Zk, Rk = self.ndt.align(pts_km1, pts_k, init_T=np.eye(4))

        # IMU 상대변환
        # T_WI at scan ends
        q0, p0 = self.desk.interpolate_pose(scan_km1_end, imu_traj)
        q1, p1 = self.desk.interpolate_pose(scan_k_end, imu_traj)
        T_WI_0 = np.eye(4)
        T_WI_0[0:3, 0:3] = Lie.R_from_q(q0)
        T_WI_0[0:3, 3] = p0
        T_WI_1 = np.eye(4)
        T_WI_1[0:3, 0:3] = Lie.R_from_q(q1)
        T_WI_1[0:3, 3] = p1

        # EKF 업데이트
        resid, K = self.ekf.update(Zk, Rk, T_WI_0, T_WI_1)
        return {
            "Zk": Zk,
            "Rk": Rk,
            "residual": resid,
            "K": K,
            "R_IL": self.ekf.RiL.copy(),
            "t_IL": self.ekf.tiL.copy(),
            "P": self.ekf.P.copy(),
        }
