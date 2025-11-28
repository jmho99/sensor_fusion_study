#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ImuStateParams:
    q_IG: np.ndarray  # (4,) IMU orientation in G
    v_G: np.ndarray  # (3,)
    p_G: np.ndarray  # (3,)
    b_g: np.ndarray  # (3,)
    b_a: np.ndarray  # (3,)
    q_IL: np.ndarray  # (4,) LiDAR->IMU extrinsic rot
    p_IL: np.ndarray  # (3,) LiDAR->IMU extrinsic trans


class EKFUpdate:
    def __init__(self):
        pass

    def ekf_update(
        self,
        params_km1: ImuStateParams,
        params_k_pred: ImuStateParams,
        P_pred: np.ndarray,
        T_LL_meas: np.ndarray,
        R_meas: np.ndarray,
    ) -> Tuple[ImuStateParams, np.ndarray, np.ndarray, np.ndarray]:
        """
        LiDAR motion constraint EKF update.

        inputs:
          params_km1    : k-1 posterior params
          params_k_pred : k prior (IMU predict 결과)
          P_pred        : k prior covariance (21x21)
          T_LL_meas     : NDT/ICP measurement (4x4)
          R_meas        : tunable measurement noise (6x6)

        returns:
          params_k_post : k posterior params
          P_post        : k posterior covariance
          r_k           : residual (6,)
          H_k           : Jacobian (6x21)
        """
        # 1) local state vector
        x0 = self.pack_state(params_k_pred)  # (21,)

        # 2) residual + Jacobian
        r_k = self.residual_fn(x0, params_km1, params_k_pred, T_LL_meas)  # (6,)
        H_k = self.numeric_H(x0, params_km1, params_k_pred, T_LL_meas)  # (6,21)

        # 3) Kalman gain
        S = H_k @ P_pred @ H_k.T + R_meas
        K = P_pred @ H_k.T @ np.linalg.inv(S)

        dx = K @ r_k  # (21,)

        # 4) state retraction
        params_k_post = self.apply_error_state(params_k_pred, dx)

        # 5) covariance update (Joseph form)
        I = np.eye(P_pred.shape[0])
        P_post = (I - K @ H_k) @ P_pred @ (I - K @ H_k).T + K @ R_meas @ K.T

        return params_k_post, P_post, r_k, H_k

    def pack_state(self, params: ImuStateParams) -> np.ndarray:
        """
        error-state 대신 numeric H를 위한 'local parameter vector'
        x = [theta_IG, v, p, b_g, b_a, theta_IL, p_IL]  (21,)
        """
        theta_IG = self.so3_log(self.quat_to_R(params.q_IG))
        theta_IL = self.so3_log(self.quat_to_R(params.q_IL))
        return np.hstack(
            [
                theta_IG,  # 0:3
                params.v_G,  # 3:6
                params.p_G,  # 6:9
                params.b_g,  # 9:12
                params.b_a,  # 12:15
                theta_IL,  # 15:18
                params.p_IL,  # 18:21
            ]
        )

    def so3_log(self, R):
        # 안전한 log
        cos_theta = (np.trace(R) - 1.0) * 0.5
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta < 1e-12:
            return np.zeros(3)
        w_hat = (R - R.T) / (2 * np.sin(theta))
        return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])

    def quat_to_R(self, q):
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )
        return R

    def unpack_state(self, x: np.ndarray, base: ImuStateParams) -> ImuStateParams:
        """
        numeric H용. x를 다시 ImuStateParams로 변환.
        (local theta -> R -> quat)
        """
        th_IG = x[0:3]
        v = x[3:6]
        p = x[6:9]
        b_g = x[9:12]
        b_a = x[12:15]
        th_IL = x[15:18]
        p_IL = x[18:21]

        q_IG = self.R_to_quat(self.so3_exp(th_IG))
        q_IL = self.R_to_quat(self.so3_exp(th_IL))

        return ImuStateParams(
            q_IG=q_IG, v_G=v, p_G=p, b_g=b_g, b_a=b_a, q_IL=q_IL, p_IL=p_IL
        )

    def so3_exp(self, w):
        """회전벡터 w(3,) -> 회전행렬 R(3,3)"""
        theta = np.linalg.norm(w)
        if theta < 1e-12:
            return np.eye(3)
        k = w / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        s = np.sin(theta)
        c = np.cos(theta)
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    def R_to_quat(self, R):
        # robust conversion
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S
        q = np.array([w, x, y, z], float)
        q = q / np.linalg.norm(q)
        return q

    def compute_TLL_pred(
        self, params_km1: ImuStateParams, params_k: ImuStateParams
    ) -> np.ndarray:
        """
        T_LL_pred = (T_GL(k-1))^-1 * T_GL(k)
        T_GL = T_GI * T_IL
        """

        R_GI_km1 = self.quat_to_R(params_km1.q_IG)
        R_GI_k = self.quat_to_R(params_k.q_IG)
        p_G_km1 = params_km1.p_G
        p_G_k = params_k.p_G

        R_IL = self.quat_to_R(params_k.q_IL)
        p_IL = params_k.p_IL

        T_GI_km1 = self.make_T(R_GI_km1, p_G_km1)  # G<-I
        T_GI_k = self.make_T(R_GI_k, p_G_k)  # G<-I
        T_IL = self.make_T(R_IL, p_IL)  # I<-L

        T_GL_km1 = T_GI_km1 @ T_IL
        T_GL_k = T_GI_k @ T_IL

        return self.inv_T(T_GL_km1) @ T_GL_k  # L_{k-1} <- L_k

    def make_T(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = t[0]
        T[1, 3] = t[1]
        T[2, 3] = t[2]
        return T

    def inv_T(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        Ti = np.eye(4)
        Ti[:3, :3] = R.T
        Ti[:3, 3] = -R.T @ t
        return Ti

    def residual_fn(
        self,
        x_vec: np.ndarray,
        params_km1: ImuStateParams,
        params_k_base: ImuStateParams,
        T_LL_meas: np.ndarray,
    ) -> np.ndarray:
        """
        r = Log( (T_pred)^-1 * T_meas )
        numeric H용: k 시점 state만 perturb
        """

        params_k = self.unpack_state(x_vec, params_k_base)

        T_pred = self.compute_TLL_pred(params_km1, params_k)
        T_err = self.inv_T(T_pred) @ T_LL_meas
        return self.se3_log(T_err)  # (6,)

    def se3_log(self, T):
        """T(4,4) -> xi(6,)"""
        R = T[:3, :3]
        t = T[:3, 3]
        phi = self.so3_log(R)
        theta = np.linalg.norm(phi)
        if theta < 1e-12:
            V_inv = np.eye(3)
        else:
            axis = phi / theta
            half_theta = 0.5 * theta
            cot_half_theta = 1 / np.tan(half_theta)
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            V_inv = (
                np.eye(3)
                - 0.5 * K
                + (1 - (theta * cot_half_theta) / 2) / (theta * theta) * (K @ K)
            )
        rho = V_inv @ t
        return np.hstack([phi, rho])  # (6,)

    def numeric_H(
        self,
        x0: np.ndarray,
        params_km1: ImuStateParams,
        params_k_base: ImuStateParams,
        T_LL_meas: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        finite-difference Jacobian: H = dr/dx
        """
        r0 = self.residual_fn(x0, params_km1, params_k_base, T_LL_meas)
        m = r0.shape[0]  # 6
        n = x0.shape[0]  # 21
        H = np.zeros((m, n))
        for j in range(n):
            xp = x0.copy()
            xp[j] += eps
            rp = self.residual_fn(xp, params_km1, params_k_base, T_LL_meas)
            H[:, j] = (rp - r0) / eps
        return H

    def apply_error_state(
        self, params: ImuStateParams, dx: np.ndarray
    ) -> ImuStateParams:
        """
        error-state retraction:
          q <- Exp(dtheta) * q
          v,p,b <- + dv,dp,db
          extrinsic도 동일
        """
        dth_IG = dx[0:3]
        dv = dx[3:6]
        dp = dx[6:9]
        dbg = dx[9:12]
        dba = dx[12:15]
        dth_IL = dx[15:18]
        dp_IL = dx[18:21]

        dq_IG = self.vec_to_quat(dth_IG)
        dq_IL = self.vec_to_quat(dth_IL)

        q_IG_new = self.quat_mul(dq_IG, params.q_IG)
        q_IG_new = q_IG_new / np.linalg.norm(q_IG_new)
        q_IL_new = self.quat_mul(dq_IL, params.q_IL)
        q_IL_new = q_IL_new / np.linalg.norm(q_IL_new)

        return ImuStateParams(
            q_IG=q_IG_new,
            v_G=params.v_G + dv,
            p_G=params.p_G + dp,
            b_g=params.b_g + dbg,
            b_a=params.b_a + dba,
            q_IL=q_IL_new,
            p_IL=params.p_IL + dp_IL,
        )

    def vec_to_quat(self, phi):
        # small angle -> quat
        theta = np.linalg.norm(phi)
        if theta < 1e-12:
            q = np.array([1.0, phi[0] / 2, phi[1] / 2, phi[2] / 2], float)
            q = q / np.linalg.norm(q)
            return q
        axis = phi / theta
        half = theta * 0.5
        return np.array([np.cos(half), *(axis * np.sin(half))], float)

    def quat_mul(self, q1, q2):
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
