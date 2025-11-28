#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple


@dataclass
class ImuStateParams:
    q_IG: np.ndarray  # quaternion [w,x,y,z], I^G q (IMU orientation w.r.t. G)
    v_G: np.ndarray  # velocity in G, shape (3,)
    p_G: np.ndarray  # position in G, shape (3,)
    b_g: np.ndarray  # gyro bias, shape (3,)
    b_a: np.ndarray  # accel bias, shape (3,)

    q_IL: np.ndarray  # quaternion [w,x,y,z], I^L q
    p_IL: np.ndarray  # translation, shape (3,)


@dataclass
class ImuNoise:
    sigma_g: float = 0.01
    sigma_a: float = 0.1
    sigma_wg: float = 1e-4
    sigma_wa: float = 1e-3


@dataclass
class CovParams:
    orient: float = 5.0
    posit: float = 1.0
    vel: float = 0.5
    gyro_b: float = 1.0
    acc_b: float = 0.1


class ImuStatePropagation:

    def __init__(self, init_state: ImuStateParams):
        self.state = init_state
        self.noise = ImuNoise()
        self.cov_params = CovParams()

    def propagate(
        self, omega_m: np.ndarray, accel_m: np.array, dt: float
    ) -> Tuple[ImuStateParams, np.ndarray]:

        new_state = self.imuStatePrediction(omega_m, accel_m, dt)
        Phi = self.computePhi(omega_m, accel_m, dt)
        G = self.computeG(omega_m, accel_m, dt)
        P, Qd = self.computePAndQ(dt)
        P_next = np.zeros((15, 15))
        P_next = Phi @ P @ Phi.T + G @ Qd @ G.T
        P_next = self.extend_P_15_to_21(P_next)

        self.state = new_state

        return new_state, P_next

    def imuStatePrediction(
        self,
        omega_m: np.ndarray,
        accel_m: np.array,
        dt: float,
        g_vec: np.ndarray = np.array([0, 0, 9.81]),
    ) -> ImuStateParams:

        q_IG = self.state.q_IG
        v_G = self.state.v_G
        p_G = self.state.p_G
        b_g = self.state.b_g
        b_a = self.state.b_a
        q_IL = self.state.q_IL
        p_IL = self.state.p_IL

        # ==============equation 8==============
        # _{G}^{I_{i+1}}q = exp( 1/2 * Ω(ω_{m,i} − b_{g,i} ) * Δt )  *  _{G}^{I_{i}}q
        omega_corr = omega_m - b_g  # (ω_m - b_g,i)
        theta = omega_corr * dt
        angle = np.linalg.norm(theta)
        if angle < 1e-12:
            # 작은 각 근사
            half = 0.5 * theta
            q = np.array([1.0, half[0], half[1], half[2]])
            dq = q / np.linalg.norm(q)
        else:
            axis = theta / angle
            half_angle = 0.5 * angle
            s = np.sin(half_angle)
            q = np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s])
            dq = q / np.linalg.norm(q)

        w_a, x_a, y_a, z_a = dq
        w_b, x_b, y_b, z_b = q_IG
        w_c = w_b * w_a - x_b * x_a - y_b * y_a - z_b * z_a
        x_c = w_b * x_a + x_b * w_a + y_b * z_a - z_b * y_a
        y_c = w_b * y_a - x_b * z_a + y_b * w_a + z_b * x_a
        z_c = w_b * z_a + x_b * y_a - y_b * x_a + z_b * w_a
        equ8_rhs = np.array([w_c, x_c, y_c, z_c])
        q_IG_next = equ8_rhs / np.linalg.norm(equ8_rhs)

        w_d, x_d, y_d, z_d = q_IG_next
        R_IG = np.array(
            [
                [
                    1 - 2 * (y_d**2 + z_d**2),
                    2 * (x_d * y_d - z_d * w_d),
                    2 * (x_d * z_d + y_d * w_d),
                ],
                [
                    2 * (x_d * y_d + z_d * w_d),
                    1 - 2 * (x_d**2 + z_d**2),
                    2 * (y_d * z_d - x_d * w_d),
                ],
                [
                    2 * (x_d * z_d - y_d * w_d),
                    2 * (y_d * z_d + x_d * w_d),
                    1 - 2 * (x_d**2 + y_d**2),
                ],
            ]
        )
        R_GI = R_IG.T
        # ==============equation 9==============
        # _{}^{G}v_{i+1} = _{}^{G}v_{i} - _{}^{G}g Δt + R_{I_{i}}^{G} (a_{m,i} - b_{a,i}) Δt
        a_corr = accel_m - b_a
        v_G_next = v_G - g_vec * dt + R_GI @ a_corr * dt
        # ==============equation 10==============
        # _{}^{G}p_{i+1} = _{}^{G}p_{I_{i}} + _{}^{G}v_{I_{i}} Δt - 0.5 _{}^{G}g Δt^2 + 0.5 R_{I_{i}}^{G} (a_{m,i} - b_{a,i}) Δt^2
        p_G_next = (
            p_G + v_G * dt - 0.5 * g_vec * dt * dt + 0.5 * (R_GI @ a_corr) * dt * dt
        )
        # ==============equation 11, 12==============
        b_g_next = b_g.copy()
        b_a_next = b_a.copy()

        # ==============equation 13, 14==============
        q_IL_next = q_IL.copy()
        p_IL_next = p_IL.copy()

        return ImuStateParams(
            q_IG=q_IG_next,
            v_G=v_G_next,
            p_G=p_G_next,
            b_g=b_g_next,
            b_a=b_a_next,
            q_IL=q_IL_next,
            p_IL=p_IL_next,
        )

    def computePhi(
        self, omega_m: np.ndarray, accel_m: np.ndarray, dt: float
    ) -> np.ndarray:
        # -----------setup------------
        q_IG = self.state.q_IG
        b_g = self.state.b_g
        b_a = self.state.b_a

        w_a, x_a, y_a, z_a = q_IG
        R_IG = np.array(
            [
                [
                    1 - 2 * (y_a**2 + z_a**2),
                    2 * (x_a * y_a - z_a * w_a),
                    2 * (x_a * z_a + y_a * w_a),
                ],
                [
                    2 * (x_a * y_a + z_a * w_a),
                    1 - 2 * (x_a**2 + z_a**2),
                    2 * (y_a * z_a - x_a * w_a),
                ],
                [
                    2 * (x_a * z_a - y_a * w_a),
                    2 * (y_a * z_a + x_a * w_a),
                    1 - 2 * (x_a**2 + y_a**2),
                ],
            ]
        )
        R_GI = R_IG.T

        omega = omega_m - b_g
        acc = accel_m - b_a
        dtheta = -omega * dt
        theta = np.linalg.norm(dtheta)

        if theta < 1e-12:
            R_Ii1_Ii = np.eye(3)
            x_a, y_a, z_a = dtheta
            skew_omega = np.array(
                [[0, -z_a, y_a], [z_a, 0, -x_a], [-y_a, x_a, 0]], dtype=float
            )
            Jr = np.eye(3) - 0.5 * skew_omega
        else:
            axis = dtheta / theta
            x_b, y_b, z_b = axis
            K = np.array([[0, -z_b, y_b], [z_b, 0, -x_b], [-y_b, x_b, 0]], dtype=float)
            R_Ii1_Ii = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

            x_c, y_c, z_c = dtheta
            skew_omega = np.array(
                [[0, -z_c, y_c], [z_c, 0, -x_c], [-y_c, x_c, 0]], dtype=float
            )
            theta2 = theta * theta
            s = np.sin(theta)
            c = np.cos(theta)

            A = (1.0 - c) / theta2
            B = (theta - s) / (theta2 * theta)
            Jr = np.eye(3) - A * skew_omega + B * (skew_omega @ skew_omega)

        Phi = np.eye(15)

        # -----------orientation------------
        Phi[0:3, 0:3] = R_Ii1_Ii
        Phi[0:3, 9:12] = -R_Ii1_Ii @ Jr * dt
        # -----------position------------
        Phi[3:6, 6:9] = np.eye(3) * dt
        x_d, y_d, z_d = acc
        skew_acc = np.array(
            [[0, -z_d, y_d], [z_d, 0, -x_d], [-y_d, x_d, 0]], dtype=float
        )
        Phi[3:6, 0:3] = -0.5 * R_GI @ skew_acc * (dt**2)
        Phi[3:6, 12:15] = -0.5 * R_GI * (dt**2)
        # -----------velocity------------
        Phi[6:9, 0:3] = -R_GI @ skew_acc * dt
        Phi[6:9, 12:15] = -R_GI * dt

        return Phi

    def computeG(
        self, omega_m: np.ndarray, accel_m: np.ndarray, dt: float
    ) -> np.ndarray:
        # -----------setup------------
        q_IG = self.state.q_IG
        b_g = self.state.b_g
        b_a = self.state.b_a

        w_a, x_a, y_a, z_a = q_IG
        R_IG = np.array(
            [
                [
                    1 - 2 * (y_a**2 + z_a**2),
                    2 * (x_a * y_a - z_a * w_a),
                    2 * (x_a * z_a + y_a * w_a),
                ],
                [
                    2 * (x_a * y_a + z_a * w_a),
                    1 - 2 * (x_a**2 + z_a**2),
                    2 * (y_a * z_a - x_a * w_a),
                ],
                [
                    2 * (x_a * z_a - y_a * w_a),
                    2 * (y_a * z_a + x_a * w_a),
                    1 - 2 * (x_a**2 + y_a**2),
                ],
            ]
        )
        R_GI = R_IG.T

        omega = omega_m - b_g
        acc = accel_m - b_a
        dtheta = -omega * dt
        theta = np.linalg.norm(dtheta)

        if theta < 1e-12:
            R_Ii1_Ii = np.eye(3)
            x_a, y_a, z_a = dtheta
            skew_omega = np.array(
                [[0, -z_a, y_a], [z_a, 0, -x_a], [-y_a, x_a, 0]], dtype=float
            )
            Jr = np.eye(3) - 0.5 * skew_omega
        else:
            axis = dtheta / theta
            x_b, y_b, z_b = axis
            K = np.array([[0, -z_b, y_b], [z_b, 0, -x_b], [-y_b, x_b, 0]], dtype=float)
            R_Ii1_Ii = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

            x_c, y_c, z_c = dtheta
            skew_omega = np.array(
                [[0, -z_c, y_c], [z_c, 0, -x_c], [-y_c, x_c, 0]], dtype=float
            )
            theta2 = theta * theta
            s = np.sin(theta)
            c = np.cos(theta)

            A = (1.0 - c) / theta2
            B = (theta - s) / (theta2 * theta)
            Jr = np.eye(3) - A * skew_omega + B * (skew_omega @ skew_omega)

        G = np.zeros((15, 12))

        # -----------orientation------------
        G[0:3, 0:3] = -R_Ii1_Ii @ Jr * dt
        # -----------position------------
        G[3:6, 3:6] = -0.5 * R_GI * (dt**2)
        # -----------velocity------------
        G[6:9, 3:6] = -R_GI * dt
        # -----------gyro------------
        G[9:12, 6:9] = np.eye(3)
        # -----------accel------------
        G[12:15, 9:12] = np.eye(3)

        return G

    def computePAndQ(self, dt: float):
        P = np.zeros((15, 15))
        P[0:3, 0:3] = (np.deg2rad(self.cov_params.orient) ** 2) * np.eye(3)
        P[3:6, 3:6] = (self.cov_params.posit**2) * np.eye(3)
        P[6:9, 6:9] = (self.cov_params.vel**2) * np.eye(3)
        P[9:12, 9:12] = (np.deg2rad(self.cov_params.gyro_b) ** 2) * np.eye(3)
        P[12:15, 12:15] = (self.cov_params.acc_b**2) * np.eye(3)

        # 3) Qd (12x12)
        Qd = np.zeros((12, 12))
        sg2 = self.noise.sigma_g**2
        sa2 = self.noise.sigma_a**2
        swg2 = self.noise.sigma_wg**2
        swa2 = self.noise.sigma_wa**2

        Qd[0:3, 0:3] = sg2 * dt * np.eye(3)  # gyro noise
        Qd[3:6, 3:6] = sa2 * dt * np.eye(3)  # accel noise
        Qd[6:9, 6:9] = swg2 * dt * np.eye(3)  # gyro bias RW
        Qd[9:12, 9:12] = swa2 * dt * np.eye(3)  # accel bias RW
        return P, Qd

    def extend_P_15_to_21(self, P_15: np.ndarray) -> np.ndarray:
        P_21 = np.zeros((21, 21))
        P_21[0:15, 0:15] = P_15
        P_21[15:21, 15:21] = np.eye(6) * 1e-6  # small cov for extrinsic
        return P_21


def main():
    data = np.genfromtxt("/home/antlab/imu_test_trim_01.csv", delimiter=",", dtype=str)
    # data = np.loadtxt("/home/antlab/imu_test_trim_01.csv", delimiter=",", skiprows=0)
    t_sec = data[:, 0].astype(float)
    t_nanosec = data[:, 1].astype(float)
    t = t_sec + t_nanosec * 1e-9
    # gyro (angular velocity)
    gx = data[:, 16].astype(float)
    gy = data[:, 17].astype(float)
    gz = data[:, 18].astype(float)

    # accel (linear acceleration)
    ax = data[:, 28].astype(float)
    ay = data[:, 29].astype(float)
    az = data[:, 30].astype(float)

    gyro = np.stack([gx, gy, gz], axis=1)
    acc = np.stack([ax, ay, az], axis=1)

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    state = ImuStateParams(
        q_IG=q0,
        v_G=np.zeros(3),
        p_G=np.zeros(3),
        b_g=np.zeros(3),
        b_a=np.zeros(3),
        q_IL=np.array([1.0, 0.0, 0.0, 0.0]),
        p_IL=np.zeros(3),
    )
    prediction = ImuStatePropagation(state)

    positions = []
    velocities = []
    orientations = []

    prev_t = t[0]

    for i in range(1, len(t)):
        dt = t[i] - prev_t
        if dt <= 0:
            continue

        omega_m = gyro[i]  # [gx, gy, gz]
        accel_m = acc[i]  # [ax, ay, az]

        states = prediction.imuStatePrediction(omega_m, accel_m, dt)

        positions.append(states.p_G.copy())
        velocities.append(states.v_G.copy())
        orientations.append(states.q_IG.copy())

        prev_t = t[i]

    positions = np.array(positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    plt.show()


if __name__ == "__main__":
    main()
