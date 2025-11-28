#!/usr/bin/env python3
import sys

sys.path.append("/home/antlab/ROS2/calib_ws/src/lid_imu_calib/python/")

from lid_ndt import NDTScanMatch
from lid_deskew import LidarDeskew, LidarDeskewParams
from imu_state_predict import ImuStatePropagation, ImuStateParams, ImuNoise, CovParams
from ekf_update import EKFUpdate, ImuStateParams

import numpy as np
import matplotlib.pyplot as plt

import os
import natsort
import shutil

from datetime import datetime

# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Vscode에서 각 파트를 셀로 확인 가능.
# 초기 T_IL 설정 -> IMU predict -> for( Lidar deskew -> Lidar ICP -> EKF update -> IMU predict update)
# 현재 imu 데이터는 csv로 저장해서 불로오고, 라이다 데이터는 각 프레임의 pcd를 불러와서 진행.
# 초기 T_IL은 다른 코드에서 진행.
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
# =========================
# T_IL init 설정
# =========================

R_IL_init = np.array(
    [
        [-0.873996, -0.476837, 0.0935768],
        [-0.336593, 0.732957, 0.591168],
        [-0.350479, 0.485181, -0.801102],
    ]
)

q_IL_init = np.zeros(4)
trace = R_IL_init[0, 0] + R_IL_init[1, 1] + R_IL_init[2, 2]
w_IL, x_IL, y_IL, z_IL = q_IL_init[0], q_IL_init[1], q_IL_init[2], q_IL_init[3]
if trace > 0:
    s = 2 * np.sqrt(1 + trace)  # s=4*qw
    w_IL = np.sqrt(1 + trace) / 4
    x_IL = (R_IL_init[2, 1] - R_IL_init[1, 2]) / s
    y_IL = (R_IL_init[0, 2] - R_IL_init[2, 0]) / s
    z_IL = (R_IL_init[1, 0] - R_IL_init[0, 1]) / s
else:
    if R_IL_init[0, 0] > R_IL_init[1, 1] and R_IL_init[0, 0] > R_IL_init[2, 2]:
        s = 2 * np.sqrt(
            1 + R_IL_init[0, 0] - R_IL_init[1, 1] - R_IL_init[2, 2]
        )  # s=4*qx
        w_IL = (R_IL_init[2, 1] - R_IL_init[1, 2]) / s
        x_IL = s / 4
        y_IL = (R_IL_init[0, 1] + R_IL_init[1, 0]) / s
        z_IL = (R_IL_init[0, 2] + R_IL_init[2, 0]) / s
    elif R_IL_init[1, 1] > R_IL_init[2, 2]:
        s = 2 * np.sqrt(
            1 - R_IL_init[0, 0] + R_IL_init[1, 1] - R_IL_init[2, 2]
        )  # s=4*qy
        w_IL = (R_IL_init[0, 2] - R_IL_init[2, 0]) / s
        x_IL = (R_IL_init[0, 1] + R_IL_init[1, 0]) / s
        y_IL = s / 4
        z_IL = (R_IL_init[1, 2] + R_IL_init[2, 1]) / s
    else:
        s = 2 * np.sqrt(
            1 - R_IL_init[0, 0] - R_IL_init[1, 1] + R_IL_init[2, 2]
        )  # s=4*qz
        w_IL = (R_IL_init[1, 0] - R_IL_init[0, 1]) / s
        x_IL = (R_IL_init[0, 2] - R_IL_init[2, 0]) / s
        y_IL = (R_IL_init[1, 2] - R_IL_init[2, 1]) / s
        z_IL = s / 4
q_IL_init = q_IL_init / np.linalg.norm(q_IL_init)
# %%
# =========================
# IMU 데이터 불러오기
# =========================
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
# %%
# =========================
# IMU predict INIT
# =========================
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
rotations = []
translations = []
prev_t = t[0]

for i in range(1, len(t)):
    dt = t[i] - prev_t
    if dt <= 0:
        continue
    omega_m = gyro[i]  # [gx, gy, gz]
    accel_m = acc[i]  # [ax, ay, az]
    states, P_pred = prediction.propagate(omega_m, accel_m, dt)
    positions.append(states.p_G.copy())
    velocities.append(states.v_G.copy())
    orientations.append(states.q_IG.copy())
    rotations.append(states.q_IL.copy())
    translations.append(states.p_IL.copy())
    prev_t = t[i]
positions = np.array(positions)
orientations = np.array(orientations)
velocities = np.array(velocities)

all_init_state = ImuStateParams(
    q_IG=orientations,
    v_G=velocities,
    p_G=positions,
    b_g=np.zeros(3),
    b_a=np.zeros(3),
    q_IL=rotations,
    p_IL=translations,
)
# %%
# =========================
# Lidar Setting
# =========================
file_path = "/home/antlab/ROS2/calib_ws/pcd_files/"
files = [f for f in os.listdir(file_path) if f.endswith(".pcd")]

if len(files) == 0:
    print("No PCD files found in the specified directory.")
    exit(1)
files = natsort.natsorted(files)
output_path = "/home/antlab/ROS2/calib_ws/deskewed_pcd_files/"
os.makedirs(output_path, exist_ok=True)
deskew_params = LidarDeskewParams(
    xyz=None,
    t_lid_pts=None,
    t_imu=t[1 : len(positions) + 1],
    q_IG=orientations,
    q_IL=rotations,
    p_IL=translations,
    p_IG=positions,
)
# %%
# =========================
# Calibration Loop
# =========================
for idx in range(len(files)):
    print(f"{[idx]} 프레임 시작 시각: ", datetime.now().time())
    # =========================
    # Lidar Deskew
    # =========================
    pcd_in_dir = os.path.join(file_path, files[idx])
    pcd_out_dir = os.path.join(
        output_path,
        files[idx].replace(".pcd", "_deskewed.pcd"),
    )

    deskew_setup = LidarDeskew(
        imu_csv="",
        pcd_in=pcd_in_dir,
        pcd_out=pcd_out_dir,
        params=deskew_params,
        imu_state=all_init_state,
    )
    deskew_result = deskew_setup.deskew_pcd_with_imu(
        pcd_in=pcd_in_dir,
        pcd_out=pcd_out_dir,
    )

    # =========================
    # Lidar ICP
    # =========================
    if idx <= 0:
        print(f"{[idx]} 프레임 완료 시각: ", datetime.now().time())
        imu_params_prev = deskew_result[0]
        pcd_target_dir = pcd_out_dir
        continue

    pcd_source = pcd_out_dir
    pcd_target = pcd_target_dir
    ndt = NDTScanMatch("", pcd_target, pcd_source, False)

    # =========================
    # EKF Update
    # =========================
    imu_params_post, P_post, residual_curr, H_curr = EKFUpdate().ekf_update(
        params_km1=imu_params_prev,
        params_k_pred=deskew_result[0],
        P_pred=P_pred,
        T_LL_meas=ndt.T_result,
        R_meas=np.eye(6) * 0.01,
    )

    # =========================
    # IMU state Update
    # =========================
    prediction = ImuStatePropagation(imu_params_post)
    positions = []
    velocities = []
    orientations = []
    rotations = []
    translations = []
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
        rotations.append(states.q_IL.copy())
        translations.append(states.p_IL.copy())
        prev_t = t[i]
    positions = np.array(positions)
    orientations = np.array(orientations)

    deskew_params = LidarDeskewParams(
        xyz=None,
        t_lid_pts=None,
        t_imu=t[1 : len(positions) + 1],
        q_IG=orientations,
        q_IL=rotations,
        p_IL=translations,
        p_IG=positions,
    )
    print(f"{[idx]} 프레임 완료 시각: ", datetime.now().time())
    T_GI_prev = deskew_result[1]
    P_pred = P_post
    imu_params_prev = imu_params_post
    pcd_target_dir = pcd_out_dir

# %%
# =========================
# print result
# =========================
w, x, y, z = imu_params_post.q_IL
R = np.array(
    [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
)

T = np.eye(4)
T[0:3, 0:3] = R
T[0:3, 3] = imu_params_post.p_IL

print("[INFO] Calibration process completed.")
print(f"[INFO] Calibrated extrinsic translation: {T}")

# %%
