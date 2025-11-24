#!/usr/bin/env python3
import sys

sys.path.append("/home/antlab/ROS2/calib_ws/src/lid_imu_calib/python/")

from lid_ndt import NDTScanMatch
from lid_deskew import LidarDeskew, LidarDeskewParams
from imu_state_predict import ImuStatePropagation, ImuStateParams, ImuNoise, CovParams

import numpy as np
import matplotlib.pyplot as plt

import os
import natsort


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

file_path = "/home/antlab/ROS2/calib_ws/pcd_files/"
files = [f for f in os.listdir(file_path) if f.endswith(".pcd")]

if len(files) == 0:
    print("No PCD files found in the specified directory.")
    exit(1)
files = natsort.natsorted(files)
output_path = "/home/antlab/ROS2/calib_ws/deskewed_pcd_files/"
os.makedirs(output_path, exist_ok=True)
deskew = LidarDeskewParams(
    xyz=None,
    t_lid_pts=None,
    t_imu=t[1 : len(positions) + 1],
    q_IG=orientations,
    q_IL=rotations,
    p_IL=translations,
)
for idx in range(len(files)):
    pcd_in_dir = os.path.join(file_path, files[idx])
    pcd_out_dir = os.path.join(output_path, files[idx].replace(".pcd", "_deskewed.pcd"))

    T_GI_0, T_IL = LidarDeskew(
        imu_csv="",
        pcd_in=pcd_in_dir,
        pcd_out=pcd_out_dir,
        params=deskew,
    )

target_name = "frame_0000_deskewed.pcd"
source_name = "frame_0001_deskewed.pcd"
ndt = NDTScanMatch(output_path, target_name, source_name)
