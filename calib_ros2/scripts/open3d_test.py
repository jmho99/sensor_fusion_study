import open3d as o3d
import numpy as np 

pcd = o3d.io.read_point_cloud("/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/d_multi_lidar_calib/origin_pointclouds/lidar0_0.pcd")
o3d.visualization.draw([pcd])