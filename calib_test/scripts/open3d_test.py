import open3d as o3d
import numpy as np 

pcd = o3d.io.read_point_cloud("/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data/e_lidar_imu_calib/room_scan2.pcd")
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
size=0.5, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd,axis])