#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D 플롯을 위한 Axes3D 임포트
from matplotlib.patches import Patch # 2D 이미지 범례를 위한 import
import cv2
from scipy.optimize import minimize
from scipy.signal import find_peaks # For finding peaks in histogram
import os # 파일 경로 처리를 위한 라이브러리
import rclpy
from rclpy.node import Node
from sensor_fusion_study_interfaces.srv import Intensity # Custom service message (Intensity로 변경)
import open3d as o3d # open3d 라이브러리 임포트 (더 이상 평면 세그멘테이션에 직접 사용되지 않지만, 다른 잠재적 용도를 위해 유지)

# =============================================================================
# 1. PCD 데이터 파싱 함수 (ASCII 문자열에서 직접 파싱으로 변경)
# =============================================================================
def parse_pcd_string(pcd_string):
    """
    Loads a PCD file and extracts XYZ coordinates and intensity by manual parsing for ASCII.
    This bypasses open3d's limitation of not exposing intensity directly in pcd.points.
    
    IMPORTANT: For accurate corner detection, the input pcd_string MUST contain
               points that clearly represent a checkerboard pattern with distinct
               intensity variations between black and white squares, and a discernible plane.
               If the input data is simple or synthetic (dummy), the algorithm will
               find "corners" based on that limited data, which may not correspond
               to actual physical checkerboard corners.
    """
    xyz_points = []
    intensities = []
    header_parsed = False
    data_started = False
    fields = []
    intensity_field_index = -1

    lines = pcd_string.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not header_parsed:
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
                if 'intensity' in fields:
                    intensity_field_index = fields.index('intensity')
                else:
                    print("Warning: 'intensity' field not found in PCD header. Using dummy intensity.")
            elif line.startswith("DATA"):
                data_type = line.split()[1]
                if data_type != "ascii":
                    raise ValueError("Only ASCII PCD data is supported for manual intensity parsing.")
                header_parsed = True
                data_started = True
        elif data_started:
            parts = line.split()
            if len(parts) != len(fields):
                # print(f"Warning: Skipping malformed data line (expected {len(fields)} fields, got {len(parts)}): {line}")
                continue

            try:
                x = float(parts[fields.index('x')])
                y = float(parts[fields.index('y')])
                z = float(parts[fields.index('z')])
                xyz_points.append([x, y, z])

                if intensity_field_index != -1 and intensity_field_index < len(parts):
                    intensities.append(float(parts[intensity_field_index]))
                else:
                    intensities.append(128.0) # Dummy intensity if not found or malformed
            except ValueError as ve:
                print(f"Warning: Could not parse numeric data in line: {line} - {ve}")
                continue
            except IndexError as ie:
                print(f"Warning: Field index error in line: {line} - {ie}")
                continue

    if not header_parsed:
        raise ValueError("PCD header not found or incomplete in string.")
    if not xyz_points:
        print("PCD string is empty or no points parsed.")
        return None

    xyz_array = np.array(xyz_points)
    intensity_array = np.array(intensities)

    if intensity_array.shape[0] != xyz_array.shape[0]:
        print("Warning: Mismatch between XYZ points and intensity values after parsing. Using dummy intensity.")
        intensity_array = np.full(xyz_array.shape[0], 128.0)

    lidar_points = np.hstack((xyz_array, intensity_array[:, np.newaxis]))
    print(f"Parsed {lidar_points.shape[0]} points from PCD string with XYZ and Intensity.")

    # Print the first few rows to the terminal as requested
    print("\nFirst 5 rows of parsed LiDAR points (XYZI):")
    print(lidar_points[:5])

    return lidar_points

# =============================================================================
# Helper function for 3D visualization of multiple point clouds and axes
# =============================================================================
def plot_single_coordinate_system_point_clouds(point_clouds_data, principal_components=None, centroid=None):
    """
    Matplotlib을 사용하여 여러 3D 점군을 하나의 좌표계에 시각화합니다.
    각 점군은 다른 색상으로 표시됩니다.
    point_clouds_data는 [(points_array_1, title_1, color_1, marker_style, size), ...] 형식의 리스트입니다.
    또한, 좌표축과 PCA 주성분 축을 함께 그려 시각적 이해를 돕습니다.
    """
    fig = plt.figure(figsize=(12, 10)) # 전체 그림 크기 조정
    ax = fig.add_subplot(111, projection='3d')

    for points, title, color, marker_style, size in point_clouds_data:
        # Check if points array is empty before scattering
        if points.size > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, marker=marker_style, label=title, zorder=1 if marker_style == '^' else 0) # zorder 추가

    # 좌표축 그리기 (LiDAR 센서 기준 좌표축)
    # 원점
    origin = [0, 0, 0]
    # 축 길이 (데이터의 스케일에 따라 조절 필요)
    all_points_for_range = np.vstack([data[0] for data in point_clouds_data if data[0].size > 0]) # Ensure points array is not empty
    if all_points_for_range.size == 0:
        max_range = 1.0 # Default if no points
    else:
        max_range = np.max(all_points_for_range) - np.min(all_points_for_range)
    axis_length = max_range * 0.2 if max_range > 0 else 1.0 # 전체 범위의 20% 정도로 설정, 0일 경우 기본값

    # X축 (빨간색)
    ax.quiver(*origin, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='LiDAR X 축')
    # Y축 (초록색)
    ax.quiver(*origin, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='LiDAR Y 축')
    # Z축 (파란색)
    ax.quiver(*origin, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='LiDAR Z 축')

    # PCA 주성분 축 그리기 (체커보드 평면의 중심에서)
    if principal_components is not None and centroid is not None:
        pca_axis_colors = ['purple', 'orange', 'cyan'] # PCA 축 색상
        for i in range(principal_components.shape[1]):
            # 주성분 벡터의 시작점은 데이터의 중심점 (centroid)
            ax.quiver(*centroid,
                      principal_components[0, i] * axis_length,
                      principal_components[1, i] * axis_length,
                      principal_components[2, i] * axis_length,
                      color=pca_axis_colors[i % len(pca_axis_colors)],
                      arrow_length_ratio=0.15, # 일반 축보다 조금 더 길게
                      label=f'PCA 축 {i+1} (평면 중심)')

        # PCA 평면 시각화 추가
        normal_vector = principal_components[:, 2] # 평면의 법선 벡터 (가장 작은 고유값에 해당)
        # 평면의 방정식: ax + by + cz = d, 여기서 (a,b,c)는 normal_vector, d = normal_vector dot centroid
        d_plane = np.dot(normal_vector, centroid)

        # 평면을 그릴 범위 설정 (점군 범위의 일부)
        # x, y 축의 min/max를 사용하여 평면을 적절한 크기로 그립니다.
        min_x_data, max_x_data = ax.get_xlim3d()
        min_y_data, max_y_data = ax.get_ylim3d()

        # 평면을 그릴 격자 생성
        xx, yy = np.meshgrid(np.linspace(min_x_data, max_x_data, 10),
                             np.linspace(min_y_data, max_y_data, 10))

        # Z 값 계산 (평면 방정식으로부터)
        # normal_vector[2]가 0에 가까울 경우 (수직 평면) 예외 처리
        if abs(normal_vector[2]) < 1e-6:
            # 매우 수직인 평면의 경우, Z 대신 X 또는 Y를 고정하여 그릴 수 있지만,
            # 여기서는 단순화를 위해 경고 메시지를 출력하고 평면을 그리지 않거나,
            # 특정 Z값에 고정된 평면으로 대체할 수 있습니다.
            # 체커보드는 보통 완전히 수직이지 않으므로 이 경우는 드뭅니다.
            zz = np.full_like(xx, centroid[2]) # 임시로 centroid의 Z 값으로 평면을 그림 (정확하지 않을 수 있음)
            print("Warning: PCA plane is nearly vertical. Z-coordinate for plane visualization might be inaccurate.")
        else:
            zz = (-normal_vector[0] * xx - normal_vector[1] * yy + d_plane) / normal_vector[2]

        ax.plot_surface(xx, yy, zz, alpha=0.2, color='cyan', label='PCA Plane', zorder=0) # 투명한 청록색 표면
        # plot_surface는 자동으로 범례에 추가되지 않으므로, 더미 scatter를 추가하여 범례에 표시
        ax.scatter([], [], [], color='cyan', alpha=0.2, label='PCA Plane')


    ax.set_title("3D 점군 및 좌표계 시각화")
    ax.set_xlabel('X (미터)')
    ax.set_ylabel('Y (미터)')
    ax.set_zlabel('Z (미터)')
    ax.set_aspect('equal', adjustable='box') # 축 비율을 동일하게 설정
    ax.legend() # 범례 표시
    plt.show()

# =============================================================================
# Helper function for Intensity Histogram
# =============================================================================
def plot_intensity_histogram(intensity_values, tau_l, tau_h, title="Intensity Histogram with Classification Thresholds"):
    """
    강도 값의 히스토그램을 그리고 흑백 분류 임계값을 표시합니다.
    """
    if len(intensity_values) == 0:
        print("히스토그램을 그릴 intensity 값이 없습니다.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(intensity_values, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(tau_l, color='red', linestyle='dashed', linewidth=2, label=f'Tau_L (흑색 임계값): {tau_l:.4f}')
    plt.axvline(tau_h, color='green', linestyle='dashed', linewidth=2, label=f'Tau_H (백색 임계값): {tau_h:.4f}')
    plt.title(title)
    plt.xlabel('강도 값')
    plt.ylabel('빈도')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# =============================================================================
# Helper function for 2D Classified Image
# =============================================================================
def plot_classified_2d_image(white_points, black_points, gray_points, tau_l, tau_h):
    """
    분류된 흑색, 백색, 회색 점군을 2D 이미지로 시각화합니다.
    이 이미지는 PCA로 정렬된 평면의 '정면' (XY 평면 투영)을 나타냅니다.
    """
    all_classified_points = []
    if white_points.size > 0:
        all_classified_points.append(white_points)
    if black_points.size > 0:
        all_classified_points.append(black_points)
    if gray_points.size > 0:
        all_classified_points.append(gray_points)

    if not all_classified_points:
        print("2D 이미지로 그릴 분류된 포인트가 없습니다.")
        return

    # PCA 정렬된 점군을 사용하므로, 이미 Z축은 평면 법선에 정렬되어 있음
    # 따라서 X, Y 좌표만 사용하면 됨
    combined_points_2d = np.vstack([p[:, :2] for p in all_classified_points if p.size > 0])

    if combined_points_2d.shape[0] == 0:
        print("유효한 2D 포인트가 없어 이미지를 그릴 수 없습니다.")
        return

    min_x, max_x = np.min(combined_points_2d[:, 0]), np.max(combined_points_2d[:, 0])
    min_y, max_y = np.min(combined_points_2d[:, 1]), np.max(combined_points_2d[:, 1])

    # 이미지 해상도 정의
    img_width_pixels = 800
    # 종횡비 유지
    x_range = max_x - min_x
    y_range = max_y - min_y

    if x_range == 0 or y_range == 0:
        print("2D 이미지로 그릴 유효한 X 또는 Y 범위가 없습니다. 평면이 너무 얇거나 단일 축에 정렬되어 있을 수 있습니다.")
        return

    pixel_size_x = x_range / img_width_pixels
    img_height_pixels = int(y_range / pixel_size_x) # 종횡비 유지를 위한 높이 계산

    # 빈 RGB 이미지 생성 (흰색 배경으로 초기화)
    image = np.ones((img_height_pixels, img_width_pixels, 3), dtype=np.uint8) * 255 # White background

    # 월드 좌표를 픽셀 좌표로 변환하는 함수
    def world_to_pixel(x, y):
        px = int((x - min_x) / x_range * (img_width_pixels - 1))
        py = int((y - min_y) / y_range * (img_height_pixels - 1))
        # 이미지 표시를 위해 Y축 반전 (원점 좌측 상단)
        py = img_height_pixels - 1 - py
        return px, py

    # 포인트 크기 설정 (픽셀 단위)
    point_size_pixels = 5 # 이 값을 조정하여 포인트 크기를 변경할 수 있습니다.
    half_point_size = point_size_pixels // 2

    # 이미지에 포인트 그리기
    # 포인트가 겹칠 경우 그리기 순서가 중요할 수 있습니다. 여기서는 고정된 순서로 그립니다.
    # 회색 포인트
    for p in gray_points:
        px, py = world_to_pixel(p[0], p[1])
        # 지정된 크기만큼 픽셀 영역 채우기
        for i in range(max(0, px - half_point_size), min(img_width_pixels, px + half_point_size + 1)):
            for j in range(max(0, py - half_point_size), min(img_height_pixels, py + half_point_size + 1)):
                if 0 <= j < img_height_pixels and 0 <= i < img_width_pixels: # Ensure within bounds
                    image[j, i] = [128, 128, 128] # Gray (RGB)

    # 흑색 포인트
    for p in black_points:
        px, py = world_to_pixel(p[0], p[1])
        for i in range(max(0, px - half_point_size), min(img_width_pixels, px + half_point_size + 1)):
            for j in range(max(0, py - half_point_size), min(img_height_pixels, py + half_point_size + 1)):
                if 0 <= j < img_height_pixels and 0 <= i < img_width_pixels: # Ensure within bounds
                    image[j, i] = [0, 0, 0] # Black (RGB)

    # 백색 포인트 - 겹칠 경우 위에 그려지도록 마지막에 그림
    for p in white_points:
        px, py = world_to_pixel(p[0], p[1])
        for i in range(max(0, px - half_point_size), min(img_width_pixels, px + half_point_size + 1)):
            for j in range(max(0, py - half_point_size), min(img_height_pixels, py + half_point_size + 1)):
                if 0 <= j < img_height_pixels and 0 <= i < img_width_pixels: # Ensure within bounds
                    image[j, i] = [255, 255, 255] # White (RGB)


    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title('강도 기반 분류 결과 (2D 이미지 - PCA 평면 뷰)')
    plt.xlabel('X (투영됨)')
    plt.ylabel('Y (투영됨)')
    plt.xticks([]) # X축 눈금 숨기기
    plt.yticks([]) # Y축 눈금 숨기기
    # 2D 이미지를 위한 간단한 범례 수동 생성
    legend_elements = [
        Patch(facecolor='white', label=f'백색 점군 (강도 > {tau_h:.4f})'),
        Patch(facecolor='black', label=f'흑색 점군 (강도 < {tau_l:.4f})'),
        Patch(facecolor='gray', label=f'회색 점군 ({tau_l:.4f} <= 강도 <= {tau_h:.4f})')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. Intensity 기반 흑백 분류 (Gray Zone 포함)
# =============================================================================
def classify_intensity_color(intensities, epsilon_g=4):
    """
    Classifies intensities into black (0), white (1), or gray zone (-1) based on min/max intensity.
    Implements Equation (4) from the paper, using min/max as R_L/R_H.
    Args:
        intensities (numpy.ndarray): Array of intensity values.
        epsilon_g (int): Constant for gray zone definition (>=2).

    Returns:
        tuple:
            - classified_colors (numpy.ndarray): Array of classified colors (0, 1, -1).
            - tau_l (float): Lower threshold for gray zone.
            - tau_h (float): Upper threshold for gray zone.
    """
    R_L = np.min(intensities)
    R_H = np.max(intensities)

    print(f"Minimum Intensity (R_L): {R_L:.4f}")
    print(f"Maximum Intensity (R_H): {R_H:.4f}")

    tau_l = ((epsilon_g - 1) * R_L + R_H) / epsilon_g
    tau_h = (R_L + (epsilon_g - 1) * R_H) / epsilon_g

    classified_colors = np.full_like(intensities, -1, dtype=int) # -1 for gray zone
    classified_colors[intensities < tau_l] = 0 # Black
    classified_colors[intensities > tau_h] = 1 # White

    return classified_colors, tau_l, tau_h


# =============================================================================
# 4. 체커보드 모델 정의 및 패턴 색상 확인
# =============================================================================
def get_checkerboard_pattern_color(x, y, grid_size_x_squares, grid_size_y_squares, checker_size_m):
    """
    Determines the color (0 for black, 1 for white) of the checkerboard pattern at a given (x, y) coordinate.
    Assumes (0,0) square (bottom-left of the entire checkerboard) is white.
    Args:
        x (float): X-coordinate in the checkerboard's local frame (origin at bottom-left square corner).
        y (float): Y-coordinate in the checkerboard's local frame.
        grid_size_x_squares (int): Number of horizontal checker squares.
        grid_size_y_squares (int): Number of vertical checker squares.
        checker_size_m (float): Size of each checker square in meters.

    Returns:
        int: 0 for black, 1 for white, or -1 if outside checkerboard bounds.
    """
    # Convert global (x, y) to grid cell (col, row)
    col = int(x / checker_size_m)
    row = int(y / checker_size_m)

    if not (0 <= col < grid_size_x_squares and 0 <= row < grid_size_y_squares):
        return -1 # Outside checkerboard

    # Checkerboard pattern: (row + col) % 2 == 0 for white, else black
    # Assuming (0,0) square is white
    return 1 if (row + col) % 2 == 0 else 0

# =============================================================================
# 5. 비용 함수 (Cost Function)
# =============================================================================
def cost_function(params, points_in_pca_plane_2d, classified_colors,
                  grid_size_x_squares, grid_size_y_squares, checker_size_m):
    """
    Cost function for optimizing checkerboard pose.
    Args:
        params (tuple): (tx, ty, theta_z) - 2D translation and rotation angle on the plane.
        points_in_pca_plane_2d (numpy.ndarray): (N, 2) array of PCA-transformed 2D points.
        classified_colors (numpy.ndarray): (N,) array of classified colors (0:black, 1:white, -1:gray).
        grid_size_x_squares (int): Number of horizontal checker squares.
        grid_size_y_squares (int): Number of vertical checker squares.
        checker_size_m (float): Size of each checker square in meters.

    Returns:
        float: Total cost.
    """
    tx, ty, theta_z = params
    cost = 0.0

    cos_theta = np.cos(theta_z)
    sin_theta = np.sin(theta_z)
    R_z = np.array([[cos_theta, -sin_theta],
                    [sin_theta, cos_theta]])

    # Define the bounding box of the checkerboard model in its local frame (0 to width, 0 to height)
    # 이제 모델의 중심이 (0,0)이므로, 경계는 -width/2 에서 +width/2 가 됩니다.
    board_width = grid_size_x_squares * checker_size_m
    board_height = grid_size_y_squares * checker_size_m
    model_min_x = -board_width / 2.0
    model_max_x = board_width / 2.0
    model_min_y = -board_height / 2.0
    model_max_y = board_height / 2.0

    # Transform all points once from PCA-frame to checkerboard's local frame (centered)
    transformed_points_2d = (R_z.T @ (points_in_pca_plane_2d - np.array([tx, ty]).reshape(1, -1)).T).T

    for i in range(transformed_points_2d.shape[0]):
        p_2d_transformed = transformed_points_2d[i]
        point_intensity_color = classified_colors[i]

        if point_intensity_color == -1: # Ignore gray zone points
            continue

        # get_checkerboard_pattern_color 함수는 (0,0)이 왼쪽 하단인 좌표계를 기대하므로,
        # 현재 점의 좌표를 다시 왼쪽 하단 기준으로 변환하여 전달합니다.
        p_2d_transformed_for_pattern_lookup_x = p_2d_transformed[0] + board_width / 2.0
        p_2d_transformed_for_pattern_lookup_y = p_2d_transformed[1] + board_height / 2.0

        model_pattern_color_at_point = get_checkerboard_pattern_color(
            p_2d_transformed_for_pattern_lookup_x, p_2d_transformed_for_pattern_lookup_y,
            grid_size_x_squares, grid_size_y_squares, checker_size_m
        )

        if model_pattern_color_at_point == -1:
            # Penalize points outside the overall model bounds.
            dist_x = max(0, model_min_x - p_2d_transformed[0], p_2d_transformed[0] - model_max_x)
            dist_y = max(0, model_min_y - p_2d_transformed[1], p_2d_transformed[1] - model_max_y)
            cost += (dist_x + dist_y) * 100 # Large penalty for points far outside
        elif point_intensity_color != model_pattern_color_at_point:
            cost += 1.0 # Penalty for color mismatch

    return cost

# =============================================================================
# Helper function to save points (x,y,z,intensity) to a file
# =============================================================================
def save_points_to_file(points_array, filename="checkerboard_points_with_intensity.txt"):
    """
    Saves a NumPy array of points (x, y, z, intensity) to a text file.
    Args:
        points_array (numpy.ndarray): (N, 4) array of points (x, y, z, intensity).
        filename (str): The name of the file to save to.
    """
    if points_array.shape[1] != 4:
        print(f"Warning: Expected points_array with 4 columns (x,y,z,intensity), but got {points_array.shape[1]}. Skipping save.")
        return

    try:
        np.savetxt(filename, points_array, fmt='%.4f', delimiter=' ', header='x y z intensity', comments='')
        print(f"Saved {points_array.shape[0]} points to {filename}")
    except Exception as e:
        print(f"Error saving points to file {filename}: {e}")

# =============================================================================
# Helper function to save corner points (x,y,z) to a file
# =============================================================================
def save_corner_points_to_file(corners_array, filename="detected_corner_points.txt"):
    """
    Saves a NumPy array of corner points (x, y, z) to a text file.
    Args:
        corners_array (numpy.ndarray): (N, 3) array of corner points (x, y, z).
        filename (str): The name of the file to save to.
    """
    if corners_array.shape[1] != 3:
        print(f"Warning: Expected corners_array with 3 columns (x,y,z), but got {corners_array.shape[1]}. Skipping save.")
        return

    try:
        np.savetxt(filename, corners_array, fmt='%.4f', delimiter=' ', header='corner_x corner_y corner_z', comments='')
        print(f"Saved {corners_array.shape[0]} corner points to {filename}")
    except Exception as e:
        print(f"Error saving corner points to file {filename}: {e}")


# =============================================================================
# 6. 논문 방식의 코너 검출 메인 함수
# =============================================================================
def estimate_chessboard_corners_paper_method(lidar_points_full,
                                             internal_corners_x, internal_corners_y, checker_size_m):
    """
    Estimates chessboard corners from LiDAR points using a simplified version of the paper's method.
    Args:
        lidar_points_full (numpy.ndarray): (N, 4) array of LiDAR points (x, y, z, intensity).
        internal_corners_x (int): Number of horizontal internal corners (e.g., 6 for 7x8 board).
        internal_corners_y (int): Number of vertical internal corners (e.g., 7 for 7x8 board).
        checker_size_m (float): Size of each checker square in meters.

    Returns:
        numpy.ndarray: (Num_corners, 3) array of estimated 3D corner coordinates in the original LiDAR frame.
    """
    print("\n--- 논문 방식의 코너 검출 시작 ---")

    # Initialize reverted_points to an empty array to prevent "referenced before assignment" error
    reverted_points = np.array([])

    # Save the full LiDAR points with intensity to a file
    save_points_to_file(lidar_points_full, "checkerboard_points_with_intensity.txt")

    points_3d = lidar_points_full[:, :3]
    intensities = lidar_points_full[:, 3]

    if points_3d.shape[0] < 4: # PCA를 수행하기 위한 최소 점 개수 확인
        print("Error: Not enough points for PCA and corner detection. Aborting.")
        return np.array([])

    # =============================================================================
    # PCA 수행
    # =============================================================================
    # 데이터 중심화
    centroid = np.mean(points_3d, axis=0) # 원본 점군의 중심
    centered_points = points_3d - centroid # 중심화된 점들

    # 공분산 행렬 계산
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # 고유값 및 고유 벡터 계산
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 고유값에 따라 고유 벡터 정렬 (내림차순)
    sorted_indices = np.argsort(eigenvalues)[::-1] # 내림차순 정렬
    principal_components = eigenvectors[:, sorted_indices] # 주성분 (pca_axes와 동일하게 사용)
    principal_eigenvalues = eigenvalues[sorted_indices] # 디버그 출력용

    # v_z_pca: 평면 법선 (가장 작은 고유값에 해당하는 고유 벡터)
    v_z_pca = principal_components[:, 2] 

    # v_z_pca (법선)이 라이다 좌표계 원점 (0,0,0)을 향하도록 방향 강제
    # 이 부분은 체커보드 평면의 법선이 항상 센서 방향을 향하도록 보장합니다.
    if np.dot(v_z_pca, -centroid) < 0:
        v_z_pca = -v_z_pca
    v_z_pca = v_z_pca / np.linalg.norm(v_z_pca)

    # v_x_pca_initial: 가장 큰 고유값에 해당하는 고유 벡터 (첫 번째 주성분)
    v_x_pca_initial = principal_components[:, 0]

    # v_x_pca가 v_z_pca에 엄격히 직교하도록 조정
    v_x_pca = v_x_pca_initial - np.dot(v_x_pca_initial, v_z_pca) * v_z_pca
    v_x_pca = v_x_pca / np.linalg.norm(v_x_pca)

    # 논문 조건 적용: mu_1과 라이다 좌표계 X축 사이의 각도가 90도를 넘지 않아야 함
    # 체커보드 X축이 LiDAR X축과 대략적으로 같은 방향을 향하도록 합니다.
    if np.dot(v_x_pca, np.array([1,0,0])) < 0:
        v_x_pca = -v_x_pca

    # 오른손 좌표계를 형성하도록 v_y_pca 계산: v_y_pca = cross(v_z_pca, v_x_pca)
    v_y_pca = np.cross(v_z_pca, v_x_pca)
    v_y_pca = v_y_pca / np.linalg.norm(v_y_pca)

    # 수치적 안정성을 위해 v_x_pca 재계산
    v_x_pca = np.cross(v_y_pca, v_z_pca)
    v_x_pca = v_x_pca / np.linalg.norm(v_x_pca)

    # 최종 회전 행렬 (PCA axes)
    pca_axes = np.column_stack((v_x_pca, v_y_pca, v_z_pca))
    R_inv = pca_axes.T # 역회전 행렬

    print("\nPCA 결과 (정렬된 축):")
    print("주성분 (고유 벡터):\n", pca_axes)
    print("고유값:\n", principal_eigenvalues)

    # --- DEBUG: Verify orthogonality of final pca_axes ---
    dot_xx = np.dot(pca_axes[:, 0], pca_axes[:, 0])
    dot_yy = np.dot(pca_axes[:, 1], pca_axes[:, 1])
    dot_zz = np.dot(pca_axes[:, 2], pca_axes[:, 2])
    dot_xy = np.dot(pca_axes[:, 0], pca_axes[:, 1])
    dot_xz = np.dot(pca_axes[:, 0], pca_axes[:, 2])
    dot_yz = np.dot(pca_axes[:, 1], pca_axes[:, 2])
    print(f"\nDEBUG: pca_axes 직교성 검증 (내적 결과):")
    print(f"  v_x . v_x = {dot_xx:.6f}")
    print(f"  v_y . v_y = {dot_yy:.6f}")
    print(f"  v_z . v_z = {dot_zz:.6f}")
    print(f"  v_x . v_y = {dot_xy:.6f}")
    print(f"  v_x . v_z = {dot_xz:.6f}")
    print(f"  v_y . v_z = {dot_yz:.6f}")
    # --- END DEBUG ---

    # 점군을 PCA 좌표계로 변환 (중심화된 점군에 pca_axes를 곱함)
    # 이제 points_in_pca_frame은 PCA 좌표계의 원점(즉, 원본 점군의 중심점)을 기준으로 합니다.
    points_in_pca_frame = np.dot(centered_points, pca_axes) 

    # 정렬된 점군을 다시 원래대로 되돌리기 (시각화 비교용)
    # points_in_pca_frame이 이미 중심화되어 있으므로, 역회전 후 중심점을 다시 더합니다.
    reverted_points = np.dot(points_in_pca_frame, R_inv) + centroid

    # projected_points_in_pca_plane_2d는 최적화에 사용되므로,
    # PCA 좌표계에서 원점을 중심으로 한 2D 투영이 필요합니다.
    # points_in_pca_frame이 이미 중심화되어 있으므로 그대로 사용합니다.
    projected_points_in_pca_plane_2d = points_in_pca_frame[:, :2]
    # =============================================================================
    # PCA 수행 끝
    # =============================================================================

    # Add debug print here
    print(f"DEBUG: projected_points_in_pca_plane_2d.shape (before masks): {projected_points_in_pca_plane_2d.shape}")

    print(f"PCA 정렬 완료. 원본 중심: {centroid}, PCA 평면의 점 수: {projected_points_in_pca_plane_2d.shape[0]}")
    print(f"PCA 축 (R 행렬):\n{pca_axes}")
    print(f"PCA 역회전 행렬 (R_inv):\n{R_inv}")
    print(f"원본 포인트 중심 (Centroid):\n{centroid}")


    # --- 체커보드 바운딩 박스 외부 점 필터링 (적응형 크롭) ---
    # PCA 변환된 점군의 실제 최소/최대 값을 사용하여 크롭 영역을 정의합니다.
    min_x_points_pca = np.min(projected_points_in_pca_plane_2d[:, 0])
    max_x_points_pca = np.max(projected_points_in_pca_plane_2d[:, 0])
    min_y_points_pca = np.min(projected_points_in_pca_plane_2d[:, 1])
    max_y_points_pca = np.max(projected_points_in_pca_plane_2d[:, 1])

    crop_margin = 0.05 # meters, 크롭 마진을 조금 더 작게 설정
    min_x_crop = min_x_points_pca - crop_margin
    max_x_crop = max_x_points_pca + crop_margin
    min_y_crop = min_y_points_pca - crop_margin
    max_y_crop = max_y_points_pca + crop_margin

    # 마스크를 명시적으로 생성
    x_mask = (projected_points_in_pca_plane_2d[:, 0] >= min_x_crop) & (projected_points_in_pca_plane_2d[:, 0] <= max_x_crop)
    y_mask = (projected_points_in_pca_plane_2d[:, 1] >= min_y_crop) & (projected_points_in_pca_plane_2d[:, 1] <= max_y_crop)
    
    # np.logical_and를 사용하여 마스크 결합
    cropped_indices = np.logical_and(x_mask, y_mask)
    
    # --- 디버깅: 인덱싱 전 차원 불일치 명시적 확인 ---
    print(f"DEBUG: points_in_pca_frame.shape: {points_in_pca_frame.shape}")
    print(f"DEBUG: intensities.shape: {intensities.shape}")
    print(f"DEBUG: projected_points_in_pca_plane_2d.shape: {projected_points_in_pca_plane_2d.shape}")
    print(f"DEBUG: x_mask.shape: {x_mask.shape}")
    print(f"DEBUG: y_mask.shape: {y_mask.shape}")
    print(f"DEBUG: cropped_indices.shape: {cropped_indices.shape}")
    print(f"DEBUG: type(cropped_indices): {type(cropped_indices)}")


    if points_in_pca_frame.shape[0] != cropped_indices.shape[0]:
        print(f"CRITICAL ERROR (Pre-Index Check): points_in_pca_frame.shape[0] ({points_in_pca_frame.shape[0]}) != cropped_indices.shape[0] ({cropped_indices.shape[0]})")
        return np.array([]) # 이 근본적인 불일치가 발생하면 중단

    if intensities.shape[0] != cropped_indices.shape[0]:
        print(f"CRITICAL ERROR (Pre-Index Check): intensities.shape[0] ({intensities.shape[0]}) != cropped_indices.shape[0] ({cropped_indices.shape[0]})")
        return np.array([]) # 이 근본적인 불일치가 발생하면 중단
    # --- 디버깅 끝 ---

    points_for_optimization_full = points_in_pca_frame[cropped_indices, :]
    intensities_for_optimization = intensities[cropped_indices]
    
    if points_for_optimization_full.shape[0] < 4:
        print("Error: Not enough points after cropping to checkerboard bounding box. Aborting.")
        return np.array([])

    cropped_points_in_pca_plane = points_for_optimization_full[:, :2]

    print(f"체커보드 영역으로 필터링 완료. 남은 점 수: {cropped_points_in_pca_plane.shape[0]}")

    classified_colors, tau_l, tau_h = classify_intensity_color(intensities_for_optimization)
    print(f"Intensity 분류 완료. Black (<{tau_l:.2f}), White (>{tau_h:.2f}), Gray Zone: [{tau_l:.2f}, {tau_h:.2f}]")
    print(f"분류된 점 수 (흑: {np.sum(classified_colors == 0)}, 백: {np.sum(classified_colors == 1)}, 회색: {np.sum(classified_colors == -1)})")

    initial_theta_guesses = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    
    best_cost = float('inf')
    best_params = None

    # 체커보드 전체 사각형 수 계산
    num_squares_x_total = internal_corners_x + 1
    num_squares_y_total = internal_corners_y + 1
    board_width = num_squares_x_total * checker_size_m
    board_height = num_squares_y_total * checker_size_m

    for initial_theta in initial_theta_guesses:
        # Initial guess for tx, ty should be the mean of the cropped points,
        # as the model's local origin is now its center.
        initial_tx = np.mean(cropped_points_in_pca_plane[:, 0])
        initial_ty = np.mean(cropped_points_in_pca_plane[:, 1])
        current_initial_guess = [initial_tx, initial_ty, initial_theta]

        print(f"최적화 시도: 초기 추정값 (tx, ty, theta_z): {current_initial_guess}")
        current_result = minimize(cost_function, current_initial_guess,
                                  args=(cropped_points_in_pca_plane, classified_colors,
                                        num_squares_x_total, num_squares_y_total, checker_size_m), # 전체 사각형 수 전달
                                  method='Powell',
                                  options={'disp': False, 'maxiter': 1000})

        if current_result.fun < best_cost:
            best_cost = current_result.fun
            best_params = current_result.x
            print(f"새로운 최적 결과 발견! 비용: {best_cost:.4f}, 파라미터: {best_params}")

    if best_params is None:
        raise RuntimeError("Optimization failed for all initial guesses.")

    optimized_tx, optimized_ty, optimized_theta_z = best_params
    print(f"최적화 완료. 최종 (tx, ty, theta_z): {optimized_tx:.4f}, {optimized_ty:.4f}, {np.degrees(optimized_theta_z):.2f} deg")
    print(f"최종 비용: {best_cost:.4f}")

    # 4. 최적화된 모델에서 3D 코너 추출
    ideal_2d_corners = []
    # 코너 좌표를 모델의 중심을 기준으로 생성
    for r in range(internal_corners_y):
        for c in range(internal_corners_x):
            # (c+1, r+1)은 내부 코너의 인덱스 (1부터 시작)
            # 이를 전체 보드의 중심을 기준으로 하는 좌표로 변환
            corner_x = (c + 1) * checker_size_m - board_width / 2.0
            corner_y = (r + 1) * checker_size_m - board_height / 2.0
            ideal_2d_corners.append([corner_x, corner_y])
    ideal_2d_corners = np.array(ideal_2d_corners)


    # Calculate final optimized 3D corners
    cos_theta = np.cos(optimized_theta_z)
    sin_theta = np.sin(optimized_theta_z)
    R_opt = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])

    # Optimized corners are now relative to the PCA plane's origin (which is the centroid of the cropped points)
    # and rotated by R_opt.
    optimized_corners_in_pca_plane = (R_opt @ ideal_2d_corners.T).T + np.array([optimized_tx, optimized_ty])
    optimized_corners_3d_in_pca_plane = np.hstack((optimized_corners_in_pca_plane, np.zeros((optimized_corners_in_pca_plane.shape[0], 1))))
    
    # 디버깅: PCA 평면에서의 최적화된 코너 출력
    print("\nDEBUG: PCA 평면에서의 최적화된 2D 코너 (첫 5개):")
    print(optimized_corners_in_pca_plane[:5])
    print("\nDEBUG: PCA 평면에서의 3D 코너 (Z=0, 첫 5개):")
    print(optimized_corners_3d_in_pca_plane[:5])

    # final_3d_corners_lidar_frame은 PCA-frame에서 원본 LiDAR frame으로 역변환됩니다.
    # points_in_pca_frame이 이미 중심화되었으므로, 역변환 시에도 centroid를 다시 더합니다.
    # 수정된 부분: 역변환 행렬 적용 방식 변경
    rotated_corners_centered_in_lidar_frame = np.dot(optimized_corners_3d_in_pca_plane, R_inv)
    final_3d_corners_lidar_frame = rotated_corners_centered_in_lidar_frame + centroid

    # 디버깅: 역회전만 적용된 코너 (LiDAR 프레임, 중심화됨)
    print("\nDEBUG: 역회전만 적용된 코너 (LiDAR 프레임, 중심화됨, 첫 5개):")
    print(rotated_corners_centered_in_lidar_frame[:5])
    print("\nDEBUG: 최종 역회전 행렬 (R_inv):\n", R_inv)
    print("\nDEBUG: 최종 중심점 (Centroid):\n", centroid)

    # Save the detected corner points to a file
    save_corner_points_to_file(final_3d_corners_lidar_frame, "detected_corner_points.txt")

    # Calculate initial PCA-aligned 3D corners (for visualization comparison)
    # 이제 initial_corners_in_pca_plane_centered도 중심을 기준으로 계산
    initial_corners_in_pca_plane_centered = ideal_2d_corners # ideal_2d_corners 자체가 이미 중심 기준
    initial_corners_3d_in_pca_plane_centered = np.hstack((initial_corners_in_pca_plane_centered, np.zeros((initial_corners_in_pca_plane_centered.shape[0], 1))))
    initial_3d_corners_in_original_frame_from_pca = np.dot(initial_corners_3d_in_pca_plane_centered, R_inv) + centroid


    print("--- 논문 방식 코너 검출 완료 ---")
    print("\nC++로 전송될 최종 3D 코너 좌표 (라이다 센서 좌표계):\n")
    # Print as lists for easy comparison with C++
    print("corners_x: [", ", ".join(f"{x:.4f}" for x in final_3d_corners_lidar_frame[:, 0].tolist()), "]")
    print("corners_y: [", ", ".join(f"{y:.4f}" for y in final_3d_corners_lidar_frame[:, 1].tolist()), "]")
    print("corners_z: [", ", ".join(f"{z:.4f}" for z in final_3d_corners_lidar_frame[:, 2].tolist()), "]")
    print(f"\n총 {final_3d_corners_lidar_frame.shape[0]}개의 코너가 검출되었습니다.")

    # 디버깅: 검출된 코너 점들의 좌표 범위 출력
    if final_3d_corners_lidar_frame.size > 0:
        print("\n검출된 코너 점들의 좌표 범위 (LiDAR 프레임):")
        print(f"  X 범위: [{np.min(final_3d_corners_lidar_frame[:, 0]):.4f}, {np.max(final_3d_corners_lidar_frame[:, 0]):.4f}]")
        print(f"  Y 범위: [{np.min(final_3d_corners_lidar_frame[:, 1]):.4f}, {np.max(final_3d_corners_lidar_frame[:, 1]):.4f}]")
        print(f"  Z 범위: [{np.min(final_3d_corners_lidar_frame[:, 2]):.4f}, {np.max(final_3d_corners_lidar_frame[:, 2]):.4f}]")

    # 디버깅: 각 코너 점의 PCA 평면으로부터의 거리 출력
    # IMPORTANT: Use pca_axes[:, 2] for normal_vector here, as this is the actual normal of the plane
    # that the points are transformed to/from.
    normal_vector_for_distance_check = pca_axes[:, 2] # Use the final, potentially flipped, v_z_pca
    d_plane_for_distance_check = np.dot(normal_vector_for_distance_check, centroid)
    print("\nDEBUG: 각 코너 점의 PCA 평면으로부터의 거리:")
    for i, corner in enumerate(final_3d_corners_lidar_frame):
        distance = np.dot(normal_vector_for_distance_check, corner) - d_plane_for_distance_check
        print(f"  코너 {i}: 평면으로부터의 거리 = {distance:.6f}")


    # =============================================================================
    # 시각화 (Visualization) - 2D PCA 변환된 평면
    # =============================================================================
    plt.figure(figsize=(10, 8))
    
    colors = np.array(['black', 'white', 'gray'])
    
    # Plot filtered points in PCA plane using cropped_points_in_pca_plane
    # classified_colors는 이미 cropped_points_in_pca_plane에 해당하므로, 이들을 사용합니다.
    black_points_pca_plane = cropped_points_in_pca_plane[classified_colors == 0]
    plt.scatter(black_points_pca_plane[:, 0], black_points_pca_plane[:, 1], color=colors[0], s=10, alpha=0.7, label='흑색 사각형 (강도 < tau_l)')
    
    white_points_pca_plane = cropped_points_in_pca_plane[classified_colors == 1]
    plt.scatter(white_points_pca_plane[:, 0], white_points_pca_plane[:, 1], color=colors[1], s=10, alpha=0.7, label='백색 사각형 (강도 > tau_h)')

    gray_points_pca_plane = cropped_points_in_pca_plane[classified_colors == -1]
    plt.scatter(gray_points_pca_plane[:, 0], gray_points_pca_plane[:, 1], color=colors[2], s=10, alpha=0.3, label='회색 영역 점')

    plt.scatter(optimized_corners_in_pca_plane[:, 0], optimized_corners_in_pca_plane[:, 1],
                color='red', marker='o', s=100, edgecolors='yellow', linewidth=2, label='검출된 코너 (2D PCA 평면)', zorder=3) # 코너 마커 zorder 추가

    # 모델 경계는 이제 중심을 기준으로 정의됩니다.
    model_corners_local_centered = np.array([
        [-board_width / 2.0, -board_height / 2.0],
        [ board_width / 2.0, -board_height / 2.0],
        [ board_width / 2.0,  board_height / 2.0],
        [-board_width / 2.0,  board_height / 2.0],
        [-board_width / 2.0, -board_height / 2.0] # Close the rectangle
    ])
    
    model_corners_transformed_to_pca_plane = (R_opt @ model_corners_local_centered.T).T + np.array([optimized_tx, optimized_ty])
    plt.plot(model_corners_transformed_to_pca_plane[:, 0], model_corners_transformed_to_pca_plane[:, 1], 'b--', label='최적화된 체커보드 모델 경계', zorder=2) # 모델 경계 zorder 추가

    # PCA 원점은 전체 점군의 평균을 사용합니다.
    pca_origin_2d = np.mean(projected_points_in_pca_plane_2d, axis=0)
    pca_x_axis_2d = pca_axes[:2, 0] * 0.5 # X축
    pca_y_axis_2d = pca_axes[:2, 1] * 0.5 # Y축
    plt.arrow(pca_origin_2d[0], pca_origin_2d[1], pca_x_axis_2d[0], pca_x_axis_2d[1],
              head_width=0.05, head_length=0.1, fc='purple', ec='purple', label='PCA X축', zorder=2) # 화살표 zorder 추가
    plt.arrow(pca_origin_2d[0], pca_origin_2d[1], pca_y_axis_2d[0], pca_y_axis_2d[1],
              head_width=0.05, head_length=0.1, fc='orange', ec='orange', label='PCA Y축', zorder=2) # 화살표 zorder 추가


    plt.title('PCA 변환된 점군, 최적화된 모델 및 검출된 코너 (2D 뷰)')
    plt.xlabel('X (미터) (PCA 변환된 프레임)')
    plt.ylabel('Y (미터) (PCA 변환된 프레임)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

    # =============================================================================
    # 시각화 (Visualization) - 좌표계 변환 비교
    # =============================================================================
    print("모든 점군을 한 화면에 시각화 중...")
    plot_single_coordinate_system_point_clouds(
        [
            (points_3d, "원본 라이다 점군 (Raw Data)", 'blue', '.', 1),
            (points_in_pca_frame, "PCA 좌표계로 변환된 점군", 'red', '.', 1), # PCA 적용된 점군
            (reverted_points, "원래 라이다 좌표계로 되돌린 점군", 'green', '.', 1), # 역변환된 점군
            (final_3d_corners_lidar_frame, "검출된 코너 점 (LiDAR 프레임)", 'magenta', '^', 1000) # 크기 1000, 삼각형 마커로 변경
        ],
        principal_components=pca_axes, # PCA 주성분 전달
        centroid=centroid # 중심점 전달
    )

    # 원본과 되돌린 점군이 거의 동일한지 확인
    # reverted_points는 이미 위에서 계산되었으므로 다시 계산할 필요 없음
    if np.allclose(points_3d, reverted_points, atol=1e-6):
        print("\n성공: 원래 점군과 되돌린 점군이 일치합니다.")
    else:
        print("\n경고: 원래 점군과 되돌린 점군이 약간 다릅니다. (허용 오차 범위 내일 수 있음)")
        diff = np.linalg.norm(points_3d - reverted_points)
        print(f"평균 차이: {diff / len(points_3d)}")

    # Intensity 히스토그램 및 2D 분류 이미지 시각화 제거됨
    # if intensities is not None and len(intensities) > 1:
    #     plot_intensity_histogram(intensities, tau_l, tau_h, "강도 히스토그램 (분류 임계값 포함)")
    #     binarized_points_black = points_in_pca_frame[classified_colors == 0]
    #     binarized_points_white = points_in_pca_frame[classified_colors == 1]
    #     binarized_points_gray = points_in_pca_frame[classified_colors == -1]
    #     plot_classified_2d_image(binarized_points_white, binarized_points_black, binarized_points_gray, tau_l, tau_h)
    # else:
    #     print("\nIntensity 값이 없거나 샘플 수가 부족하여 추가적인 시각화를 수행할 수 없습니다.")


    return final_3d_corners_lidar_frame

# =============================================================================
# ROS2 Service Server Implementation
# =============================================================================
class LidarCornerDetectionService(Node):
    def __init__(self):
        super().__init__('lidar_corner_detection_service')
        # 서비스 타입도 Intensity로 변경
        self.srv = self.create_service(Intensity, 'detect_lidar_corners', self.detect_corners_callback)
        self.get_logger().info('Lidar Corner Detection Service Ready.')

    def detect_corners_callback(self, request, response):
        self.get_logger().info('Received request for corner detection.')
        
        pcd_data_ascii = request.pcd_data_ascii
        internal_corners_x = int(request.grid_size_x)
        internal_corners_y = int(request.grid_size_y)
        checker_size_m = float(request.checker_size_m)
        
        try:
            lidar_points_full = parse_pcd_string(pcd_data_ascii)

            if lidar_points_full is None or lidar_points_full.shape[0] == 0:
                self.get_logger().error("Failed to parse PCD data or no points received. Please ensure valid PCD data is sent.")
                response.corners_x = []
                response.corners_y = []
                response.corners_z = []
                return response

            final_3d_corners_lidar_frame = estimate_chessboard_corners_paper_method(
                lidar_points_full,
                internal_corners_x=internal_corners_x,
                internal_corners_y=internal_corners_y,
                checker_size_m=checker_size_m
            )

            if final_3d_corners_lidar_frame.size > 0:
                response.corners_x = final_3d_corners_lidar_frame[:, 0].tolist()
                response.corners_y = final_3d_corners_lidar_frame[:, 1].tolist()
                response.corners_z = final_3d_corners_lidar_frame[:, 2].tolist()
                self.get_logger().info(f'Detected {len(final_3d_corners_lidar_frame)} corners and sent response.')
            else:
                self.get_logger().warn("No corners detected after processing. Sending empty response.")
                response.corners_x = []
                response.corners_y = []
                response.corners_z = []

        except Exception as e:
            self.get_logger().error(f"Error during corner detection: {e}")
            response.corners_x = []
            response.corners_y = []
            response.corners_z = []

        return response

def main(args=None):
    rclpy.init(args=args)
    lidar_corner_detection_service = LidarCornerDetectionService()
    rclpy.spin(lidar_corner_detection_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
