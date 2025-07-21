#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D 플롯을 위한 Axes3D 임포트
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
    
    IMPORTANT: For accurate corner detection, the input pcd_string should contain
               points that clearly represent a checkerboard pattern with distinct
               intensity variations between black and white squares.
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
# 2. PCA를 이용한 평면 정렬 및 초기 변환 계산
# =============================================================================
def pca_align_points(points_3d):
    """
    Performs PCA to align a 3D point cloud to the XY plane and center it at the origin.
    Derives Z-axis (plane normal) from PCA, forces Z to point towards the origin.
    Then, attempts to align PCA X-axis with the global X-axis (or longest extent within the plane),
    and derives Y-axis to ensure a right-handed system.
    Args:
        points_3d (numpy.ndarray): (N, 3) array of 3D points (x, y, z).

    Returns:
        tuple:
            - aligned_points (numpy.ndarray): (N, 3) array of aligned points.
            - R_inv (numpy.ndarray): Inverse rotation matrix to transform back to original frame.
            - centroid (numpy.ndarray): Centroid of original points.
            - pca_axes (numpy.ndarray): The 3x3 rotation matrix for the PCA axes (v_x, v_y, v_z).
    """
    centroid = np.mean(points_3d, axis=0)
    centered_points = points_3d - centroid

    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # v_z_pca: Normal to the plane (eigenvector corresponding to the smallest eigenvalue)
    v_z_pca = eigenvectors[:, sorted_indices[2]] 

    # Force v_z_pca (normal) to point towards the origin (0,0,0)
    # The vector from the centroid to the origin is -centroid.
    # If the dot product of v_z_pca with this vector is negative, flip v_z_pca.
    if np.dot(v_z_pca, -centroid) < 0:
        v_z_pca = -v_z_pca
    
    # Now, define v_x_pca and v_y_pca.
    # We want v_x_pca to be as parallel as possible to the global X-axis [1,0,0]
    # while being in the plane orthogonal to v_z_pca.
    
    # Project global X-axis onto the plane orthogonal to v_z_pca
    global_x_ref = np.array([1.0, 0.0, 0.0])
    v_x_pca = global_x_ref - np.dot(global_x_ref, v_z_pca) * v_z_pca
    v_x_pca = v_x_pca / np.linalg.norm(v_x_pca) # Normalize

    # Calculate v_y_pca to form a right-handed system: v_y_pca = cross(v_z_pca, v_x_pca)
    v_y_pca = np.cross(v_z_pca, v_x_pca)
    v_y_pca = v_y_pca / np.linalg.norm(v_y_pca) # Normalize

    # Re-calculate v_x_pca to ensure perfect orthogonality with the derived v_y_pca and v_z_pca.
    # This step is for numerical stability and ensuring a perfect right-handed system.
    v_x_pca = np.cross(v_y_pca, v_z_pca)
    v_x_pca = v_x_pca / np.linalg.norm(v_x_pca) # Normalize


    # Construct the final rotation matrix R
    R = np.column_stack((v_x_pca, v_y_pca, v_z_pca))
    pca_axes = R
    aligned_points = centered_points @ R
    R_inv = R.T

    print(f"PCA X-axis (v_x_pca) after orthogonality adjustment: {v_x_pca}")
    print(f"PCA Y-axis (v_y_pca) after orthogonality adjustment: {v_y_pca}")
    print(f"PCA Z-axis (v_z_pca) (normal to plane, forced towards origin): {v_z_pca}")
    print(f"Dot product of v_z_pca with vector from centroid to origin (-centroid): {np.dot(v_z_pca, -centroid):.4f}")
    print(f"Dot product of v_x_pca and v_y_pca: {np.dot(v_x_pca, v_y_pca):.4f}")
    print(f"Dot product of v_x_pca and v_z_pca: {np.dot(v_x_pca, v_z_pca):.4f}")
    print(f"Dot product of v_y_pca and v_z_pca: {np.dot(v_y_pca, v_z_pca):.4f}")
    print(f"Determinant of PCA axes matrix (should be ~1 for right-handed): {np.linalg.det(pca_axes):.4f}")


    return aligned_points, R_inv, centroid, pca_axes

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
    
    print(f"Minimum Intensity (R_L): {R_L:.2f}")
    print(f"Maximum Intensity (R_H): {R_H:.2f}")

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
def cost_function(params, aligned_points_2d, classified_colors,
                  grid_size_x_squares, grid_size_y_squares, checker_size_m):
    """
    Cost function for optimizing checkerboard pose.
    Args:
        params (tuple): (tx, ty, theta_z) - 2D translation and rotation angle on the plane.
        aligned_points_2d (numpy.ndarray): (N, 2) array of PCA-aligned 2D points.
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
    model_min_x = 0
    model_max_x = grid_size_x_squares * checker_size_m
    model_min_y = 0
    model_max_y = grid_size_y_squares * checker_size_m

    for i in range(aligned_points_2d.shape[0]):
        p_2d = aligned_points_2d[i]
        point_intensity_color = classified_colors[i]

        if point_intensity_color == -1: # Ignore gray zone points
            continue

        # Transform point from PCA-aligned frame to checkerboard's local frame
        # Inverse rotation and inverse translation
        transformed_p_2d = R_z.T @ (p_2d - np.array([tx, ty]))

        model_pattern_color = get_checkerboard_pattern_color(
            transformed_p_2d[0], transformed_p_2d[1],
            grid_size_x_squares, grid_size_y_squares, checker_size_m
        )

        if model_pattern_color == -1:
            # Penalize points outside the model bounds. This encourages the model to encompass the points.
            dist_x = max(0, model_min_x - transformed_p_2d[0], transformed_p_2d[0] - model_max_x)
            dist_y = max(0, model_min_y - transformed_p_2d[1], transformed_p_2d[1] - model_max_y)
            cost += (dist_x + dist_y) * 100 # Large penalty for points far outside
        elif point_intensity_color != model_pattern_color:
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

    # Save the full LiDAR points with intensity to a file
    save_points_to_file(lidar_points_full, "checkerboard_points_with_intensity.txt")

    # Initialize visualization variables to empty arrays to prevent NameError
    projected_points_on_plane = np.array([])
    final_3d_corners_lidar_frame = np.array([])
    initial_3d_corners_pca_aligned_model = np.array([])

    points_3d = lidar_points_full[:, :3]
    intensities = lidar_points_full[:, 3]

    if points_3d.shape[0] < 4: # PCA를 수행하기 위한 최소 점 개수 확인
        print("Error: Not enough points for PCA and corner detection. Aborting.")
        return np.array([])

    aligned_points_3d, R_inv, centroid, pca_axes = pca_align_points(points_3d)
    aligned_points_2d = aligned_points_3d[:, :2] # Z-component is effectively 0 after PCA alignment to plane

    print(f"PCA 정렬 완료. 원본 중심: {centroid}, 정렬된 평면의 점 수: {aligned_points_2d.shape[0]}")
    print(f"PCA 축 (R 행렬):\n{pca_axes}")
    print(f"PCA 역회전 행렬 (R_inv):\n{R_inv}")
    print(f"원본 포인트 중심 (Centroid):\n{centroid}")


    # --- Step 0.5: Filter points outside the checkerboard bounding box in the PCA-aligned plane ---
    # Calculate the expected bounding box of the checkerboard in the PCA-aligned frame
    # (assuming the checkerboard's origin (0,0) is its bottom-left corner)
    board_width = (internal_corners_x + 1) * checker_size_m
    board_height = (internal_corners_y + 1) * checker_size_m

    # Initial guess for the checkerboard's center in the PCA-aligned frame
    # This is used to define a crop box around the expected checkerboard location.
    # The `optimized_tx`, `optimized_ty` from the cost function will refine this.
    # For initial cropping, we assume the checkerboard is roughly centered on the mean of the points.
    mean_x_aligned = np.mean(aligned_points_2d[:, 0])
    mean_y_aligned = np.mean(aligned_points_2d[:, 1])

    # Define a crop box centered around the mean, with dimensions slightly larger than the board.
    # This helps to initially isolate the checkerboard region.
    # The actual checkerboard model will be placed by the optimizer.
    crop_margin = 0.1 # meters, small margin around the board
    min_x_crop = mean_x_aligned - (board_width / 2.0) - crop_margin
    max_x_crop = mean_x_aligned + (board_width / 2.0) + crop_margin
    min_y_crop = mean_y_aligned - (board_height / 2.0) - crop_margin
    max_y_crop = mean_y_aligned + (board_height / 2.0) + crop_margin

    # Create a mask for points within the crop box
    x_mask = (aligned_points_2d[:, 0] >= min_x_crop) & (aligned_points_2d[:, 0] <= max_x_crop)
    y_mask = (aligned_points_2d[:, 1] >= min_y_crop) & (aligned_points_2d[:, 1] <= max_y_crop)
    
    cropped_indices = x_mask & y_mask
    
    cropped_aligned_points_full = aligned_points_3d[cropped_indices, :]
    intensities_for_optimization = intensities[cropped_indices] # FIX: Crop intensities as well
    
    if cropped_aligned_points_full.shape[0] < 4:
        print("Error: Not enough points after cropping to checkerboard bounding box. Aborting.")
        return np.array([])

    # Update points_3d, intensities, aligned_points_2d with the cropped data
    points_3d_for_optimization = cropped_aligned_points_full[:, :3]
    aligned_points_2d_for_optimization = cropped_aligned_points_full[:, :2]

    print(f"체커보드 영역으로 필터링 완료. 남은 점 수: {aligned_points_2d_for_optimization.shape[0]}") # Corrected variable name

    classified_colors, tau_l, tau_h = classify_intensity_color(intensities_for_optimization)
    print(f"Intensity 분류 완료. Black (<{tau_l:.2f}), White (>{tau_h:.2f}), Gray Zone: [{tau_l:.2f}, {tau_h:.2f}]")
    print(f"분류된 점 수 (흑: {np.sum(classified_colors == 0)}, 백: {np.sum(classified_colors == 1)}, 회색: {np.sum(classified_colors == -1)})")

    num_squares_x = internal_corners_x + 1
    num_squares_y = internal_corners_y + 1

    initial_theta_guesses = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    
    best_cost = float('inf')
    best_params = None

    for initial_theta in initial_theta_guesses:
        # Initial guess for tx, ty should be based on the *cropped* points' mean
        initial_tx = np.mean(aligned_points_2d_for_optimization[:, 0]) - (num_squares_x * checker_size_m / 2.0)
        initial_ty = np.mean(aligned_points_2d_for_optimization[:, 1]) - (num_squares_y * checker_size_m / 2.0)
        current_initial_guess = [initial_tx, initial_ty, initial_theta]

        print(f"최적화 시도: 초기 추정값 (tx, ty, theta_z): {current_initial_guess}")
        current_result = minimize(cost_function, current_initial_guess,
                                  args=(aligned_points_2d_for_optimization, classified_colors,
                                        num_squares_x, num_squares_y, checker_size_m),
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
    for r in range(internal_corners_y):
        for c in range(internal_corners_x):
            ideal_2d_corners.append([ (c + 1) * checker_size_m, (r + 1) * checker_size_m ])
    ideal_2d_corners = np.array(ideal_2d_corners)

    # Calculate final optimized 3D corners
    cos_theta = np.cos(optimized_theta_z)
    sin_theta = np.sin(optimized_theta_z)
    R_opt = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])

    transformed_2d_corners_aligned_frame = (R_opt @ ideal_2d_corners.T).T + np.array([optimized_tx, optimized_ty])
    transformed_2d_corners_3d_aligned = np.hstack((transformed_2d_corners_aligned_frame, np.zeros((transformed_2d_corners_aligned_frame.shape[0], 1))))
    
    # final_3d_corners_lidar_frame은 PCA-aligned frame에서 원본 LiDAR frame으로 역변환됩니다.
    # 따라서 Z축 방향을 강제한 효과는 PCA-aligned frame 내에서만 적용되며,
    # 최종 결과는 원본 LiDAR 센서의 좌표계로 다시 변환되므로 추가적인 "되돌리기" 작업은 필요하지 않습니다.
    final_3d_corners_lidar_frame = (R_inv @ transformed_2d_corners_3d_aligned.T).T + centroid

    # Save the detected corner points to a file
    save_corner_points_to_file(final_3d_corners_lidar_frame, "detected_corner_points.txt")

    # Calculate initial PCA-aligned 3D corners (for visualization comparison)
    # These corners are placed at the centroid with PCA axes alignment, but no further 2D optimization
    initial_2d_corners_aligned_frame_pca_only = ideal_2d_corners - np.array([board_width/2.0, board_height/2.0]) # Center the ideal corners around (0,0) in PCA frame
    initial_2d_corners_3d_aligned_pca_only = np.hstack((initial_2d_corners_aligned_frame_pca_only, np.zeros((initial_2d_corners_aligned_frame_pca_only.shape[0], 1))))
    initial_3d_corners_pca_aligned_model = (R_inv @ initial_2d_corners_3d_aligned_pca_only.T).T + centroid


    print("--- 논문 방식 코너 검출 완료 ---")
    print("\nC++로 전송될 최종 3D 코너 좌표 (라이다 센서 좌표계):\n")
    # Print as lists for easy comparison with C++
    print("corners_x: [", ", ".join(f"{x:.4f}" for x in final_3d_corners_lidar_frame[:, 0].tolist()), "]")
    print("corners_y: [", ", ".join(f"{y:.4f}" for y in final_3d_corners_lidar_frame[:, 1].tolist()), "]")
    print("corners_z: [", ", ".join(f"{z:.4f}" for z in final_3d_corners_lidar_frame[:, 2].tolist()), "]")
    print(f"\n총 {final_3d_corners_lidar_frame.shape[0]}개의 코너가 검출되었습니다.")


    # =============================================================================
    # 시각화 (Visualization) - 2D PCA 정렬된 평면
    # =============================================================================
    plt.figure(figsize=(10, 8))
    
    colors = np.array(['black', 'white', 'gray'])
    
    # Plot cropped points
    black_points_cropped = aligned_points_2d_for_optimization[classified_colors == 0]
    plt.scatter(black_points_cropped[:, 0], black_points_cropped[:, 1], color=colors[0], s=10, alpha=0.7, label='Black Squares (Intensity < tau_l)')
    
    white_points_cropped = aligned_points_2d_for_optimization[classified_colors == 1]
    plt.scatter(white_points_cropped[:, 0], white_points_cropped[:, 1], color=colors[1], s=10, alpha=0.7, label='White Squares (Intensity > tau_h)')

    gray_points_cropped = aligned_points_2d_for_optimization[classified_colors == -1]
    plt.scatter(gray_points_cropped[:, 0], gray_points_cropped[:, 1], color=colors[2], s=10, alpha=0.3, label='Gray Zone Points')

    plt.scatter(transformed_2d_corners_aligned_frame[:, 0], transformed_2d_corners_aligned_frame[:, 1],
                color='red', marker='o', s=100, edgecolors='yellow', linewidth=2, label='Detected Corners (2D Aligned)')

    model_corners_local = np.array([
        [0, 0],
        [num_squares_x * checker_size_m, 0],
        [num_squares_x * checker_size_m, num_squares_y * checker_size_m],
        [0, num_squares_y * checker_size_m],
        [0, 0] # Close the rectangle
    ])
    
    model_corners_aligned = (R_opt @ model_corners_local.T).T + np.array([optimized_tx, optimized_ty])
    plt.plot(model_corners_aligned[:, 0], model_corners_aligned[:, 1], 'b--', label='Optimized Checkerboard Model Boundary')

    pca_origin_2d = np.mean(aligned_points_2d, axis=0) # Use original aligned points mean for PCA axis visualization
    pca_x_axis_2d = pca_axes[:2, 0] * 0.5 # X축
    pca_y_axis_2d = pca_axes[:2, 1] * 0.5 # Y축
    plt.arrow(pca_origin_2d[0], pca_origin_2d[1], pca_x_axis_2d[0], pca_x_axis_2d[1],
              head_width=0.05, head_length=0.1, fc='purple', ec='purple', label='PCA X-axis')
    plt.arrow(pca_origin_2d[0], pca_origin_2d[1], pca_y_axis_2d[0], pca_y_axis_2d[1],
              head_width=0.05, head_length=0.1, fc='orange', ec='orange', label='PCA Y-axis')


    plt.title('PCA Aligned Points (Cropped), Optimized Model, and Detected Corners (2D View)')
    plt.xlabel('X (meters) in PCA Aligned Frame')
    plt.ylabel('Y (meters) in PCA Aligned Frame')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

    # =============================================================================
    # 시각화 (Visualization) - 3D 라이다 평면, 최적화 코너, PCA 초기 코너 (모두 라이다 좌표계)
    # =============================================================================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 라이다 포인트 (PCA 평면으로 투영된 점들 - 원본 라이다 좌표계)
    centered_original_points = lidar_points_full[:, :3] - centroid
    aligned_original_points_for_proj = centered_original_points @ pca_axes
    
    # Check if aligned_original_points_for_proj is not empty before proceeding
    if aligned_original_points_for_proj.shape[0] > 0:
        aligned_original_points_for_proj[:, 2] = 0 # Flatten Z to 0 on the PCA plane
        projected_points_on_plane = (R_inv @ aligned_original_points_for_proj.T).T + centroid
    else:
        print("Warning: No points to project onto PCA plane for visualization.")
        # projected_points_on_plane remains empty array as initialized at the top of the function.

    # Use the projected points for plotting the plane in 3D
    if projected_points_on_plane.shape[0] > 0:
        scatter_original = ax.scatter(projected_points_on_plane[:, 0], projected_points_on_plane[:, 1], projected_points_on_plane[:, 2],
                                      c=lidar_points_full[:, 3], cmap='viridis', s=5, alpha=0.7, label='Lidar Points Projected onto PCA Plane (Original Lidar Coords, Intensity)')
    else:
        print("Skipping scatter plot for projected_points_on_plane as it's empty.")

    # 2. 최적화된 3D 코너 (C++로 전송될 결과 - 원본 라이다 좌표계)
    if final_3d_corners_lidar_frame.shape[0] > 0:
        ax.scatter(final_3d_corners_lidar_frame[:, 0], final_3d_corners_lidar_frame[:, 1], final_3d_corners_lidar_frame[:, 2],
                   c='red', marker='o', s=150, edgecolors='yellow', linewidth=2, label='Optimized Detected Corners (Original Lidar Coords, to C++)')
    else:
        print("Skipping plotting of final_3d_corners_lidar_frame as it's empty.")

    # 3. PCA 기반 초기 3D 코너 (비교용 - 원본 라이다 좌표계)
    if initial_3d_corners_pca_aligned_model.shape[0] > 0:
        ax.scatter(initial_3d_corners_pca_aligned_model[:, 0], initial_3d_corners_pca_aligned_model[:, 1], initial_3d_corners_pca_aligned_model[:, 2],
                   c='blue', marker='x', s=150, linewidth=3, label='Initial PCA-Aligned Model Corners (Original Lidar Coords, for comparison)')
    else:
        print("Skipping plotting of initial_3d_corners_pca_aligned_model as it's empty.")


    ax.set_xlabel('X (meters) in Original Lidar Frame')
    ax.set_ylabel('Y (meters) in Original Lidar Frame')
    ax.set_zlabel('Z (meters) in Original Lidar Frame')
    ax.set_title('Detected Chessboard Corners in Original Lidar Coordinate System (3D View)')
    plt.legend()
    plt.grid(True)
    plt.show()

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
        # grid_size_x and grid_size_y are now interpreted as INTERNAL corner counts
        internal_corners_x = int(request.grid_size_x)
        internal_corners_y = int(request.grid_size_y)
        checker_size_m = float(request.checker_size_m)

        try:
            lidar_points_full = parse_pcd_string(pcd_data_ascii)

            if lidar_points_full is None or lidar_points_full.shape[0] == 0:
                self.get_logger().error("Failed to parse PCD data or no points received.")
                response.corners_x = []
                response.corners_y = []
                response.corners_z = []
                return response

            # Call the corner estimation function
            final_3d_corners_lidar_frame = estimate_chessboard_corners_paper_method(
                lidar_points_full,
                internal_corners_x=internal_corners_x,
                internal_corners_y=internal_corners_y,
                checker_size_m=checker_size_m
            )

            # Populate the response with the 3D corners in the LiDAR sensor's coordinate system
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
