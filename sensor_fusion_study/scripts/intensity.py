#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize
from scipy.signal import find_peaks # For finding peaks in histogram
import os # 파일 경로 처리를 위한 라이브러리
import rclpy
from rclpy.node import Node
from sensor_fusion_study_interfaces.srv import Intensity # Custom service message (Intensity로 변경)

# =============================================================================
# 1. PCD 데이터 파싱 함수 (ASCII 문자열에서 직접 파싱으로 변경)
# =============================================================================
def parse_pcd_string(pcd_string):
    """
    Parses a PCD ASCII string and extracts XYZ coordinates and intensity.
    Assumes the PCD string is in ASCII format and explicitly contains 'intensity'.
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
    The principal components are used as the new axes.
    Args:
        points_3d (numpy.ndarray): (N, 3) array of 3D points (x, y, z).

    Returns:
        tuple:
            - aligned_points (numpy.ndarray): (N, 3) array of aligned points.
            - R_inv (numpy.ndarray): Inverse rotation matrix to transform back to original frame.
            - centroid (numpy.ndarray): Centroid of original points.
    """
    centroid = np.mean(points_3d, axis=0)
    centered_points = points_3d - centroid

    # Compute covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by descending eigenvalues
    # The eigenvector corresponding to the smallest eigenvalue is the normal vector of the plane (new Z-axis)
    # The other two correspond to the principal directions within the plane (new X and Y axes)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # R is the rotation matrix that aligns the principal components with the standard axes
    # New X-axis = eigenvectors[:, 0]
    # New Y-axis = eigenvectors[:, 1]
    # New Z-axis = eigenvectors[:, 2] (plane normal)
    R = eigenvectors

    aligned_points = centered_points @ R # Apply rotation
    
    R_inv = R.T # Inverse of orthogonal matrix (like rotation matrix) is its transpose

    # Note on PCA axis ambiguity: PCA only defines the axes, not their positive direction.
    # The optimization's `theta_z` parameter is crucial for finding the correct chessboard orientation
    # within this PCA-aligned plane, accounting for potential 90/180/270 degree rotations.
    # If the initial PCA X or Y axis is flipped, the optimization should correct it via theta_z.

    return aligned_points, R_inv, centroid

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
        grid_size_x_squares (int): Number of horizontal checker squares (e.g., 7 for 7x8 board).
        grid_size_y_squares (int): Number of vertical checker squares (e.g., 8 for 7x8 board).
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

    points_3d = lidar_points_full[:, :3]
    intensities = lidar_points_full[:, 3]

    aligned_points_3d, R_inv, centroid = pca_align_points(points_3d)
    aligned_points_2d = aligned_points_3d[:, :2] # Z-component is effectively 0 after PCA alignment to plane

    print(f"PCA 정렬 완료. 원본 중심: {centroid}, 정렬된 평면의 점 수: {aligned_points_2d.shape[0]}")

    classified_colors, tau_l, tau_h = classify_intensity_color(intensities)
    print(f"Intensity 분류 완료. Black (<{tau_l:.2f}), White (>{tau_h:.2f}), Gray Zone: [{tau_l:.2f}, {tau_h:.2f}]")
    print(f"분류된 점 수 (흑: {np.sum(classified_colors == 0)}, 백: {np.sum(classified_colors == 1)}, 회색: {np.sum(classified_colors == -1)})")

    # Calculate number of squares based on internal corners
    num_squares_x = internal_corners_x + 1
    num_squares_y = internal_corners_y + 1

    # Initial guess for (tx, ty, theta_z)
    # The model's origin (bottom-left of the checkerboard squares) should align with the points.
    # We estimate the center of the aligned points and use that as an initial translation offset
    # to roughly center the model on the point cloud.
    # This initial guess assumes the checkerboard's origin (bottom-left of the first square)
    # is roughly at the mean of the aligned points, adjusted by half the board size.
    initial_tx = np.mean(aligned_points_2d[:, 0]) - (num_squares_x * checker_size_m / 2.0)
    initial_ty = np.mean(aligned_points_2d[:, 1]) - (num_squares_y * checker_size_m / 2.0)
    initial_guess = [initial_tx, initial_ty, 0.0] # Start with no rotation relative to PCA axes

    print(f"최적화 시작. 초기 추정값 (tx, ty, theta_z): {initial_guess}")

    result = minimize(cost_function, initial_guess,
                      args=(aligned_points_2d, classified_colors,
                            num_squares_x, num_squares_y, checker_size_m),
                      method='Powell', # Powell method is robust for derivative-free optimization
                      options={'disp': True, 'maxiter': 1000})

    optimized_tx, optimized_ty, optimized_theta_z = result.x
    print(f"최적화 완료. 최종 (tx, ty, theta_z): {optimized_tx:.4f}, {optimized_ty:.4f}, {np.degrees(optimized_theta_z):.2f} deg")
    print(f"최종 비용: {result.fun:.4f}")

    # 4. 최적화된 모델에서 3D 코너 추출
    # Define ideal 2D internal corners in the checkerboard's own coordinate system
    # Origin is at the bottom-left of the *internal corner grid*.
    # These are the (X, Y) coordinates of the internal corners if the checkerboard
    # were perfectly aligned with its origin at (0,0,0) and lying on the XY plane.
    ideal_2d_corners = []
    for r in range(internal_corners_y):
        for c in range(internal_corners_x):
            # The internal corners are located at (c+1)*square_size, (r+1)*square_size
            # relative to the checkerboard's origin (bottom-left of the first square).
            ideal_2d_corners.append([ (c + 1) * checker_size_m, (r + 1) * checker_size_m ])
    ideal_2d_corners = np.array(ideal_2d_corners)

    # Apply optimized 2D transformation (rotation and translation) to ideal corners
    # This transforms the ideal corners from the checkerboard's local frame
    # to the PCA-aligned 2D plane.
    cos_theta = np.cos(optimized_theta_z)
    sin_theta = np.sin(optimized_theta_z)
    R_opt = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])

    # Transform ideal corners from checkerboard local frame to PCA-aligned frame
    # The optimized transformation (tx, ty, theta_z) maps the checkerboard's origin
    # to (tx, ty) in the PCA-aligned frame, and rotates it by theta_z.
    # So, to get the corners in the PCA-aligned frame, we rotate them by R_opt
    # and then translate them by (tx, ty).
    transformed_2d_corners_aligned_frame = (R_opt @ ideal_2d_corners.T).T + np.array([optimized_tx, optimized_ty])

    # Transform back to original 3D LiDAR coordinate system
    # Add a Z-component (0) for the 2D corners in the aligned plane before transforming back to 3D
    transformed_2d_corners_3d_aligned = np.hstack((transformed_2d_corners_aligned_frame, np.zeros((transformed_2d_corners_aligned_frame.shape[0], 1))))
    
    # Apply inverse PCA rotation and add back centroid
    # This brings the points from the PCA-aligned frame back to the original LiDAR sensor frame.
    final_3d_corners_lidar_frame = (R_inv @ transformed_2d_corners_3d_aligned.T).T + centroid

    print("--- 논문 방식 코너 검출 완료 ---")

    # =============================================================================
    # 시각화 (Visualization)
    # =============================================================================
    plt.figure(figsize=(10, 8))
    
    # PCA 정렬된 점들 (intensity에 따라 색상 구분)
    # classified_colors: 0 (black), 1 (white), -1 (gray)
    colors = np.array(['black', 'white', 'gray'])
    
    # Plot black points
    black_points = aligned_points_2d[classified_colors == 0]
    plt.scatter(black_points[:, 0], black_points[:, 1], color=colors[0], s=10, alpha=0.7, label='Black Squares (Intensity < tau_l)')
    
    # Plot white points
    white_points = aligned_points_2d[classified_colors == 1]
    plt.scatter(white_points[:, 0], white_points[:, 1], color=colors[1], s=10, alpha=0.7, label='White Squares (Intensity > tau_h)')

    # Plot gray points
    gray_points = aligned_points_2d[classified_colors == -1]
    plt.scatter(gray_points[:, 0], gray_points[:, 1], color=colors[2], s=10, alpha=0.3, label='Gray Zone Points')

    # 최적화된 코너 표시
    plt.scatter(transformed_2d_corners_aligned_frame[:, 0], transformed_2d_corners_aligned_frame[:, 1],
                color='red', marker='o', s=100, edgecolors='yellow', linewidth=2, label='Detected Corners (2D Aligned)')

    # 최적화된 체커보드 모델의 경계선 그리기
    # 체커보드 모델의 4개 코너 (로컬 좌표계)
    model_corners_local = np.array([
        [0, 0],
        [num_squares_x * checker_size_m, 0],
        [num_squares_x * checker_size_m, num_squares_y * checker_size_m],
        [0, num_squares_y * checker_size_m],
        [0, 0] # Close the rectangle
    ])
    
    # PCA 정렬된 프레임으로 변환
    model_corners_aligned = (R_opt @ model_corners_local.T).T + np.array([optimized_tx, optimized_ty])
    plt.plot(model_corners_aligned[:, 0], model_corners_aligned[:, 1], 'b--', label='Optimized Checkerboard Model Boundary')

    plt.title('PCA Aligned Points, Optimized Model, and Detected Corners (2D View)')
    plt.xlabel('X (meters) in PCA Aligned Frame')
    plt.ylabel('Y (meters) in PCA Aligned Frame')
    plt.axis('equal')
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

            if lidar_points_full is None:
                self.get_logger().error("Failed to parse PCD data from string.")
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
            response.corners_x = final_3d_corners_lidar_frame[:, 0].tolist()
            response.corners_y = final_3d_corners_lidar_frame[:, 1].tolist()
            response.corners_z = final_3d_corners_lidar_frame[:, 2].tolist()
            self.get_logger().info(f'Detected {len(final_3d_corners_lidar_frame)} corners and sent response.')

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
