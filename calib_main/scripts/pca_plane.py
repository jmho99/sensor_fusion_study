import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D 플롯을 위한 import
from matplotlib.patches import Patch # 2D 이미지 범례를 위한 import

# PCD 파일 경로 설정
# Replace this with the actual path to your PCD file.
pcd_file_path = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/cam_lidar_calib/lidar_plane_points.pcd"

def plot_single_coordinate_system_point_clouds(point_clouds_data, principal_components=None, centroid=None):
    """
    Matplotlib을 사용하여 여러 3D 점군을 하나의 좌표계에 시각화합니다.
    각 점군은 다른 색상으로 표시됩니다.
    point_clouds_data는 [(points_array_1, title_1, color_1), ...] 형식의 리스트입니다.
    또한, 좌표축과 PCA 주성분 축을 함께 그려 시각적 이해를 돕습니다.
    """
    fig = plt.figure(figsize=(10, 8)) # 전체 그림 크기 조정
    ax = fig.add_subplot(111, projection='3d')

    for points, title, color in point_clouds_data:
        # Check if points array is empty before scattering
        if points.size > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=color, marker='.', label=title)

    # 좌표축 그리기
    # 원점
    origin = [0, 0, 0]
    # 축 길이 (데이터의 스케일에 따라 조절 필요)
    # 데이터의 최대/최소 값 범위에서 적절한 축 길이를 추정합니다.
    all_points_for_range = np.vstack([data[0] for data in point_clouds_data if data[0].size > 0]) # Ensure points array is not empty
    if all_points_for_range.size == 0:
        max_range = 1.0 # Default if no points
    else:
        max_range = np.max(all_points_for_range) - np.min(all_points_for_range)
    axis_length = max_range * 0.2 if max_range > 0 else 1.0 # 전체 범위의 20% 정도로 설정, 0일 경우 기본값

    # X축 (빨간색)
    ax.quiver(*origin, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='X Axis')
    # Y축 (초록색)
    ax.quiver(*origin, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Y Axis')
    # Z축 (파란색)
    ax.quiver(*origin, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Z Axis')

    # PCA 주성분 축 그리기
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
                      label=f'PCA Axis {i+1}')


    ax.set_title("PCA 기반 점군 정렬 및 이진화 (단일 좌표계)")
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_aspect('equal', adjustable='box') # 축 비율을 동일하게 설정
    ax.legend() # 범례 표시
    plt.show()

def plot_intensity_histogram(intensity_values, tau_l, tau_h, title="Intensity Histogram with Classification Thresholds"):
    """
    강도 값의 히스토그램을 그리고 흑백 분류 임계값을 표시합니다.
    """
    if len(intensity_values) == 0:
        print("히스토그램을 그릴 intensity 값이 없습니다.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(intensity_values, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(tau_l, color='red', linestyle='dashed', linewidth=2, label=f'Tau_L (Black Threshold): {tau_l:.4f}')
    plt.axvline(tau_h, color='green', linestyle='dashed', linewidth=2, label=f'Tau_H (White Threshold): {tau_h:.4f}')
    plt.title(title)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_classified_2d_image(white_points, black_points, gray_points, tau_l, tau_h):
    """
    분류된 흑색, 백색, 회색 점군을 2D 이미지로 시각화합니다.
    이 이미지는 정렬된 평면의 '정면' (XY 평면 투영)을 나타냅니다.
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

    combined_points = np.vstack(all_classified_points)

    # 정렬된 평면의 X, Y 좌표를 사용하여 2D 이미지 투영
    # PCA를 통해 평면의 법선이 Z축에 정렬되었으므로, X와 Y는 평면 내의 주요 차원을 나타냅니다.
    min_x, max_x = np.min(combined_points[:, 0]), np.max(combined_points[:, 0])
    min_y, max_y = np.min(combined_points[:, 1]), np.max(combined_points[:, 1])

    # 이미지 해상도 정의
    img_width_pixels = 800
    # 종횡비 유지
    x_range = max_x - min_x
    y_range = max_y - min_y

    if x_range == 0 or y_range == 0:
        print("2D 이미지로 그릴 유효한 X 또는 Y 범위가 없습니다. 평면이 너무 얇거나 단일 축에 정렬되어 있을 수 있습니다.")
        # 이 경우, 이미지 생성을 건너뛰거나 다른 투영을 고려할 수 있습니다.
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
                image[j, i] = [128, 128, 128] # Gray (RGB)

    # 흑색 포인트
    for p in black_points:
        px, py = world_to_pixel(p[0], p[1])
        for i in range(max(0, px - half_point_size), min(img_width_pixels, px + half_point_size + 1)):
            for j in range(max(0, py - half_point_size), min(img_height_pixels, py + half_point_size + 1)):
                image[j, i] = [0, 0, 0] # Black (RGB)

    # 백색 포인트 - 겹칠 경우 위에 그려지도록 마지막에 그림
    for p in white_points:
        px, py = world_to_pixel(p[0], p[1])
        for i in range(max(0, px - half_point_size), min(img_width_pixels, px + half_point_size + 1)):
            for j in range(max(0, py - half_point_size), min(img_height_pixels, py + half_point_size + 1)):
                image[j, i] = [255, 255, 255] # White (RGB)


    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Intensity 기반 분류 결과 (2D 이미지 - 정면 뷰)")
    plt.xlabel('X (투영됨)')
    plt.ylabel('Y (투영됨)')
    plt.xticks([]) # X축 눈금 숨기기
    plt.yticks([]) # Y축 눈금 숨기기
    # 2D 이미지를 위한 간단한 범례 수동 생성
    legend_elements = [
        Patch(facecolor='white', label=f'백색 점군 (Intensity > {tau_h:.4f})'),
        Patch(facecolor='black', label=f'흑색 점군 (Intensity < {tau_l:.4f})'),
        Patch(facecolor='gray', label=f'회색 점군 ({tau_l:.4f} <= Intensity <= {tau_h:.4f})')
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


def perform_pca_and_align(pcd_path):
    """
    PCD 파일을 로드하고, PCA를 수행하여 점군을 정렬한 다음,
    다시 원래대로 되돌리는 과정을 시연합니다.
    정렬된 평면에서 intensity를 사용해서 이진화합니다.
    """
    print(f"PCD 파일 로드 중: {pcd_file_path}")
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"PCD 파일을 로드하는 데 실패했습니다: {e}")
        print("파일 경로를 확인하거나, 파일이 유효한 PCD 형식인지 확인하십시오.")
        return

    if not pcd.has_points():
        print("로드된 PCD에 포인트가 없습니다. 유효한 파일인지 확인하십시오.")
        return

    points = np.asarray(pcd.points)
    print(f"로드된 포인트 수: {len(points)}")

    # 1. Intensity 값 추출
    intensity_values = None
    if pcd.has_colors():
        intensity_values = np.asarray(pcd.colors)[:, 0]
        print(f"PCD에서 Intensity 값 (색상 채널에서 추출) 범위: [{np.min(intensity_values):.4f}, {np.max(intensity_values):.4f}]")
    else:
        print("PCD에 색상 정보가 없거나, Open3D가 intensity를 색상으로 매핑하지 않았습니다.")
        print("데모를 위해 임의의 intensity 값 생성 (0-1 범위).")
        intensity_values = np.random.rand(len(points)) # 0-1 범위의 임의 값 생성
        print(f"생성된 Intensity 값 범위: [{np.min(intensity_values):.4f}, {np.max(intensity_values):.4f}]")

    # Ensure intensity_values is a 1D array
    if intensity_values is not None and intensity_values.ndim > 1:
        intensity_values = intensity_values.flatten()


    # 2. PCA 수행
    # 데이터 중심화
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # 공분산 행렬 계산
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # 고유값 및 고유 벡터 계산
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 고유값에 따라 고유 벡터 정렬 (내림차순)
    # principal_components의 열은 정규화된 고유 벡터이며, 고유값의 내림차순으로 정렬됩니다.
    # 즉, principal_components[:,0]은 가장 큰 분산 방향, principal_components[:,1]은 두 번째,
    # principal_components[:,2]는 가장 작은 분산 방향(평면의 법선)입니다.
    sorted_indices = np.argsort(eigenvalues)[::-1] # 내림차순 정렬
    principal_components = eigenvectors[:, sorted_indices]
    principal_eigenvalues = eigenvalues[sorted_indices]

    print("\nPCA 결과:")
    print("주성분 (고유 벡터):\n", principal_components)
    print("고유값:\n", principal_eigenvalues)

    # 3. 주성분에 따라 점군 정렬 (OpenCV 체커보드 평면 설정 기준)
    # PCA의 고유 벡터를 직접 사용하여 점군을 정렬합니다.
    # 이렇게 하면 principal_components[:,0]이 새로운 X축, principal_components[:,1]이 새로운 Y축,
    # principal_components[:,2] (평면 법선)가 새로운 Z축이 됩니다.
    # 이는 평면이 XY 평면에 놓이게 하여 2D 투영 시 "정면" 뷰를 제공합니다.
    aligned_points = np.dot(centered_points, principal_components) + centroid

    # 4. 정렬된 점군을 다시 원래대로 되돌리기
    # 역회전 행렬을 적용합니다. (직교 행렬의 역행렬은 전치 행렬과 같습니다.)
    reverted_points = np.dot(aligned_points - centroid, principal_components.T) + centroid

    # 5. Intensity 기반 흑백 분류 (논문 방식)
    classified_colors = np.array([])
    tau_l = None
    tau_h = None
    binarized_points_white = np.array([])
    binarized_points_black = np.array([])
    binarized_points_gray = np.array([])

    if intensity_values is not None and len(intensity_values) > 1: # 최소 2개 이상의 샘플 필요
        classified_colors, tau_l, tau_h = classify_intensity_color(intensity_values)

        # 분류된 색상에 따라 포인트 분리
        binarized_points_black = aligned_points[classified_colors == 0]
        binarized_points_white = aligned_points[classified_colors == 1]
        binarized_points_gray = aligned_points[classified_colors == -1]

        print(f"분류된 포인트 수: 흑색={len(binarized_points_black)}, 백색={len(binarized_points_white)}, 회색={len(binarized_points_gray)}")

        # 이진화 결과 히스토그램 별도 이미지로 표시
        plot_intensity_histogram(intensity_values, tau_l, tau_h, "Intensity Histogram with Black/White/Gray Classification")

        # 분류된 평면을 2D 이미지로 별도의 창으로 보여줌 (정면 뷰)
        plot_classified_2d_image(binarized_points_white, binarized_points_black, binarized_points_gray, tau_l, tau_h)

    else:
        print("\nIntensity 값이 없거나 샘플 수가 부족하여 흑백 분류를 수행할 수 없습니다.")


    # 모든 점군을 한 번에 시각화
    print("모든 점군을 한 화면에 시각화 중...")
    plot_single_coordinate_system_point_clouds(
        [
            (points, "원본 점군", 'blue'),
            (aligned_points, "정렬된 점군 (PCA 기반)", 'red'),
            (reverted_points, "원래대로 되돌린 점군", 'green')
            # 분류된 점군은 별도 창에서 2D 이미지로 표시되므로 여기서는 제외
        ],
        principal_components=principal_components, # PCA 주성분 전달
        centroid=centroid # 중심점 전달
    )

    # 원본과 되돌린 점군이 거의 동일한지 확인
    if np.allclose(points, reverted_points, atol=1e-6):
        print("\n성공: 원래 점군과 되돌린 점군이 일치합니다.")
    else:
        print("\n경고: 원래 점군과 되돌린 점군이 약간 다릅니다. (허용 오차 범위 내일 수 있음)")
        diff = np.linalg.norm(points - reverted_points)
        print(f"평균 차이: {diff / len(points)}")


# 함수 실행
if __name__ == "__main__":
    perform_pca_and_align(pcd_file_path)
