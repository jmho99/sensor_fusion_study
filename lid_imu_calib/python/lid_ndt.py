#!/usr/bin/env python3
import numpy as np
import open3d as o3d


class NDTScanMatch:
    def __init__(
        self, file_path="", target_name="target.pcd", source_name="source.pcd"
    ):
        self.file_path = file_path
        self.target_path = file_path + target_name
        self.source_path = file_path + source_name
        self.align_path = file_path + "frame_0000_aligned.pcd"
        self.all_process()

    def all_process(self):
        target_header, target_field, target_pcd = self.load_pcd_ascii_with_t(
            self.target_path
        )
        source_header, source_field, source_pcd = self.load_pcd_ascii_with_t(
            self.source_path
        )

        target_o3d = o3d.geometry.PointCloud()
        target_o3d.points = o3d.utility.Vector3dVector(target_pcd)
        source_o3d = o3d.geometry.PointCloud()
        source_o3d.points = o3d.utility.Vector3dVector(source_pcd)
        source_o3d_default = o3d.geometry.PointCloud()
        source_o3d_default.points = o3d.utility.Vector3dVector(source_pcd)

        target_o3d.points = self.clean_points(target_o3d).points
        source_o3d.points = self.clean_points(source_o3d).points

        voxel_size = 0.1
        target_down = self.voxel_downsample(target_o3d, voxel_size)
        source_down = self.voxel_downsample(source_o3d, voxel_size)

        init_T = np.eye(4)
        threshold = voxel_size * 2.0
        result_icp = self.run_icp(source_down, target_down, threshold, init_T)
        T_result = result_icp.transformation
        self.align_and_save(
            source_o3d_default, T_result, self.file_path, target_header, target_field
        )

    def clean_points(self, pcd):
        pts = np.asarray(pcd.points)
        mask = np.isfinite(pts).all(axis=1)  # 각 행(x,y,z)이 전부 finite인지
        pts_clean = pts[mask]
        print(f"[DEBUG] removed {len(pts) - len(pts_clean)} invalid points")
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(pts_clean)
        return pcd_clean

    def voxel_downsample(self, pcd, voxel_size):
        down_pcd = pcd.voxel_down_sample(voxel_size)
        return down_pcd

    def run_icp(self, source, target, threshold, init_T):
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        return result

    def align_and_save(self, source, T, output_path, header_lines, fields):
        source_aligned = source.transform(T.copy())
        self.save_pcd_ascii(
            self.align_path, header_lines, fields, np.asarray(source_aligned.points)
        )
        print(f"[INFO] Aligned PCD saved to: {self.align_path}")
        result_txt_path = output_path + "0_T_result_4x4.txt"
        np.savetxt(result_txt_path, T, fmt="%.9f")
        print(f"[INFO] T (4x4) saved to: T_result_4x4.txt")

    def load_pcd_ascii_with_t(self, path):
        """
        ASCII PCD 파일을 읽어 header 정보와 데이터(x,y,z, t 포함)를 반환.
        가정: FIELDS 중에 x, y, z, 그리고 t(time)가 있음.
        """
        with open(path, "r") as f:
            lines = f.readlines()

        header_lines = []
        data_start_idx = None

        for i, line in enumerate(lines):
            if line.strip().startswith("DATA"):
                header_lines = lines[: i + 1]
                data_start_idx = i + 1
                break

        if data_start_idx is None:
            raise ValueError("PCD: DATA 라인을 찾지 못했습니다.")

        # 헤더에서 FIELDS 파싱
        fields_line = None
        for line in header_lines:
            if line.startswith("FIELDS"):
                fields_line = line
                break
        if fields_line is None:
            raise ValueError("PCD: FIELDS 라인을 찾지 못했습니다.")

        fields = fields_line.strip().split()[
            1:
        ]  # 'FIELDS x y z t' -> ['x','y','z','t']

        # x,y,z,t 필드 인덱스 찾기
        try:
            ix = fields.index("x")
            iy = fields.index("y")
            iz = fields.index("z")
            iintensity = fields.index("intensity")
            it = fields.index("t")
            isec = fields.index("sec")
            inanosec = fields.index("nanosec")
        except ValueError:
            raise ValueError(
                f"PCD: FIELDS에 포함되지 않은 인덱스를 포함하고 있습니다: {fields}"
            )

        # 데이터 부분 읽기
        data_str = lines[data_start_idx:]
        # 공백으로 구분된 float/정수들이므로 numpy로 파싱
        data = np.loadtxt(data_str, dtype=float)  # (N, num_fields)

        x = data[:, ix]
        y = data[:, iy]
        z = data[:, iz]
        intensity = data[:, iintensity]
        t_offset = data[:, it]  # ns 기준이라고 가정
        sec = data[:, isec]
        nanosec = data[:, inanosec]

        return (
            header_lines,
            fields,
            np.vstack([x, y, z]).T,
        )

    def save_pcd_ascii(self, path, header_lines, fields, xyz_t):
        """
        header_lines: 기존 PCD 헤더 (DATA 줄까지)
        fields: FIELDS의 필드 리스트 (ex. ['x','y','z','intensity','t'])
        xyz_t: (N,4) [x,y,z,t] 형태 (t는 그대로 유지)
        """
        # fields 내에서 x,y,z, 그리고 t 의 위치만 쓸 거라고 가정 (간단한 버전)
        ix = fields.index("x")
        iy = fields.index("y")
        iz = fields.index("z")

        N = xyz_t.shape[0]

        new_header_lines = []
        for line in header_lines:
            strip = line.strip().lower()

            # POINTS N
            if strip.startswith("points"):
                new_line = f"POINTS {N}\n"
                new_header_lines.append(new_line)
                continue
            # WIDTH N
            if strip.startswith("width"):
                new_line = f"WIDTH {N/128}\n"
                new_header_lines.append(new_line)
                continue
            # FIELDS x y z
            if strip.startswith("fields"):
                new_line = "FIELDS x y z\n"
                new_header_lines.append(new_line)
                continue

            # SIZE 4 4 4
            if strip.startswith("size"):
                new_line = "SIZE 4 4 4\n"
                new_header_lines.append(new_line)
                continue

            # TYPE F F F
            if strip.startswith("type"):
                new_line = "TYPE F F F\n"
                new_header_lines.append(new_line)
                continue

            # COUNT 1 1 1
            if strip.startswith("count"):
                new_line = "COUNT 1 1 1\n"
                new_header_lines.append(new_line)
                continue
            # HEIGHT 는 보통 1이라면 그대로 두고 싶으면 그냥 패스
            new_header_lines.append(line)

        with open(path, "w") as f:
            # 헤더 쓰기
            for line in new_header_lines:
                f.write(line)
            # DATA ascii 뒤부터 데이터 작성
            # (header_lines 안에 DATA ascii 줄이 포함되어 있어야 함)
            # xyz_t는 x,y,z,t 순서, fields 순서를 그대로 쓰려면 fields 길이만큼 라인 만들 수도 있음.
            for i in range(N):
                x, y, z = xyz_t[i]
                # 여기서는 x y z t만 쓰는 단순 버전
                # 다른 필드가 있었던 PCD였다면, 필요 시 확장해야 함.
                line_vals = [
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{z:.6f}",
                ]
                f.write(" ".join(line_vals) + "\n")


def main():
    file_path = "/home/antlab/ROS2/calib_ws/pcd_files/"
    target_name = "frame_0000_deskewed.pcd"
    source_name = "frame_0001_deskewed.pcd"
    ndt = NDTScanMatch(file_path, target_name, source_name)


if __name__ == "__main__":
    main()
