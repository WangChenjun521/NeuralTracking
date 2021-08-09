import sys
import os
import shutil

import cv2
import open3d as o3d
import open3d.core as o3c
import numpy as np
from scipy.spatial.transform.rotation import Rotation

from rendering.pytorch3d_renderer import PyTorch3DRenderer
from data import StandaloneFrameDataset
import data.presets as presets
import fusion2.default_voxel_grid
import data.camera
from settings import settings_general

PROGRAM_EXIT_SUCCESS = 0


def main():
    frame_dataset: StandaloneFrameDataset = presets.StandaloneFramePreset.BERLIN_0.value

    device = o3c.Device("cuda:0")
    volume: o3d.t = fusion2.default_voxel_grid.make_default_tsdf_voxel_grid(device)

    depth_image = frame_dataset.load_depth_image_open3d(device)
    color_image = frame_dataset.load_color_image_open3d(device)
    intrinsics_open3d_cpu, _ = data.camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(frame_dataset.get_intrinsics_path(),
                                                                                              frame_dataset.get_depth_image_path())
    intrinsics_open3d_cuda = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix, o3d.core.Dtype.Float32, device)
    extrinsics_open3d_cuda = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32, device)

    volume.integrate(depth_image, color_image, intrinsics_open3d_cuda, extrinsics_open3d_cuda, settings_general.depth_scale, 3.0)
    original_mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
    renderer = PyTorch3DRenderer((depth_image.rows, depth_image.columns), device, intrinsics_open3d_cuda)

    frame_count = 6
    rotation_increment = 2.0  # degrees

    rotation_center = np.array([0.0855289, -0.03289237, 2.79831315], dtype=np.float32)

    def rotate_mesh_z(mesh: o3d.geometry.TriangleMesh, center_point: np.ndarray, angle: float) -> o3d.geometry.TriangleMesh:
        vertices = np.array(mesh.vertices)
        rotation_matrix = Rotation.from_euler("z", angle, degrees=True).as_matrix()
        centralized_vertices = vertices - center_point
        rotated_vertices = center_point + rotation_matrix.dot(centralized_vertices.T).T

        _rotated_mesh = o3d.geometry.TriangleMesh(o3d.cuda.pybind.utility.Vector3dVector(rotated_vertices), mesh.triangles)
        _rotated_mesh.vertex_colors = mesh.vertex_colors
        return _rotated_mesh

    # prepare folders
    root_output_directory = os.path.join(settings_general.output_directory, "berlin_z_rotation_sequence")
    depth_output_directory = os.path.join(root_output_directory, "depth")
    if not os.path.exists(depth_output_directory):
        os.makedirs(depth_output_directory)
    color_output_directory = os.path.join(root_output_directory, "color")
    if not os.path.exists(color_output_directory):
        os.makedirs(color_output_directory)

    # record animation rendering output
    for i_frame in range(0, frame_count):
        rotated_mesh = rotate_mesh_z(original_mesh, rotation_center, rotation_increment * i_frame)
        depth, color = renderer.render_mesh(rotated_mesh, depth_scale=1000.0)
        color_path = os.path.join(color_output_directory, f"{i_frame:06d}.jpg")
        depth_path = os.path.join(depth_output_directory, f"{i_frame:06d}.png")
        cv2.imwrite(color_path, color)
        cv2.imwrite(depth_path, depth.astype(np.uint16))

    shutil.copy(frame_dataset.get_intrinsics_path(), os.path.join(root_output_directory, "intrinsics.txt"))

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())