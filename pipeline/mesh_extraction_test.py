#!/usr/bin/python3

# A minimal example that generates a TSDF based on a depth image and then extracts the mesh from it
# Copyright 2021 Gregory Kramida
import os
import sys
import open3d as o3d
import torch
import numpy as np

import camera

PROGRAM_EXIT_SUCCESS = 0


def main():
    # Options

    use_mask = True
    segment = None

    #####################################################################################################
    # === open3d device, image paths, camera intrinsics ===
    #####################################################################################################

    # === device config ===

    device = o3d.core.Device('cuda:0')

    # === compile image paths ===

    frames_directory = "/mnt/Data/Reconstruction/real_data/deepdeform/val/seq014/"
    frame_index = 400
    color_image_filename_mask = frames_directory + "color/{:06d}.jpg"
    color_image_path = color_image_filename_mask.format(frame_index)
    depth_image_filename_mask = frames_directory + "depth/{:06d}.png"
    depth_image_path = depth_image_filename_mask.format(frame_index)

    mask_image_folder = frames_directory + "mask"
    if segment is None:
        segment = os.path.splitext(os.listdir(mask_image_folder)[0])[0].split('_')[1]
    mask_image_path = os.path.join(mask_image_folder, "{:06d}_{:s}.png".format(frame_index, segment))

    # === handle intrinsics ===

    depth_intrinsics_path = "/mnt/Data/Reconstruction/real_data/deepdeform/val/seq014/intrinsics.txt"
    intrinsics_open3d_cpu = camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(depth_intrinsics_path, depth_image_path)
    intrinsics_open3d_gpu = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix, o3d.core.Dtype.Float32, device)

    extrinsics_numpy = np.eye(4)
    extrinsics_open3d_gpu = o3d.core.Tensor(extrinsics_numpy, o3d.core.Dtype.Float32, device)

    #####################################################################################################
    # === open3d volume representation parameters ===
    #####################################################################################################

    voxel_size = 0.008  # voxel resolution in meters
    sdf_trunc = 0.04  # truncation distance in meters
    block_resolution = 16  # 16^3 voxel blocks
    initial_block_count = 1000  # initially allocated number of voxel blocks

    volume = o3d.t.geometry.TSDFVoxelGrid(
        {
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        block_resolution=block_resolution,
        block_count=initial_block_count,
        device=device)

    #####################################################################################################
    # === load images, fuse into TSDF, extract & visualize mesh ===
    #####################################################################################################

    # === images & TSDF integration/fusion ===

    if use_mask:
        depth_image_numpy = np.array(o3d.io.read_image(depth_image_path))
        color_image_numpy = np.array(o3d.io.read_image(color_image_path))
        mask_image_numpy = np.array(o3d.io.read_image(mask_image_path))
        mask_image_numpy_color = np.dstack((mask_image_numpy, mask_image_numpy, mask_image_numpy)).astype(np.uint8)
        # apply mask
        depth_image_numpy &= mask_image_numpy
        color_image_numpy &= mask_image_numpy_color
        depth_image_open3d_legacy = o3d.geometry.Image(depth_image_numpy)
        color_image_open3d_legacy = o3d.geometry.Image(color_image_numpy)
    else:
        depth_image_open3d_legacy = o3d.io.read_image(depth_image_path)
        color_image_open3d_legacy = o3d.io.read_image(color_image_path)

    depth_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(depth_image_open3d_legacy, device=device)
    color_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(color_image_open3d_legacy, device=device)

    volume.integrate(depth_image_gpu, color_image_gpu, intrinsics_open3d_gpu, extrinsics_open3d_gpu, 1000.0, 3.0)

    # === mesh extraction ===

    mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
    mesh.compute_vertex_normals()

    # === visualization ===

    o3d.visualization.draw_geometries([mesh],
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1.0, 0],
                                      zoom=0.7)

    o3d.io.write_triangle_mesh("../output/mesh_{:06d}_red_shorts.ply".format(frame_index), mesh)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())