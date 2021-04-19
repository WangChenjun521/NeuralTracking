# Copyright (c) 2018 Andy Zeng

# Library imports
import numpy as np
import time
from numba import njit, prange
from skimage import measure
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


try:
  import pycuda.driver as cuda
  from pycuda.compiler import SourceModule
  FUSION_GPU_MODE = 1
except Exception as err:
  print('Warning: {}'.format(err))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')
  FUSION_GPU_MODE = 0

# Modules
import options as opt

cuda.init()
pycuda_ctx = cuda.Device(0).retain_primary_context()
torch.cuda.init()  # any torch cuda initialization before pycuda calls, torch.randn(10).cuda() works too

def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]

class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, max_depth, voxel_size,cam_intr, use_gpu=True):
    """Constructor.

    Args:
      max_depth (float): Maximum depth in the sequence 
      voxel_size (float): The volume discretization in meters.cam_pose
      cam_intr (np.array(4)): fx,fy,cx,cy (camera interincs parameters)
    """

    print(f"Initializing TSDF Volume:\n\tmax_depth:{max_depth}\n\tvoxel_size:{voxel_size}")

    self.cam_intr = np.eye(3)
    self.cam_intr[0,0] = cam_intr[0]
    self.cam_intr[1,1] = cam_intr[1]
    self.cam_intr[0,2] = cam_intr[2]
    self.cam_intr[1,2] = cam_intr[3]


    # Get corners of 3D camera view frustum of depth image
    im_h = opt.image_height
    im_w = opt.image_width
    view_frust_pts = np.array([
      (np.array([0,0,0,im_w,im_w])-self.cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/self.cam_intr[0,0],
      (np.array([0,0,im_h,0,im_h])-self.cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/self.cam_intr[1,1],
      np.array([0,max_depth,max_depth,max_depth,max_depth])
    ])

    # Estimate the bounding box for the volume
    vol_bnds = np.asarray([np.min(view_frust_pts,axis=1),np.max(view_frust_pts,axis=1)]).T
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    # Define voxel volume parameters
    self._vol_bnds = vol_bnds
    self._voxel_size = float(voxel_size)
    self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
    self._color_const = 256 * 256

    # Adjust volume bounds and ensure C-order contiguous
    self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
    self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
    self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)
    print("Voxel Origin:",self._vol_origin)
    print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
      self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
      self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
    )

    # Initialize pointers to voxel volume in CPU memory
    self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    # for computing the cumulative moving average of observations per voxel
    self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

    self.gpu_mode = use_gpu and FUSION_GPU_MODE

    # Copy voxel volumes to GPU
    if self.gpu_mode:

      # Determine block/grid size on GPU
      gpu_dev = pycuda_ctx.get_device()
      pycuda_ctx.push()
      self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
      n_blocks = int(np.ceil(float(np.prod(self._vol_dim))/float(self._max_gpu_threads_per_block)))
      grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,int(np.floor(np.cbrt(n_blocks))))
      grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,int(np.floor(np.sqrt(n_blocks/grid_dim_x))))
      grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,int(np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
      self._max_gpu_grid_dim = np.array([grid_dim_x,grid_dim_y,grid_dim_z]).astype(int)
      self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim))/float(np.prod(self._max_gpu_grid_dim)*self._max_gpu_threads_per_block)))


      self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
      cuda.memcpy_htod(self._tsdf_vol_gpu,self._tsdf_vol_cpu)
      self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
      cuda.memcpy_htod(self._weight_vol_gpu,self._weight_vol_cpu)
      self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
      cuda.memcpy_htod(self._color_vol_gpu,self._color_vol_cpu)


      # Cuda kernel function (C++)
      self._cuda_src_mod = SourceModule("""
        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          // World coordinates to camera coordinates
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          // Camera coordinates to image pixels
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          // Skip if outside view frustum
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;
          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;
          // Integrate TSDF
          float trunc_margin = other_params[4];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
          weight_vol[voxel_idx] = w_new;
          tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;

          // Integrate color
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
        }""")

      self._cuda_integrate = self._cuda_src_mod.get_function("integrate")
      pycuda_ctx.pop()

    else:
      # Get voxel grid coordinates
      xv, yv, zv = np.meshgrid(
        range(self._vol_dim[0]),
        range(self._vol_dim[1]),
        range(self._vol_dim[2]),
        indexing='ij'
      )
      self.vox_coords = np.concatenate([
        xv.reshape(1,-1),
        yv.reshape(1,-1),
        zv.reshape(1,-1)
      ], axis=0).astype(int).T

  @staticmethod
  @njit(parallel=True)
  def vox2world(vol_origin, vox_coords, vox_size):
    """Convert voxel grid coordinates to world coordinates.
    """
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    for i in prange(vox_coords.shape[0]):
      for j in range(3):
        cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
    return cam_pts

  @staticmethod
  @njit(parallel=True)
  def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates.
    """
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
      pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
      pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix

  @staticmethod
  @njit(parallel=True)
  def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
    """Integrate the TSDF volume.
    """
    tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
    w_new = np.empty_like(w_old, dtype=np.float32)
    for i in prange(len(tsdf_vol)):
      w_new[i] = w_old[i] + obs_weight
      tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
    return tsdf_vol_int, w_new

  def integrate(self, color_im, depth_im, obs_weight=1.):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
    """

    assert len(color_im.shape) == 3 and len(depth_im.shape) == 2 , "Input not correct\nDepth:{depth_im.shape} should be (H,W)\nColor:{color_im.shape} should be (H,W,3)"
    im_h, im_w = depth_im.shape

    # Fold RGB color image into a single channel image
    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[...,2]*self._color_const + color_im[...,1]*256 + color_im[...,0])

    cam_pose = np.eye(4)

    if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
      pycuda_ctx.push()
      for gpu_loop_idx in range(self._n_gpu_loops):
        self._cuda_integrate(self._tsdf_vol_gpu,
                            self._weight_vol_gpu,
                            self._color_vol_gpu,
                            cuda.InOut(self._vol_dim.astype(np.float32)),
                            cuda.InOut(self._vol_origin.astype(np.float32)),
                            cuda.InOut(self.cam_intr.reshape(-1).astype(np.float32)),
                            cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                            cuda.InOut(np.asarray([
                              gpu_loop_idx,
                              self._voxel_size,
                              im_h,
                              im_w,
                              self._trunc_margin,
                              obs_weight
                            ], np.float32)),
                            cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                            cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                            block=(self._max_gpu_threads_per_block,1,1),
                            grid=(
                              int(self._max_gpu_grid_dim[0]),
                              int(self._max_gpu_grid_dim[1]),
                              int(self._max_gpu_grid_dim[2]),
                            )
        )
      pycuda_ctx.pop()
    else:  # CPU mode: integrate voxel volume (vectorized implementation)
      # Convert voxel grid coordinates to pixel coordinates
      cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
      cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
      pix_z = cam_pts[:, 2]
      pix = self.cam2pix(cam_pts, self.cam_intr)
      pix_x, pix_y = pix[:, 0], pix[:, 1]

      # Eliminate pixels outside view frustum
      valid_pix = np.logical_and(pix_x >= 0,
                  np.logical_and(pix_x < im_w,
                  np.logical_and(pix_y >= 0,
                  np.logical_and(pix_y < im_h,
                  pix_z > 0))))
      depth_val = np.zeros(pix_x.shape)
      depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

      # Integrate TSDF
      depth_diff = depth_val - pix_z
      valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
      dist = np.minimum(1, depth_diff / self._trunc_margin)
      valid_vox_x = self.vox_coords[valid_pts, 0]
      valid_vox_y = self.vox_coords[valid_pts, 1]
      valid_vox_z = self.vox_coords[valid_pts, 2]
      w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      valid_dist = dist[valid_pts]
      tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
      self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
      self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

      # Integrate color
      old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      old_b = np.floor(old_color / self._color_const)
      old_g = np.floor((old_color-old_b*self._color_const)/256)
      old_r = old_color - old_b*self._color_const - old_g*256
      new_color = color_im[pix_y[valid_pts],pix_x[valid_pts]]
      new_b = np.floor(new_color / self._color_const)
      new_g = np.floor((new_color - new_b*self._color_const) /256)
      new_r = new_color - new_b*self._color_const - new_g*256
      new_b = np.minimum(255., np.round((w_old*old_b + obs_weight*new_b) / w_new))
      new_g = np.minimum(255., np.round((w_old*old_g + obs_weight*new_g) / w_new))
      new_r = np.minimum(255., np.round((w_old*old_r + obs_weight*new_r) / w_new))
      self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*self._color_const + new_g*256 + new_r

  def get_volume(self):
    if self.gpu_mode:
      pycuda_ctx.push()
      cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
      cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
      pycuda_ctx.pop()
    return self._tsdf_vol_cpu, self._color_vol_cpu

  def get_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """
    tsdf_vol, color_vol = self.get_volume()

    # Marching cubes
    verts = measure.marching_cubes(tsdf_vol, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + self._vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._color_const)
    colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
    colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

  def get_mesh(self):
    """Compute a mesh from the voxel volume using marching cubes.
    """
    tsdf_vol, color_vol = self.get_volume()

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size+self._vol_origin  # voxel grid coordinates to world coordinates

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
    colors_b = np.floor(rgb_vals/self._color_const)
    colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
    colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
    colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
    colors = colors.astype(np.uint8)
    return verts, faces, norms, colors