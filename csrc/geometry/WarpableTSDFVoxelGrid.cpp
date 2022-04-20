//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/6/21.
//  Copyright (c) 2021 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#include "WarpableTSDFVoxelGrid.h"

#include <open3d/t/geometry/kernel/TSDFVoxelGrid.h>
#include <open3d/core/TensorKey.h>
#include <utility>
#include <geometry/kernel/Defines.h>

#include "geometry/kernel/WarpableTSDFVoxelGrid.h"
#include "geometry/kernel/WarpableTSDFVoxelGrid_Analytics.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
using namespace open3d::t::geometry;

namespace nnrtl {
namespace geometry {

o3c::Tensor WarpableTSDFVoxelGrid::ExtractVoxelCenters() {
	o3c::Tensor active_block_indices;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_indices);

	o3c::Tensor voxel_centers;
	kernel::tsdf::ExtractVoxelCenters(
			active_block_indices.To(o3c::Dtype::Int64),
			block_hashmap->GetKeyTensor(), block_hashmap->GetValueTensor(),
			voxel_centers, this->GetBlockResolution(), this->GetVoxelSize());

	return voxel_centers;
}


open3d::core::Tensor WarpableTSDFVoxelGrid::ExtractTSDFValuesAndWeights() {
	// Query active blocks and their nearest neighbors to handle boundary cases.
	o3c::Tensor active_block_indices;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_indices);

	o3c::Tensor voxel_values;
	kernel::tsdf::ExtractTSDFValuesAndWeights(
			active_block_indices.To(o3c::Dtype::Int64),
			block_hashmap->GetValueTensor(),
			voxel_values, this->GetBlockResolution());

	return voxel_values;
}

open3d::core::Tensor WarpableTSDFVoxelGrid::ExtractValuesInExtent(int min_voxel_x, int min_voxel_y, int min_voxel_z,
                                                                  int max_voxel_x, int max_voxel_y, int max_voxel_z) {
	o3c::Tensor active_block_indices;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_indices);

	o3c::Tensor voxel_values;
	kernel::tsdf::ExtractValuesInExtent(
			min_voxel_x, min_voxel_y, min_voxel_z,
			max_voxel_x, max_voxel_y, max_voxel_z,
			active_block_indices.To(o3c::Dtype::Int64), block_hashmap->GetKeyTensor(), block_hashmap->GetValueTensor(),
			voxel_values, this->GetBlockResolution());

	return voxel_values;
}

inline
static void PrepareDepthAndColorForIntegration(o3c::Tensor& depth_tensor, o3c::Tensor& color_tensor, const Image& depth, const Image& color,
											   const std::unordered_map<std::string, o3c::Dtype>& attr_dtype_map_){
	if (depth.IsEmpty()) {
		o3u::LogError(
				"[TSDFVoxelGrid] input depth is empty for integration.");
	}

	depth_tensor = depth.AsTensor().To(o3c::Dtype::Float32).Contiguous();

	if (color.IsEmpty()) {
		o3u::LogDebug(
				"[TSDFIntegrateWarped] color image is empty, perform depth "
				"integration only.");
	} else if (color.GetRows() == depth.GetRows() &&
	           color.GetCols() == depth.GetCols() && color.GetChannels() == 3) {
		if (attr_dtype_map_.count("color") != 0) {
			color_tensor = color.AsTensor().To(o3c::Dtype::Float32).Contiguous();
		} else {
			o3u::LogWarning(
					"[TSDFIntegrateWarped] color image is ignored since voxels do "
					"not contain colors.");
		}
	} else {
		o3u::LogWarning(
				"[TSDFIntegrateWarped] color image is ignored for the incompatible "
				"shape.");
	}
}

o3c::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedEuclideanDQ(const Image& depth, const Image& color,
                                                  const o3c::Tensor& depth_normals, const o3c::Tensor& intrinsics,
                                                  const o3c::Tensor& extrinsics, const o3c::Tensor& warp_graph_nodes,
                                                  const o3c::Tensor& node_dual_quaternion_transformations,
                                                  float node_coverage, int anchor_count, int minimum_valid_anchor_count,
                                                  float depth_scale, float depth_max) {

	o3c::AssertTensorDtype(intrinsics, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(extrinsics, o3c::Dtype::Float32);

	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		o3u::LogError("anchor_count is {}, but is required to satisfy 0 < anchor_count < {}", anchor_count, MAX_ANCHOR_COUNT);
	}

	// TODO note the difference from TSDFVoxelGrid::Integrate:
	//  IntegrateWarpedEuclideanDQ currently assumes that all of the relevant hash blocks have already been activated. This will probably change in the future.
	o3c::Tensor depth_tensor, color_tensor;
	PrepareDepthAndColorForIntegration(depth_tensor, color_tensor, depth, color, this->GetAttrDtypeMap());

	// Query active blocks and their nearest neighbors to handle boundary cases.
	o3c::Tensor active_block_addresses;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_addresses);
	o3c::Tensor block_values = block_hashmap->GetValueTensor();

	o3c::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedEuclideanDQ(
			active_block_addresses.To(o3c::Dtype::Int64), block_hashmap->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal,  this->GetBlockResolution(), this->GetVoxelSize(), this->GetSDFTrunc(),
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
			node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count,
			depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}

o3c::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedEuclideanDQ(const Image& depth, const o3c::Tensor& depth_normals, const o3c::Tensor& intrinsics,
                                                                       const o3c::Tensor& extrinsics, const o3c::Tensor& warp_graph_nodes,
                                                                       const o3c::Tensor& node_dual_quaternion_transformations, float node_coverage,
                                                                       int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	Image empty_color;
	return IntegrateWarpedEuclideanDQ(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
	                                  node_dual_quaternion_transformations,
	                                  node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

o3c::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedEuclideanMat(const Image& depth, const Image& color, const o3c::Tensor& depth_normals,
                                                                        const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics,
                                                                        const o3c::Tensor& warp_graph_nodes,
                                                                        const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations,
                                                                        float node_coverage, int anchor_count, int minimum_valid_anchor_count,
                                                                        float depth_scale, float depth_max) {
	// TODO note the difference from TSDFVoxelGrid::Integrate:
	//  IntegrateWarpedEuclideanDQ currently assumes that all of the relevant hash blocks have already been activated. This will probably change in the future.

	o3c::AssertTensorDtype(intrinsics, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(extrinsics, o3c::Dtype::Float32);

	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		o3u::LogError("anchor_count is {}, but is required to satisfy 0 < anchor_count < {}", anchor_count, MAX_ANCHOR_COUNT);
	}


	// float downsampling_factor = 0.5;
	// auto depth_downsampled = depth.Resize(downsampling_factor, Image::InterpType::Linear);
	// o3c::Tensor intrinsics_downsampled = intrinsics * downsampling_factor;
	//
	// o3c::Tensor active_indices;
	// block_hashmap_->GetActiveIndices(active_indices);
	// o3c::Tensor coordinates_of_inactive_neighbors_of_active_blocks =
	// 		BufferCoordinatesOfInactiveNeighborBlocks(active_indices);
	// o3c::Tensor blocks_to_activate_mask;
	// kernel::tsdf::DetermineWhichBlocksToActivateWithWarp(
	// 		blocks_to_activate_mask,
	// 		coordinates_of_inactive_neighbors_of_active_blocks,
	// 		depth_downsampled.AsTensor().To(o3c::Dtype::Float32).Contiguous(),
	// 		intrinsics_downsampled, extrinsics, warp_graph_nodes, node_rotations, node_translations,
	// 		node_coverage, block_resolution_, voxel_size_, sdf_trunc_);
	// o3c::Tensor block_coords;

	o3c::Tensor depth_tensor, color_tensor;
	PrepareDepthAndColorForIntegration(depth_tensor, color_tensor, depth, color, this->GetAttrDtypeMap());

	// Query active blocks and their nearest neighbors to handle boundary cases.
	o3c::Tensor active_block_addresses;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_addresses);
	o3c::Tensor block_values = block_hashmap->GetValueTensor();


	o3c::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedEuclideanMat(
			active_block_addresses.To(o3c::Dtype::Int64), block_hashmap->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal, this->GetBlockResolution(), this->GetVoxelSize(), this->GetSDFTrunc(),
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
			node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}



o3c::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedEuclideanMat(const Image& depth, const o3c::Tensor& depth_normals, const o3c::Tensor& intrinsics,
                                                                        const o3c::Tensor& extrinsics, const o3c::Tensor& warp_graph_nodes,
                                                                        const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations,
                                                                        float node_coverage, int anchor_count, int minimum_valid_anchor_count,
                                                                        float depth_scale, float depth_max) {
	Image empty_color;
	return IntegrateWarpedEuclideanMat(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes, node_rotations, node_translations,
	                                   node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

o3c::Tensor
WarpableTSDFVoxelGrid::IntegrateWarpedShortestPathDQ(const open3d::t::geometry::Image& depth, const open3d::t::geometry::Image& color,
                                                     const o3c::Tensor& depth_normals, const o3c::Tensor& intrinsics,
                                                     const o3c::Tensor& extrinsics, const o3c::Tensor& warp_graph_nodes,
                                                     const o3c::Tensor& warp_graph_edges,
                                                     const o3c::Tensor& node_dual_quaternion_transformations, float node_coverage,
                                                     int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {

	// TODO: note the difference from TSDFVoxelGrid::Integrate:
	//  IntegrateWarpedEuclideanDQ assumes that all of the relevant hash blocks have already been activated.
	o3c::AssertTensorDtype(intrinsics, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(extrinsics, o3c::Dtype::Float32);

	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		o3u::LogError("anchor_count is {}, but is required to satisfy 0 < anchor_count < {}", anchor_count, MAX_ANCHOR_COUNT);
	}

	o3c::Tensor depth_tensor, color_tensor;
	PrepareDepthAndColorForIntegration(depth_tensor, color_tensor, depth, color, this->GetAttrDtypeMap());

	// Query active blocks and their nearest neighbors to handle boundary cases.
	o3c::Tensor active_block_addresses;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_addresses);
	o3c::Tensor block_values = block_hashmap->GetValueTensor();


	o3c::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedShortestPathDQ(
			active_block_addresses.To(o3c::Dtype::Int64), block_hashmap->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal,this->GetBlockResolution(), this->GetVoxelSize(), this->GetSDFTrunc(),
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
			node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count,
			depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}

o3c::Tensor
WarpableTSDFVoxelGrid::IntegrateWarpedShortestPathDQ(const open3d::t::geometry::Image& depth, const o3c::Tensor& depth_normals,
                                                     const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics,
                                                     const o3c::Tensor& warp_graph_nodes, const o3c::Tensor& warp_graph_edges,
                                                     const o3c::Tensor& node_dual_quaternion_transformations, float node_coverage,
                                                     int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	Image empty_color;
	return IntegrateWarpedShortestPathDQ(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
	                                     node_dual_quaternion_transformations,
	                                     node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

o3c::Tensor
WarpableTSDFVoxelGrid::IntegrateWarpedShortestPathMat(const open3d::t::geometry::Image& depth, const open3d::t::geometry::Image& color,
                                                      const o3c::Tensor& depth_normals, const o3c::Tensor& intrinsics,
                                                      const o3c::Tensor& extrinsics, const o3c::Tensor& warp_graph_nodes,
                                                      const o3c::Tensor& warp_graph_edges, const o3c::Tensor& node_rotations,
                                                      const o3c::Tensor& node_translations, float node_coverage, int anchor_count,
                                                      int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	// TODO: note the difference from TSDFVoxelGrid::Integrate:
	//  IntegrateWarpedEuclideanDQ assumes that all of the relevant hash blocks have already been activated.

	o3c::AssertTensorDtype(intrinsics, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(extrinsics, o3c::Dtype::Float32);

	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		o3u::LogError("anchor_count is {}, but is required to satisfy 0 < anchor_count < {}", anchor_count, MAX_ANCHOR_COUNT);
	}

	o3c::Tensor depth_tensor, color_tensor;
	PrepareDepthAndColorForIntegration(depth_tensor, color_tensor, depth, color, this->GetAttrDtypeMap());


	// Query active blocks and their nearest neighbors to handle boundary cases.
	o3c::Tensor active_block_addresses;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_addresses);
	o3c::Tensor block_values = block_hashmap->GetValueTensor();


	o3c::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedShortestPathMat(
			active_block_addresses.To(o3c::Dtype::Int64), block_hashmap->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal, this->GetBlockResolution(), this->GetVoxelSize(), this->GetSDFTrunc(),
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
			node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}


o3c::Tensor
WarpableTSDFVoxelGrid::IntegrateWarpedShortestPathMat(const open3d::t::geometry::Image& depth, const o3c::Tensor& depth_normals,
                                                      const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics,
                                                      const o3c::Tensor& warp_graph_nodes, const o3c::Tensor& warp_graph_edges,
                                                      const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations,
                                                      float node_coverage, int anchor_count, int minimum_valid_anchor_count, float depth_scale,
                                                      float depth_max) {
	Image empty_color;
	return IntegrateWarpedShortestPathMat(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
	                                      warp_graph_edges, node_rotations, node_translations,
	                                      node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

o3c::Tensor WarpableTSDFVoxelGrid::BufferCoordinatesOfInactiveNeighborBlocks(const o3c::Tensor& active_block_addresses) {
	//TODO: shares most code with TSDFVoxelGrid::BufferRadiusNeighbors (DRY violation)
	o3c::Tensor key_buffer_int3_tensor = this->GetBlockHashMap()->GetKeyTensor();

	o3c::Tensor active_keys = key_buffer_int3_tensor.IndexGet(
			{active_block_addresses.To(o3c::Dtype::Int64)});
	int64_t n = active_keys.GetShape()[0];

	// Fill in radius nearest neighbors.
	o3c::Tensor keys_nb({27, n, 3}, o3c::Dtype::Int32, this->GetDevice());
	for (int nb = 0; nb < 27; ++nb) {
		int dz = nb / 9;
		int dy = (nb % 9) / 3;
		int dx = nb % 3;
		o3c::Tensor dt = o3c::Tensor(std::vector<int>{dx - 1, dy - 1, dz - 1},
		                               {1, 3}, o3c::Dtype::Int32, this->GetDevice());
		keys_nb[nb] = active_keys + dt;
	}
	keys_nb = keys_nb.View({27 * n, 3});

	o3c::Tensor neighbor_block_addresses, neighbor_mask;
	this->GetBlockHashMap()->Find(keys_nb, neighbor_block_addresses, neighbor_mask);

	// ~ binary "or" to get the inactive address/coordinate mask instead of the active one
	neighbor_mask = neighbor_mask.LogicalNot();

	return keys_nb.GetItem(o3c::TensorKey::IndexTensor(neighbor_mask));
}

int64_t WarpableTSDFVoxelGrid::ActivateSleeveBlocks() {
	o3c::Tensor active_indices;
	this->GetBlockHashMap()->GetActiveIndices(active_indices);
	o3c::Tensor inactive_neighbor_of_active_blocks_coordinates =
			BufferCoordinatesOfInactiveNeighborBlocks(active_indices);

	o3c::Tensor neighbor_block_addresses, neighbor_mask;
	this->GetBlockHashMap()->Activate(inactive_neighbor_of_active_blocks_coordinates, neighbor_block_addresses, neighbor_mask);

	return inactive_neighbor_of_active_blocks_coordinates.GetShape()[0];
}

} // namespace geometry
} // namespace nnrt
