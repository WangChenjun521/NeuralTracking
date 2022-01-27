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
#include <open3d/core/Tensor.h>
#include <open3d/utility/Console.h>
// #include <open3d/core/hashmap/Hashmap.h>

#include "geometry/kernel/WarpableTSDFVoxelGrid.h"

using namespace open3d;

namespace nnrt::geometry::kernel::tsdf {


// void DetermineWhichBlocksToActivateWithWarp(core::Tensor& blocks_to_activate_mask, const core::Tensor& candidate_block_coordinates,
//                                             const core::Tensor& depth_downsampled, const core::Tensor& intrinsics_downsampled,
//                                             const core::Tensor& extrinsics, const core::Tensor& graph_nodes, const core::Tensor& node_rotations,
//                                             const core::Tensor& node_translations, float node_coverage, int64_t block_resolution, float voxel_size,
//                                             float sdf_truncation_distance) {
// 	core::Device device = candidate_block_coordinates.GetDevice();
// 	core::Device::DeviceType device_type = device.GetType();
// 	if (device_type == core::Device::DeviceType::CPU) {
//
// 		DetermineWhichBlocksToActivateWithWarp<core::Device::DeviceType::CPU>(
// 				blocks_to_activate_mask, candidate_block_coordinates, depth_downsampled,
// 				intrinsics_downsampled, extrinsics, graph_nodes, node_rotations,
// 				node_translations, node_coverage, block_resolution, voxel_size, sdf_truncation_distance
// 		);
// 	} else if (device_type == core::Device::DeviceType::CUDA) {
// #ifdef BUILD_CUDA_MODULE
// 		DetermineWhichBlocksToActivateWithWarp<core::Device::DeviceType::CUDA>(
// 				blocks_to_activate_mask, candidate_block_coordinates, depth_downsampled,
// 				intrinsics_downsampled, extrinsics, graph_nodes, node_rotations,
// 				node_translations, node_coverage, block_resolution, voxel_size, sdf_truncation_distance
// 		);
// #else
// 		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
// #endif
// 	} else {
// 		utility::LogError("Unimplemented device");
// 	}
// }


} // namespace nnrt::geometry::kernel::tsdf