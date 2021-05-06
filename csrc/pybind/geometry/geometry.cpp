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
#include <open3d/t/geometry/TSDFVoxelGrid.h>
#include <open3d/geometry/Image.h>

#include "geometry/ExtendedTSDFVoxelGrid.h"
#include "geometry.h"

using namespace open3d;
using namespace open3d::t::geometry;

namespace nnrt {
namespace geometry {
void pybind_geometry(py::module& m) {
	py::module m_submodule = m.def_submodule(
			"geometry", "Open3D-tensor-based geometry defining module.");

	pybind_extended_tsdf_voxelgrid(m_submodule);
}

void pybind_extended_tsdf_voxelgrid(pybind11::module& m) {


	py::class_<ExtendedTSDFVoxelGrid, open3d::t::geometry::TSDFVoxelGrid> extended_tsdf_voxel_grid(
			m, "ExtendedTSDFVoxelGrid", "A voxel grid for TSDF and/or color integration, extended with custom functions.");
	extended_tsdf_voxel_grid.def(
			py::init<const std::unordered_map<std::string, core::Dtype>&, float,
					float, int64_t, int64_t, const core::Device&>(),
			"map_attrs_to_dtypes"_a =
					std::unordered_map<std::string, core::Dtype>{
							{"tsdf",   core::Dtype::Float32},
							{"weight", core::Dtype::UInt16},
							{"color",  core::Dtype::UInt16},
					},
			"voxel_size"_a = 3.0 / 512, "sdf_trunc"_a = 0.04,
			"block_resolution"_a = 16, "block_count"_a = 100,
			"device"_a = core::Device("CPU:0"));

	extended_tsdf_voxel_grid.def("integrate",
	                             py::overload_cast<const Image&, const core::Tensor&,
			                             const core::Tensor&, float, float>(
			                             &TSDFVoxelGrid::Integrate),
	                             "depth"_a, "intrinsics"_a, "extrinsics"_a,
	                             "depth_scale"_a, "depth_max"_a);

	extended_tsdf_voxel_grid.def(
			"integrate",
			py::overload_cast<const Image&, const Image&, const core::Tensor&,
					const core::Tensor&, float, float>(
					&TSDFVoxelGrid::Integrate),
			"depth"_a, "color"_a, "intrinsics"_a, "extrinsics"_a,
			"depth_scale"_a, "depth_max"_a);

	extended_tsdf_voxel_grid.def(
			"raycast", &TSDFVoxelGrid::RayCast, "intrinsics"_a, "extrinsics"_a,
			"width"_a, "height"_a, "depth_scale"_a = 1000.0,
			"depth_min"_a = 0.1f, "depth_max"_a = 3.0f,
			"weight_threshold"_a = 3.0f,
			"raycast_result_mask"_a = TSDFVoxelGrid::SurfaceMaskCode::DepthMap |
			                          TSDFVoxelGrid::SurfaceMaskCode::ColorMap);
	extended_tsdf_voxel_grid.def(
			"extract_surface_points", &TSDFVoxelGrid::ExtractSurfacePoints,
			"estimate_number"_a = -1, "weight_threshold"_a = 3.0f,
			"surface_mask"_a = TSDFVoxelGrid::SurfaceMaskCode::VertexMap |
			                   TSDFVoxelGrid::SurfaceMaskCode::ColorMap);
	extended_tsdf_voxel_grid.def("extract_surface_mesh",
	                             &TSDFVoxelGrid::ExtractSurfaceMesh,
	                             "weight_threshold"_a = 3.0f);

	extended_tsdf_voxel_grid.def("to", &TSDFVoxelGrid::To, "device"_a, "copy"_a = false);
	extended_tsdf_voxel_grid.def("clone", &TSDFVoxelGrid::Clone);
	extended_tsdf_voxel_grid.def("cpu", &TSDFVoxelGrid::CPU);
	extended_tsdf_voxel_grid.def("cuda", &TSDFVoxelGrid::CUDA, "device_id"_a);
	extended_tsdf_voxel_grid.def("get_block_hashmap", &TSDFVoxelGrid::GetBlockHashmap);
	extended_tsdf_voxel_grid.def("get_device", &TSDFVoxelGrid::GetDevice);

	// =============================== CUSTOM METHODS =======================================================

	extended_tsdf_voxel_grid.def("extract_voxel_centers", &ExtendedTSDFVoxelGrid::ExtractVoxelCenters);
	extended_tsdf_voxel_grid.def("extract_values_in_extent", &ExtendedTSDFVoxelGrid::ExtractValuesInExtent,
	                   "min_x"_a, "min_y"_a, "min_z"_a,
	                   "max_x"_a, "max_y"_a, "max_z"_a);



}

} // namespace geometry
} //namespace nnrt