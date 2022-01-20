//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/8/21.
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
#pragma once

#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/core/TensorList.h>

#include <pybind11/pybind11.h>

#include "core/KdTree.h"

namespace py = pybind11;

namespace nnrt::geometry {

open3d::t::geometry::PointCloud
WarpPointCloudMat(const open3d::t::geometry::PointCloud& input_point_cloud, const open3d::core::Tensor& nodes,
                  const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                  int anchor_count, float node_coverage,
                  int minimum_valid_anchor_count = 0);

open3d::t::geometry::PointCloud
WarpPointCloudMat(const open3d::t::geometry::PointCloud& input_point_cloud, const open3d::core::Tensor& nodes,
                  const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                  const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
                  int minimum_valid_anchor_count = 0);

open3d::t::geometry::TriangleMesh WarpTriangleMeshMat(const open3d::t::geometry::TriangleMesh& input_mesh, const open3d::core::Tensor& nodes,
                                                      const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                                                      int anchor_count, float node_coverage, bool threshold_nodes_by_distance = true,
                                                      int minimum_valid_anchor_count = 0);

open3d::t::geometry::TriangleMesh WarpTriangleMeshDQ(const open3d::t::geometry::TriangleMesh& input_mesh, const open3d::core::Tensor& nodes,
                                                     const open3d::core::Tensor& node_transformations,
                                                     int anchor_count, float node_coverage, bool threshold_nodes_by_distance = true,
                                                     int minimum_valid_anchor_count = 0);

void ComputeAnchorsAndWeightsEuclidean(open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points,
                                       const open3d::core::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
                                       float node_coverage);

py::tuple ComputeAnchorsAndWeightsEuclidean(const open3d::core::Tensor& points, const open3d::core::Tensor& nodes, int anchor_count,
                                            int minimum_valid_anchor_count, float node_coverage);

void ComputeAnchorsAndWeightsShortestPath(open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points,
                                          const open3d::core::Tensor& nodes, const open3d::core::Tensor& edges, int anchor_count,
                                          float node_coverage);

py::tuple ComputeAnchorsAndWeightsShortestPath(const open3d::core::Tensor& points, const open3d::core::Tensor& nodes,
                                               const open3d::core::Tensor& edges, int anchor_count, float node_coverage);

class GraphWarpField {
	friend class WarpableTSDFVoxelGrid;

public:
	GraphWarpField(open3d::core::Tensor nodes, open3d::core::Tensor edges, open3d::core::Tensor edge_weights, open3d::core::Tensor clusters,
	               float node_coverage = 0.05, bool threshold_nodes_by_distance = false, int anchor_count = 4, int minimum_valid_anchor_count = 0);
	virtual ~GraphWarpField() = default;

	open3d::core::Tensor GetWarpedNodes() const;
	open3d::core::TensorList GetNodeExtent() const;
	open3d::t::geometry::TriangleMesh WarpMesh(const open3d::t::geometry::TriangleMesh& input_mesh) const;

	Eigen::Vector3f WarpPoint(const Eigen::Vector3f& input_point);

	//TODO: gradually hide these fields and expose only on a need-to-know basis
	open3d::core::Tensor nodes;
	open3d::core::Tensor edges;
	open3d::core::Tensor edge_weights;
	open3d::core::Tensor clusters;

	open3d::core::Tensor translations;
	open3d::core::Tensor rotations;

	const float node_coverage;
	const int anchor_count;
	const bool threshold_nodes_by_distance;
	const int minimum_valid_anchor_count;
private:
	// std::shared_ptr<core::KdTree> KDTreeIndex;
	//
	// virtual void FindKNearestNodesToPoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances,
	//                                        const open3d::core::Tensor& query_points, int32_t k);

};


} // namespace nnrt::geometry