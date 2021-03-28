#include "cpu/image_proc.h"
#include "cpu/graph_proc.h"

#define XSTRINGIFY(s) STRINGIFY(s)
#define STRINGIFY(s) #s

using namespace pybind11::literals;


int add(int i, int j) {
	return i + j;
}

// Definitions of all methods in the module.
PYBIND11_MODULE(nnrt, m) {

	m.def("compute_augmented_flow_from_rotation",
	      &image_proc::compute_augmented_flow_from_rotation,
	      "flow_image_rot_sa2so"_a, "flow_image_so2to"_a, "flow_image_rot_to2ta"_a, "height"_a, "width"_a,
	      "Compute an optical flow image that reflects the augmentation applied to the source and target images.");

	//TODO: define what the hell this is and what it's supposed to do or remove if it's garbage
	m.def("count_tp1", &image_proc::count_tp1, "");
	m.def("count_tp2", &image_proc::count_tp2, "");
	m.def("count_tp3", &image_proc::count_tp3, "");

	//TODO: define what the hell this is and what it's supposed to do or remove if it's garbage
	m.def("extend3", &image_proc::extend3, "");


	//[image] --> [ordered point cloud (point image)] and [point image] --> [mesh (vertex positions, vertex colors, face vertex indices)]

	m.def("backproject_depth_ushort",
	      py::overload_cast<
			      py::array_t<unsigned short>&, py::array_t<float>&, float, float, float, float, float
	      >(&image_proc::backproject_depth_ushort),
	      "image_in"_a, "point_image_out"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a, "normalizer"_a,
	      "Back-project depth image into 3D points. Stores output in point_image_out as array of shape (3, h, w).");

	m.def("backproject_depth_ushort",
	      py::overload_cast<
			      py::array_t<unsigned short>&, float, float, float, float, float
	      >(&image_proc::backproject_depth_ushort),
	      "image_in"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a, "normalizer"_a,
	      "Back-project depth image into 3D points. Returns ordered point cloud as array of shape (3, h, w).");

	m.def("backproject_depth_float", &image_proc::backproject_depth_float, "image_in"_a, "point_image_out"_a,
	      "fx"_a, "fy"_a, "cx"_a, "cy"_a, "Back-project depth image into 3D points");

	m.def("compute_mesh_from_depth", &image_proc::compute_mesh_from_depth, "point_image"_a, "max_triangle_edge_distance"_a,
	      "vertex_position"_a, "face_indices"_a, "Computes a mesh using back-projected points and pixel connectivity");

	m.def("compute_mesh_from_depth_and_color", &image_proc::compute_mesh_from_depth_and_color, "point_image"_a, "background_image"_a,
	      "max_triangle_edge_distance"_a, "vertex_positions"_a, "vertex_colors"_a, "face_indices"_a,
	      "Compute a mesh using back-projected points and pixel connectivity. Additionally, extracts colors for each vertex");

	m.def("compute_mesh_from_depth_and_flow", &image_proc::compute_mesh_from_depth_and_flow,
	   "point_image_in"_a, "flow_image_in"_a, "max_triangle_edge_distance"_a, "vertex_positions_out"_a,
   "vertex_flows_out"_a, "vertex_pixels_out"_a, "face_indices_out"_a,
	   "Compute a mesh using backprojected points and pixel connectivity. Additionally, extracts flows for each vertex");

	// image filtering
	m.def("filter_depth", py::overload_cast<py::array_t<unsigned short>&, py::array_t<unsigned short>&, int>(&image_proc::filter_depth),
	      "depth_image_in"_a, "depth_image_out"_a, "radius"_a,
	      "Run a median filter on input depth image, outputs to provided output image. Does not modify the original.");
	m.def("filter_depth", py::overload_cast<py::array_t<unsigned short>&, int>(&image_proc::filter_depth),
	      "depth_image_in"_a, "radius"_a,
	      "Run a median filter on provided depth image and outputs a new image with the result. Does not modify the original.");




	// warping via bilinear/trilinear interpolation

	m.def("warp_flow", &image_proc::warp_flow, "image"_a, "flow"_a, "mask"_a,
	      "Warp image (RGB) using provided 2D flow inside masked region using bilinear interpolation.\n"
	      "We assume:\n    image shape: (3, h, w)\n    flow shape: (2, h, w)\n    mask shape: (2, h, w)");

	m.def("warp_rigid", &image_proc::warp_rigid, "rgbxyz_image"_a, "rotation"_a, "translation"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a,
	      "Warp image (concatenated RGB + XYZ ordered point cloud, i.e. 6 channels) using provided depth map and rigid pose. Assumed rotation shape: (9), translation shape: (2).");

	m.def("warp_3d", &image_proc::warp_3d, "Warp image inside masked region using provided warped point cloud and trilinear interpolation).",
	      "We assume:\n    image shape: (6, h, w)\n    flow shape: (3, h, w)\n    mask shape: (h, w)");

	// procedures for deformation graph node sampling from a point-cloud-based mesh

	m.def("get_vertex_erosion_mask", &graph_proc::get_vertex_erosion_mask, "vertex_positions"_a, "face_indices"_a, "iteration_count"_a,
	      "min_neighbors"_a,
	      "Compile a vertex mask that can be used to erode the provided mesh (iteratively mask out vertices at surface discontinuities, leave only non-eroded vertices)");

	m.def("sample_nodes", &graph_proc::sample_nodes, "vertex_positions_in"_a, "vertex_erosion_mask_in"_a, "node_coverage"_a,
	      "use_only_non_eroded_indices"_a, "random_shuffle"_a,"Samples graph canonical_node_positions that cover given vertices.");

	// procedures for deformation graph processing

	m.def("compute_edges_geodesic", &graph_proc::compute_edges_geodesic, "vertex_positions"_a, "valid_vertices"_a,
	      "face_indices"_a, "node_indices"_a, "max_neighbor_count"_a, "node_coverage"_a,
	      "graph_edges"_a, "graph_edge_weights"_a, "graph_edge_distances"_a, "node_to_vertex_distances"_a,
	      "allow_only_valid_vertices"_a, "enforce_total_num_neighbors"_a,
	      "Compute geodesic edges between given graph canonical_node_positions (subsampled vertices on given mesh)\n"
	      " using a priority-queue-based implementation of Djikstra's algorithm.\n"
	      "Output is returned as an array of dimensions (node_count, max_neighbor_count), where row index represents a source node index and\n"
	      " the row's entries, if >=0, represent destination node indices, ordered by geodesic distance between source and destination. \n"
	      "If the source node has no geodesic neighbors, the nearest euclidean neighbor node's index will appear as the first and only entry in the node.");

	//TODO: a tuple-return-type overload of the above
	m.def("compute_edges_euclidean", &graph_proc::compute_edges_euclidean, "canonical_node_positions"_a, "max_neighbor_count"_a,
	      "Compute Euclidean edges between given graph canonical_node_positions.\n"
	      "The output is returned as an array of (node_count, max_neighbor_count), where row index represents a source node index and\n"
	      "the row's entries, if >=0, represent destination node indices, ordered by euclidean distance between source and destination.");

	m.def("compute_pixel_anchors_geodesic", py::overload_cast<const py::array_t<float>&,
			const py::array_t<int>&, const py::array_t<float>&, const py::array_t<int>&,  py::array_t<int>&, py::array_t<float>&, int, int, float>(
					&graph_proc::compute_pixel_anchors_geodesic),
	      "node_to_vertex_distance"_a, "valid_nodes_mask"_a, "vertices"_a, "vertex_pixels"_a, "pixel_anchors"_a,
	      "pixel_weights"_a, "width"_a, "height"_a, "node_coverage"_a,
	      "Compute anchor ids and skinning weights for every pixel using graph connectivity.\n"
	      "Output pixel anchors array (height, width, K) contains indices of K graph canonical_node_positions that \n"
	      "influence the corresponding point in the point_image. K is currently hard-coded to " STRINGIFY(GRAPH_K) ". \n"
	      "\n The output pixel weights array of the same dimensions contains the corresponding node weights based "
	      "\n on distance d from point to node: weight = e^( -d^(2) / (2*node_coverage^(2)) ).");
	m.def("compute_pixel_anchors_geodesic", py::overload_cast<const py::array_t<float>&,
			      const py::array_t<int>&, const py::array_t<float>&, const py::array_t<int>&, int, int, float>(
			&graph_proc::compute_pixel_anchors_geodesic),
	      "node_to_vertex_distance"_a, "valid_nodes_mask"_a, "vertices"_a, "vertex_pixels"_a,
	      "width"_a, "height"_a, "node_coverage"_a,
	      "Compute anchor ids and skinning weights for every pixel using graph connectivity.\n"
	      "Output pixel anchors array (height, width, K) contains indices of K graph canonical_node_positions that \n"
	      "influence the corresponding point in the point_image. K is currently hard-coded to " STRINGIFY(GRAPH_K) ". \n"
	      "\n The output pixel weights array of the same dimensions contains the corresponding node weights based "
	      "\n on distance d from point to node: weight = e^( -d^(2) / (2*node_coverage^(2)) ).");

	m.def("compute_pixel_anchors_euclidean", &graph_proc::compute_pixel_anchors_euclidean,
	      "graph_nodes"_a, "point_image"_a, "node_coverage"_a, "pixel_anchors"_a, "pixel_weights"_a,
	      "Compute anchor ids and skinning weights for every pixel using Euclidean distances.\n"
	      "Output pixel anchors array (height, width, K) contains indices of K graph canonical_node_positions that \n"
	      "influence the corresponding point in the point_image. K is currently hard-coded to " STRINGIFY(GRAPH_K) ". \n"
	      "\n The output pixel weights array of the same dimensions contains the corresponding node weights based "
	      "\n on distance d from point to node: weight = e^( -d^(2) / (2*node_coverage^(2)) ).");


	m.def("node_and_edge_clean_up", &graph_proc::node_and_edge_clean_up,
	   "graph_edges"_a, "valid_nodes_mask"_a, "Remove invalid nodes");

	m.def("compute_clusters", &graph_proc::compute_clusters,
	   "graph_edges"_a, "graph_clusters"_a, "Computes graph node clusters");

	m.def("update_pixel_anchors", &graph_proc::update_pixel_anchors, "node_id_mapping"_a,
	   "pixel_anchors"_a,
	   "Update pixel anchor after node id change");

	m.def("construct_regular_graph",
	      py::overload_cast<const py::array_t<float>&, int, int, float, float, float, py::array_t<float>&, py::array_t<int>&, py::array_t<int>&, py::array_t<float>&>(
			      &graph_proc::construct_regular_graph),
	      "point_image"_a, "x_nodes"_a, "y_nodes"_a,
	      "edge_threshold"_a, "max_point_to_node_distance"_a, "max_depth"_a,
	      "graph_nodes"_a, "graph_edges"_a, "pixel_anchors"_a, "pixel_weights"_a,
	      "Sample graph uniformly in pixel space, and compute pixel anchors");

	m.def("construct_regular_graph",
	      py::overload_cast<const py::array_t<float>&, int, int, float, float, float>(
			      &graph_proc::construct_regular_graph),
	      "point_image"_a, "x_nodes"_a, "y_nodes"_a,
	      "edge_threshold"_a, "max_point_to_node_distance"_a, "max_depth"_a,
	      "Samples graph uniformly in pixel space, and computes pixel anchors");


}