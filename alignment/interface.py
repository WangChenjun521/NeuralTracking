# Copyright 2021 Gregory Kramida

import open3d.core
import torch
import torch.utils.dlpack
import numpy as np
from multipledispatch import dispatch

from alignment.deform_net import DeformNet
from warp_field.graph import DeformationGraphNumpy, DeformationGraphOpen3D


@dispatch(DeformNet, np.ndarray, np.ndarray, np.ndarray, np.ndarray, DeformationGraphNumpy, np.ndarray, open3d.core.Device)
def run_non_rigid_alignment(deform_net: DeformNet,
                            source_rgbxyz: np.ndarray,
                            target_rgbxyz: np.ndarray,
                            pixel_anchors: np.ndarray,
                            pixel_weights: np.ndarray,
                            graph: DeformationGraphNumpy,
                            cropped_intrinsics: np.ndarray,
                            device: open3d.core.Device) -> dict:
    torch_device = torch.device(repr(device).lower())

    source_cuda = torch.from_numpy(source_rgbxyz).to(torch_device).unsqueeze(0)
    target_cuda = torch.from_numpy(target_rgbxyz).to(torch_device).unsqueeze(0)
    pixel_anchors_cuda = torch.from_numpy(pixel_anchors).to(torch_device).unsqueeze(0)
    pixel_weights_cuda = torch.from_numpy(pixel_weights).to(torch_device).unsqueeze(0)

    graph_nodes_cuda = torch.from_numpy(graph.get_warped_nodes()).to(torch_device).unsqueeze(0)
    graph_edges_cuda = torch.from_numpy(graph.edges).to(torch_device).unsqueeze(0)
    graph_edges_weights_cuda = torch.from_numpy(graph.edge_weights).to(torch_device).unsqueeze(0)
    graph_clusters_cuda = torch.from_numpy(graph.clusters.reshape(-1, 1)).to(torch_device).unsqueeze(0)

    intrinsics_cuda = torch.from_numpy(cropped_intrinsics).to(torch_device).unsqueeze(0)

    node_count = np.array(graph.nodes.shape[0], dtype=np.int64)
    node_count_cuda = torch.from_numpy(node_count).to(torch_device).unsqueeze(0)

    with torch.no_grad():
        deform_net_data = deform_net(
            source_cuda, target_cuda,
            graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
            pixel_anchors_cuda, pixel_weights_cuda,
            node_count_cuda, intrinsics_cuda,
            evaluate=True, split="test"
        )

    return deform_net_data


@dispatch(DeformNet, np.ndarray, np.ndarray, np.ndarray, np.ndarray, DeformationGraphOpen3D, np.ndarray, open3d.core.Device)
def run_non_rigid_alignment(deform_net: DeformNet,
                            source_rgbxyz: np.ndarray,
                            target_rgbxyz: np.ndarray,
                            pixel_anchors: np.ndarray,
                            pixel_weights: np.ndarray,
                            graph: DeformationGraphOpen3D,
                            cropped_intrinsics: np.ndarray,
                            device: open3d.core.Device) -> dict:
    torch_device = torch.device(repr(device).lower())

    source_cuda = torch.from_numpy(source_rgbxyz).to(torch_device).unsqueeze(0)
    target_cuda = torch.from_numpy(target_rgbxyz).to(torch_device).unsqueeze(0)
    pixel_anchors_cuda = torch.from_numpy(pixel_anchors).to(torch_device).unsqueeze(0)
    pixel_weights_cuda = torch.from_numpy(pixel_weights).to(torch_device).unsqueeze(0)

    warped_nodes = graph.get_warped_nodes()
    graph_nodes_cuda = torch.utils.dlpack.from_dlpack(warped_nodes.to_dlpack()).unsqueeze(0)
    graph_edges_cuda = torch.utils.dlpack.from_dlpack(graph.edges.to_dlpack()).unsqueeze(0)
    graph_edges_weights_cuda = torch.utils.dlpack.from_dlpack(graph.edge_weights.to_dlpack()).unsqueeze(0)
    graph_clusters_cuda = torch.utils.dlpack.from_dlpack(graph.clusters.to_dlpack()).reshape(-1, 1).unsqueeze(0)

    intrinsics_cuda = torch.from_numpy(cropped_intrinsics).to(torch_device).unsqueeze(0)

    node_count = np.array(graph.nodes.shape[0], dtype=np.int64)
    node_count_cuda = torch.from_numpy(node_count).cuda().unsqueeze(0)
    # endregion
    #####################################################################################################
    # region === run the motion prediction & optimization ====
    #####################################################################################################
    with torch.no_grad():
        deform_net_data = deform_net(
            source_cuda, target_cuda,
            graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
            pixel_anchors_cuda, pixel_weights_cuda,
            node_count_cuda, intrinsics_cuda,
            evaluate=True, split="test"
        )
    return deform_net_data
