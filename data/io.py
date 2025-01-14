import os
import sys
import struct
import re

import numpy as np
from telemetry import visualization as im_aux
from telemetry.visualization import flow


def save_rgb_image(filename, image_numpy):
    import skimage.io as sio
    image_to_save = np.copy(image_numpy)

    if image_to_save.shape[0] == 3:
        image_to_save = np.moveaxis(image_to_save, 0, -1)

    assert image_to_save.shape[-1] == 3, "image has {} channels, so it's not rgb, you liar!".format(
        image_to_save.shape[-1])

    if image_to_save.dtype == "float32":
        assert np.max(image_to_save) <= 1.0

        image_to_save = image_to_save * 255.0
        image_to_save = image_to_save.astype(np.uint8)

    sio.imsave(filename, image_to_save)


def save_grayscale_image(filename, image_numpy):
    import skimage.io as sio
    image_to_save = np.copy(image_numpy)
    image_to_save = (image_to_save * 255).astype(np.uint8)

    if len(image_to_save.shape) == 2:
        sio.imsave(filename, image_to_save)
    elif len(image_to_save.shape) == 3:
        assert image_to_save.shape[0] == 1 or image_to_save.shape[-1] == 1
        image_to_save = image_to_save[0]
        sio.imsave(filename, image_to_save)


def overlay_images_and_save(filename, image_original_1, image_original_2, alpha=0.5):
    final_image = im_aux.overlay_images(image_original_1, image_original_2, alpha)
    final_image.save(filename, "PNG")


def overlay_mask_and_save(filename, image_original, mask_original, alpha=0.5):
    masked_image = im_aux.overlay_mask(image_original, mask_original, alpha)
    masked_image.save(filename, "PNG")


def load_PFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def save_PFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def load_flow_binary(filename):
    # Flow is stored row-wise in order [channels, height, width].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    flow = None
    with open(filename, 'rb') as fin:
        width = struct.unpack('I', fin.read(4))[0]
        height = struct.unpack('I', fin.read(4))[0]
        channels = struct.unpack('I', fin.read(4))[0]
        n_elems = height * width * channels

        flow = struct.unpack('f' * n_elems, fin.read(n_elems * 4))
        flow = np.asarray(flow, dtype=np.float32).reshape([channels, height, width])

    return flow


def save_flow_binary(filename, flow):
    # Flow is stored row-wise in order [channels, height, width].
    assert len(flow.shape) == 3

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', flow.shape[2]))
        fout.write(struct.pack('I', flow.shape[1]))
        fout.write(struct.pack('I', flow.shape[0]))
        fout.write(struct.pack('={}f'.format(flow.size), *flow.flatten("C")))


def load_flow_middlebury(filename):
    f = open(filename, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def save_flow_middlebury(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def load_flow(filename):
    if filename.endswith('.pfm') or filename.endswith('.PFM'):
        return load_PFM(filename)[0][:, :, 0:2]
    elif filename.endswith('.oflow') or filename.endswith('.OFLOW'):
        return load_flow_binary(filename)
    elif filename.endswith('.sflow') or filename.endswith('.SFLOW'):
        return load_flow_binary(filename)
    elif filename.endswith('.flo') or filename.endswith('.FLO'):
        return load_flow_middlebury(filename)
    else:
        print("Wrong flow extension: {}".format(filename))
        exit()


def save_flow(filename, flow):
    if filename.endswith('.pfm') or filename.endswith('.PFM'):
        save_PFM(filename, flow)
    elif filename.endswith('.oflow') or filename.endswith('.OFLOW'):
        save_flow_binary(filename, flow)
    elif filename.endswith('.sflow') or filename.endswith('.SFLOW'):
        save_flow_binary(filename, flow)
    elif filename.endswith('.flo') or filename.endswith('.FLO'):
        save_flow_middlebury(filename, flow)
    else:
        print("Wrong flow extension: {}".format(filename))
        exit()


def load_graph_nodes_or_deformations(filename):
    # Node positions are stored row-wise in order [num_nodes, 3].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    nodes = None
    with open(filename, 'rb') as fin:
        num_nodes = struct.unpack('I', fin.read(4))[0]

        nodes = struct.unpack('f' * num_nodes * 3, fin.read(num_nodes * 3 * 4))
        nodes = np.asarray(nodes, dtype=np.float32).reshape([num_nodes, 3])

    return nodes


def save_graph_nodes(filename, nodes):
    # Node positions are stored row-wise in order [num_nodes, 3].
    assert len(nodes.shape) == 2
    assert (nodes.shape[1] == 3)

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', nodes.shape[0]))
        fout.write(struct.pack('={}f'.format(nodes.size), *nodes.flatten("C")))


def load_graph_edges(filename):
    # Graph edges are stored row-wise in order [num_nodes, num_edges].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    edges = None
    with open(filename, 'rb') as fin:
        num_nodes = struct.unpack('I', fin.read(4))[0]
        num_neighbors = struct.unpack('I', fin.read(4))[0]

        edges = struct.unpack('i' * num_nodes * num_neighbors, fin.read(num_nodes * num_neighbors * 4))
        edges = np.asarray(edges, dtype=np.int32).reshape([num_nodes, num_neighbors])

    return edges


def save_graph_edges(filename, edges):
    # Graph edges are stored row-wise in order [num_nodes, num_edges].
    assert len(edges.shape) == 2

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', edges.shape[0]))
        fout.write(struct.pack('I', edges.shape[1]))
        fout.write(struct.pack('={}i'.format(edges.size), *edges.flatten("C")))


def load_graph_edges_weights(filename):
    # Graph edges are stored row-wise in order [num_nodes, num_edges].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    edges_weights = None
    with open(filename, 'rb') as fin:
        num_nodes = struct.unpack('I', fin.read(4))[0]
        num_neighbors = struct.unpack('I', fin.read(4))[0]

        edges_weights = struct.unpack('f' * num_nodes * num_neighbors, fin.read(num_nodes * num_neighbors * 4))
        edges_weights = np.asarray(edges_weights, dtype=np.float32).reshape([num_nodes, num_neighbors])

    return edges_weights


def save_graph_edges_weights(filename, edges_weights):
    # Graph edges are stored row-wise in order [num_nodes, num_edges].
    assert len(edges_weights.shape) == 2

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', edges_weights.shape[0]))
        fout.write(struct.pack('I', edges_weights.shape[1]))
        fout.write(struct.pack('={}f'.format(edges_weights.size), *edges_weights.flatten("C")))


def save_graph_node_deformations(filename, node_deformations):
    # Node deformations are stored row-wise in order [num_nodes, 3].
    assert len(node_deformations.shape) == 2
    assert (node_deformations.shape[1] == 3)

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', node_deformations.shape[0]))
        fout.write(struct.pack('={}f'.format(node_deformations.size), *node_deformations.flatten("C")))


def load_graph_node_translations(filename):
    # Node translations are stored row-wise in order [num_nodes, 3].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    node_deformations = None
    with open(filename, 'rb') as fin:
        num_nodes = struct.unpack('I', fin.read(4))[0]

        translations_vec = struct.unpack('f' * num_nodes * 3, fin.read(num_nodes * 3 * 4))
        translations_vec = np.asarray(translations_vec, dtype=np.float32).reshape([num_nodes, 3])

    return translations_vec


def save_graph_node_translations(filename, translations_vec):
    # Node positions are stored row-wise in order [num_nodes, 3].
    assert len(translations_vec.shape) == 2
    assert (translations_vec.shape[1] == 3)

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', translations_vec.shape[0]))
        fout.write(struct.pack('={}f'.format(translations_vec.size), *translations_vec.flatten("C")))


def load_graph_node_rotations(filename):
    # Node rotations are stored row-wise in order [num_nodes, 3, 3].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    node_deformations = None
    with open(filename, 'rb') as fin:
        num_nodes = struct.unpack('I', fin.read(4))[0]

        rotations_mat = struct.unpack('f' * num_nodes * 3 * 3, fin.read(num_nodes * 3 * 3 * 4))
        rotations_mat = np.asarray(rotations_mat, dtype=np.float32).reshape([num_nodes, 3, 3])

    return rotations_mat


def save_graph_node_rotations(filename, rotations_mat):
    assert len(rotations_mat.shape) == 3
    assert (rotations_mat.shape[1] == 3)
    assert (rotations_mat.shape[2] == 3)

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', rotations_mat.shape[0]))
        fout.write(struct.pack('={}f'.format(rotations_mat.size), *rotations_mat.flatten("C")))


def load_graph_clusters(filename):
    # Graph clusters are stored row-wise in order [num_nodes, 1].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    clusters = None
    with open(filename, 'rb') as fin:
        num_nodes = struct.unpack('I', fin.read(4))[0]
        _ = struct.unpack('I', fin.read(4))[0]

        clusters = struct.unpack('i' * num_nodes, fin.read(num_nodes * 4))
        clusters = np.asarray(clusters, dtype=np.int32).reshape([num_nodes, 1])

    return clusters


def save_graph_clusters(filename, clusters):
    # Graph clusters are stored row-wise in order [num_nodes, 1].
    assert len(clusters.shape) == 2

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', clusters.shape[0]))
        fout.write(struct.pack('I', clusters.shape[1]))
        fout.write(struct.pack('={}i'.format(clusters.size), *clusters.flatten("C")))


def load_float_image(filename):
    # Image is stored row-wise in order [xdim, ydim, zdim].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    image = None
    with open(filename, 'rb') as fin:
        zdim = struct.unpack('I', fin.read(4))[0]
        ydim = struct.unpack('I', fin.read(4))[0]
        xdim = struct.unpack('I', fin.read(4))[0]
        n_elems = xdim * ydim * zdim

        image = struct.unpack('f' * n_elems, fin.read(n_elems * 4))
        image = np.asarray(image, dtype=np.float32).reshape([xdim, ydim, zdim])

    return image


def save_float_image(filename, image_input):
    image = np.copy(image_input)

    # Image is stored row-wise in order [xdim, ydim, zdim].
    assert len(image.shape) == 3

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', image.shape[2]))
        fout.write(struct.pack('I', image.shape[1]))
        fout.write(struct.pack('I', image.shape[0]))
        fout.write(struct.pack('={}f'.format(image.size), *image.flatten("C")))


def save_int_image(filename, image_input):
    image = np.copy(image_input)

    # Image is stored row-wise in order [xdim, ydim, zdim].
    assert len(image.shape) == 3

    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', image.shape[2]))
        fout.write(struct.pack('I', image.shape[1]))
        fout.write(struct.pack('I', image.shape[0]))
        fout.write(struct.pack('={}i'.format(image.size), *image.flatten("C")))


def load_int_image(filename):
    # Image is stored row-wise in order [xdim, ydim, zdim].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    image = None
    with open(filename, 'rb') as fin:
        zdim = struct.unpack('I', fin.read(4))[0]
        ydim = struct.unpack('I', fin.read(4))[0]
        xdim = struct.unpack('I', fin.read(4))[0]
        n_elems = xdim * ydim * zdim

        image = struct.unpack('i' * n_elems, fin.read(n_elems * 4))
        image = np.asarray(image, dtype=np.int32).reshape([xdim, ydim, zdim])

    return image


def draw_optical_flow_and_save(flow_image, filename):
    save_rgb_image(filename, flow.flow_to_color_safe(flow_image))
