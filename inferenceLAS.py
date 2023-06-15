import argparse
import os
import logging
import numpy as np
import laspy
#from openpoints.dataset.data_util import voxelize

input_path = "/Volumes/LHH-WD-1TB/data/PointView_implementation_test/ready4pointnext/npy_files/cloud (12)_sampled.npy"
output_path = "/Volumes/LHH-WD-1TB/data/PointView_implementation_test/ready4pointnext/npy_files/"

def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

def load_data(data_path):
    label, feat = None, None
    data = np.load(data_path)  # xyzrgb
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
    feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    # coord = data[['x', 'y', 'z']].view((np.float64, len(data.dtype.names)))
    # feat = data[['r', 'g', 'b']].view((np.int8, len(data.dtype.names)))
    # label = data['classification']

    idx_points = []
    voxel_idx, reverse_idx_part, reverse_idx_sort = None, None, None
    voxel_size = 0.04

    if voxel_size is not None:
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if False: #cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max() + 1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle]  # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0)  # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort

def write_las(points, colors, labels, out_filename):
    N = points.shape[0]

    # Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="Classification", type=np.uint8))
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.00001, 0.00001, 0.00001])

    # Create a LasData object
    las_data = laspy.LasData(header)

    # Assign data to LasData object
    las_data.x = points[:, 0]
    las_data.y = points[:, 1]
    las_data.z = points[:, 2]
    las_data.Classification = labels.astype(np.uint8)

    # Set RGB color
    las_data.red = colors[:, 0].astype(np.uint16)
    las_data.green = colors[:, 1].astype(np.uint16)
    las_data.blue = colors[:, 2].astype(np.uint16)

    # Write to LAS file
    las_data.write(out_filename)

data = np.load(input_path) 
coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

# output as LAS-file
write_las(coord, feat, label, os.path.join(output_path, "test00001.las"))



