#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os


def vis_multi_points(points, colors=None, labels=None, 
                     opacity=1.0, point_size=5.0,
                     color_map='Paired', save_fig=False, save_name='example', orbit=False):
    """Visualize a point cloud

    Args:
        points (list): a list of 2D numpy array. 
        colors (list, optional): [description]. Defaults to None.
    
    Example:
        vis_multi_points([points, pts], labels=[self.sub_clouds_points_labels[cloud_ind], labels])
    """
    import pyvista as pv
    import numpy as np
    from pyvista import themes
    from matplotlib import cm

    my_theme = themes.DefaultTheme()
    my_theme.color = 'black'
    my_theme.lighting = True
    my_theme.show_edges = True
    my_theme.edge_color = 'white'
    my_theme.background = 'white'
    pv.set_plot_theme(my_theme)

    n_clouds = len(points)
    plotter = pv.Plotter(shape=(1, n_clouds), border=False)

    if colors is None:
        colors = [None] * n_clouds
    if labels is None:
        labels = [None] * n_clouds

    for i in range(n_clouds):
        plotter.subplot(0, i)
        if len(points[i].shape) == 3: points[i] = points[i][0]
        if colors[i] is not None and len(colors[i].shape) == 3: colors[i] = colors[i][0]
        if colors[i] is None and labels[i] is not None:
            color_maps = cm.get_cmap(color_map, labels[i].max() + 1)
            colors[i] = color_maps(labels[i])[:, :3]
            if colors[i].min() <0:
                colors[i] = np.array((colors[i] - colors[i].min) / (colors[i].max() - colors[i].min()) *255).astype(np.int8)
                
        plotter.add_points(points[i], opacity=opacity, point_size=point_size, render_points_as_spheres=True, scalars=colors[i], rgb=True)
    plotter.link_views()
    if save_fig:
        # plotter.show(auto_close=False)
        # plotter.screenshot(filename=f'{save_name}.png')
        plotter.show(screenshot='airplane.png')
        plotter.close()
    elif orbit:
        #viewup = [0.5, 0.5, 1]
        plotter.camera.zoom(3)
        plotter.show(auto_close=False)
        path = plotter.generate_orbital_path(n_points=50, shift=20)
        plotter.open_gif("orbitcloud3x20.gif")
        plotter.orbit_on_path(path, write_frames=True, step=0.5)
        plotter.close()
    else:
        plotter.show()
        plotter.close()


def read_obj(filename):
    values = np.loadtxt(filename, usecols=(1,2,3,4,5,6))
    return values[:, :3], values[:, 3:6]


# --------------------------------
# 1,4,5,40 
idx = 13
#data_dir = '/home/lasse/Git/PointNeXt/log/s3dis/s3dis-train-pointnext-xl-ngpus1-seed1045-20221008-170028-9LKswMZCFEnShmwcCgsFdY_ckpt_best.pth_20221008-233159-CiCAoPyMoVReUYhFcoHAM4/visualization'
#data_dir = '/Users/lhh/Downloads/visualization_s3dis'
#dataset_name = 's3dis-Area5'
data_dir = '/Volumes/LHH-WD-1TB/data/Novafos-3D/Visualization/visualization_snog_vest/visualization/'
dataset_name = 'novafos3d-cloud'
# --------------------------------

method_names = ['pred']
file_list = [os.path.join(data_dir, f'input-{dataset_name}_{idx}.obj')]
for method_name in method_names:
    file_list.append(os.path.join(data_dir, f'{method_name}-{dataset_name}_{idx}.obj'))

#file_list.append(os.path.join(data_dir, f'gt-{dataset_name}_{idx}.obj'))

input_points, input_colors =read_obj(file_list[0])
valid_idx = input_points[:, 2] < 6
input_points = input_points[valid_idx]
#gt_points, gt_colors =read_obj(file_list[-1])
method1_points, method1_colors =read_obj(file_list[1])
#method2_points, method2_colors =read_obj(file_list[2])

#vis_multi_points([input_points, input_points, input_points, input_points], [input_colors[valid_idx]*255.,  method1_colors[valid_idx], method2_colors[valid_idx], gt_colors[valid_idx]])

# 3 window view | Input cloud in colors | Prediction | Ground Truth | 
# vis_multi_points([input_points, input_points, input_points], [input_colors[valid_idx], method1_colors[valid_idx], gt_colors[valid_idx]])

# 2 window view | Prediction | Ground Truth |
# vis_multi_points([input_points, input_points], [method1_colors[valid_idx], gt_colors[valid_idx]])

# Inference results | 2 window view | Input cloud in colors | Prediction |
vis_multi_points([input_points, input_points], [input_colors[valid_idx], method1_colors[valid_idx]])