import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from source.visual import get_semseg_palette

def debug_visualize_loaded_pointclouds(pred_class, class_names, pred_xyz, gt_xyz, gt_class, keep_index):
    n_classes = max(keep_index) + 1
    class2color = get_semseg_palette(n_classes)

    fig, ax = plt.subplots(figsize=(8, n_classes // 5))  # Adjust the figure size based on number of colors

    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_classes)
    ax.axis("off")

    # Display each color with its name
    for i, idx in enumerate(keep_index):
        ax.add_patch(plt.Rectangle((0, i), 9, 1, color=class2color[idx]))  # Color swatch
        ax.text(9.2, i + 0.5, class_names[i], va='center', fontsize=10)    # Color name

    plt.show()
    
    # predicted point cloud in open3d
    print("Before resampling")
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz.numpy())
    pred_pcd.colors = o3d.utility.Vector3dVector(class2color[pred_class.numpy()])
    # o3d.visualization.draw_geometries([pred_pcd])
    o3d.io.write_point_cloud("before_resampling.ply", pred_pcd)
    
    print(np.unique(pred_class.numpy()))
    
     # predicted point cloud in open3d
    print("After resampling")
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz.numpy())
    pred_pcd.colors = o3d.utility.Vector3dVector(class2color[pred_class.numpy()])
    # o3d.visualization.draw_geometries([pred_pcd])
    o3d.io.write_point_cloud("after_resampling.ply", pred_pcd)

    # GT point cloud in open3d
    print("GT pointcloud")
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_xyz.numpy())
    gt_pcd.colors = o3d.utility.Vector3dVector(class2color[gt_class.numpy()])
    # o3d.visualization.draw_geometries([gt_pcd])
    o3d.io.write_point_cloud("gt.ply", gt_pcd)
    
    print(np.unique(gt_class.numpy()))

    print("Merged pointcloud")
    # o3d.visualization.draw_geometries([gt_pcd, pred_pcd])
    o3d.io.write_point_cloud("merged.ply", gt_pcd + pred_pcd)