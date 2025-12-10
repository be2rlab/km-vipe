import argparse
import copy
import os

import torch
import numpy as np
import open3d as o3d

# from pytorch3d.ops import knn_points
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree

from source.debug import debug_visualize_loaded_pointclouds


# def compute_knn_associations(src_xyz, dst_xyz, k=1):
#     knn_pred = knn_points(
#         src_xyz.unsqueeze(0).cuda().contiguous().float(),
#         dst_xyz.unsqueeze(0).cuda().contiguous().float(),
#         lengths1=None,
#         lengths2=None,
#         return_nn=True,
#         return_sorted=True,
#         K=k,
#     )
    
#     dst_to_src_idx = knn_pred.idx.squeeze(0)
    
#     return dst_to_src_idx


def compute_knn_associations(src_xyz, dst_xyz, k=1):
    src_np = src_xyz.cpu().numpy()
    dst_np = dst_xyz.cpu().numpy()
    
    tree = BallTree(dst_np, metric="minkowski")
    dist, indices = tree.query(src_np, k=k)
    
    dst_to_src_idx = torch.tensor(indices, device=src_xyz.device)
    
    return dst_to_src_idx


def load_slam_reconstructed_gt(args, scene_id):
    '''Load the SLAM reconstruction results, to ensure fair comparison'''
    slam_path = os.path.join(args.replica_root, scene_id, "rgb_cloud")
    
    slam_pointclouds = o3d.io.read_point_cloud(os.path.join(slam_path, "pointcloud.pcd"))
    slam_xyz = torch.tensor(np.asarray(slam_pointclouds.points))
    
    return slam_xyz


def load_gt_pointcloud(gt_pc_path, semantic_info, dataset):
    gt_pc_ext = gt_pc_path.suffix

    from source.pointcloud import load_gt_pointcloud_replica, load_gt_pointcloud_aria
    if dataset == 'replica':
        return load_gt_pointcloud_replica(gt_pc_path, semantic_info)
    elif dataset == 'aria':
        return load_gt_pointcloud_aria(gt_pc_path)
    
    # if gt_pc_ext == '.ply':
    #     from source.pointcloud import load_gt_pointcloud_ply
    #     return load_gt_pointcloud_ply(gt_pc_path, semantic_info)
    # elif gt_pc_ext == '.pcd':
    #     from source.pointcloud import load_gt_pointcloud_pcd
    #     return load_gt_pointcloud_pcd(gt_pc_path, semantic_info)
    # else:
    #     raise ValueError(f"Unknown GT pointcloud extension: {gt_pc_path}")


def load_pred_pointcloud(approach_name, *args, **kwargs):
    if approach_name in ['cg', 'conceptgraphs']:
        from adaptors import conceptgraph as cg
        return cg.load_pred_pointcloud(*args, **kwargs)
    elif approach_name in ['bbq', 'beyondbarequeries']:
        from adaptors import bbq
        return bbq.load_pred_pointcloud(*args, **kwargs)
    elif approach_name in ['bbq_experimental']:
        from adaptors import bbq_experimental
        return bbq_experimental.load_pred_pointcloud(*args, **kwargs)
    elif approach_name in ['hovsg', 'hov-sg']:
        from adaptors import hovsg
        return hovsg.load_pred_pointcloud(*args, **kwargs)
    elif approach_name in ["openscene", "OpenScene"]:
        from adaptors import openscene
        return openscene.load_pred_pointcloud(*args, **kwargs)
    elif approach_name in ["rtkm"]:
        from adaptors import rtkm
        return rtkm.load_pred_pointcloud(*args, **kwargs)
    else:
        raise ValueError(f"Unknown approach name: {approach_name}")
    

def preprocess_point_cloud(pcd, voxel_size=0.01):
    """Preprocess point cloud for better ICP"""
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Remove statistical outliers
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    
    # Estimate normals
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
    )
    
    return pcd_clean

def icp_registration(source_xyz, target_xyz, max_distance=0.05, init_transformation=None):
    """
    Perform ICP registration between two point clouds
    
    Args:
        source_xyz: numpy array of source points (N, 3)
        target_xyz: numpy array of target points (M, 3)
        max_distance: maximum correspondence distance
        init_transformation: initial transformation matrix (4x4)
    
    Returns:
        transformed_source: aligned source points
        transformation: final transformation matrix
        fitness: alignment quality metric
    """
    
    # Convert numpy arrays to Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_xyz)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_xyz)

    # print("Preprocessing point clouds...")
    # source_processed = preprocess_point_cloud(source_pcd, voxel_size=0.02)
    # target_processed = preprocess_point_cloud(target_pcd, voxel_size=0.02)
    
    # Estimate normals for better point-to-plane ICP
    print("Estimating normals...")
    source_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    
    # Set initial transformation
    # if init_transformation is None:
    #     init_transformation = np.identity(4)
        # init_transformation[2, 2] = -1

    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, 0])

    init_transformation = np.eye(4)
    init_transformation[:3, :3] = rotation_matrix

    init_transformation = np.array([
        [-6.12323400e-17,  1.00000000e+00, -6.12323400e-17, -1.11402249e+00],
        [ 1.22464680e-16,  6.12323400e-17,  1.00000000e+00,  2.17550659e+00],
        [ 1.00000000e+00,  6.12323400e-17, -1.22464680e-16,  2.73588204e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ])

    
    # Perform point-to-plane ICP
    print("Running ICP...")
    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd, 
        target_pcd, 
        max_distance,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        # o3d.pipelines.registration.TransformationEstimationPointToPlane(), # TransformationEstimationPointToPoint
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=10_000,  # Increase for better convergence
            relative_fitness=1e-7,
            relative_rmse=1e-7
        )
    )
    
    print(f"ICP Fitness: {reg_result.fitness:.3f}")
    print(f"ICP Inlier RMSE: {reg_result.inlier_rmse:.6f}")
    print("Transformation matrix:")
    print(reg_result.transformation)
    
    # Apply transformation to source point cloud
    transformed_source_pcd = copy.deepcopy(source_pcd)
    transformed_source_pcd.transform(reg_result.transformation)
    
    # Convert back to numpy
    transformed_source = np.asarray(transformed_source_pcd.points)
    
    return transformed_source, reg_result.transformation, reg_result.fitness


def evaluate_scen(
    gt_pointcloud,
    pred_pointcloud, 
    class_feats,
    nn_count = 5
):
    gt_xyz, gt_class = gt_pointcloud
    pred_xyz, pred_color, pred_class = pred_pointcloud

    pred_xyz, _, _ = icp_registration(pred_xyz, gt_xyz, max_distance=5, init_transformation=None)
    pred_xyz = torch.tensor(pred_xyz)

    debug_visualize_loaded_pointclouds(pred_class, class_feats['names'], pred_xyz, gt_xyz, gt_class, class_feats['ids'])

    pred_to_gt_idx = compute_knn_associations(gt_xyz, pred_xyz, k=nn_count).cpu()
    
    class_feats['ids'] = list(class_feats['ids']) + [-1]
    
    abandoned_gt_points_idx = torch.isin(gt_class, torch.tensor(class_feats['ids']))
    gt_class_mapped = gt_class[abandoned_gt_points_idx]
    pred_class_mapped = torch.mode(pred_class[pred_to_gt_idx], dim=-1)[0]
    pred_class_mapped = pred_class_mapped[abandoned_gt_points_idx]
    
    confmatrix = confusion_matrix(
        y_true = gt_class_mapped.cpu().numpy(),
        y_pred = pred_class_mapped.cpu().numpy(),
        labels = class_feats['ids']
    )
    
    # assert confmatrix.sum(0)[ignore_index].sum() == 0
    # assert confmatrix.sum(1)[ignore_index].sum() == 0
    
    return {
        "conf_matrix": torch.tensor(confmatrix),
        "labels": torch.tensor(class_feats['ids']),
    }


def eval_loop(args, class_feats, exclude_class, id_to_class_dict, class_to_id_dict):
    conf_matrices = {}
    scene_ids = list(args.scene_ids_str.split())
    
    for scene_id in scene_ids:
        print("Evaluating on:", scene_id)
        conf_matrix, keep_index = evaluate_scen(
            scene_id = scene_id,
            id_to_class_dict = id_to_class_dict,
            class_to_id_dict = class_to_id_dict,
            class_feats = class_feats,
            args = args,
            exclude_class_idx = exclude_class,
        )
        
        conf_matrix = conf_matrix.detach().cpu()

        conf_matrices[scene_id] = {
            "conf_matrix": conf_matrix,
            "keep_index": keep_index,
        }
    
    conf_matrix_all = np.sum([conf_matrix["conf_matrix"].numpy() for conf_matrix in conf_matrices.values()], axis=0)
    keep_index_all = np.unique([conf_matrix["keep_index"] for conf_matrix in conf_matrices.values()])
    
    conf_matrices["all"] = {
        "conf_matrix": torch.tensor(conf_matrix_all),
        "keep_index": torch.tensor(keep_index_all),
    }
    
    return conf_matrices