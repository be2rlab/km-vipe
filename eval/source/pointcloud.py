import json
import os
import pickle

import plyfile
import torch
import numpy as np
import open3d as o3d

from collections import defaultdict, Counter


def has_tie(lst):
    counts = Counter(lst)
    if not counts:
        return False  # Empty list
    max_count = max(counts.values())
    return list(counts.values()).count(max_count) > 1


def load_gt_pointcloud_replica(gt_pc_path, semantic_info):
    plydata = plyfile.PlyData.read(gt_pc_path)

    object_to_class_mapping = {obj["id"]: obj["class_id"] for obj in semantic_info["objects"]}

    vertices_xyz = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    
    each_face_vertices = plydata["face"]["vertex_indices"]
    each_face_object_id = plydata["face"]["object_id"]

    vertex_id_to_semantic = defaultdict(list)

    for i, face_vertices_idx in enumerate(each_face_vertices):
        for idx in face_vertices_idx:
            vertex_id_to_semantic[idx].append(
                object_to_class_mapping.get(each_face_object_id[i], -1)
            )
            
    vertex_labels = np.array([Counter(vertex_id_to_semantic.get(i, [-1])).most_common(1)[0][0] for i in range(vertices_xyz.shape[0])])
    vertex_with_ties = np.array([has_tie(vertex_id_to_semantic.get(i, [-1])) for i in range(vertices_xyz.shape[0])])
    vertex_labels[vertex_with_ties] = -1
    
    gt_xyz = torch.tensor(vertices_xyz)
    gt_class = torch.tensor(vertex_labels, dtype=int)
    
    return gt_xyz, gt_class


def load_gt_pointcloud_aria(gt_pc_path):
    pcd = o3d.t.io.read_point_cloud(str(gt_pc_path))

    gt_xyz = pcd.point.positions.numpy()
    gt_class = pcd.point.labels.numpy().flatten()


    gt_xyz = torch.tensor(gt_xyz)
    gt_class = torch.tensor(gt_class, dtype=int)

    gt_class[gt_class == 105] = 0 

    return gt_xyz, gt_class


def save_pointcloud(save_dir, o3d_pcd=None, xyz=None, colors=None, semantics=None, annotations=None):
    if o3d_pcd is not None and (xyz is not None or colors is not None):
        raise ValueError("Provide either 'o3d_pcd' or 'xyz/colors', not both.")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if o3d_pcd is None:
        o3d_pcd = o3d.geometry.PointCloud()
        if xyz is not None:
            o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz))
        if colors is not None:
            o3d_pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    
    pcd_saving_path = os.path.join(save_dir, "pointcloud.pcd")
    print(f'Saving .pcd file to "{pcd_saving_path}"')
    
    o3d.io.write_point_cloud(
        pcd_saving_path, 
        o3d_pcd
    )
        
    if semantics is not None:           
        sem_saving_path = os.path.join(save_dir, "semantic.npy")
        print(f'Saving semantics to "{sem_saving_path}"')
        
        np_semantics = np.asarray(semantics).astype(int)
        
        np.save(sem_saving_path, np_semantics)
        
        if annotations is not None:
            ann_saving_path = os.path.join(save_dir, "annotations.json")
            print(f'Saving annotations to "{ann_saving_path}"')
            
            with open(ann_saving_path, 'w') as f:
                json.dump(annotations, f, indent=4)