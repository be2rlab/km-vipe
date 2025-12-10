import glob
import gzip
import os
import pickle

import torch
import numpy as np

from conceptgraph.slam.slam_classes import MapObjectList


def load_pred_pointcloud(pred_pc_path, class_feats, device='cuda'):   
    with gzip.open(pred_pc_path, "rb") as f:
            results = pickle.load(f)
        
    objects = MapObjectList()
    objects.load_serializable(results['objects'])

    # Compute the CLIP similarity for the mapped objects and assign class to them
    object_feats = objects.get_stacked_values_torch("clip_ft").to(device)
    object_class_sim = torch.nn.functional.cosine_similarity(
        object_feats.unsqueeze(1), class_feats['feats'].unsqueeze(0), dim=-1
    )
    
    class_ids = torch.tensor(class_feats['ids'])
    object_class = class_ids[object_class_sim.argmax(dim=-1).detach().cpu()] # (num_objects,)
    
    pred_xyz = []
    pred_color = []
    pred_class = []
    for i in range(len(objects)):
        obj_pcd = objects[i]['pcd']
        pred_xyz.append(np.asarray(obj_pcd.points))
        pred_color.append(np.asarray(obj_pcd.colors))
        pred_class.append(np.ones(len(obj_pcd.points)) * object_class[i].item())
        
    pred_xyz = torch.from_numpy(np.concatenate(pred_xyz, axis=0))
    pred_color = torch.from_numpy(np.concatenate(pred_color, axis=0))
    pred_class = torch.from_numpy(np.concatenate(pred_class, axis=0)).long()
    
    return pred_xyz, pred_color, pred_class