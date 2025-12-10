import os
import numpy as np
import open3d as o3d
import torch

def load_pred_pointcloud(pred_pc_path, class_feats, device='cuda', batch_size=10_000):
    pcd_filename = os.path.join(pred_pc_path, "gt.ply")
    pred_filename = os.path.join(pred_pc_path, "predictions.npy")

    pcd = o3d.io.read_point_cloud(pcd_filename)
    pred_xyz = torch.tensor(np.asarray(pcd.points))
    pred_color = torch.tensor(np.asarray(pcd.colors))

    feats = torch.tensor(np.load(pred_filename)).to(torch.float)
    
    pred_classes = []
    
    for i in range(0, feats.shape[0], batch_size):
        batch_feats = feats[i:i+batch_size].to(device)
        
        class_sim = torch.nn.functional.cosine_similarity(
            batch_feats.unsqueeze(1), class_feats['feats'].unsqueeze(0), dim=-1
        )
    
        class_ids = torch.tensor(class_feats['ids'])
        pred_class_batch = class_ids[class_sim.argmax(dim=-1).detach().cpu()] # (num_objects,)
        pred_classes.append(pred_class_batch)
        
    pred_class = torch.cat(pred_classes, dim=0)    
    
    return pred_xyz, pred_color, pred_class
