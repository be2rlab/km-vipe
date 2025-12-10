import os

import torch
import numpy as np
import open3d as o3d


def load_feature_map(path, normalize=True):
    """
    Load features map from disk, mask_feats.pt and objects/pcd_i.ply
    :param path: path to feature map
    :param normalize: whether to normalize features
    :return: mask_pcds, mask_feats
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Feature map not found in {}".format(path))
    
    if not os.path.exists(os.path.join(path, "objects")):
        raise FileNotFoundError("objects directory not found in {}".format(path))
    
    mask_feats = torch.load(os.path.join(path, "mask_feats.pt")).float()
    if normalize:
        mask_feats = torch.nn.functional.normalize(mask_feats, p=2, dim=-1).cpu().numpy()
    else:
        mask_feats = mask_feats.cpu().numpy()
    
    mask_pcds = []
    number_of_pcds = len(os.listdir(os.path.join(path, "objects")))
    not_found = []
    for i in range(number_of_pcds):
        if os.path.exists(os.path.join(path, "objects", "pcd_{}.ply".format(i))):
            mask_pcds.append(
                o3d.io.read_point_cloud(os.path.join(path, "objects", "pcd_{}.ply".format(i)))
            )
        else:
            print("masked pcd {} not found in {}".format(i, path))
            not_found.append(i)

    mask_feats = np.delete(mask_feats, not_found, axis=0)

    return mask_pcds, mask_feats
    
    
def load_pred_pointcloud(pred_pc_path, class_feats, device='cuda'):
    object_pcds, object_feats = load_feature_map(pred_pc_path, normalize=False)
    
    object_feats = torch.from_numpy(object_feats).to(device)
    object_class_sim = torch.nn.functional.cosine_similarity(
        object_feats.unsqueeze(1), class_feats['feats'].unsqueeze(0), dim=-1
    )
    
    class_ids = torch.tensor(class_feats['ids'])
    object_class = class_ids[object_class_sim.argmax(dim=-1).detach().cpu()] # (num_objects,)
    
    pred_xyz = []
    pred_color = []
    pred_class = []
    for i in range(len(object_pcds)):
        obj_pcd = object_pcds[i]
        pred_xyz.append(np.asarray(obj_pcd.points))
        pred_color.append(np.asarray(obj_pcd.colors))
        pred_class.append(np.ones(len(obj_pcd.points)) * object_class[i].item())
        
    pred_xyz = torch.from_numpy(np.concatenate(pred_xyz, axis=0))
    pred_color = torch.from_numpy(np.concatenate(pred_color, axis=0))
    pred_class = torch.from_numpy(np.concatenate(pred_class, axis=0)).long()
    
    return pred_xyz, pred_color, pred_class