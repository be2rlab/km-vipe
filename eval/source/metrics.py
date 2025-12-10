import torch
import numpy as np
import pandas as pd


def compute_metrics(confmatrix, class_names):
    '''
    iou - jaccard index 
    '''
    if isinstance(confmatrix, torch.Tensor):
        confmatrix = confmatrix.cpu().numpy()

    tp = np.diag(confmatrix)
    fp = confmatrix.sum(axis=0) - tp
    fn = confmatrix.sum(axis=1) - tp
    
    ious = tp / np.maximum(fn + fp + tp, 1e-7)
    miou = ious.mean()
    f_miou = (ious * (tp + fn) / confmatrix.sum()).sum()
    
    precision = tp / np.maximum(tp + fp, 1e-7)
    recall = tp / np.maximum(tp + fn, 1e-7)
    
    f1score = 2 * precision * recall / np.maximum(precision + recall, 1e-7)

    mdict = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "iou": ious.tolist(),
        "miou": miou.item(),
        "fmiou": f_miou.item(),
        "acc0.15": (ious > 0.15).sum().item(),
        "acc0.25": (ious > 0.25).sum().item(),
        "acc0.50": (ious > 0.50).sum().item(),
        "acc0.75": (ious > 0.75).sum().item(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1score": f1score.tolist()
    }

    return mdict


def metrics_loop(conf_matrices, class_names):
    results = []
    for scene_id, res in conf_matrices.items():
        conf_matrix = res["conf_matrix"]
        keep_index = res["keep_index"]
        conf_matrix = conf_matrix[keep_index, :][:, keep_index]
        keep_class_names = [class_names[i] for i in keep_index]

        mdict = compute_metrics(conf_matrix, keep_class_names)
        results.append(
            {
                "scene_id": scene_id,
                "miou": mdict["miou"] * 100.0,
                "mrecall": np.mean(mdict["recall"]) * 100.0,
                "mprecision": np.mean(mdict["precision"]) * 100.0,
                "mf1score": np.mean(mdict["f1score"]) * 100.0,
                "fmiou": mdict["fmiou"] * 100.0,
            }
        )
        
    df_result = pd.DataFrame(results)
    
    return df_result