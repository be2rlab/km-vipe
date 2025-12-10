import os
import torch
import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", 
        type=Path, 
        help='Directory containing the *_conf_matrix.pkl files'
    )
    
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=None, 
        help='Output directory to save metrics'
    )
    
    parser.add_argument(
        "--excluded", 
        type=str, 
        default="",
        help='Space-separated list of class indices to exclude from the metrics computation, e.g., "-1 0"'
    )

    parser.add_argument(
        "--chunks", 
        type=str, 
        default=None,
        help="JSON-formatted string specifying class chunks, e.g., '{\"head\": [...], \"common\": [...], \"tail\": [...]}'"
    )
    
    return parser


def compute_metrics(confmatrix, labels=None, excluded=None, existed=None):
    nonzero_mask = (confmatrix.sum(axis=1) != 0)
    
    if labels is not None:
        if excluded is not None:
            nonzero_mask = nonzero_mask * np.isin(labels, np.array(excluded), invert=True)
        if existed is not None:
            nonzero_mask = nonzero_mask * np.isin(labels, np.array(existed))
    
    tp = np.diag(confmatrix)[nonzero_mask]
    fp = confmatrix.sum(axis=0)[nonzero_mask] - tp
    fn = confmatrix.sum(axis=1)[nonzero_mask] - tp
    
    ious = tp / np.maximum(fn + fp + tp, 1e-7)
    miou = ious.mean()
    f_miou = (ious * (tp + fn) / (tp + fn).sum()).sum()
    
    precision = tp / np.maximum(tp + fp, 1e-7)
    recall = tp / np.maximum(tp + fn, 1e-7)
    
    f1score = 2 * precision * recall / np.maximum(precision + recall, 1e-7)

    macc = recall.mean()

    mdict = {
        # "iou": ious.tolist(),
        "miou": miou.item(),
        "fmiou": f_miou.item(),
        "macc": macc.item(),
        # "acc0.15": (ious > 0.15).sum().item(),
        # "acc0.25": (ious > 0.25).sum().item(),
        # "acc0.50": (ious > 0.50).sum().item(),
        # "acc0.75": (ious > 0.75).sum().item(),
        # "precision": precision.tolist(),
        # "recall": recall.tolist(),
        # "f1score": f1score.tolist()
    }

    return mdict


def load_matrices(results_dir):
    matrices = {}
    
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith("_conf_matrix.pkl"):
            scene_name = filename.replace("_conf_matrix.pkl", "")
            file_path = os.path.join(results_dir, filename)
            
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                
            matrices[scene_name] = data
            
    return matrices


def get_overall_conf_matrix(matrices):
    overall_labels = np.unique([idx for data in matrices.values() for idx in data['labels']])
    
    overall_conf_matrix = np.zeros((len(overall_labels), len(overall_labels)), dtype=int)
    
    for data in matrices.values():
        labels = data['labels']
        conf_matrix = data["conf_matrix"]
        
        index_map = {int(val): idx for idx, val in enumerate(overall_labels)}
        indices = np.array([index_map[int(val)] for val in labels])
        
        I, J = np.meshgrid(indices, indices, indexing='ij')
        overall_conf_matrix[I, J] += conf_matrix.numpy()
        
    return overall_conf_matrix, overall_labels
        

def process_scenes(results_dir, excluded=None):
    scene_metrics = []
    
    matrices = load_matrices(results_dir)

    for scene_name, data in matrices.items():       
        conf_matrix = data["conf_matrix"].numpy()
        
        metrics = compute_metrics(conf_matrix, data["labels"], excluded=excluded)
        metrics["scene"] = scene_name
        scene_metrics.append(metrics)

    metrics_df = pd.DataFrame(scene_metrics)
    
    overall_conf_matrix, overall_labels = get_overall_conf_matrix(matrices)
    
    return overall_conf_matrix, overall_labels, metrics_df


def main(args):   
    excluded = list(map(int, args.excluded.split()))

    overall_conf_matrix, overall_labels, metrics_df = \
        process_scenes(args.results_dir, excluded=excluded)
    
    overall_metrics_list = []
    mean_overall = metrics_df.iloc[:, :-1].mean().to_dict()
    mean_overall["scene"] = "overall_mean"
    overall_metrics_list.append(mean_overall)
    
    overall_metrics = compute_metrics(overall_conf_matrix, overall_labels, excluded=excluded)
    overall_metrics["scene"] = "overall"
    overall_metrics_list.append(overall_metrics)
    
    if args.chunks is not None:
        for chunk_name, existed in json.loads(args.chunks).items():
            chunk_metrics = compute_metrics(overall_conf_matrix, overall_labels, excluded=excluded, existed=existed)
            chunk_metrics["scene"] = chunk_name
            overall_metrics_list.append(chunk_metrics)
    
    metrics_df = pd.concat([metrics_df, pd.DataFrame(overall_metrics_list)], ignore_index=True)
    print(metrics_df)

    output_dir = args.output_dir if args.output_dir is not None else args.results_dir
    output_file = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    main(args)