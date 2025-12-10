import argparse
import json
import os
import pickle
from pathlib import Path

import open_clip
import torch
import numpy as np
import open3d as o3d

from source.eval import evaluate_scen, load_gt_pointcloud, load_pred_pointcloud
from source.pointcloud import save_pointcloud


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic segmentation predictions")
    
    parser.add_argument(
        "--approach", type=str, required=True,
        help="Name of the approach being evaluated"
    )
    
    parser.add_argument(
        "--semantic_info_path", type=Path, required=True,
        help="Path to the semantic info JSON file"
    )
    
    parser.add_argument(
        "--gt_pc_path", type=Path, required=True,
        help="Path to the ground-truth point cloud file"
    )
    
    parser.add_argument(
        "--pred_pc_path", type=Path, required=True,
        help="Path to the predicted point cloud results"
    )
    
    parser.add_argument(
        "--pred_pc_save_dir", type=Path, default=None,
        help="Optional directory to save predicted point clouds"
    )
    
    parser.add_argument(
        "--existed_classes", type=str, default=None,
        help="Space-separated list of class indices to include (e.g., \"1 2 3\")"
    )
    
    parser.add_argument(
        "--excluded_classes", type=str, default=None,
        help="Space-separated list of class indices to exclude (e.g., \"0\")"
    )
    
    parser.add_argument(
        '--scene_label_set', action='store_true',
        help="If set, use a per-scene label set during evaluation"
    )
    
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to use for evaluation"
    )
    
    parser.add_argument(
        "--output_path", type=Path, required=True,
        help="Directory to save final evaluation metrics and outputs"
    )
    
    parser.add_argument(
        "--result_tag", type=str, required=True,
        help="Unique identifier for this result (used to label outputs)"
    )
    
    parser.add_argument(
        "--clip_batch_size", type=int, default=64,
        help="Batch size for CLIP feature extraction"
    )
    
    parser.add_argument(
        "--clip_name", type=str, default="ViT-H-14",
        help="Name of the OpenCLIP model architecture"
    )
    
    parser.add_argument(
        "--clip_pretrained", type=str, default="laion2b_s32b_b79k",
        help="Pretrained weights identifier for the CLIP model"
    )
    
    parser.add_argument(
        "--clip_prompts", type=str, default="{}",
        help=(
            'Semicolon-separated prompt templates for CLIP, e.g., "{}; an image of {}". '
            'Prompts averaged across templates'
        )
    )

    parser.add_argument(
        "--dataset", type=str, default="replica"
    )
    
    parser.add_argument(
        "--nn_count", type=int, default=5,
        help="Number of nearest neighbors used in k-NN when assigning predicted points to ground truth"
    )

    return parser


def get_semseg_class_names(semantic_info, existed_classes, excluded_classes):        
    class_id_to_label_mapping = {class_param['id']: class_param['name'] for class_param in semantic_info['classes']}
    class_id_to_label_mapping[0] = "background"

    all_classes = set(class_id_to_label_mapping.keys())
    
    if isinstance(existed_classes, str):
        existed_classes = set(map(int, existed_classes.split()))
        
    existed_classes = all_classes if existed_classes is None else existed_classes
    excluded_classes = set([]) if excluded_classes is None else set(map(int, excluded_classes.split()))
        
    # assert all_classes >= existed_classes
    
    abandoned_classes = sorted(list(
        (all_classes & existed_classes) - excluded_classes
    ))
    
    print("Using classes: ", [(i, class_id_to_label_mapping[i]) for i in abandoned_classes])
    
    return abandoned_classes, class_id_to_label_mapping


def get_prompts_feats(prompts, clip_model, clip_tokenizer, device='cuda', batch_size=64):
    text = clip_tokenizer(prompts)
    text = text.to(device)
    class_feats = []
    for i in range(int(np.ceil(len(text) / batch_size))):
        with torch.no_grad():
            class_feats.append(clip_model.encode_text(text[i * batch_size : (i+1) * batch_size]))
        
    class_feats = torch.cat(class_feats, dim=0)
    class_feats /= class_feats.norm(dim=-1, keepdim=True) # (num_classes, D)
    
    return class_feats


def compute_clip_embeddings(class_ids, class_id_to_label_mapping, device='cuda', batch_size=64, 
                            model_name="ViT-H-14", pretrained="laion2b_s32b_b79k", prompt_templates = ['{}']):
    class_ids = sorted(class_ids)
    class_names = [class_id_to_label_mapping[idx] for idx in class_ids]

    clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained)
    clip_model = clip_model.to(device)
    clip_tokenizer = open_clip.get_tokenizer(model_name)
    
    template_feats = []
    
    for template in prompt_templates:
        prompts = [template.format(name.replace('_', ' ')) for name in class_names]
        
        template_feats.append(get_prompts_feats(prompts, clip_model, clip_tokenizer, device, batch_size))
        
    class_feats = torch.stack(template_feats, dim=-1).mean(dim=-1)
    class_feats /= class_feats.norm(dim=-1, keepdim=True)
    
    return {'feats': class_feats, 'names': class_names, 'ids': class_ids}


def save_results(output_path, result_tag, conf_matrix=None, df_result=None):   
    os.makedirs(output_path, exist_ok=True)
    
    if conf_matrix is not None:
        save_path = os.path.join(
            output_path,
            f"{result_tag}_conf_matrix.pkl"
        )
                
        pickle.dump(conf_matrix, open(save_path, "wb"))
        
    if df_result is not None:
        save_path = os.path.join(
            output_path,
            f"{result_tag}_results.csv"
        )
            
        df_result.to_csv(save_path, index=False)
        

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    semantic_info = json.load(open(args.semantic_info_path))
    
    gt_pointcloud = load_gt_pointcloud(args.gt_pc_path, semantic_info, dataset=args.dataset)
    
    if args.scene_label_set:
        gt_class = gt_pointcloud[-1]
        args.existed_classes = set(gt_class.tolist())
    
    abandoned_classes, class_id_to_label_mapping = \
        get_semseg_class_names(
            semantic_info = semantic_info,
            existed_classes = args.existed_classes,
            excluded_classes = args.excluded_classes
        )
    
    class_feats = compute_clip_embeddings(
        class_ids = abandoned_classes, 
        class_id_to_label_mapping = class_id_to_label_mapping,
        device = args.device,
        batch_size = args.clip_batch_size,
        model_name = args.clip_name,
        pretrained = args.clip_pretrained,
        prompt_templates = list(map(str.strip, args.clip_prompts.split(';')))
    )
    
    pred_pointcloud = load_pred_pointcloud(args.approach, args.pred_pc_path, class_feats, args.device)
    
    if args.pred_pc_save_dir is not None:
        pred_xyz, pred_color, pred_class = pred_pointcloud
        
        save_pointcloud(
            save_dir = args.pred_pc_save_dir,
            xyz = pred_xyz,
            colors = pred_color,
            semantics = pred_class,
            annotations = class_id_to_label_mapping
        )
    
    conf_matrix = evaluate_scen(
        gt_pointcloud,
        pred_pointcloud, 
        class_feats,
        nn_count = args.nn_count
    )
    
    save_results(args.output_path, args.result_tag, conf_matrix=conf_matrix)


if __name__ == '__main__':
    main()