import argparse
import glob
import gzip
import os
import pickle
import re
import yaml
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

import open_clip

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", 
        type=Path,
    )
    
    parser.add_argument(
        "--output_name", 
        type=str,
    )
    
    return parser


def get_xyxy_from_mask(mask):
    non_zero_indices = np.nonzero(mask)

    if non_zero_indices[0].sum() == 0:
        return (0, 0, 0, 0)
    x_min = np.min(non_zero_indices[1])
    y_min = np.min(non_zero_indices[0])
    x_max = np.max(non_zero_indices[1])
    y_max = np.max(non_zero_indices[0])

    return (x_min, y_min, x_max, y_max)


def crop_image(image, mask, padding=30):
    image = np.array(image)
    x1, y1, x2, y2 = get_xyxy_from_mask(mask)

    if image.shape[:2] != mask.shape:
        logger.critical(
            "Shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape)
        )
        raise RuntimeError

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image
    image_crop = image[y1:y2, x1:x2]

    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)

    return image_crop


@torch.no_grad()
def get_obg_feat(obj, image_paths, clip_preprocess, clip_model, device="cuda"):
    image = Image.open(image_paths[obj['color_image_idx']]).convert("RGB")
    mask = obj["mask"]
    image = image.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
    image_crop = crop_image(image, mask)
    # image_crop = crop_image(image, mask, padding=0)

    clip_image_crop = clip_preprocess(image_crop).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(clip_image_crop)

    return image_features


def get_obj_descriptions(scene_dir, objects, start=0, end=-1, stride=1, device="cuda"):
    paths = glob.glob(
        os.path.join(
            scene_dir, "results/frame*.jpg"
        )
    )

    image_paths = sorted(paths)

    # image_paths = {}

    # for path in paths:
    #     filename = os.path.basename(path)
    #     match = re.search(r"frame(\d+)\.jpg", filename)
    #     if match:
    #         index = int(match.group(1))
    #         image_paths[index] = path
    
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "EVA02-B-16", pretrained="merged2b_s8b_b131k"
    )

    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    #     "ViT-H-14", pretrained="laion2b_s32b_b79k"
    # )

    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    #     "ViT-B-32", pretrained="laion2b_s34b_b79k"
    # )

    clip_model = clip_model.to('cuda')

    objects_with_feats = objects.copy()
    data_slice = slice(
        start, 
        end + 1 if end >= 0 else len(image_paths) + 1 + end, 
        stride
    )

    for obj in objects_with_feats:
        clip_ft = get_obg_feat(
            obj = obj, 
            image_paths = image_paths[data_slice], 
            clip_preprocess = clip_preprocess, 
            clip_model = clip_model,
            device = device
        )

        obj['clip_ft'] = clip_ft

    return objects_with_feats


def load_pred_pointcloud(pred_pc_path, class_feats, device='cuda'):
    with gzip.open(pred_pc_path, "rb") as f:
        results = pickle.load(f)

    objects = results['objects']

    object_feats = torch.cat([obj['clip_ft'] for obj in objects]).to(device)
    object_class_sim = torch.nn.functional.cosine_similarity(
        object_feats.unsqueeze(1), class_feats['feats'].unsqueeze(0), dim=-1
    )

    class_ids = torch.tensor(class_feats['ids'])
    object_class = class_ids[object_class_sim.argmax(dim=-1).detach().cpu()] # (num_objects,)

    pred_xyz = []
    pred_color = []
    pred_class = []
    for i in range(len(objects)):
        # obj_pcd = objects[i]['pcd_np']
        pred_xyz.append(objects[i]['pcd_np'])
        pred_color.append(objects[i]['pcd_color_np'])
        pred_class.append(np.ones(objects[i]['pcd_np'].shape[0]) * object_class[i].item())

    pred_xyz = torch.from_numpy(np.concatenate(pred_xyz, axis=0))
    pred_color = torch.from_numpy(np.concatenate(pred_color, axis=0))
    pred_class = torch.from_numpy(np.concatenate(pred_class, axis=0)).long()

    return pred_xyz, pred_color, pred_class


def main(args):
    with open(args.config_path) as file:
        config = yaml.full_load(file)

    pred_pc_path = os.path.join(
        config['nodes_constructor']['output_path'], 
        config['nodes_constructor']['output_name_objects']
    )
    with gzip.open(pred_pc_path, "rb") as f:
        results = pickle.load(f)


    scene_dir = os.path.join(
        config['dataset']['base_dir'], 
        config['dataset']['sequence']
    )
    objects_with_feats = get_obj_descriptions(
        scene_dir = scene_dir, 
        objects = results['objects'],
        start = config['dataset']['start'],
        end = config['dataset']['end'], 
        stride = config['dataset']['stride'],
        device = config['dataset']['device']
    )

    output_pc_path = os.path.join(
        config['nodes_constructor']['output_path'], 
        args.output_name
    )
    
    with gzip.open(output_pc_path, 'wb') as file:
        pickle.dump(
            {'objects': objects_with_feats}, file
        )


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)