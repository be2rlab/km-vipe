import glob
import gzip
import os
import pickle
import sys

import torch
import numpy as np

from pathlib import Path
from typing import Any


sys.path.insert(0, "/home/user/km-vipe/dino/Talk2DINO/")
from src.model import ProjectionLayer

def _resolve_device(device: str | torch.device) -> torch.device:
    """Normalize device inputs to torch.device."""
    return device if isinstance(device, torch.device) else torch.device(device)


def load_slam_map(path: Path, device: str = "cpu"):
    """Load the SLAM map from disk."""
    torch_device = _resolve_device(device)
    data = torch.load(path, map_location=torch_device)
    return data


def _extract_pca_basis(state: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Attempt to locate (mean, components) tensors inside an arbitrary container saved on disk.
    Supports plain dicts, nested dicts, or objects exposing .mean/.components (e.g., PCABasis).
    """
    if isinstance(state, dict):
        if "mean" in state and "components" in state:
            return state["mean"], state["components"]
        for value in state.values():
            if isinstance(value, dict):
                try:
                    return _extract_pca_basis(value)
                except (KeyError, TypeError):
                    continue
    elif hasattr(state, "mean") and hasattr(state, "components"):
        return getattr(state, "mean"), getattr(state, "components")
    raise KeyError("Unable to locate 'mean' and 'components' tensors in the provided PCA state.")


def load_pca_basis(path: Path, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load PCA basis tensors (mean, components) from disk.

    Args:
        path: Path to a .pt/.pth file that stores a PCAProjector.state_dict() or equivalent.
        device: Device to map the tensors to when loading.
    """
    torch_device = _resolve_device(device)
    state = torch.load(path, map_location=torch_device)
    mean, components = _extract_pca_basis(state)
    if not isinstance(mean, torch.Tensor):
        mean = torch.as_tensor(mean, device=torch_device, dtype=torch.float32)
    if not isinstance(components, torch.Tensor):
        components = torch.as_tensor(components, device=torch_device, dtype=torch.float32)
    return mean, components


def decode_embeddings_with_pca(
    encoded_embeddings: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    """
    Decode PCA-compressed embeddings back to the original feature dimension.

    Args:
        encoded_embeddings: (N, K) tensor containing PCA codes.
        mean: (C,) tensor representing the feature mean used during PCA fit.
        components: (C, K) tensor with PCA components.
    """
    if encoded_embeddings is None:
        raise ValueError("Cannot decode embeddings because the map does not contain any.")
    if encoded_embeddings.dim() != 2:
        raise ValueError(f"Expected encoded embeddings to be 2D, got shape {encoded_embeddings.shape}.")

    comps = components.to(device=encoded_embeddings.device, dtype=encoded_embeddings.dtype)
    mean = mean.to(device=encoded_embeddings.device, dtype=encoded_embeddings.dtype)

    if encoded_embeddings.shape[1] != comps.shape[1]:
        raise ValueError(
            f"PCA code dimension mismatch: encoded dim {encoded_embeddings.shape[1]} "
            f"!= components dim {comps.shape[1]}."
        )

    decoded = encoded_embeddings @ comps.transpose(0, 1) + mean.unsqueeze(0)
    print(f"decoding shape {decoded.shape}")
    return decoded


def get_full_embedding_map(map_path: Path, pca_basis_path: Path, device: str = "cpu"):
    """
    Load a saved SLAM map and attach decoded full-dimensional embeddings under
    the key ``dense_disp_embeddings_full``.
    """
    data = load_slam_map(map_path, device=device)
    if data.get("dense_disp_embeddings_full") is not None:
        return data

    embeddings = data.get("dense_disp_embeddings")
    if embeddings is None:
        raise ValueError("SLAM map does not contain embeddings to decode.")

    mean, components = load_pca_basis(pca_basis_path, device=device)
    decoded = decode_embeddings_with_pca(embeddings, mean, components)
    data["dense_disp_embeddings_full"] = decoded
    return data

def create_talk2dino(device='cuda'):
    proj_name = 'vitb_mlp_infonce'
    # proj_name = 'dinov3_vitb_mlp_infonce'
    config_path = os.path.join("/home/user/km-vipe/dino/Talk2DINO/configs", f"{proj_name}.yaml")
    weights_path = os.path.join("/home/user/km-vipe/dino/Talk2DINO/weights", f"{proj_name}.pth")

    talk2dino = ProjectionLayer.from_config(config_path)
    talk2dino.load_state_dict(torch.load(weights_path, map_location=device))
    talk2dino.to(device)
    
    return talk2dino

# def project_talk2dino(talk2dino, img_features, text_features):
#     import clip
#     from src.model import ProjectionLayer
#     import torch, os

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # Load Talk2DINO projection layer
#     proj_name = 'vitb_mlp_infonce'
#     config_path = os.path.join("configs", f"{proj_name}.yaml")
#     weights_path = os.path.join("weights", f"{proj_name}.pth")

#     talk2dino = ProjectionLayer.from_config(config_path)
#     talk2dino.load_state_dict(torch.load(weights_path, map_location=device))
#     talk2dino.to(device)

#     # Load CLIP model
#     clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
#     tokenizer = clip.tokenize

#     # Example: Tokenize and project text features
#     texts = ["a cat"]
#     text_tokens = tokenizer(texts).to(device)
#     text_features = clip_model.encode_text(text_tokens)
#     projected_text_features = talk2dino.project_clip_txt(text_features)


def load_pred_pointcloud(pred_pc_path, class_feats, device='cuda', batch_size=10_000):
    
    map_path = os.path.join(pred_pc_path, 'rgb_slam_map.pt')
    pca_basis_path = os.path.join(pred_pc_path, 'pca_basis.pt')
    
    results = get_full_embedding_map(map_path, pca_basis_path)
    
    pred_xyz = results['dense_disp_xyz']
    pred_xyz[..., 0] *= -1
    # pred_xyz[..., 1] *= -1
    pred_xyz[..., 2] *= -1
    pred_color = results['dense_disp_rgb']
    feats = results['dense_disp_embeddings_full']
    
    talk2dino = create_talk2dino(device=device)
    projected_text_features = talk2dino.project_clip_txt(class_feats['feats'])
    
    pred_classes = []
    
    for i in range(0, feats.shape[0], batch_size):
        batch_feats = feats[i:i+batch_size].to(device)
        batch_feats = talk2dino.get_visual_embed(batch_feats)
        
        class_sim = torch.nn.functional.cosine_similarity(
            batch_feats.unsqueeze(1), projected_text_features.unsqueeze(0), dim=-1
        )
    
        class_ids = torch.tensor(class_feats['ids'])
        pred_class_batch = class_ids[class_sim.argmax(dim=-1).detach().cpu()] # (num_objects,)
        pred_classes.append(pred_class_batch)
        
    pred_class = torch.cat(pred_classes, dim=0)    
    
    return pred_xyz, pred_color, pred_class