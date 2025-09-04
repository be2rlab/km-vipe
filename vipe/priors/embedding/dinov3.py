from __future__ import annotations
import math
import os
from typing import List, Tuple, Optional, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from torch import Tensor, nn
import time
import contextlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DinoV3Variant(str, Enum):
    VITS   = "vits16"
    VITSP  = "vits16plus"
    VITB   = "vitb16"
    VITL   = "vitl16"
    VITHP  = "vith16plus"
    VIT7B  = "vit7b16"

# --- Config -----------------------------------------------------------------

@dataclass(frozen=True)
class DinoV3Config:
    hub_id: str              # e.g. "dinov3_vits16"
    num_layers: int          # transformer depth
    weights_filename: Optional[str] = None  # filename or None if not provided

# A single registry is easier to maintain than 3 separate dicts.
REGISTRY: Dict[DinoV3Variant, DinoV3Config] = {
    DinoV3Variant.VITS:  DinoV3Config("dinov3_vits16",      12, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    DinoV3Variant.VITSP: DinoV3Config("dinov3_vits16plus",  12, "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
    DinoV3Variant.VITB:  DinoV3Config("dinov3_vitb16",      12, "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    DinoV3Variant.VITL:  DinoV3Config("dinov3_vitl16",      24, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    DinoV3Variant.VITHP: DinoV3Config("dinov3_vith16plus",  32, "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
    DinoV3Variant.VIT7B: DinoV3Config("dinov3_vit7b16",     40, "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"),
}

Alias = Union[DinoV3Variant, str]

_ALIAS_NORMALIZATION = {
    "vits": "vits16", "vits16": "vits16",
    "vitsp": "vits16plus", "vits16plus": "vits16plus",
    "vitb": "vitb16", "vitb16": "vitb16",
    "vitl": "vitl16", "vitl16": "vitl16",
    "vithp": "vith16plus", "vith": "vith16plus", "vith16plus": "vith16plus",
    "vit7b": "vit7b16", "vit7b16": "vit7b16",
}

def _normalize_alias(x: str) -> str:
    s = x.lower().replace("_", "").replace("-", "")
    if s.startswith("dinov3"):
        s = s.replace("dinov3", "", 1)
    s = s.strip()
    return _ALIAS_NORMALIZATION.get(s, s)

def get_config(variant: Alias) -> DinoV3Config:
    """Accepts DinoV3Variant or a string alias ('vitl', 'dinov3_vitl16', etc.)."""
    if isinstance(variant, DinoV3Variant):
        return REGISTRY[variant]
    norm = _normalize_alias(variant)
    for v in DinoV3Variant:
        if v.value == norm:
            return REGISTRY[v]
    raise KeyError(f"Unknown DINOv3 variant: {variant!r}")


@torch.compile(disable=True)
def _no_op_compile_guard(x):
    return x

class ResizeTransform(nn.Module):
    """Resize image to a fixed size."""
    
    def __init__(self, image_size: int = 768, patch_size: int = 16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
    
    def forward(self, img):
        w, h = img.size
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        return TVTF.resize(img, (h_patches * self.patch_size, w_patches * self.patch_size), interpolation=TVT.InterpolationMode.BICUBIC)


class ResizeToMultiple(nn.Module):
    """Resize image to make dimensions multiples of a given value while maintaining aspect ratio."""
    
    def __init__(self, short_side: int, multiple: int):
        super().__init__()
        self.short_side = short_side
        self.multiple = multiple

    def _round_up(self, side: float) -> int:
        return math.ceil(side / self.multiple) * self.multiple

    def forward(self, img):
        old_width, old_height = TVTF.get_image_size(img)
        if old_width > old_height:
            new_height = self._round_up(self.short_side)
            new_width = self._round_up(old_width * new_height / old_height)
        else:
            new_width = self._round_up(self.short_side)
            new_height = self._round_up(old_height * new_width / old_width)
        return TVTF.resize(img, [new_height, new_width], interpolation=TVT.InterpolationMode.BICUBIC)


class DINOv3EmbeddingEngine:
    """
    A comprehensive class for embedding using DINOv3 features.
    
    This class handles:
    - DINOv3 model initialization and loading
    - Frame embedding and dense feature extraction
    - Mask-based feature embedding
    - Visualization of results
    """
    
    def __init__(
        self,
        model: Alias = DinoV3Variant.VITL,
        weights_path: Optional[Union[str, Path]] = None,
        weights_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        short_side: int = 768,
    ):
        """
        Initialize the DINOv3 engine.
        
        Args:
            model: Variant enum or alias string ('vitl', 'dinov3_vitl16', etc.)
            weights_path: Explicit path to weights (overrides registry)
            weights_dir: If provided, will join with registry filename
            device: 'cuda' or 'cpu'
            short_side: Target short side for preprocessing (used by ResizeTransform)
        """
        self.device = device
        self.short_side = short_side

        # Resolve config from alias/enum
        self.cfg = get_config(model)
        self.n_layers = self.cfg.num_layers

        # Resolve weights path
        if weights_path is None and self.cfg.weights_filename and weights_dir is not None:
            weights_path = Path(weights_dir) / self.cfg.weights_filename
        self.weights_path = str(weights_path) if isinstance(weights_path, Path) else weights_path
        print(f"Using weights path: {self.weights_path}")
        # Initialize model
        self.model = self._load_model(self.cfg.hub_id, self.weights_path)
        self.patch_size = getattr(self.model, "patch_size", 16)
        self.embed_dim = getattr(self.model, "embed_dim", None)

        # Initialize transform
        self.transform = self._create_transform()

        # Initialize tracking state
        self.reset_state()
        
        print(f"Initialized DINOv3 engine:")
        print(f"  Variant hub_id: {self.cfg.hub_id}")
        print(f"  Num layers: {self.n_layers}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Embedding dimension: {self.embed_dim}")
        print(f"  Device: {self.device}")
    
    def _load_model(self, hub_id: str, weights_path: Optional[str]) -> nn.Module:
        """Load DINOv3 model via torch.hub with optional local weights."""
        try:
            model = torch.hub.load(
                repo_or_dir="facebookresearch/dinov3",
                model=hub_id,
                source="github",
            )

            if weights_path and os.path.exists(weights_path):
                state = torch.load(weights_path, map_location="cpu")
                # Some DINO checkpoints need strict=False
                model.load_state_dict(state, strict=False)

            model.to(self.device)
            model.eval()
            torch.set_grad_enabled(False)
            return model

        except Exception as e:
            print(f"Error loading model from hub ({e}). Falling back to local repo path...")
            model = torch.hub.load(
                repo_or_dir="/home/user/dinov3",
                model=hub_id,
                source="local",
            )
            if weights_path and os.path.exists(weights_path):
                state = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state, strict=False)
            model.to(self.device)
            model.eval()
            return model
    
    def _create_transform(self) -> TVT.Compose:
        """Create image preprocessing transform."""
        return TVT.Compose([
            ResizeTransform(image_size=self.short_side, patch_size=self.patch_size),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def reset_state(self):
        """Reset the tracking state for processing a new video."""
        self.features_queue: List[Tensor] = []
        self.probs_queue: List[Tensor] = []
        self.first_feats: Optional[Tensor] = None
        self.first_probs: Optional[Tensor] = None
        self.neighborhood_mask: Optional[Tensor] = None
        self.num_masks: int = 0
        self.frame_height: int = 0
        self.frame_width: int = 0
        self.feats_height: int = 0
        self.feats_width: int = 0
    
    def embed_frame(self, image: Union[Image.Image, Tensor]) -> Tensor:
        """
        Extract dense pixel-wise features from a single frame.

        Returns:
            Tensor [h, w, D] on self.device
        """
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image).to(self.device, non_blocking=True)  # [C,H,W]
        elif isinstance(image, Tensor):
            img_tensor = image.to(self.device, non_blocking=True)
        else:
            raise ValueError("Input image must be a PIL Image or a Tensor")

        self.model.eval()

        # Proper AMP context
        if self.device == "cuda":
            use_bf16 = torch.cuda.is_bf16_supported()
            amp_ctx = torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if use_bf16 else torch.float16
            )
        else:
            amp_ctx = contextlib.nullcontext()

        with torch.inference_mode(), amp_ctx:
            # Ensure batch dimension: [1, C, H, W]
            x = img_tensor.unsqueeze(0)

            # DINO(v2/v3) API commonly returns a list of feature maps
            # Here we assume reshape=True,norm=True yields [B, D, h, w] per layer
            feats_list = self.model.get_intermediate_layers(
                x, n=range(self.n_layers), reshape=True, norm=True
            )
            # Pick the desired layer (default last)
            feat_idx = getattr(self, "feat_layer", -1)
            feats = feats_list[feat_idx]       # [1, D, h, w]
            feats = feats[0]                   # [D, h, w] (explicit, no .squeeze())
            feats = feats.permute(1, 2, 0).contiguous()  # [h, w, D]

        # Keep on device to avoid later device mismatches
        # (Move to CPU only where/when you actually need it.)
        return feats

    def embed_frame_with_masks(
        self,
        image: Union[Image.Image, Tensor],
        masks: np.ndarray,
        return_mask_probs: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[int, Tensor]]:
        """
        Args:
            image: Input frame
            masks: HxW integer mask image where each value encodes an instance/class.
            return_mask_probs: if True returns one-hot [h, w, M]; can be large.

        Returns:
            dense_features: [h, w, D] (on self.device)
            mask_probs: [h, w, M] or None (on self.device, float32)
            mask_embeddings: {mask_id: [D]} (each on self.device)
        """
        dense_features = self.embed_frame(image)  # [h,w,D] on device
        h, w, D = dense_features.shape

        # Cache shapes for later consumers if needed
        self.feats_height, self.feats_width = int(h), int(w)

        if isinstance(image, Image.Image):
            processed_img = self.transform(image).to(self.device, non_blocking=True)  # [C,H',W']
            _, self.frame_height, self.frame_width = processed_img.shape

        # --- Masks ---
        # (1) to device, long dtype
        masks_tensor = torch.as_tensor(masks, device=self.device, dtype=torch.long)  # [H,W]

        # (2) resize to feature resolution with nearest kernel
        masks_resized = F.interpolate(
            masks_tensor[None, None].float(),  # [1,1,H,W]
            size=(h, w),
            mode="nearest"
        )[0, 0].long()  # [h,w]

        # (3) number of masks (assume background == 0)
        num_masks = int(masks_resized.max().item()) + 1
        self.num_masks = num_masks

        # (4) optional one-hot (warning: can be large [h,w,M])
        mask_probs = None
        if return_mask_probs:
            mask_probs = F.one_hot(masks_resized, num_classes=num_masks).to(torch.float32)

        # --- Vectorized mask embeddings ---
        # flatten
        labels = masks_resized.view(-1)           # [N]
        feats = dense_features.view(-1, D)        # [N,D]

        # sum features per mask id
        sums = torch.zeros(num_masks, D, device=self.device, dtype=feats.dtype)
        sums.index_add_(0, labels, feats)         # accumulate rows into sums[labels]

        # count per mask id
        counts = torch.bincount(labels, minlength=num_masks).clamp_min(1).unsqueeze(1)  # [M,1]

        means = sums / counts                      # [M,D]

        # Build dict (skip background id=0)
        mask_embeddings: Dict[int, Tensor] = {
            int(i): means[i] for i in range(1, num_masks) if counts[i].item() > 0
        }

        return dense_features, mask_probs, mask_embeddings

    
    @staticmethod
    def mask_to_rgb(mask: Union[np.ndarray, Tensor], num_masks: int) -> np.ndarray:
        """Convert segmentation mask to RGB visualization."""
        if isinstance(mask, Tensor):
            mask = mask.cpu().numpy()

        # Exclude background
        background = mask == 0
        mask = mask - 1
        num_masks = num_masks - 1

        # Choose palette
        if num_masks <= 10:
            mask_rgb = plt.get_cmap("tab10")(mask)[..., :3]
        elif num_masks <= 20:
            mask_rgb = plt.get_cmap("tab20")(mask)[..., :3]
        else:
            mask_rgb = plt.get_cmap("gist_rainbow")(mask / (num_masks - 1))[..., :3]

        mask_rgb = (mask_rgb * 255).astype(np.uint8)
        mask_rgb[background, :] = 0
        return mask_rgb
    
    def visualize_embeddings(
        self,
        features: Tensor,
        image: Optional[Union[Image.Image, Tensor]] = None,
        method: str = "pca",
        num_components: int = 3,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        Visualize DINOv3 feature embeddings using dimensionality reduction.
        
        Args:
            features: Feature tensor of shape [h, w, D]
            image: Original image for overlay (optional)
            method: Reduction method ('pca', 'tsne', 'mean', 'std', 'norm')
            num_components: Number of components for PCA/t-SNE (1-3)
            save_path: Path to save visualization
            title: Custom title for the plot
        """
        h, w, D = features.shape
        features_np = features.cpu().numpy()
        
        if method == "pca":
            # Use PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            
            features_flat = features_np.reshape(-1, D)
            pca = PCA(n_components=num_components)
            reduced_features = pca.fit_transform(features_flat)
            
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            
            if num_components == 1:
                vis_features = reduced_features.reshape(h, w)
                self._plot_single_channel(vis_features, f"PCA Component 1", save_path, title)
            elif num_components == 2:
                vis_features = reduced_features.reshape(h, w, 2)
                self._plot_dual_channel(vis_features, "PCA Components 1&2", save_path, title)
            else:  # num_components == 3
                vis_features = reduced_features.reshape(h, w, 3)
                self._plot_rgb_features(vis_features, "PCA RGB Visualization", image, save_path, title)
                
        elif method == "tsne":
            # Use t-SNE for dimensionality reduction
            from sklearn.manifold import TSNE
            
            features_flat = features_np.reshape(-1, D)
            # Subsample for t-SNE if too many points
            if features_flat.shape[0] > 10000:
                indices = np.random.choice(features_flat.shape[0], 10000, replace=False)
                features_subset = features_flat[indices]
                tsne = TSNE(n_components=num_components, random_state=42, perplexity=30)
                reduced_subset = tsne.fit_transform(features_subset)
                
                # Interpolate back to full resolution
                reduced_features = np.zeros((features_flat.shape[0], num_components))
                reduced_features[indices] = reduced_subset
            else:
                tsne = TSNE(n_components=num_components, random_state=42, perplexity=30)
                reduced_features = tsne.fit_transform(features_flat)
            
            if num_components == 1:
                vis_features = reduced_features.reshape(h, w)
                self._plot_single_channel(vis_features, "t-SNE Component 1", save_path, title)
            elif num_components == 2:
                vis_features = reduced_features.reshape(h, w, 2)
                self._plot_dual_channel(vis_features, "t-SNE Components 1&2", save_path, title)
            else:  # num_components == 3
                vis_features = reduced_features.reshape(h, w, 3)
                self._plot_rgb_features(vis_features, "t-SNE RGB Visualization", image, save_path, title)
                
        elif method == "mean":
            # Visualize mean activation across all dimensions
            vis_features = features_np.mean(axis=-1)
            self._plot_single_channel(vis_features, "Mean Activation", save_path, title)
            
        elif method == "std":
            # Visualize standard deviation across all dimensions
            vis_features = features_np.std(axis=-1)
            self._plot_single_channel(vis_features, "Feature Std Deviation", save_path, title)
            
        elif method == "norm":
            # Visualize L2 norm of features
            vis_features = np.linalg.norm(features_np, axis=-1)
            self._plot_single_channel(vis_features, "Feature L2 Norm", save_path, title)
            
        else:
            raise ValueError(f"Unknown visualization method: {method}")
    
    def _plot_single_channel(self, features: np.ndarray, method_name: str, save_path: Optional[str], title: Optional[str]):
        """Plot single-channel feature visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        im = ax.imshow(features, cmap='viridis')
        ax.set_title(title or f"DINOv3 Features - {method_name}")
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def _plot_dual_channel(self, features: np.ndarray, method_name: str, save_path: Optional[str], title: Optional[str]):
        """Plot dual-channel feature visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Component 1
        im1 = axes[0].imshow(features[:, :, 0], cmap='viridis')
        axes[0].set_title("Component 1")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Component 2
        im2 = axes[1].imshow(features[:, :, 1], cmap='viridis')
        axes[1].set_title("Component 2")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Combined visualization (as RG channels)
        combined = np.zeros((features.shape[0], features.shape[1], 3))
        combined[:, :, 0] = (features[:, :, 0] - features[:, :, 0].min()) / (features[:, :, 0].max() - features[:, :, 0].min())
        combined[:, :, 1] = (features[:, :, 1] - features[:, :, 1].min()) / (features[:, :, 1].max() - features[:, :, 1].min())
        
        axes[2].imshow(combined)
        axes[2].set_title("Combined (R=Comp1, G=Comp2)")
        axes[2].axis('off')
        
        fig.suptitle(title or f"DINOv3 Features - {method_name}")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def _plot_rgb_features(self, features: np.ndarray, method_name: str, image: Optional[Union[Image.Image, Tensor]], save_path: Optional[str], title: Optional[str]):
        """Plot RGB feature visualization with optional image overlay."""
        # Normalize features to [0, 1] for RGB display
        features_norm = features.copy()
        for i in range(3):
            channel = features_norm[:, :, i]
            features_norm[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        
        if image is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            if isinstance(image, Tensor):
                if image.dim() == 3:  # [C, H, W]
                    img_np = image.permute(1, 2, 0).cpu().numpy()
                    # Denormalize if needed
                    if img_np.min() < 0:  # Likely normalized
                        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        img_np = np.clip(img_np, 0, 1)
                else:
                    img_np = image.cpu().numpy()
            else:
                img_np = np.array(image) / 255.0
            
            # Resize image to match feature resolution
            from skimage.transform import resize
            img_resized = resize(img_np, (features.shape[0], features.shape[1]), anti_aliasing=True)
            
            axes[0].imshow(img_resized)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Feature visualization
            axes[1].imshow(features_norm)
            axes[1].set_title(f"{method_name}")
            axes[1].axis('off')
            
            # Overlay
            alpha = 0.6
            overlay = alpha * img_resized + (1 - alpha) * features_norm
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay (60% Image + 40% Features)")
            axes[2].axis('off')
            
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(features_norm)
            ax.set_title(title or f"DINOv3 Features - {method_name}")
            ax.axis('off')
        
        fig.suptitle(title or f"DINOv3 Features - {method_name}")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def visualize_feature_similarity(
        self,
        features: Tensor,
        query_points: List[Tuple[int, int]],
        image: Optional[Union[Image.Image, Tensor]] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        Visualize feature similarity maps for specific query points.
        
        Args:
            features: Feature tensor of shape [h, w, D]
            query_points: List of (y, x) coordinates to use as query points
            image: Original image for reference
            save_path: Path to save visualization
            title: Custom title
        """
        h, w, D = features.shape
        num_queries = len(query_points)
        
        fig, axes = plt.subplots(2, num_queries + 1, figsize=(4 * (num_queries + 1), 8))
        if num_queries == 0:
            return
        
        # Show original image if available
        if image is not None:
            if isinstance(image, Tensor):
                if image.dim() == 3:
                    img_np = image.permute(1, 2, 0).cpu().numpy()
                    if img_np.min() < 0:
                        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        img_np = np.clip(img_np, 0, 1)
                else:
                    img_np = image.cpu().numpy()
            else:
                img_np = np.array(image) / 255.0
            
            from skimage.transform import resize
            img_resized = resize(img_np, (h, w), anti_aliasing=True)
            
            axes[0, 0].imshow(img_resized)
            axes[1, 0].imshow(img_resized)
        else:
            # Show feature norm as reference
            feature_norm = torch.norm(features, dim=-1).cpu().numpy()
            axes[0, 0].imshow(feature_norm, cmap='viridis')
            axes[1, 0].imshow(feature_norm, cmap='viridis')
        
        axes[0, 0].set_title("Reference Image")
        axes[1, 0].set_title("Query Points")
        
        # Mark query points
        for i, (y, x) in enumerate(query_points):
            axes[1, 0].plot(x, y, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
            axes[1, 0].text(x, y-2, f'Q{i+1}', color='white', fontweight='bold', ha='center')
        
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Compute and show similarity maps
        features_flat = features.view(-1, D)  # [hw, D]
        
        for i, (qy, qx) in enumerate(query_points):
            # Get query feature
            query_feat = features[qy, qx].unsqueeze(0)  # [1, D]
            
            # Compute similarities
            similarities = F.cosine_similarity(query_feat, features_flat, dim=1)  # [hw]
            sim_map = similarities.view(h, w).cpu().numpy()
            
            # Show similarity map
            im1 = axes[0, i+1].imshow(sim_map, cmap='hot', vmin=0, vmax=1)
            axes[0, i+1].set_title(f'Similarity to Query {i+1}\n({qx}, {qy})')
            axes[0, i+1].axis('off')
            plt.colorbar(im1, ax=axes[0, i+1], shrink=0.8)
            
            # Show thresholded similarity (top 10%)
            threshold = np.percentile(sim_map, 90)
            sim_thresh = (sim_map > threshold).astype(float)
            axes[1, i+1].imshow(sim_thresh, cmap='Reds', alpha=0.7)
            if image is not None:
                axes[1, i+1].imshow(img_resized, alpha=0.3)
            axes[1, i+1].set_title(f'Top 10% Similar\n(threshold: {threshold:.3f})')
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved similarity visualization to {save_path}")
        
        plt.show()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / (2**30),
                "reserved_gb": torch.cuda.memory_reserved() / (2**30),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (2**30),
            }
        return {"message": "CUDA not available"}


def demo_usage():
    tracker = DINOv3EmbeddingEngine(
        model=DinoV3Variant.VITHP,                # or "vithp" / "dinov3_vith16plus"
        weights_dir="/home/user/km-vipe/weights/dinov3"         # used if registry has a filename
    )
    
    dummy_frames = []
    for i in range(20):
        dummy_frames.append(Image.open(f"/data/{str(i).zfill(6)}.jpg"))
        test_image = dummy_frames[-1]
        t0 = time.time()
        features = tracker.embed_frame(test_image)  # Warm-up
        print(f"Frame {i} embedded: features shape {features.shape}, dtype {features.dtype}, device {features.device}")
        t1 = time.time()
        print(f"Warm-up embedding time: {t1 - t0:.3f} seconds")
        tracker.visualize_embeddings(features, test_image, method="pca", num_components=3, 
                                   title="DINOv3 Features - PCA RGB")
        

if __name__ == "__main__":
    # Run demonstration
    demo_usage()
