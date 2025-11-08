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


class DinoV2Variant(str, Enum):
    VITS        = "vits14"
    VITB        = "vitb14"
    VITL        = "vitl14"
    VITG        = "vitg14"
    # VITS_REG    = "vits14_reg"
    # VITB_REG    = "vitb14_reg"
    # VITL_REG    = "vitl14_reg"
    # VITG_REG    = "vitg14_reg"

# --- Config -----------------------------------------------------------------

@dataclass(frozen=True)
class DinoV2Config:
    hub_id: str              # e.g. "dinov2_vits16"
    # num_layers: int          # transformer depth
    weights_filename: Optional[str] = None  # filename or None if not provided

# A single registry is easier to maintain than 3 separate dicts.
REGISTRY: Dict[DinoV2Variant, DinoV2Config] = {
    DinoV2Variant.VITS:     DinoV2Config("dinov2_vits14",       "dinov2_vits14_pretrain.pth"),
    DinoV2Variant.VITB:     DinoV2Config("dinov2_vitb14",       "dinov2_vitb14_pretrain.pth"),
    DinoV2Variant.VITL:     DinoV2Config("dinov2_vitl14",       "dinov2_vitl14_pretrain.pth"),
    DinoV2Variant.VITG:     DinoV2Config("dinov2_vitg14",       "dinov2_vitg14_pretrain.pth"),
    # DinoV2Variant.VITS_REG: DinoV2Config("dinov2_vits14_reg",   "dinov2_vits14_reg4_pretrain.pth"),
    # DinoV2Variant.VITB_REG: DinoV2Config("dinov2_vitb14_reg",   "dinov2_vitb14_reg4_pretrain.pth"),
    # DinoV2Variant.VITL_REG: DinoV2Config("dinov2_vitl14_reg",   "dinov2_vitl14_reg4_pretrain.pth"),
    # DinoV2Variant.VITG_REG: DinoV2Config("dinov2_vitg14_reg",   "dinov2_vitg14_reg4_pretrain.pth"),
}

Alias = Union[DinoV2Variant, str]

_ALIAS_NORMALIZATION = {
    "vits": "vits14", "vits14": "vits14",
    "vitb": "vitb14", "vitb14": "vitb14",
    "vitl": "vitl14", "vitl14": "vitl14",
    "vitg": "vitg14", "vitg14": "vitg14",
    # "vits_reg": "vits14_reg", "vits14_reg": "vits14_reg",
    # "vitb_reg": "vitb14_reg", "vitb14_reg": "vitb14_reg",
    # "vitl_reg": "vitl14_reg", "vitl14_reg": "vitl14_reg",
    # "vitg_reg": "vitg14_reg", "vitg14_reg": "vitg14_reg",
}

def _normalize_alias(x: str) -> str:
    s = x.lower().replace("_", "").replace("-", "")
    if s.startswith("dinov2"):
        s = s.replace("dinov2", "", 1)
    s = s.strip()
    return _ALIAS_NORMALIZATION.get(s, s)

def get_config(variant: Alias) -> DinoV2Config:
    """Accepts DinoV2Variant or a string alias ('vitl', 'dinov2_vitl14', etc.)."""
    if isinstance(variant, DinoV2Variant):
        return REGISTRY[variant]
    norm = _normalize_alias(variant)
    for v in DinoV2Variant:
        if v.value == norm:
            return REGISTRY[v]
    raise KeyError(f"Unknown DINOv2 variant: {variant!r}")


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


class DINOv2EmbeddingEngine:
    """
    A comprehensive class for embedding using DINOv2 features.
    
    This class handles:
    - DINOv2 model initialization and loading
    - Frame embedding and dense feature extraction
    - Mask-based feature embedding
    - Visualization of results
    """
    
    def __init__(
        self,
        model: Alias = DinoV2Variant.VITL,
        weights_path: Optional[Union[str, Path]] = None,
        weights_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        short_side: int = 768, # TODO: check this number
    ):
        """
        Initialize the DINOv2 engine.
        
        Args:
            model: Variant enum or alias string ('vitl', 'dinov2_vitl16', etc.)
            weights_path: Explicit path to weights (overrides registry)
            weights_dir: If provided, will join with registry filename
            device: 'cuda' or 'cpu'
            short_side: Target short side for preprocessing (used by ResizeTransform)
        """
        self.device = device
        self.short_side = short_side

        # Resolve config from alias/enum
        self.cfg = get_config(model)
        # self.n_layers = self.cfg.num_layers

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
        
        print(f"Initialized DINOv2 engine:")
        print(f"  Variant hub_id: {self.cfg.hub_id}")
        # print(f"  Num layers: {self.n_layers}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Embedding dimension: {self.embed_dim}")
        print(f"  Device: {self.device}")
    
    def _load_model(self, hub_id: str, weights_path: Optional[str]) -> nn.Module:
        """Load DINOv2 model via torch.hub with optional local weights."""
        try:
            model = torch.hub.load(
                repo_or_dir="facebookresearch/dinov2",
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
                repo_or_dir="/home/user/km-vipe/dino/dinov2",
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

            # DINO(v2/v2) API commonly returns a list of feature maps
            # Here we assume reshape=True,norm=True yields [B, D, h, w] per layer
            feats_list = self.model.get_intermediate_layers(
                x, n=1, reshape=True, norm=True
            )

            feats = feats_list[0]
            feats = feats[0]                            # [D, h, w] (explicit, no .squeeze())
            feats = feats.permute(1, 2, 0).contiguous() # [h, w, D]

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


def demo_usage():
    tracker = DINOv2EmbeddingEngine(
        model=DinoV2Variant.VITL,                               # or "vithp" / "dinov2_vith16plus"
        weights_dir="/home/user/km-vipe/weights/dinov2"         # used if registry has a filename
    )
    
    dummy_frames = []
    for i in range(1):
        dummy_frames.append(Image.open(f"/data/{str(i).zfill(6)}.jpg"))
        test_image = dummy_frames[-1]
        t0 = time.time()
        features = tracker.embed_frame(test_image)  # Warm-up
        print(f"Frame {i} embedded: features shape {features.shape}, dtype {features.dtype}, device {features.device}")
        t1 = time.time()
        print(f"Warm-up embedding time: {t1 - t0:.3f} seconds")
        # tracker.visualize_embeddings(features, test_image, method="pca", num_components=3, 
        #                            title="DINOv2 Features - PCA RGB")
        

if __name__ == "__main__":
    # Run demonstration
    demo_usage()