import numpy as np
import torch

from vipe.streams.base import VideoFrame

from .dinov3 import DINOv3EmbeddingEngine, DinoV3Variant


class EmbeddingsPipeline:
    """Base class for embedding pipelines."""

    def __init__(self, model_variant: DinoV3Variant = DinoV3Variant.VITHP, weights_dir: str = None) -> None:
        self.engine = DINOv3EmbeddingEngine(model=model_variant, weights_dir="/home/user/km-vipe/weights/dinov3")

    def process_frame(self, frame_data: VideoFrame) -> tuple[torch.Tensor, int]:
        """Process a single video frame to extract embeddings."""
        # Convert to RGB numpy images
        rgb_frame = (frame_data.rgb * 255).permute(2, 0, 1)  # [C, H, W]

        feats = self.engine.embed_frame(rgb_frame)
        # print(f"Embedding shape: {feats.shape}")
        # self.engine.visualize_embeddings(
        #     feats, rgb_frame, method="pca", num_components=3, title="DINOv3 Features - PCA RGB"
        # )
        return feats, self.engine.patch_size  # (H', W', C) torch tensor, patch size
