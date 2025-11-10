import numpy as np
import torch

from vipe.streams.base import VideoFrame

from .dinov3 import DINOv3EmbeddingEngine, DinoV3Variant


class EmbeddingsPipeline:
    """Base class for embedding pipelines."""

    def __init__(
        self,
        model_variant: DinoV3Variant = DinoV3Variant.VITL,
        weights_dir: str | None = None,
    ) -> None:
        if weights_dir is None:
            weights_dir = "/home/user/km-vipe/weights/dinov3"

        self.engine = DINOv3EmbeddingEngine(model=model_variant, weights_dir=weights_dir)
        # Store the ImageNet statistics that the DINO models expect so that we can
        # normalize frames without relying on the engine's resize-heavy transforms.
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _prepare_image_tensor(self, frame_data: VideoFrame) -> torch.Tensor:
        """
        Convert a frame into a channel-first tensor that is normalized the same way
        DINO was trained (ImageNet mean/std). We keep the original resolution so
        that the embeddings stay aligned with the pixels the SLAM stack optimizes.
        """
        rgb_tensor = frame_data.rgb.permute(2, 0, 1).contiguous().float()
        rgb_tensor = rgb_tensor.clamp_(0.0, 1.0)

        mean = self._norm_mean.to(rgb_tensor.device)
        std = self._norm_std.to(rgb_tensor.device)
        return (rgb_tensor - mean) / std

    def process_frame(self, frame_data: VideoFrame) -> tuple[torch.Tensor, int]:
        """Process a single video frame to extract embeddings."""
        orig_device = frame_data.rgb.device
        normalized = self._prepare_image_tensor(frame_data)

        feats = self.engine.embed_frame(normalized).detach()
        # Keep feature tensors on the same device as the frame so that subsequent
        # processors (resize/crop) do not pay extra transfer costs.
        feats = feats.to(orig_device, non_blocking=True).float()

        return feats, self.engine.patch_size  # (H', W', C) torch tensor, patch size
