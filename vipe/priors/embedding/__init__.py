from dataclasses import dataclass

import numpy as np
import torch

from vipe.streams.base import VideoFrame

from .dinov3 import DINOv3EmbeddingEngine, DinoV3Variant


@dataclass
class PCABasis:
    mean: torch.Tensor  # (C,)
    components: torch.Tensor  # (C, K)


class PCAProjector:
    """
    Torch-based PCA projector for per-pixel embeddings.
    - Fit: learns mean and top-K components on sampled pixels.
    - Encode: projects to K-dim.
    - Decode: reconstructs to original C-dim (approximate).
    """

    def __init__(self, target_dim: int = 64, max_samples: int = 200_000, seed: int = 0):
        self.target_dim = target_dim
        self.max_samples = max_samples
        self.seed = seed
        self._basis: PCABasis | None = None

    @torch.no_grad()
    def fit(self, feats: torch.Tensor) -> PCABasis:
        """
        feats: (H, W, C) or (N, H, W, C) tensor of per-pixel features (float).
        We randomly sample up to max_samples pixels to learn PCA.
        """
        # Flatten to (P, C)
        if feats.dim() == 3:
            H, W, C = feats.shape
            X = feats.reshape(H * W, C)
        elif feats.dim() == 4:
            N, H, W, C = feats.shape
            X = feats.reshape(N * H * W, C)
        else:
            raise ValueError("feats must be (H, W, C) or (N, H, W, C)")

        device = X.device
        P, C = X.shape

        # Sample pixels
        g = torch.Generator(device=device)
        g.manual_seed(self.seed)
        if P > self.max_samples:
            idx = torch.randint(low=0, high=P, size=(self.max_samples,), generator=g, device=device)
            Xs = X[idx]
        else:
            Xs = X

        # Center
        mean = Xs.mean(dim=0)  # (C,)
        Xc = Xs - mean

        # SVD on (P, C). We only need the right singular vectors (Vh).
        # full_matrices=False is cheaper and is what we want for PCA.
        # Xc = U S V^T  -> top-K components are columns of V (size CxC), take first K.
        # torch.linalg.svd returns U, S, Vh; V = Vh^T
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        V = Vh.transpose(0, 1)  # (C, C)

        K = min(self.target_dim, V.shape[1])
        components = V[:, :K].contiguous()  # (C, K)

        self._basis = PCABasis(mean=mean, components=components)
        return self._basis

    def is_fit(self) -> bool:
        return self._basis is not None

    @torch.no_grad()
    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (H, W, C) float tensor
        returns: (H, W, K) float tensor
        """
        if self._basis is None:
            raise RuntimeError("PCAProjector not fit yet. Call fit() first.")
        mean, comps = self._basis.mean, self._basis.components  # (C,), (C, K)
        device = feats.device
        # Move basis to the right device (in case fit occurred elsewhere)
        mean = mean.to(device=device, dtype=feats.dtype)
        comps = comps.to(device=device, dtype=feats.dtype)

        H, W, C = feats.shape
        X = feats.reshape(-1, C)  # (P, C)
        Xc = X - mean
        Z = Xc @ comps  # (P, K)
        return Z.reshape(H, W, comps.shape[1])

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: (H, W, K) float tensor
        returns: (H, W, C) float tensor (approximate reconstruction)
        """
        if self._basis is None:
            raise RuntimeError("PCAProjector not fit yet. Call fit() first.")
        mean, comps = self._basis.mean, self._basis.components  # (C,), (C, K)
        device = codes.device
        mean = mean.to(device=device, dtype=codes.dtype)
        comps = comps.to(device=device, dtype=codes.dtype)

        H, W, K = codes.shape
        Z = codes.reshape(-1, K)  # (P, K)
        Xr = Z @ comps.transpose(0, 1) + mean  # (P, C)
        C = comps.shape[0]
        return Xr.reshape(H, W, C)

    def state_dict(self) -> dict:
        if self._basis is None:
            raise RuntimeError("Projector has no basis yet.")
        return {"mean": self._basis.mean, "components": self._basis.components}

    def load_state_dict(self, state: dict) -> None:
        self._basis = PCABasis(mean=state["mean"], components=state["components"])


class EmbeddingsPipeline:
    """Base class for embedding pipelines."""

    def __init__(
        self,
        model_variant: DinoV3Variant = DinoV3Variant.VITSP,
        weights_dir: str | None = None,
        pca_dim: int | None = 32,  # e.g., 32 or 64; set None to disable
        pca_max_samples: int = 200_000,
        pca_seed: int = 0,
    ) -> None:
        if weights_dir is None:
            weights_dir = "/home/user/km-vipe/weights/dinov3"

        self.engine = DINOv3EmbeddingEngine(
            model=model_variant, weights_dir=weights_dir, pyramid_scales=[2.0, 1.0, 0.75]
        )
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        # Optional PCA projector
        self.projector: PCAProjector | None = None
        if pca_dim is not None:
            self.projector = PCAProjector(target_dim=pca_dim, max_samples=pca_max_samples, seed=pca_seed)

    def _prepare_image_tensor(self, frame_data: VideoFrame) -> torch.Tensor:
        rgb_tensor = frame_data.rgb.permute(2, 0, 1).contiguous().float()
        rgb_tensor = rgb_tensor.clamp_(0.0, 1.0)
        mean = self._norm_mean.to(rgb_tensor.device)
        std = self._norm_std.to(rgb_tensor.device)
        return (rgb_tensor - mean) / std

    def process_frame(self, frame_data: VideoFrame) -> tuple[torch.Tensor, int]:
        """Process a single video frame to extract (optionally PCA-encoded) embeddings."""
        orig_device = frame_data.rgb.device
        normalized = self._prepare_image_tensor(frame_data)
        pyr_feats = self.engine.embed_frame_multiscale(normalized)
        upfeats = self.engine.upsample_features(
            features=pyr_feats,
            target_size=(frame_data.rgb.shape[0], frame_data.rgb.shape[1]),
            method="pyramid_weighted",
        ).detach()
        del pyr_feats

        feats = upfeats.to(orig_device, non_blocking=True).float()  # (H, W, C)
        del upfeats

        if feats.dim() == 3 and feats.shape[0] < feats.shape[-1]:
            feats = feats.permute(1, 2, 0).contiguous()

        # Optional PCA compress
        if self.projector is not None:
            if not self.projector.is_fit():
                self.projector.fit(feats)

            codes = self.projector.encode(feats)  # (H, W, K)
            del feats
            self.engine.visualize_embeddings(
                codes,
                method="naive_rgb",
                entity_path="visualizations/naive_rgb",
                frame_idx=0,
            )
            return codes, self.engine.patch_size

        # self.engine.visualize_embeddings(
        #     feats,
        #     method="naive_rgb",
        #     entity_path="visualizations/naive_rgb",
        #     frame_idx=0,
        # )
        return feats, self.engine.patch_size  # (H, W, C)
