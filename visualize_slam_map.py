#!/usr/bin/env python3
"""
Standalone script to visualize SLAM maps saved with SLAMMap.save()
Displays two point clouds: one with RGB colors, one with PCA-colored embeddings.

Usage:
    python visualize_slam_map.py <path_to_saved_map.pt> [--pca-basis projector_state.pt]
"""

import argparse

from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch


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


def pca_to_rgb(embeddings: torch.Tensor, n_components: int = 3) -> np.ndarray:
    """
    Apply PCA to embeddings and convert to RGB visualization.

    Args:
        embeddings: (N, D) tensor of embeddings
        n_components: Number of PCA components (3 for RGB)

    Returns:
        (N, 3) RGB array with values in [0, 255]
    """
    # Convert to numpy and ensure float64 for numerical stability
    emb_np = embeddings.cpu().numpy().astype(np.float64)

    # Center the data
    mean = emb_np.mean(axis=0)
    centered = emb_np - mean

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project onto top n_components
    pca_features = centered @ eigenvectors[:, :n_components]

    # Normalize each component independently to [0, 1]
    rgb = np.zeros((pca_features.shape[0], 3))
    for i in range(min(n_components, 3)):
        channel = pca_features[:, i]
        min_val, max_val = channel.min(), channel.max()
        if max_val - min_val > 1e-8:
            rgb[:, i] = (channel - min_val) / (max_val - min_val)
        else:
            rgb[:, i] = 0.5

    # Convert to uint8
    return (rgb * 255).astype(np.uint8)


def visualize_slam_map(map_path: Path, device: str = "cpu", pca_basis_path: Path | None = None):
    """
    Main visualization function.

    Args:
        map_path: Path to the saved SLAM map (.pt file)
    """
    print(f"Loading SLAM map from: {map_path}")
    if pca_basis_path is not None:
        print(f"Decoding embeddings using PCA basis from: {pca_basis_path}")
        data = get_full_embedding_map(map_path, pca_basis_path, device=device)
    else:
        data = load_slam_map(map_path, device=device)

    # Extract data
    xyz = data["dense_disp_xyz"]
    rgb = data["dense_disp_rgb"]
    full_embeddings = data.get("dense_disp_embeddings_full")
    embeddings = data.get("dense_disp_embeddings")
    embedding_valid = data.get("dense_disp_embedding_valid")

    print(f"Point cloud size: {xyz.shape[0]} points")
    if full_embeddings is not None:
        print("Using decoded full-dimensional embeddings for visualization.")
        embeddings_for_vis = full_embeddings
    else:
        embeddings_for_vis = embeddings
    print(f"Has embeddings: {embeddings_for_vis is not None}")

    # Initialize Rerun
    rr.init("SLAM Map Viewer", spawn=True)

    # Convert to numpy
    xyz_np = xyz.cpu().numpy()
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)

    # Log RGB point cloud
    print("Logging RGB point cloud...")
    rr.log(
        "world/point_cloud/rgb",
        rr.Points3D(
            positions=xyz_np,
            colors=rgb_np,
            radii=0.01,
        ),
    )

    # Log embedding-colored point cloud if available
    if embeddings_for_vis is not None:
        print("Processing embeddings with PCA...")

        # Filter by validity mask if available
        if embedding_valid is not None:
            valid_mask = embedding_valid.cpu().numpy()
            valid_xyz = xyz_np[valid_mask]
            valid_embeddings = embeddings_for_vis[embedding_valid]
            print(f"Valid embeddings: {valid_embeddings.shape[0]} / {embeddings_for_vis.shape[0]}")
        else:
            valid_xyz = xyz_np
            valid_embeddings = embeddings_for_vis

        # Apply PCA to get RGB colors
        pca_colors = pca_to_rgb(valid_embeddings, n_components=3)

        print("Logging PCA-colored point cloud...")
        rr.log(
            "world/point_cloud/pca_embeddings",
            rr.Points3D(
                positions=valid_xyz,
                colors=pca_colors,
                radii=0.01,
            ),
        )

        # Log embedding statistics
        emb_mean = valid_embeddings.mean().item()
        emb_std = valid_embeddings.std().item()
        emb_dim = valid_embeddings.shape[1]
        source = "decoded_full" if full_embeddings is not None else "encoded"

        rr.log("stats/embedding_dim", rr.TextLog(f"{source} embedding dimension: {emb_dim}"))
        rr.log("stats/embedding_source", rr.TextLog(f"Embedding source: {source}"))
        rr.log("stats/embedding_mean", rr.Scalar(emb_mean))
        rr.log("stats/embedding_std", rr.Scalar(emb_std))
    else:
        print("No embeddings found in the map.")

    # Log metadata
    rr.log("stats/total_points", rr.Scalar(xyz.shape[0]))
    rr.log("stats/map_path", rr.TextLog(str(map_path)))

    # Compute and log bounding box
    bbox_min = xyz_np.min(axis=0)
    bbox_max = xyz_np.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min

    print(f"\nBounding box:")
    print(f"  Center: {bbox_center}")
    print(f"  Size: {bbox_size}")

    rr.log(
        "world/bounding_box",
        rr.Boxes3D(
            sizes=[bbox_size],
            centers=[bbox_center],
            colors=[[255, 255, 0, 128]],
        ),
    )

    print("\nVisualization complete! Use the Rerun viewer to explore.")
    print("Toggle between point clouds in the left panel:")
    print("  - world/point_cloud/rgb: Original RGB colors")
    if embeddings_for_vis is not None:
        print("  - world/point_cloud/pca_embeddings: PCA-colored embeddings")


def main():
    parser = argparse.ArgumentParser(description="Visualize SLAM map with RGB and PCA-colored embeddings")
    parser.add_argument(
        "map_path",
        type=Path,
        help="Path to the saved SLAM map (.pt file)",
    )
    parser.add_argument(
        "--pca-basis",
        type=Path,
        default=None,
        help="Optional path to a PCA basis (.pt/.pth) with 'mean' and 'components' tensors to decode full embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load tensors on (default: cpu)",
    )

    args = parser.parse_args()

    if not args.map_path.exists():
        print(f"Error: Map file not found: {args.map_path}")
        return

    if args.pca_basis is not None and not args.pca_basis.exists():
        print(f"Error: PCA basis file not found: {args.pca_basis}")
        return

    visualize_slam_map(args.map_path, device=args.device, pca_basis_path=args.pca_basis)


if __name__ == "__main__":
    main()
