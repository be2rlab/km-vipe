import datetime
import functools
import math
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from torch import Tensor, nn
from tqdm import tqdm


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
        model_name: str = "dinov3_vitl16",
        weights_path: Optional[str] = None,
        device: str = "cuda",
        short_side: int = 680,
        max_context_length: int = 7,
        neighborhood_size: float = 12.0,
        neighborhood_shape: str = "circle",
        topk: int = 5,
        temperature: float = 0.2
    ):
        """
        Initialize the DINOv3 segmentation tracker.
        
        Args:
            model_name: Name of the DINOv3 model to use
            weights_path: Path to custom weights file (optional)
            device: Device to run on ('cuda' or 'cpu')
            short_side: Target size for the short side of input images
            max_context_length: Maximum number of context frames to keep
            neighborhood_size: Size of neighborhood for patch matching
            neighborhood_shape: Shape of neighborhood ('circle' or 'square')
            topk: Number of top similar patches to consider
            temperature: Temperature for softmax in similarity computation
        """
        self.device = device
        self.short_side = short_side
        self.max_context_length = max_context_length
        self.neighborhood_size = neighborhood_size
        self.neighborhood_shape = neighborhood_shape
        self.topk = topk
        self.temperature = temperature
        
        # Initialize model
        self.model = self._load_model(model_name, weights_path)
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        
        # Initialize transform
        self.transform = self._create_transform()
        
        # Initialize tracking state
        self.reset_state()
        
        print(f"Initialized DINOv3 tracker:")
        print(f"  Model: {model_name}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Embedding dimension: {self.embed_dim}")
        print(f"  Device: {self.device}")
    
    def _load_model(self, model_name: str, weights_path: Optional[str]) -> nn.Module:
        """Load DINOv3 model with optional custom weights."""
        try:
            if weights_path and os.path.exists(weights_path):
                # Load with custom weights
                model = torch.hub.load(
                    repo_or_dir="facebookresearch/dinov3",
                    model=model_name,
                    source="github",
                    weights=weights_path
                )
            else:
                # Load with default weights
                model = torch.hub.load(
                    repo_or_dir="facebookresearch/dinov3",
                    model=model_name,
                    source="github"
                )
            
            model.to(self.device)
            model.eval()
            torch.set_grad_enabled(False)
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to local loading...")
            # Fallback for local loading (adjust path as needed)
            model = torch.hub.load(
                repo_or_dir="/home/user/dinov3",
                model=model_name,
                source="local"
            )
            model.to(self.device)
            model.eval()
            return model
    
    def _create_transform(self) -> TVT.Compose:
        """Create image preprocessing transform."""
        return TVT.Compose([
            ResizeToMultiple(short_side=self.short_side, multiple=self.patch_size),
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
    
    @torch.compile(disable=True)
    def embed_frame(self, image: Union[Image.Image, Tensor]) -> Tensor:
        """
        Extract dense pixel-wise features from a single frame.
        
        Args:
            image: Input image (PIL Image or preprocessed tensor)
            
        Returns:
            Dense features tensor of shape [h, w, D]
        """
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image).to(self.device)
        else:
            img_tensor = image
        
        # Extract features using DINOv3
        feats = self.model.get_intermediate_layers(
            img_tensor.unsqueeze(0), n=1, reshape=True
        )[0]  # [1, D, h, w]
        
        feats = feats.movedim(-3, -1)  # [1, h, w, D]
        feats = F.normalize(feats, dim=-1, p=2)  # L2 normalize
        
        return feats.squeeze(0)  # [h, w, D]
    
    def embed_frame_with_masks(
        self, 
        image: Union[Image.Image, Tensor], 
        masks: np.ndarray
    ) -> Tuple[Tensor, Tensor, Dict[int, Tensor]]:
        """
        Extract features from a frame and compute mask-specific embeddings.
        
        Args:
            image: Input image
            masks: Segmentation masks array where each unique value represents a mask
            
        Returns:
            Tuple of (dense_features, mask_probs, mask_embeddings)
            - dense_features: [h, w, D] dense feature map
            - mask_probs: [h, w, M] one-hot mask probabilities
            - mask_embeddings: Dict mapping mask_id to average feature embedding
        """
        # Get dense features
        dense_features = self.embed_frame(image)
        h, w, D = dense_features.shape
        
        # Store dimensions for later use
        self.feats_height, self.feats_width = h, w
        if isinstance(image, Image.Image):
            processed_img = self.transform(image).to(self.device)
            _, self.frame_height, self.frame_width = processed_img.shape
        
        # Process masks
        masks_tensor = torch.from_numpy(masks).to(self.device, dtype=torch.long)
        
        # Resize masks to feature resolution
        masks_resized = F.interpolate(
            masks_tensor[None, None, :, :].float(),
            (h, w),
            mode="nearest-exact"
        )[0, 0].long()
        
        # Get number of masks and create one-hot encoding
        self.num_masks = int(masks.max() + 1)
        mask_probs = F.one_hot(masks_resized, self.num_masks).float()
        
        # Compute mask-specific embeddings
        mask_embeddings = {}
        for mask_id in range(self.num_masks):
            if mask_id == 0:  # Skip background
                continue
                
            mask = (masks_resized == mask_id).float()
            if mask.sum() > 0:
                # Get average feature for this mask
                mask_features = dense_features * mask.unsqueeze(-1)
                mask_embedding = mask_features.sum(dim=(0, 1)) / mask.sum()
                mask_embeddings[mask_id] = mask_embedding
        
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
    
    def analyze_feature_statistics(self, features: Tensor) -> Dict[str, float]:
        """
        Analyze and return statistics about the feature embeddings.
        
        Args:
            features: Feature tensor of shape [h, w, D]
            
        Returns:
            Dictionary with various statistics
        """
        h, w, D = features.shape
        features_flat = features.view(-1, D).cpu().numpy()
        
        stats = {
            "spatial_resolution": f"{h}x{w}",
            "feature_dimension": D,
            "total_patches": h * w,
            "mean_feature_norm": float(torch.norm(features, dim=-1).mean()),
            "std_feature_norm": float(torch.norm(features, dim=-1).std()),
            "mean_feature_value": float(features.mean()),
            "std_feature_value": float(features.std()),
            "min_feature_value": float(features.min()),
            "max_feature_value": float(features.max()),
        }
        
        # Compute pairwise similarities statistics
        print("Computing pairwise similarity statistics...")
        similarities = np.dot(features_flat, features_flat.T)
        
        stats.update({
            "mean_cosine_similarity": float(similarities.mean()),
            "std_cosine_similarity": float(similarities.std()),
            "min_cosine_similarity": float(similarities.min()),
            "max_cosine_similarity": float(similarities.max()),
        })
        
        return stats
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / (2**30),
                "reserved_gb": torch.cuda.memory_reserved() / (2**30),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (2**30),
            }
        return {"message": "CUDA not available"}


# # Example usage and demo
# def demo_usage():
#     """Demonstrate how to use the DINOv3SegmentationTracker class."""
    
#     # Initialize tracker
#     tracker = DINOv3SegmentationTracker(
#         model_name="dinov3_vith16plus",
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         short_side=680*2,
#         max_context_length=7,
#         neighborhood_size=12.0,
#         topk=5,
#         temperature=0.2,
#         weights_path="/home/user/km-vipe/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
#     )
    
#     dummy_frames = []
#     for i in range(20):
#         dummy_frames.append(Image.open(f"/data/{str(i).zfill(6)}.jpg"))
    
    
#     # results = model(dummy_frames[0], device="cuda:0")
#     # dummy_masks = results[0].masks.data.cpu().numpy()[0]  # Use first mask for demo
#     test_image = dummy_frames[0]
    
#     features = tracker.embed_frame(dummy_frames[0])  # Warm-up
#     h, w, D = features.shape
#     tracker.visualize_embeddings(features, test_image, method="pca", num_components=3, 
#                                title="DINOv3 Features - PCA RGB")
    
#     # 2. PCA 2-component visualization  
#     print("\n2. PCA 2-Component Visualization...")
#     tracker.visualize_embeddings(features, test_image, method="pca", num_components=2,
#                                title="DINOv3 Features - PCA 2D")
    
#     # 3. Statistical visualizations
#     print("\n3. Statistical Visualizations...")
#     tracker.visualize_embeddings(features, test_image, method="mean", 
#                                title="Mean Feature Activation")
#     tracker.visualize_embeddings(features, test_image, method="std", 
#                                title="Feature Standard Deviation")
#     tracker.visualize_embeddings(features, test_image, method="norm", 
#                                title="Feature L2 Norm")
    
#     # 4. Similarity visualization with multiple query points
#     print("\n4. Feature Similarity Analysis...")
    
#     # Select query points from different regions
#     query_points = [
#         (h//4, w//4),      # Top-left region
#         (h//2, w//2),      # Center
#         (3*h//4, 3*w//4),  # Bottom-right
#         (h//4, 3*w//4),    # Top-right
#     ]
    
#     tracker.visualize_feature_similarity(features, query_points, test_image,
#                                        title="Feature Similarity Maps")
    
#     # 5. Feature statistics analysis
#     print("\n5. Feature Statistics Analysis...")
#     stats = tracker.analyze_feature_statistics(features)
    
#     print("\nDetailed Feature Statistics:")
#     print("=" * 50)
#     for key, value in stats.items():
#         print(f"{key:.<25} {value}")

#     print(f"Extracted features shape: {feats.shape}")
#     exit()
    
    
#     results[0].save(f"/data/mask_demo.png")
#     print(f"Created {len(dummy_frames)} dummy frames")
#     print(f"Mask shape: {dummy_masks.shape}, unique values: {np.unique(dummy_masks)}")
    
#     # Track the video
#     print("\nTracking video...")
#     mask_probabilities, predicted_masks = tracker.track_video(dummy_frames, dummy_masks)
    
#     print(f"Tracking results:")
#     print(f"  Mask probabilities shape: {mask_probabilities.shape}")
#     print(f"  Predicted masks shape: {predicted_masks.shape}")
    
#     # Visualize results
#     print("\nVisualizing results...")
#     tracker.visualize_tracking_results(
#         dummy_frames, 
#         predicted_masks,
#         mask_probabilities,
#         selected_frames=[0, 2, 4]
#     )
    
#     # Print memory usage
#     memory_stats = tracker.get_memory_usage()
#     print(f"\nMemory usage: {memory_stats}")
    
#     return tracker, dummy_frames, predicted_masks, mask_probabilities


if __name__ == "__main__":
    # Run demonstration
    tracker, frames, masks, probs = demo_usage()