import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
from typing import List, Tuple, Optional, Literal
from torch import Tensor
import matplotlib.pyplot as plt
from guided_filter_pytorch.guided_filter import FastGuidedFilter
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
import gc

# ============================================
# CONFIGURATION
# ============================================
image_path = "/home/user/km-vipe/weights/frame000019.jpg"
REPO_DIR = "/home/user/km-vipe/weights/dinov3"
weights_path = "/home/user/km-vipe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
BENCHMARK_ITERATIONS = 50

# ============================================
# PYRAMID UPSAMPLER
# ============================================
class PyramidUpsampler:
    """Handles multi-scale feature upsampling with different blending strategies."""
    
    def __init__(
        self,
        scales: Optional[List[float]] = None,
        blend_mode: Literal["weighted", "average", "max"] = "weighted",
        device: str = "cuda",
    ):
        self.scales = scales or [1.0, 0.75, 0.5]
        self.blend_mode = blend_mode
        self.device = device
    
    def upsample_single_scale(
        self,
        features: Tensor,
        target_size: Tuple[int, int],
        mode: Literal["bilinear", "bicubic"] = "bilinear",
    ) -> Tensor:
        """Upsample features to target size using single-scale interpolation."""
        device = features.device
        h, w = features.shape[:2] if features.dim() == 3 else features.shape[2:4]
        
        if (h, w) == target_size:
            return features if features.dim() == 3 else features.squeeze(0).permute(1, 2, 0)
        
        if features.dim() == 3:
            features = features.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
        
        interp_kwargs = {"size": target_size, "mode": mode, "antialias": True}
        if mode == "bilinear":
            interp_kwargs["align_corners"] = False
        
        upsampled = F.interpolate(features, **interp_kwargs)
        result = upsampled.squeeze(0).permute(1, 2, 0).contiguous()
        
        return result

# ============================================
# GUIDED FILTER UPSAMPLER (Memory Efficient)
# ============================================
class GuidedFilterUpsampler:
    """Edge-aware upsampling using Fast Guided Filter with memory-efficient batching."""
    
    def __init__(self, radius: int = 2, eps: float = 1e-4, batch_size: int = 32):
        self.gf = FastGuidedFilter(r=radius, eps=eps)
        self.batch_size = batch_size
    
    def upsample(
        self,
        features: Tensor,  # [H, W, D] or [1, D, H, W]
        guidance_image: Tensor,  # [1, 3, H_hr, W_hr]
        target_size: Tuple[int, int]
    ) -> Tensor:
        """
        Upsample features using guided filter with memory-efficient batching.
        
        Args:
            features: Low-res features
            guidance_image: High-res RGB guidance
            target_size: (H_target, W_target)
            
        Returns:
            Upsampled features: [H_target, W_target, D]
        """
        # Convert to [1, D, H, W] if needed
        if features.dim() == 3:
            features = features.permute(2, 0, 1).unsqueeze(0)
        
        B, D, H_lr, W_lr = features.shape
        H_hr, W_hr = target_size
        
        # Create low-res and high-res grayscale guidance
        with torch.no_grad():
            hr_gray = TVTF.rgb_to_grayscale(guidance_image)  # [1, 1, H_hr, W_hr]
            lr_gray = F.interpolate(
                hr_gray, 
                size=(H_lr, W_lr), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Process features in batches to save memory
        refined_channels = []
        num_batches = (D + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, D)
            
            # Process batch
            batch_feats = features[:, start_idx:end_idx, :, :]  # [1, batch_size, H_lr, W_lr]
            batch_refined = []
            
            for i in range(batch_feats.shape[1]):
                feat_channel = batch_feats[:, i:i+1, :, :]
                filtered = self.gf(lr_gray, feat_channel, hr_gray)
                batch_refined.append(filtered)
                
                # Clear intermediate tensors
                del feat_channel, filtered
            
            # Concatenate batch results
            batch_result = torch.cat(batch_refined, dim=1)
            refined_channels.append(batch_result)
            
            # Clean up
            del batch_feats, batch_refined, batch_result
            torch.cuda.empty_cache()
        
        # Concatenate all batches
        refined = torch.cat(refined_channels, dim=1)
        
        # Clean up guidance
        del hr_gray, lr_gray
        torch.cuda.empty_cache()
        
        # Convert back to [H, W, D]
        result = refined.squeeze(0).permute(1, 2, 0).contiguous()
        del refined
        
        return result

# ============================================
# UTILITY FUNCTIONS
# ============================================
class ResizeTransform(torch.nn.Module):
    """Resize image to preserve aspect ratio."""
    
    def __init__(self, image_size: int = 768, patch_size: int = 16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
    
    def forward(self, img):
        w, h = img.size
        h_patches = self.image_size // self.patch_size
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        return TVTF.resize(
            img, 
            (h_patches * self.patch_size, w_patches * self.patch_size)
        )

def compute_quality_metrics(feats_orig: Tensor, feats_upsampled: Tensor) -> dict:
    """
    Compute quality metrics between original and upsampled features.
    
    Args:
        feats_orig: [1, D, H, W] or [H, W, D]
        feats_upsampled: [1, D, H_hr, W_hr] or [H_hr, W_hr, D]
    """
    # Normalize to [1, D, H, W]
    if feats_orig.dim() == 3:
        feats_orig = feats_orig.permute(2, 0, 1).unsqueeze(0)
    if feats_upsampled.dim() == 3:
        feats_upsampled = feats_upsampled.permute(2, 0, 1).unsqueeze(0)
    
    # Downsample upsampled features back to original resolution
    with torch.no_grad():
        feats_down = F.interpolate(
            feats_upsampled, 
            size=feats_orig.shape[-2:], 
            mode='bilinear',
            align_corners=False
        )
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(
        feats_orig.flatten(2),
        feats_down.flatten(2),
        dim=1
    )
    
    # L2 distance
    l2_dist = (feats_orig - feats_down).pow(2).sum(dim=1).sqrt()
    
    # Spatial correlation (smoothness metric)
    def spatial_correlation(f):
        f_flat = f.flatten(2)
        if f_flat.shape[2] <= 1:
            return torch.tensor(1.0)
        corr = F.cosine_similarity(f_flat[:, :, :-1], f_flat[:, :, 1:], dim=1)
        return corr.mean()
    
    orig_corr = spatial_correlation(feats_orig)
    refined_corr = spatial_correlation(feats_down)
    
    # Clean up
    del feats_down
    torch.cuda.empty_cache()
    
    return {
        "cosine_similarity_mean": cos_sim.mean().item(),
        "cosine_similarity_min": cos_sim.min().item(),
        "cosine_similarity_std": cos_sim.std().item(),
        "l2_distance_mean": l2_dist.mean().item(),
        "l2_distance_max": l2_dist.max().item(),
        "spatial_correlation_orig": orig_corr.item(),
        "spatial_correlation_refined": refined_corr.item(),
        "spatial_correlation_diff": abs(orig_corr - refined_corr).item(),
    }

def visualize_comparison(
    original: Tensor, 
    pyramid: Tensor, 
    guided: Tensor,
    save_path: str = "upsampling_comparison.png"
):
    """Visualize first 3 channels as RGB."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, feat, title in zip(
        axes, 
        [original, pyramid, guided],
        ["Original (Low-Res)", "Pyramid (High-Res)", "Guided Filter (High-Res)"]
    ):
        # Take first 3 channels and normalize
        if feat.dim() == 4:
            feat = feat.squeeze(0).permute(1, 2, 0)
        
        rgb = feat[..., :3].cpu().numpy()
        for i in range(3):
            channel = rgb[..., i]
            rgb[..., i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison saved to: {save_path}")

# ============================================
# MAIN BENCHMARKING
# ============================================
def main():
    print("="*60)
    print("GUIDED FILTER vs PYRAMID UPSAMPLER BENCHMARK")
    print("="*60)
    
    # Load model
    print("\nLoading DINOv3 model...")
    model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=weights_path)
    model.cuda().eval()
    print("✓ Model loaded")
    
    # Load and preprocess image
    print("\nLoading image...")
    preprocess = TVT.Compose([
        ResizeTransform(image_size=768, patch_size=16),
        TVT.ToTensor(),
        TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).cuda()
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    print(f"Image shape: {img_tensor.shape}")
    
    # Extract features
    print("\nExtracting features...")
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        target_layer = 23  # Last layer
        feats_list = model.get_intermediate_layers(
            img_tensor, 
            n=[target_layer], 
            reshape=True, 
            norm=True
        )
        feats = feats_list[0][0]  # [D, h, w]
        feats = feats.float().permute(1, 2, 0).contiguous()  # [h, w, D]
    
    print(f"Feature shape: {feats.shape}")
    h_feat, w_feat = feats.shape[0], feats.shape[1]
    
    # Clear model from memory
    del model, feats_list
    torch.cuda.empty_cache()
    gc.collect()
    
    # Initialize upsamplers
    pyramid_upsampler = PyramidUpsampler(
        scales=[1.0, 0.75, 0.5],
        blend_mode="weighted",
        device="cuda"
    )
    
    guided_upsampler = GuidedFilterUpsampler(radius=2, eps=1e-4, batch_size=64)
    
    target_size = (H, W)
    
    # ============================================
    # ACCURACY COMPARISON
    # ============================================
    print("\n" + "="*60)
    print("ACCURACY COMPARISON")
    print("="*60)
    
    # Pyramid upsampling
    print("\nPyramid upsampling...")
    with torch.no_grad():
        pyramid_result = pyramid_upsampler.upsample_single_scale(
            feats, 
            target_size, 
            mode="bilinear"
        )
    pyramid_metrics = compute_quality_metrics(feats, pyramid_result)
    
    # Guided filter upsampling
    print("Guided filter upsampling...")
    with torch.no_grad():
        guided_result = guided_upsampler.upsample(
            feats,
            img_tensor,
            target_size
        )
    guided_metrics = compute_quality_metrics(feats, guided_result)
    
    # Print metrics
    print("\n📊 Pyramid Upsampler Metrics:")
    for key, value in pyramid_metrics.items():
        print(f"   {key:30s}: {value:.6f}")
    
    print("\n📊 Guided Filter Metrics:")
    for key, value in guided_metrics.items():
        print(f"   {key:30s}: {value:.6f}")
    
    # Comparison
    print("\n📊 Relative Comparison:")
    cos_diff = guided_metrics['cosine_similarity_mean'] - pyramid_metrics['cosine_similarity_mean']
    l2_diff = guided_metrics['l2_distance_mean'] - pyramid_metrics['l2_distance_mean']
    corr_diff = guided_metrics['spatial_correlation_diff'] - pyramid_metrics['spatial_correlation_diff']
    
    print(f"   Cosine Similarity Diff: {cos_diff:+.6f} {'(Guided better)' if cos_diff > 0 else '(Pyramid better)'}")
    print(f"   L2 Distance Diff:       {l2_diff:+.6f} {'(Guided better)' if l2_diff < 0 else '(Pyramid better)'}")
    print(f"   Spatial Corr Diff:      {corr_diff:+.6f} {'(Guided better)' if abs(corr_diff) < abs(pyramid_metrics['spatial_correlation_diff']) else '(Pyramid better)'}")
    
    # ============================================
    # SPEED BENCHMARKING
    # ============================================
    print("\n" + "="*60)
    print("SPEED BENCHMARKING")
    print("="*60)
    
    # Warmup
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        with torch.no_grad():
            _ = pyramid_upsampler.upsample_single_scale(feats, target_size)
            _ = guided_upsampler.upsample(feats, img_tensor, target_size)
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Benchmark Pyramid
    print(f"\nBenchmarking Pyramid Upsampler ({BENCHMARK_ITERATIONS} iterations)...")
    pyramid_times = []
    for i in range(BENCHMARK_ITERATIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = pyramid_upsampler.upsample_single_scale(feats, target_size)
        torch.cuda.synchronize()
        end = time.perf_counter()
        pyramid_times.append((end - start) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{BENCHMARK_ITERATIONS}")
    
    pyramid_mean = np.mean(pyramid_times)
    pyramid_std = np.std(pyramid_times)
    pyramid_fps = 1000.0 / pyramid_mean
    
    # Benchmark Guided Filter
    print(f"\nBenchmarking Guided Filter ({BENCHMARK_ITERATIONS} iterations)...")
    guided_times = []
    for i in range(BENCHMARK_ITERATIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = guided_upsampler.upsample(feats, img_tensor, target_size)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        end = time.perf_counter()
        guided_times.append((end - start) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{BENCHMARK_ITERATIONS}")
    
    guided_mean = np.mean(guided_times)
    guided_std = np.std(guided_times)
    guided_fps = 1000.0 / guided_mean
    
    speedup = guided_mean / pyramid_mean
    
    # ============================================
    # RESULTS SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\n⚡ Speed Comparison:")
    print(f"   Pyramid Upsampler:")
    print(f"      Mean:  {pyramid_mean:7.3f} ms ± {pyramid_std:5.3f} ms ({pyramid_fps:6.2f} FPS)")
    print(f"   Guided Filter:")
    print(f"      Mean:  {guided_mean:7.3f} ms ± {guided_std:5.3f} ms ({guided_fps:6.2f} FPS)")
    print(f"   Relative Speed: Pyramid is {abs(1/speedup):.2f}x faster than Guided")
    
    print("\n🎯 Quality Comparison:")
    print(f"   Cosine Similarity:")
    print(f"      Pyramid: {pyramid_metrics['cosine_similarity_mean']:.6f}")
    print(f"      Guided:  {guided_metrics['cosine_similarity_mean']:.6f}")
    print(f"      Winner:  {'Guided ✓' if guided_metrics['cosine_similarity_mean'] > pyramid_metrics['cosine_similarity_mean'] else 'Pyramid ✓'}")
    
    print(f"\n   L2 Distance (lower is better):")
    print(f"      Pyramid: {pyramid_metrics['l2_distance_mean']:.6f}")
    print(f"      Guided:  {guided_metrics['l2_distance_mean']:.6f}")
    print(f"      Winner:  {'Guided ✓' if guided_metrics['l2_distance_mean'] < pyramid_metrics['l2_distance_mean'] else 'Pyramid ✓'}")
    
    print(f"\n   Spatial Correlation Preservation:")
    print(f"      Pyramid: {pyramid_metrics['spatial_correlation_diff']:.6f}")
    print(f"      Guided:  {guided_metrics['spatial_correlation_diff']:.6f}")
    print(f"      Winner:  {'Guided ✓' if guided_metrics['spatial_correlation_diff'] < pyramid_metrics['spatial_correlation_diff'] else 'Pyramid ✓'}")
    
    # ============================================
    # VISUALIZATION
    # ============================================
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualize_comparison(
        feats,
        pyramid_result,
        guided_result,
        save_path="/home/user/km-vipe/upsampling_comparison.png"
    )
    
    print("\n✅ BENCHMARKING COMPLETE!")

if __name__ == "__main__":
    main()