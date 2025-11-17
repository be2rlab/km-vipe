import torch
from torchvision import transforms
from torch2trt import TRTModule
import tensorrt as trt
import torch.nn as nn
from PIL import Image
import numpy as np
import time
from typing import Dict
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TVTF

image_path = "/home/user/km-vipe/weights/frame000019.jpg"
engine_path = "/home/user/km-vipe/weights/dinov3_vitl16_bf16_768.engine"
REPO_DIR = "/home/user/km-vipe/weights/dinov3"
weights_path = "/home/user/km-vipe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
IMG_SIZE = 768
PATCH_SIZE = 16
BENCHMARK_ITERATIONS = 100

# Fixed dimensions for both models
TRT_HEIGHT = 768
TRT_WIDTH = 1024

class AspectPreservingResize(nn.Module):
    """Resize image preserving aspect ratio to multiples of patch_size."""
    def __init__(self, image_size: int = 768, patch_size: int = 16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
    
    def forward(self, img):
        # img is [B, C, H, W]
        B, C, h, w = img.shape
        
        # Calculate number of patches: height fixed to image_size, width preserves ratio
        h_patches = self.image_size // self.patch_size
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        
        new_h = h_patches * self.patch_size
        new_w = w_patches * self.patch_size
        
        return TVTF.resize(img, (new_h, new_w), antialias=True)

class DinoBackboneDenseONNX(nn.Module):
    def __init__(self, model, image_size=768, patch_size=16):
        super().__init__()
        self.resize = AspectPreservingResize(image_size, patch_size)
        self.model = model
        self.patch_size = patch_size
        self.h_patches = image_size // patch_size  # Fixed: 768 // 16 = 48

    def forward(self, x):
        # Resize input preserving aspect ratio
        x_resized = self.resize(x)
        
        feats_dict = self.model.forward_features(x_resized)

        # pick dense patch embeddings
        feats = feats_dict["x_norm_patchtokens"]  # [B, N, D]

        # reshape [B, N, D] -> [B, D, H, W]
        B, N, D = feats.shape
        h = self.h_patches  # Fixed height: 48 patches
        w = N // h  # Calculate width from total patches
        patches = feats.permute(0, 2, 1).reshape(B, D, h, w)
        return patches

def compute_cosine_similarity(feat1: Tensor, feat2: Tensor) -> Dict[str, float]:
    """
    Compute cosine similarity metrics between two feature tensors.
    
    Args:
        feat1: First feature tensor [B, D, H, W]
        feat2: Second feature tensor [B, D, H, W]
    
    Returns:
        Dictionary with mean, min, max cosine similarity
    """
    # Flatten spatial dimensions
    feat1_flat = feat1.flatten(2)  # [B, D, H*W]
    feat2_flat = feat2.flatten(2)  # [B, D, H*W]
    
    # Normalize along feature dimension
    feat1_norm = torch.nn.functional.normalize(feat1_flat, dim=1)
    feat2_norm = torch.nn.functional.normalize(feat2_flat, dim=1)
    
    # Compute cosine similarity for each spatial location
    cos_sim = (feat1_norm * feat2_norm).sum(dim=1)  # [B, H*W]
    
    return {
        "mean": cos_sim.mean().item(),
        "min": cos_sim.min().item(),
        "max": cos_sim.max().item(),
        "std": cos_sim.std().item()
    }

# ============================================
# Load and Preprocess Image
# ============================================
print("="*60)
print("IMAGE PREPROCESSING")
print("="*60)
print(f"\nLoading image from: {image_path}")
img = Image.open(image_path).convert("RGB")
w, h = img.size
print(f"Original image size: {w}x{h}")

# Resize to fixed dimensions for fair comparison
print(f"Resizing to fixed dimensions: {TRT_WIDTH}x{TRT_HEIGHT}")
print("⚠️  Note: Aspect ratio NOT preserved for fair TensorRT comparison")
img_resized = img.resize((TRT_WIDTH, TRT_HEIGHT), Image.Resampling.BILINEAR)

# Transform to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img_resized).unsqueeze(0).cuda()

print(f"Input tensor shape: {img_tensor.shape}")
print(f"Expected output shape: [1, 1024, 48, 64] (48={TRT_HEIGHT//PATCH_SIZE}, 64={TRT_WIDTH//PATCH_SIZE})")

# ============================================
# Load PyTorch FP32 Model
# ============================================
print("\n" + "="*60)
print("LOADING MODELS")
print("="*60)
print("\nLoading PyTorch FP32 model...")
model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=weights_path)
model.cuda().eval()

if hasattr(model, "mask_token"):
    model.mask_token = model.mask_token.to("cuda")

pytorch_model = DinoBackboneDenseONNX(model, IMG_SIZE, PATCH_SIZE).cuda().eval()
print("✓ PyTorch model loaded")

# ============================================
# Load TensorRT Engine
# ============================================
print("\nLoading TensorRT engine...")
with open(engine_path, "rb") as f:
    engine_bytes = f.read()
trt_logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(trt_logger)
engine = runtime.deserialize_cuda_engine(engine_bytes)
trt_model = TRTModule(
    engine=engine,
    input_names=["input_image"],
    output_names=["dense_features"]
)
trt_model.eval()
print("✓ TensorRT engine loaded")

# ============================================
# Single Inference for Accuracy Comparison
# ============================================
print("\n" + "="*60)
print("ACCURACY COMPARISON")
print("="*60)

with torch.no_grad():
    # PyTorch inference
    pytorch_feats = pytorch_model(img_tensor).float()
    
    # TensorRT inference
    trt_feats = trt_model(img_tensor).float()

print(f"\nPyTorch output shape:  {pytorch_feats.shape}")
print(f"TensorRT output shape: {trt_feats.shape}")

if pytorch_feats.shape != trt_feats.shape:
    print("\n❌ ERROR: Output shapes don't match!")
    print("   Cannot perform accuracy comparison.")
    exit(1)

# Compute similarity metrics
similarity_metrics = compute_cosine_similarity(pytorch_feats, trt_feats)

print(f"\n📊 Cosine Similarity Metrics:")
print(f"   Mean:   {similarity_metrics['mean']:.6f}")
print(f"   Std:    {similarity_metrics['std']:.6f}")
print(f"   Min:    {similarity_metrics['min']:.6f}")
print(f"   Max:    {similarity_metrics['max']:.6f}")

# Compute absolute differences
abs_diff = (pytorch_feats - trt_feats).abs()
rel_diff = abs_diff / (pytorch_feats.abs() + 1e-8)

print(f"\n📊 Absolute Difference Statistics:")
print(f"   Mean:   {abs_diff.mean().item():.6f}")
print(f"   Std:    {abs_diff.std().item():.6f}")
print(f"   Max:    {abs_diff.max().item():.6f}")
print(f"   Median: {abs_diff.median().item():.6f}")

print(f"\n📊 Relative Difference (%):")
print(f"   Mean:   {rel_diff.mean().item() * 100:.4f}%")
print(f"   Max:    {rel_diff.max().item() * 100:.4f}%")

# ============================================
# Speed Benchmarking
# ============================================
print("\n" + "="*60)
print("SPEED BENCHMARKING")
print("="*60)

# Warmup
print("\nWarming up models (10 iterations)...")
for _ in range(10):
    with torch.no_grad():
        _ = pytorch_model(img_tensor)
        _ = trt_model(img_tensor)

torch.cuda.synchronize()

# Benchmark PyTorch
print(f"Benchmarking PyTorch ({BENCHMARK_ITERATIONS} iterations)...")
pytorch_times = []
for i in range(BENCHMARK_ITERATIONS):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = pytorch_model(img_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    pytorch_times.append((end - start) * 1000)  # Convert to ms
    
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i + 1}/{BENCHMARK_ITERATIONS}")

pytorch_mean = np.mean(pytorch_times)
pytorch_std = np.std(pytorch_times)
pytorch_min = np.min(pytorch_times)
pytorch_max = np.max(pytorch_times)
pytorch_fps = 1000.0 / pytorch_mean

# Benchmark TensorRT
print(f"\nBenchmarking TensorRT ({BENCHMARK_ITERATIONS} iterations)...")
trt_times = []
for i in range(BENCHMARK_ITERATIONS):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = trt_model(img_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    trt_times.append((end - start) * 1000)  # Convert to ms
    
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i + 1}/{BENCHMARK_ITERATIONS}")

trt_mean = np.mean(trt_times)
trt_std = np.std(trt_times)
trt_min = np.min(trt_times)
trt_max = np.max(trt_times)
trt_fps = 1000.0 / trt_mean

speedup = pytorch_mean / trt_mean

# ============================================
# Results Summary
# ============================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print("\n⚡ Speed Comparison:")
print(f"   TensorRT:")
print(f"      Mean:  {trt_mean:7.3f} ms ± {trt_std:5.3f} ms ({trt_fps:6.2f} FPS)")
print(f"      Range: {trt_min:7.3f} ms - {trt_max:7.3f} ms")
print(f"   PyTorch FP32:")
print(f"      Mean:  {pytorch_mean:7.3f} ms ± {pytorch_std:5.3f} ms ({pytorch_fps:6.2f} FPS)")
print(f"      Range: {pytorch_min:7.3f} ms - {pytorch_max:7.3f} ms")
print(f"   Speedup: {speedup:.2f}x")

print("\n🎯 Accuracy Metrics:")
print(f"   Cosine Similarity:  {similarity_metrics['mean']:.6f} (±{similarity_metrics['std']:.6f})")
print(f"   Mean Abs Diff:      {abs_diff.mean().item():.6f}")
print(f"   Max Abs Diff:       {abs_diff.max().item():.6f}")
print(f"   Mean Rel Diff:      {rel_diff.mean().item() * 100:.4f}%")

# Determine quality assessment
if similarity_metrics['mean'] > 0.99:
    quality = "Excellent ✓"
elif similarity_metrics['mean'] > 0.95:
    quality = "Good ✓"
elif similarity_metrics['mean'] > 0.90:
    quality = "Acceptable ~"
else:
    quality = "Poor ✗"

print(f"\n   Overall Quality: {quality}")

# ============================================
# Combined Visualization
# ============================================
print("\n" + "="*60)
print("GENERATING COMBINED VISUALIZATION")
print("="*60)

# Prepare feature visualizations
trt_feats_viz = trt_feats.squeeze(0).permute(1, 2, 0).contiguous()
pytorch_feats_viz = pytorch_feats.squeeze(0).permute(1, 2, 0).contiguous()
diff_viz = abs_diff.squeeze(0).permute(1, 2, 0).contiguous()

# Process visualizations
def process_features(features, method="naive_rgb"):
    """Process features for visualization"""
    if method == "naive_rgb":
        vis_features_rgb = features[..., :3].cpu().numpy()
        for i in range(3):
            channel = vis_features_rgb[..., i]
            min_val, max_val = channel.min(), channel.max()
            vis_features_rgb[..., i] = (channel - min_val) / (max_val - min_val + 1e-8)
        return vis_features_rgb
    elif method == "mean":
        vis_features = features.mean(axis=-1).cpu().numpy()
        return vis_features
    return None

pytorch_vis = process_features(pytorch_feats_viz, "naive_rgb")
trt_vis = process_features(trt_feats_viz, "naive_rgb")
diff_vis = process_features(diff_viz, "mean")

# Create combined figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Original resized image
axes[0, 0].imshow(img_resized)
axes[0, 0].set_title(f'Original Image (Resized to {TRT_WIDTH}x{TRT_HEIGHT})', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Plot 2: PyTorch features
axes[0, 1].imshow(pytorch_vis)
axes[0, 1].set_title(f'PyTorch FP32 Features\n(CosSim: {similarity_metrics["mean"]:.4f})', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Plot 3: TensorRT features
axes[1, 0].imshow(trt_vis)
axes[1, 0].set_title(f'TensorRT BF16 Features\n({speedup:.2f}x faster)', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Plot 4: Difference heatmap
im = axes[1, 1].imshow(diff_vis, cmap='viridis')
axes[1, 1].set_title(f'Absolute Difference\n(Mean: {abs_diff.mean().item():.4f}, Max: {abs_diff.max().item():.4f})', 
                      fontsize=12, fontweight='bold')
axes[1, 1].axis('off')
plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
save_path = "/home/user/km-vipe/combined_visualization.png"
plt.savefig(save_path, bbox_inches='tight', dpi=150)
print(f"\n✓ Combined visualization saved to: {save_path}")
plt.close()

print("\n" + "="*60)
print("✅ BENCHMARKING COMPLETE")
print("="*60)
print(f"\nGenerated file:")
print(f"  • {save_path}")