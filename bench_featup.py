import torch
import torch.nn as nn
from PIL import Image
import time
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TVTF
from contextlib import contextmanager

# ==================== GPU Memory Tracking ====================
@contextmanager
def track_memory(description):
    """Context manager to track GPU memory usage"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024**3  # GB
    
    yield
    
    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated() / 1024**3
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\n{description}")
    print(f"  Start memory: {start_mem:.3f} GB")
    print(f"  End memory:   {end_mem:.3f} GB")
    print(f"  Peak memory:  {peak_mem:.3f} GB")
    print(f"  Delta:        {end_mem - start_mem:.3f} GB")

# ==================== Timing Utilities ====================
def benchmark_function(func, *args, num_warmup=3, num_runs=10, **kwargs):
    """Benchmark a function with warmup runs"""
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = func(*args, **kwargs)
        torch.cuda.synchronize()
    
    # Actual benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            result = func(*args, **kwargs)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    return {
        'mean': times.mean(),
        'std': times.std(),
        'min': times.min(),
        'max': times.max(),
        'median': np.median(times),
        'result': result
    }

# ==================== ResizeTransform ====================
class ResizeTransform(nn.Module):
    def __init__(self, image_size: int = 512, patch_size: int = 14):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
    
    def forward(self, img):
        w, h = img.size
        h_patches = self.image_size // self.patch_size
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        new_h = h_patches * self.patch_size
        new_w = w_patches * self.patch_size
        return TVTF.resize(img, (new_h, new_w))

# ==================== Main Benchmark ====================
def run_benchmark():
    print("="*80)
    print("FeatUp + DINOv2 Benchmark")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load image
    img_path = '/home/user/km-vipe/weights/frame000019.jpg'
    image = Image.open(img_path).convert("RGB")
    original_H, original_W = image.height, image.width
    print(f"\nOriginal image size: {original_W}x{original_H}")
    
    # Test different resolutions
    test_sizes = [224, 336, 448, 512, 672]
    
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    # Load FeatUp
    with track_memory("FeatUp Model Loading"):
        start = time.time()
        use_norm = False
        upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
        upsampler.eval()
        load_time = time.time() - start
        print(f"  Loading time: {load_time:.2f}s")
    
    results = []
    
    for image_size in test_sizes:
        print("\n" + "="*80)
        print(f"BENCHMARK: image_size = {image_size}")
        print("="*80)
        
        # Preprocess
        preprocess = T.Compose([
            ResizeTransform(image_size=image_size, patch_size=14),
            T.ToTensor(),
            norm
        ])
        
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        _, _, H, W = image_tensor.shape
        print(f"Processed shape: {image_tensor.shape}")
        print(f"Number of pixels: {H * W:,}")
        
        # Benchmark DINOv2 Backbone
        print("\n--- DINOv2 Backbone ---")
        with track_memory(f"DINOv2 backbone (size={image_size})"):
            backbone_stats = benchmark_function(
                upsampler.model,
                image_tensor,
                num_warmup=3,
                num_runs=10
            )
            lr_feats = backbone_stats['result']
        
        print(f"  Output shape: {lr_feats.shape}")
        print(f"  Mean time: {backbone_stats['mean']*1000:.2f} ± {backbone_stats['std']*1000:.2f} ms")
        print(f"  Min time:  {backbone_stats['min']*1000:.2f} ms")
        print(f"  Max time:  {backbone_stats['max']*1000:.2f} ms")
        
        # Benchmark FeatUp Upsampling
        print("\n--- FeatUp Upsampling ---")
        with track_memory(f"FeatUp upsampling (size={image_size})"):
            featup_stats = benchmark_function(
                upsampler,
                image_tensor,
                num_warmup=3,
                num_runs=10
            )
            hr_feats = featup_stats['result']
        
        print(f"  Output shape: {hr_feats.shape}")
        print(f"  Mean time: {featup_stats['mean']*1000:.2f} ± {featup_stats['std']*1000:.2f} ms")
        print(f"  Min time:  {featup_stats['min']*1000:.2f} ms")
        print(f"  Max time:  {featup_stats['max']*1000:.2f} ms")
        
        # Total pipeline
        total_time = backbone_stats['mean'] + featup_stats['mean']
        fps = 1.0 / total_time
        
        print(f"\n--- Total Pipeline ---")
        print(f"  Total time: {total_time*1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        
        results.append({
            'image_size': image_size,
            'processed_shape': (H, W),
            'lr_shape': lr_feats.shape,
            'hr_shape': hr_feats.shape,
            'backbone_time_ms': backbone_stats['mean'] * 1000,
            'upsampling_time_ms': featup_stats['mean'] * 1000,
            'total_time_ms': total_time * 1000,
            'fps': fps
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Size':<8} {'Processed':<15} {'LR Shape':<20} {'HR Shape':<20} {'Backbone':<12} {'Upsampling':<12} {'Total':<12} {'FPS':<8}")
    print("-" * 120)
    
    for r in results:
        print(f"{r['image_size']:<8} "
              f"{r['processed_shape'][1]}x{r['processed_shape'][0]:<11} "
              f"{str(r['lr_shape']):<20} "
              f"{str(r['hr_shape']):<20} "
              f"{r['backbone_time_ms']:<12.2f} "
              f"{r['upsampling_time_ms']:<12.2f} "
              f"{r['total_time_ms']:<12.2f} "
              f"{r['fps']:<8.2f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_benchmark()