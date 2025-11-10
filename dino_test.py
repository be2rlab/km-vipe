import torch
from torchvision import transforms
from torch2trt import TRTModule
import tensorrt as trt
import torch.nn as nn
from PIL import Image
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Union
from torch import Tensor
import matplotlib.pyplot as plt


# --- CONFIGURATION ---
image_path = "/home/user/km-vipe/weights/frame000000.jpg"
engine_path = "/home/user/km-vipe/weights/dinov3_vitl16_backbone_dense_bf16_768.engine"
REPO_DIR = "/home/user/km-vipe/weights/dinov3"
weights_path = "/home/user/km-vipe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
IMG_SIZE = 768
BENCHMARK_ITERATIONS = 100
# ---------------------


# --- 1. MODEL DEFINITION ---

class DinoBackboneDenseONNX(nn.Module):
    """Wrapper class to extract dense patch features for ONNX export/comparison."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats_dict = self.model.forward_features(x)
        feats = feats_dict["x_norm_patchtokens"]  # [B, N, D]

        # reshape [B, N, D] -> [B, D, H, W]
        B, N, D = feats.shape
        h = w = int(N**0.5) 
        patches = feats.permute(0, 2, 1).reshape(B, D, h, w)
        return patches


# --- 2. SETUP: MODEL & INPUTS ---

# Preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# Load image and input tensor
image = Image.open(image_path).convert("RGB")
img_tensor = preprocess(image).unsqueeze(0).cuda()  # [1, 3, H, W]


# Load PyTorch Model
# print("--- Setup ---")
# print("Loading DINOv3 backbone (PyTorch)...")
# model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=weights_path)
# model.cuda().eval()
# if hasattr(model, "mask_token"):
#     model.mask_token = model.mask_token.to("cuda")
# dense_model = DinoBackboneDenseONNX(model).cuda().eval()

# Load TensorRT Engine
print("Loading TensorRT engine...")
with open(engine_path, "rb") as f:
    engine_bytes = f.read()
trt_logger = trt.Logger()
runtime = trt.Runtime(trt_logger)
engine = runtime.deserialize_cuda_engine(engine_bytes)
trt_model = TRTModule(
    engine=engine,
    input_names=["input_image"],
    output_names=["dense_features"]
)
trt_model.eval()

with torch.no_grad():
    trt_feats = trt_model(img_tensor).float()

print(trt_feats.shape)
trt_feats = trt_feats.squeeze(0).permute(1, 2, 0).contiguous()
print(trt_feats.shape)

def visualize_embeddings_plt(
    features: Tensor,
    method = "naive_rgb",
    save_path: Optional[str] = None, 
):
    """
    Visualize DINOv3 feature embeddings using simple channel stats with Matplotlib.

    Args:
        features: Feature tensor of shape [h, w, D]
        method: Reduction method ('mean', 'std', 'norm', 'naive_rgb')
        save_path: Optional path to save the resulting image file (e.g., 'viz.png').

    Returns:
        A NumPy array representing the visualization (H x W for single-channel,
        H x W x 3 for 'naive_rgb').
    """
    vis_features_np = None

    if method == "mean":
        vis_features = features.mean(axis=-1)
        vis_features_np = vis_features.cpu().numpy()
    elif method == "std":
        vis_features = features.std(axis=-1)
        vis_features_np = vis_features.cpu().numpy()
    elif method == "norm":
        vis_features = torch.linalg.norm(features, dim=-1)
        vis_features_np = vis_features.cpu().numpy()
    elif method == "naive_rgb":
        if features.shape[-1] < 3:
            print("Warning: 'naive_rgb' requires >= 3 feature dimensions. Returning None.")
            return None

        # Take first 3 channels and normalize them *independently*
        vis_features_rgb = features[..., :3].cpu().numpy()
        for i in range(3):
            channel = vis_features_rgb[..., i]
            min_val, max_val = channel.min(), channel.max()
            # Normalize to [0, 1]
            vis_features_rgb[..., i] = (channel - min_val) / (max_val - min_val + 1e-8)

        vis_features_np = vis_features_rgb
    else:
        raise ValueError(f"Unknown visualization method: {method}")

    # --- Matplotlib Plotting and Saving ---
    if vis_features_np is not None:
        plt.figure()
        
        # Determine the colormap based on the visualization type
        cmap = 'viridis' # Default for single-channel data
        if method == "naive_rgb":
            # For 'naive_rgb', the data is already in 3 channels [0, 1]
            plt.imshow(vis_features_np)
        else:
            # For 'mean', 'std', 'norm', use a colormap and treat as a single channel image
            plt.imshow(vis_features_np, cmap=cmap)
            plt.colorbar() # Add a colorbar for quantitative methods
        
        plt.title(f"Feature Visualization: {method}")
        plt.axis('off') # Hide axes for cleaner image visualization

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
            plt.close() # Close the figure to free memory
        else:
            # If no save path is provided, show the plot (optional)
            # plt.show()
            pass
            
        return vis_features_np
    
    return None

visualize_embeddings_plt(trt_feats, save_path = "/home/user/km-vipe/featmap.png")



# # --- 3. ACCURACY CHECK (MAE) ---

# # Get PyTorch reference output (using FP32 for max fidelity)
# print("\n--- Accuracy Check (MAE) ---")
# with torch.no_grad():
#     torch_feats_fp32 = dense_model(img_tensor).float()

# # Get TensorRT BF16 output
# with torch.no_grad():
#     trt_feats_bf16 = trt_model(img_tensor).float()

# # Calculate MAE
# mae = torch.mean(torch.abs(torch_feats_fp32 - trt_feats_bf16)).item()

# print(f"PyTorch Output Shape: {torch_feats_fp32.shape}")
# print(f"TensorRT Output Shape: {trt_feats_bf16.shape}")
# print(f"Mean Absolute Error (MAE) between PyTorch (FP32) and TensorRT (BF16): {mae:.6e}")


# # --- 4. BENCHMARKING ---

# print(f"\n--- Benchmarking ({BENCHMARK_ITERATIONS} Iterations) ---")

# # Run PyTorch Benchmark (using BF16 for fair comparison)
# print("Benchmarking PyTorch (BF16)...")
# torch_times = []
# with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
#     # Warmup
#     for _ in range(10):
#         dense_model(img_tensor)
    
#     # Measure
#     for _ in range(BENCHMARK_ITERATIONS):
#         start = time.time()
#         dense_model(img_tensor)
#         torch.cuda.synchronize()
#         end = time.time()
#         torch_times.append(end - start)

# torch_avg_time = np.mean(torch_times) * 1000
# torch_fps = 1 / (torch_avg_time / 1000)


# # Run TensorRT Benchmark
# print("Benchmarking TensorRT (BF16)...")
# trt_times = []
# with torch.no_grad():
#     # Warmup
#     for _ in range(10):
#         trt_model(img_tensor)
    
#     # Measure
#     for _ in range(BENCHMARK_ITERATIONS):
#         start = time.time()
#         trt_model(img_tensor)
#         torch.cuda.synchronize()
#         end = time.time()
#         trt_times.append(end - start)

# trt_avg_time = np.mean(trt_times) * 1000
# trt_fps = 1 / (trt_avg_time / 1000)


# # --- 5. LOG RESULTS ---

# print("\n--- Benchmark Results ---")
# print(f"| Model    | Precision | Avg Latency (ms) | FPS |")
# print(f"| PyTorch  | BF16      | {torch_avg_time:.3f} | {torch_fps:.1f} |")
# print(f"| TensorRT | BF16      | {trt_avg_time:.3f} | {trt_fps:.1f} |")
# print(f"MAE (PyTorch FP32 vs TRT BF16): {mae:.6e}")