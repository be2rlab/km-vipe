import torch
from torchvision import transforms
from torch2trt import TRTModule
import tensorrt as trt
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Union
from torch import Tensor
import matplotlib.pyplot as plt
from guided_filter_pytorch.guided_filter import FastGuidedFilter
import torchvision.transforms.functional as TF
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF


# --- CONFIGURATION ---
image_path = "/home/user/km-vipe/weights/frame000019.jpg"
engine_path = "/home/user/km-vipe/weights/dinov3_vitl16_bf16_768.engine"

REPO_DIR = "/home/user/km-vipe/weights/dinov3"
weights_path = "/home/user/km-vipe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=weights_path)
model.cuda().eval()

class ResizeTransform(nn.Module):
    """Resize image to a fixed size."""

    def __init__(self, image_size: int = 768, patch_size: int = 16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

    def forward(self, img):
        w, h = img.size
        h_patches = self.image_size // self.patch_size
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        return TVTF.resize(img, (h_patches * self.patch_size, w_patches * self.patch_size))

# Preprocessing
preprocess = TVT.Compose(
            [
                ResizeTransform(image_size=768, patch_size=16),
                TVT.ToTensor(),
                TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

# Load image and input tensor
image = Image.open(image_path).convert("RGB")
img_tensor = preprocess(image).unsqueeze(0).cuda()  # [1, 3, H, W]
print(img_tensor.shape)
H, W = img_tensor.shape[2], img_tensor.shape[3]

# print("Loading TensorRT engine...")
# with open(engine_path, "rb") as f:
#     engine_bytes = f.read()
# trt_logger = trt.Logger()
# runtime = trt.Runtime(trt_logger)
# engine = runtime.deserialize_cuda_engine(engine_bytes)
# trt_model = TRTModule(
#     engine=engine,
#     input_names=["input_image"],
#     output_names=["dense_features"]
# )
# trt_model.eval()

# with torch.no_grad():
#     trt_feats = trt_model(img_tensor).float()

# print(trt_feats.shape)
# trt_feats = trt_feats.squeeze(0).permute(1, 2, 0).contiguous()
# print(trt_feats.shape) # [D, Hp, Wp] # [1024, 48, 48]

amp_dtype = torch.bfloat16
with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
    x = img_tensor
    target_layer = (24 + -1) % 24

    feats_list = model.get_intermediate_layers(x, n=[target_layer], reshape=True, norm=True)
    feats = feats_list[0][0]  # [D, h, w]
    feats = feats.permute(1, 2, 0).contiguous() 

print(feats.shape)

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

visualize_embeddings_plt(feats, method="naive_rgb", save_path = "/home/user/km-vipe/featmap.png")

batched_feats = feats.permute(2, 0, 1).unsqueeze(0).contiguous()
print(f"batched_feats: {batched_feats.shape}")

img_tensor_b = img_tensor  # [1,3,768,768]

# def bilateral_grid_upsample(feats, img, grid_size=16):
#     """
#     More sophisticated than simple guided filter
#     """
#     B, C, H, W = feats.shape
#     target_H, target_W = img.shape[-2:]
    
#     # Create bilateral grid based on image
#     # This is complex, but the idea is:
#     # - Bin features by spatial location AND color similarity
#     # - Interpolate in this higher-dimensional space
    
#     # Simplified version: bicubic + edge-aware blending
#     feats_bicubic = F.interpolate(feats, size=(target_H, target_W), 
#                                   mode='bicubic', align_corners=False)
#     feats_nearest = F.interpolate(feats, size=(target_H, target_W), 
#                                   mode='nearest')
    
#     # Edge map from image
#     img_gray = TF.rgb_to_grayscale(img)
#     edges = torch.abs(
#         F.conv2d(img_gray, 
#                  torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
#                              device=img.device).float().view(1, 1, 3, 3),
#                  padding=1)
#     )
#     edges = torch.sigmoid(edges * 10)  # Soft edge mask
    
#     # Blend: nearest at edges (preserve boundaries), bicubic elsewhere
#     refined = edges * feats_nearest + (1 - edges) * feats_bicubic
    
#     return refined

# refined = bilateral_grid_upsample(batched_feats, img_tensor)

print(f"img_tensor_b: {img_tensor_b.shape}")
hr_gray = TF.rgb_to_grayscale(img_tensor_b)  # [1,1,768,768]
lr_gray = F.interpolate(hr_gray, size=(H//16, W//16), mode='bilinear', align_corners=False)
print(f"lr_gray: {lr_gray.shape}")

gf = FastGuidedFilter(r=2, eps=1e-4)
refined_channels = []
for i in range(batched_feats.shape[1]):
    feat_channel = batched_feats[:, i:i+1, :, :]  # [1, 1, 48, 48]

    filtered = gf(lr_gray, feat_channel, hr_gray)
    refined_channels.append(filtered)

refined = torch.cat(refined_channels, dim=1)
print(refined.shape)  # [1,1024,768,768]

visualize_embeddings_plt(refined.squeeze(0).permute(1, 2, 0), method="naive_rgb", save_path = "/home/user/km-vipe/featmap_up.png")

refined_down = F.interpolate(refined, size=(H//16, W//16), mode='bilinear')

# Compute cosine similarity per spatial location
cos_sim = F.cosine_similarity(
    batched_feats.flatten(2),  # [1, 1024, 2304]
    refined_down.flatten(2),
    dim=1
)
print(f"Mean cosine similarity: {cos_sim.mean().item():.4f}")
print(f"Min cosine similarity: {cos_sim.min().item():.4f}")

def visualize_features(feats_before, feats_after, num_channels=8):
    fig, axes = plt.subplots(2, num_channels, figsize=(20, 5))
    
    for i in range(num_channels):
        # Before upsampling
        axes[0, i].imshow(feats_before[0, i].cpu(), cmap='viridis')
        axes[0, i].set_title(f'Ch {i} Before')
        axes[0, i].axis('off')
        
        # After upsampling
        axes[1, i].imshow(feats_after[0, i].cpu(), cmap='viridis')
        axes[1, i].set_title(f'Ch {i} After')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_comparison.png', dpi=150)
    plt.close()

# Compare
up_bilinear = F.interpolate(batched_feats, size=(768, 1024), mode='bilinear')
visualize_features(up_bilinear, refined, num_channels=8)

from sklearn.decomposition import PCA

def pca_visualization(feats_48, feats_768):
    # Flatten spatial dimensions
    f1 = feats_48[0].permute(1, 2, 0).reshape(-1, 1024).cpu().numpy()  # [2304, 1024]
    f2 = feats_768[0].permute(1, 2, 0).reshape(-1, 1024).cpu().numpy()  # [589824, 1024]
    
    # PCA to 3 components
    pca = PCA(n_components=3)
    pca.fit(f1)
    
    rgb1 = pca.transform(f1).reshape(H//16, W//16, 3)
    rgb2 = pca.transform(f2).reshape(H, W, 3)
    
    # Normalize to [0, 1]
    rgb1 = (rgb1 - rgb1.min()) / (rgb1.max() - rgb1.min())
    rgb2 = (rgb2 - rgb2.min()) / (rgb2.max() - rgb2.min())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(rgb1)
    ax1.set_title('Original Features (PCA)')
    ax2.imshow(rgb2)
    ax2.set_title('Refined Features (PCA)')
    plt.savefig('pca_comparison.png')
    plt.close()

pca_visualization(batched_feats, refined)

def comprehensive_check(feats_orig, feats_refined):
    # Downsample refined back
    feats_down = F.interpolate(feats_refined, size=feats_orig.shape[-2:], 
                               mode='bilinear')
    
    print("="*50)
    cos_sim = F.cosine_similarity(
        feats_orig.flatten(2),
        feats_down.flatten(2),
        dim=1
    ).mean()
    print(f"Cosine Similarity: {cos_sim.item():.4f}")
    
    l2_dist = (feats_orig - feats_down).pow(2).sum(dim=1).sqrt().mean()
    print(f"L2 Distance: {l2_dist.item():.4f}")
    
    def spatial_correlation(f):
        # Correlation between adjacent patches
        f_flat = f.flatten(2)  # [B, C, H*W]
        corr_h = F.cosine_similarity(f_flat[:, :, :-1], f_flat[:, :, 1:], dim=1)
        return corr_h.mean()
    
    orig_corr = spatial_correlation(feats_orig)
    refined_corr = spatial_correlation(feats_down)
    print(f"Spatial Correlation (orig): {orig_corr.item():.4f}")
    print(f"Spatial Correlation (refined): {refined_corr.item():.4f}")
    print(f"Spatial Correlation Loss: {abs(orig_corr - refined_corr).item():.4f}")
    
    print("="*50)

comprehensive_check(batched_feats, refined)