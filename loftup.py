import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from torch import Tensor
from typing import Optional
import torchvision.transforms as T
import torchvision.transforms.functional as TVTF

class ResizeTransform(nn.Module):
    """Resize image to maintain aspect ratio with short side = image_size,
    and ensure dimensions are multiples of patch_size."""
    
    def __init__(self, image_size: int = 512, patch_size: int = 14):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
    
    def forward(self, img):
        w, h = img.size
        # Calculate patches for height and width
        h_patches = self.image_size // self.patch_size
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        
        new_h = h_patches * self.patch_size
        new_w = w_patches * self.patch_size
        
        print(f"Resizing {w}x{h} -> {new_w}x{new_h} (patches: {w_patches}x{h_patches})")
        return TVTF.resize(img, (new_h, new_w))

norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnorm = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                      std=[1/0.229, 1/0.224, 1/0.225])

def visualize_embeddings_plt(
    features: Tensor,
    method = "naive_rgb",
    save_path: Optional[str] = None, 
):
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
        vis_features_rgb = features[..., :3].cpu().numpy()
        for i in range(3):
            channel = vis_features_rgb[..., i]
            min_val, max_val = channel.min(), channel.max()
            vis_features_rgb[..., i] = (channel - min_val) / (max_val - min_val + 1e-8)
        vis_features_np = vis_features_rgb
    else:
        raise ValueError(f"Unknown visualization method: {method}")

    if vis_features_np is not None:
        plt.figure(figsize=(12, 8))
        cmap = 'viridis'
        if method == "naive_rgb":
            plt.imshow(vis_features_np)
        else:
            plt.imshow(vis_features_np, cmap=cmap)
            plt.colorbar()
        plt.title(f"Feature Visualization: {method}")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to: {save_path}")
            plt.close()
        return vis_features_np
    return None

def pca_visualization(feats_patch, feats_pixel, save_path='/home/user/km-vipe/pca_comparison.png'):
    # Flatten spatial dimensions
    f1 = feats_patch[0].permute(1, 2, 0).reshape(-1, feats_patch.shape[1]).cpu().numpy()
    f2 = feats_pixel[0].permute(1, 2, 0).reshape(-1, feats_pixel.shape[1]).cpu().numpy()
    
    H_patch, W_patch = feats_patch.shape[2], feats_patch.shape[3]
    H_pixel, W_pixel = feats_pixel.shape[2], feats_pixel.shape[3]
    
    # PCA to 3 components
    pca = PCA(n_components=3)
    pca.fit(f1)
    
    rgb1 = pca.transform(f1).reshape(H_patch, W_patch, 3)
    rgb2 = pca.transform(f2).reshape(H_pixel, W_pixel, 3)
    
    # Normalize to [0, 1]
    rgb1 = (rgb1 - rgb1.min()) / (rgb1.max() - rgb1.min())
    rgb2 = (rgb2 - rgb2.min()) / (rgb2.max() - rgb2.min())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.imshow(rgb1)
    ax1.set_title(f'Patch Features (PCA) [{H_patch}x{W_patch}]', fontsize=14)
    ax1.axis('off')
    ax2.imshow(rgb2)
    ax2.set_title(f'Upsampled Features (PCA) [{H_pixel}x{W_pixel}]', fontsize=14)
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PCA comparison saved to: {save_path}")
    plt.close()

img_path = '/home/user/km-vipe/weights/frame000019.jpg'
image = Image.open(img_path).convert("RGB")
original_H, original_W = image.height, image.width
print(f"Original image size: {original_W}x{original_H}")

# 设置参数
patch_size = 14
image_size = 224 # short side

# 加载模型
model = AutoModel.from_pretrained('facebook/dinov2-small').cuda()
model.eval()

upsampler = torch.hub.load('andrehuang/loftup', "loftup_dinov2s", pretrained=True).cuda()
upsampler.eval()

preprocess = T.Compose([
    ResizeTransform(image_size=image_size, patch_size=patch_size),
    T.ToTensor(),
    norm
])

normalized_img = preprocess(image).unsqueeze(0).cuda()
_, _, H_processed, W_processed = normalized_img.shape
print(f"Processed image shape: {normalized_img.shape}")

with torch.no_grad():
    outputs = model(pixel_values=normalized_img)
    patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [1, num_patches, 384]
    
B, num_patches, D = patch_tokens.shape
h_patches = H_processed // patch_size
w_patches = W_processed // patch_size
print(f"Patch tokens: {num_patches} = {h_patches}x{w_patches}, dim={D}")

# Reshape to spatial format: [1, 384, h_patches, w_patches]
lr_feats = patch_tokens.reshape(1, h_patches, w_patches, D).permute(0, 3, 1, 2)
print(f"Low-res features shape: {lr_feats.shape}")

transform_original = T.Compose([
    T.ToTensor(),
    norm
])

normalized_img_original = transform_original(image).unsqueeze(0).cuda()
print(f"Original guidance image shape: {normalized_img_original.shape}")

# ==================== 3. 使用loftup上采样到原始尺寸 ====================
with torch.no_grad():
    hr_feats = upsampler(lr_feats, normalized_img_original)  # [1, 384, original_H, original_W]

print(f"High-res features shape: {hr_feats.shape}")

# ==================== 4. 可视化 ====================
# 转换为 [H, W, D] 格式
trt_feats = hr_feats[0].permute(1, 2, 0)  # [original_H, original_W, 384]

print(f"\n正在生成可视化...")
visualize_embeddings_plt(trt_feats, method="naive_rgb", 
                         save_path="/home/user/km-vipe/featmap_upsampled.png")

pca_visualization(lr_feats, hr_feats, save_path='/home/user/km-vipe/pca_comparison.png')

print("\n✅ Done! Generated files:")
print(f"  - /home/user/km-vipe/featmap_upsampled.png (naive RGB, {original_W}x{original_H})")
print(f"  - /home/user/km-vipe/pca_comparison.png (PCA: {w_patches}x{h_patches} vs {original_W}x{original_H})")