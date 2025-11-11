import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from torch import Tensor
from typing import Optional
import torchvision.transforms as T
import torchvision.transforms.functional as TVTF

norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnorm = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                      std=[1/0.229, 1/0.224, 1/0.225])

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
        print(f"Resizing {w}x{h} -> {new_w}x{new_h} (patches: {w_patches}x{h_patches})")
        return TVTF.resize(img, (new_h, new_w))

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
    f1 = feats_patch[0].permute(1, 2, 0).reshape(-1, feats_patch.shape[1]).cpu().numpy()
    f2 = feats_pixel[0].permute(1, 2, 0).reshape(-1, feats_pixel.shape[1]).cpu().numpy()
    
    H_patch, W_patch = feats_patch.shape[2], feats_patch.shape[3]
    H_pixel, W_pixel = feats_pixel.shape[2], feats_pixel.shape[3]
    
    pca = PCA(n_components=3)
    pca.fit(f1)
    
    rgb1 = pca.transform(f1).reshape(H_patch, W_patch, 3)
    rgb2 = pca.transform(f2).reshape(H_pixel, W_pixel, 3)
    
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
patch_size = 14
image_size = 680 

print("Loading FeatUp upsampler...")
use_norm = False 
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
upsampler.eval()

preprocess = T.Compose([
    ResizeTransform(image_size=image_size, patch_size=patch_size),
    T.ToTensor(),
    norm
])

image_tensor = preprocess(image).unsqueeze(0).to(device)
_, _, H_processed, W_processed = image_tensor.shape
print(f"Processed image shape: {image_tensor.shape}")

with torch.no_grad():
    hr_feats = upsampler(image_tensor)  # [1, 384, H_processed, W_processed]
    lr_feats = upsampler.model(image_tensor)  # [1, 384, H_patch, W_patch]

print(f"Low-res features shape: {lr_feats.shape}")
print(f"High-res features shape: {hr_feats.shape}")

visualize_embeddings_plt(lr_feats.squeeze(0).permute(1,2,0), method="naive_rgb", save_path = "/home/user/km-vipe/featmap.png")
visualize_embeddings_plt(hr_feats.squeeze(0).permute(1,2,0), method="naive_rgb", save_path = "/home/user/km-vipe/featmap_up.png")

trt_feats = hr_feats[0].permute(1, 2, 0)  # [H, W, 384]

print(f"\nGenerating visulizations...")
visualize_embeddings_plt(trt_feats, method="naive_rgb", 
                         save_path="/home/user/km-vipe/featmap_upsampled.png")

pca_visualization(lr_feats, hr_feats, save_path='/home/user/km-vipe/pca_comparison.png')

print("\nDone! Generated files:")
print(f"  - /home/user/km-vipe/featmap_upsampled.png (naive RGB, {W_processed}x{H_processed})")
print(f"  - /home/user/km-vipe/pca_comparison.png (PCA comparison)")