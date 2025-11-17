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

image_path = "/home/user/km-vipe/weights/frame000019.jpg"
engine_path = "/home/user/km-vipe/weights/dinov3_vitl16_bf16_768.engine"
REPO_DIR = "/home/user/km-vipe/weights/dinov3"
weights_path = "/home/user/km-vipe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
IMG_SIZE = 768

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