import torch
import torch.nn as nn
import torchvision.transforms.functional as TVTF

REPO_DIR = "/home/user/km-vipe/weights/dinov3"
weights_path = "/home/user/km-vipe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
onnx_path = "/home/user/km-vipe/weights/dinov3_vitl16_backbone_dense_768.onnx"
IMG_SIZE = 768
PATCH_SIZE = 16

print("Loading DINOv3 backbone...")
model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=weights_path)
model.cuda().eval()

# move mask_token to CUDA if needed
if hasattr(model, "mask_token"):
    model.mask_token = model.mask_token.to("cuda")

STABLE_EPS = 1e-5 
print(f"Setting all LayerNorm eps values to {STABLE_EPS} for FP16 stability...")

# Change all LayerNorms in the blocks
for block in model.blocks:
    if hasattr(block, 'norm1'):
        block.norm1.eps = STABLE_EPS
    if hasattr(block, 'norm2'):
        block.norm2.eps = STABLE_EPS

# Change the final LayerNorm
if hasattr(model, 'norm'):
    model.norm.eps = STABLE_EPS

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

dense_model = DinoBackboneDenseONNX(model, IMG_SIZE, PATCH_SIZE).cuda().eval()

# Test with different aspect ratios
test_shapes = [
    (1, 3, 768, 768),   # Square
    (1, 3, 768, 1024),  # 4:3
    (1, 3, 480, 640),   # Different size
]

for shape in test_shapes:
    dummy_input = torch.randn(*shape, device="cuda")
    with torch.no_grad():
        output = dense_model(dummy_input)
        print(f"Input: {dummy_input.shape} -> Output: {output.shape}")

# Export with a standard size
dummy_input = torch.randn(1, 3, 768, 1024, device="cuda")

torch.onnx.export(
    dense_model,
    (dummy_input,),
    onnx_path,
    input_names=["input_image"],
    output_names=["dense_features"],
    opset_version=17,
    do_constant_folding=True,
)

print("✅ ONNX export complete:", onnx_path)