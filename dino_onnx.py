import torch
import torch.nn as nn

REPO_DIR = "/home/user/km-vipe/weights/dinov3"
weights_path = "/home/user/km-vipe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
onnx_path = "/home/user/km-vipe/weights/dinov3_vitl16_backbone_dense_448.onnx"
IMG_SIZE = 448

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

class DinoBackboneDenseONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats_dict = self.model.forward_features(x)

        # pick dense patch embeddings
        feats = feats_dict["x_norm_patchtokens"]  # [B, N, D]

        # reshape [B, N, D] -> [B, D, H, W]
        B, N, D = feats.shape
        h = w = int(N**0.5) 
        patches = feats.permute(0, 2, 1).reshape(B, D, h, w)
        return patches

dense_model = DinoBackboneDenseONNX(model).cuda().eval()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device="cuda")

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
