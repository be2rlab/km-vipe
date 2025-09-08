#############
# DOWNLOAD LINKS
# backbone: https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoieXRiOTI4d3VwNWt2NXZndmtuNm5wYThwIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTc1Mzc5NTJ9fX1dfQ__&Signature=cJkGICbE3q-8UnO8-U1GEVcaxHiCg1p6P5ySWzDBojahN19QOHLXkuH1S2mDto7Cw-sHbf3d2GGQhG4B9BZ-3gKbFBbwTzvYwoDZgdyX0xArDpnwXCUs6BCHwXf1yCdPvYK9R7xlDLzIukL43Bujl8JWJwRC3qvM3w5QjDJ0M49FyD8MmSD5YQh81SRqvzGRePLjvlefaHq-W%7El7pvG-DtxMAKE2pEIVpZLY1LxgCS3cbES202DOYerM0twyOFjvdwuDYeZ7jhiz3sovmFCHLzj%7ELpijyhF4tDwyLh7wouOIqP0RZJjmZi41HWLFogRfYIi%7EBDSQw1hljHeyKHD1Hg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2300273540405481
# image+text head: https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoieXRiOTI4d3VwNWt2NXZndmtuNm5wYThwIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTc1Mzc5NTJ9fX1dfQ__&Signature=cJkGICbE3q-8UnO8-U1GEVcaxHiCg1p6P5ySWzDBojahN19QOHLXkuH1S2mDto7Cw-sHbf3d2GGQhG4B9BZ-3gKbFBbwTzvYwoDZgdyX0xArDpnwXCUs6BCHwXf1yCdPvYK9R7xlDLzIukL43Bujl8JWJwRC3qvM3w5QjDJ0M49FyD8MmSD5YQh81SRqvzGRePLjvlefaHq-W%7El7pvG-DtxMAKE2pEIVpZLY1LxgCS3cbES202DOYerM0twyOFjvdwuDYeZ7jhiz3sovmFCHLzj%7ELpijyhF4tDwyLh7wouOIqP0RZJjmZi41HWLFogRfYIi%7EBDSQw1hljHeyKHD1Hg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2300273540405481
#############

import torch
from torchvision import transforms
from PIL import Image
import requests
from skimage import data

device = "cuda" if torch.cuda.is_available() else "cpu"

REPO_DIR = "dinov3/"

backbone_weights = f"./weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
weights = f"./weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"

# DINOv3
dino_backbone, tokenizer = torch.hub.load(REPO_DIR, 'dinov3_vitl16_dinotxt_tet1280d20h24l', source='local', weights=weights, backbone_weights=backbone_weights)

# ---- Example dataset ----
images = {
    "astronaut": data.astronaut(),
    "coffee": data.coffee(),
    "chelsea": data.chelsea(),   # cat
}

class_names = [
    "a photo of an astronaut",
    "a photo of a cup of coffee",
    "a photo of a cat"
]

# ---- Model & preprocessing ----
dino_backbone.to(device).eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# ---- Preprocess images ----
img_tensors = []
for name, img in images.items():
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)   # if numpy array (skimage style)
    img_tensor = preprocess(img).unsqueeze(0)
    img_tensors.append(img_tensor)

img_batch = torch.cat(img_tensors).to(device)

# ---- Encode text ----
text_tensor = tokenizer.tokenize(class_names).to(device)

# ---- Get features ----
with torch.inference_mode():
    img_feats = dino_backbone.encode_image(img_batch)     # [N, D]
    text_feats = dino_backbone.encode_text(text_tensor)   # [M, D]

print("Image embedding shape:", img_feats.shape)
print("Text embedding shape:", text_feats.shape)

# ---- Compare similarity ----
# cosine similarity: [N, M]
sim = torch.nn.functional.cosine_similarity(
    img_feats.unsqueeze(1), text_feats.unsqueeze(0), dim=-1
)

print("Similarity matrix:\n", sim)

# ---- Optional: match best class for each image ----
best_match = sim.argmax(dim=1)
for i, (name, _) in enumerate(images.items()):
    print(f"{name} → {class_names[best_match[i]]} (score={sim[i, best_match[i]].item():.4f})")
