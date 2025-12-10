#############
# DOWNLOAD LINKS from https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ 
# You have to fill a form to get access to the models.
# This access is temporarily, download all models would be better, otherwise you need to submit the form again.
#############

import torch
from torchvision import transforms
from PIL import Image
import requests
from skimage import data
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

dino_version = 'v2'

if dino_version == 'v2':
    REPO_PATH = "dino/dinov2" # Specify a local path to the repository (or use installed package instead)
    sys.path.append(REPO_PATH)
    
    from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer
    
    # repo_dir = "dino/dinov2/"
    # model_name = "dinov3_vitl16_dinotxt_tet1280d20h24l"
    # backbone_weights = f"./weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    # weights = f"./weights/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    
    dino_backbone = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(device)
    tokenizer = get_tokenizer()
    
    
elif dino_version == 'v3':
    repo_dir = "dino/dinov3/"
    model_name = "dinov3_vitl16_dinotxt_tet1280d20h24l"
    backbone_weights = f"./weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    weights = f"./weights/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    
    # DINOv3
    dino_backbone, tokenizer = torch.hub.load(repo_dir, model_name, source='local', weights=weights, backbone_weights=backbone_weights)
    
else:
    raise ValueError(f"Unsupported DINO version: {dino_version}")

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
