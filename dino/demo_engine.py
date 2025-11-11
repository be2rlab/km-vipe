import torch
import numpy as np
from omegaconf import OmegaConf
from torchvision.io import read_image
import matplotlib.pyplot as plt
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import torch.nn.functional as F

sys.path.insert(0, "Talk2DINO/")
from src.model import VisualProjectionLayer, DoubleMLP


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_root)
from vipe.priors.embedding import dinov2

sys.path.insert(0, "Talk2DINO/src/open_vocabulary_segmentation")
from models.dinotext import DINOText
from models import build_model


device = "cuda"

def plot_qualitative(image, sim, output_path, palette):
    qualitative_plot = np.zeros((sim.shape[0], sim.shape[1], 3)).astype(np.uint8)

    for j in list(np.unique(sim)):
        qualitative_plot[sim == j] = np.array(palette[j])
    plt.axis('off')
    plt.imshow(image)
    plt.imshow(qualitative_plot, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


@torch.no_grad()
def generate_masks(
        self, image, img_metas, text_emb, classnames, text_is_token=False, apply_pamr=False, background_func="weighted_average_sigmoid", lambda_bg=0.2,
        # kp_w=0.3,
):
    H, W = image.height, image.width
    
    engine = dinov2.DINOv2EmbeddingEngine(
        model=dinov2.DinoV2Variant.VITB_REG,  
        device=device,
        short_side=448,  # adjust based on your needs
        pyramid_scales=[2.0, 1.0, 0.75, 0.5],
        enable_rerun=False  # disable for inference
    )
    
    # Extract multi-scale features using DINOv2 engine
    features_pyramid = engine.embed_frame_multiscale(image)
    
    # Upsample to original resolution
    image_feat = engine.upsample_features(
        features=features_pyramid, 
        target_size=(H, W), 
        method="pyramid_weighted"
    )

    #image_feat = engine.embed_frame(image)
    
    image_feat = image_feat.reshape(1, -1, image_feat.shape[-1])

    print(image_feat.shape)

    if type(self.proj) == VisualProjectionLayer:
        image_feat = self.proj.project_dino(image_feat.float())
    if type(self.proj) == DoubleMLP:
        image_feat = self.proj.project_visual(image_feat.float())


    b, n_patches, c = image_feat.shape
    image_feat = image_feat.reshape(b, H, W, c).permute(0, 3, 1, 2)
    
    mask, simmap = self.masker.forward_seg(image_feat, text_emb, hard=False)  # [B, N, H', W']

    # resize
    mask = F.interpolate(mask, (H, W), mode='bilinear', align_corners=True)  # [B, N, H, W]

    assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"

    return mask, simmap

def demo_usage(config_path, input_path, output_path, with_background, text):
    cfg = OmegaConf.load(config_path)
    text = text.replace("_", " ").split(",")

    model = build_model(cfg.model)
    model.to(device).eval()

    img = Image.open(input_path).convert("RGB")

    # Create a colormap
    num_classes = len(text)
    cmap = plt.get_cmap("tab20", num_classes)

    palette = [[int(c * 255) for c in cmap(i)[:3]] for i in np.linspace(0, 1, num_classes)]
    if len(text) > len(palette):
        for _ in range(len(text) - len(palette)):
            palette.append([np.random.randint(0, 255) for _ in range(3)])
            
    if with_background:
        palette.insert(0, [0, 0, 0])
        model.with_bg_clean = True

    with torch.no_grad():
        text_emb = model.build_dataset_class_tokens("sub_imagenet_template", text)
        text_emb = model.build_text_embedding(text_emb)

    mask, _ = generate_masks(model, img, img_metas=None, text_emb=text_emb, classnames=text, apply_pamr=True)
    if with_background:
        background = torch.ones_like(mask[:, :1]) * 0.55
        mask = torch.cat([background, mask], dim=1)
    
    mask = mask.argmax(dim=1)
    plot_qualitative(np.array(img), mask.cpu()[0].numpy(), output_path, palette)   



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for Talk2DINO")
    parser.add_argument("--config", type=str, default="Talk2DINO/src/open_vocabulary_segmentation/configs/cityscapes/dinotext_cityscapes_vitb_mlp_infonce.yml")
    parser.add_argument("--output", type=str, default="seg_result.png")
    parser.add_argument("--input", type=str, default="/data/Replica/office1/results/frame000437.jpg")
    parser.add_argument("--with_background", action="store_true")
    parser.add_argument("--textual_categories", type=str, default="house,backpack,bed,bench,bin,blanket,blinds,book,bottle,box,bowl,cabinet,chair,clock,cloth,clothing,cushion,curtain,desk,door,floor,handrail,lamp,monitor,mouse,pan,phone,pillow,plate,rack,remote-control,shelf,sink,sofa,stool,table,tablet,towel,tv-screen,tv-stand,vase,wall,wardrobe,window,rug,bag,set-of-clothing")
    args = parser.parse_args()

    demo_usage(args.config, args.input, args.output, args.with_background, args.textual_categories)