import math
import numpy as np
from typing import Optional, Tuple, Union, List

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model


class ClipEmbedding:
    """
    Compute one CLIP embedding per instance mask, and build a pixel-wise embedding map
    by assigning each pixel the embedding of the mask it belongs to.

    Typical usage:
        clipper = ClipEmbedding(model_name='MobileCLIP2-S4', weights_path='weights/mobileclip2_s4.pt', device='cuda')
        mask_embeds, index_map = clipper.embed_masks(frame_rgb, masks_bool)  # [M,D], [H,W]
        # Optional (memory heavy):
        pixelwise = clipper.materialize_pixel_embeddings(mask_embeds, index_map)  # [H,W,D]
    """

    def __init__(
        self,
        model_name: str = "MobileCLIP2-S4",
        weights_path: Optional[str] = "weights/mobileclip2_s4.pt",
        device: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        preprocess=None,
        use_autocast: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if model is None or preprocess is None:
            # open_clip supports checkpoint IDs or local paths via `pretrained=`
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=weights_path
            )
        # Reparameterize for inference
        self.model = reparameterize_model(model.eval()).to(self.device)
        self.preprocess = preprocess
        self.use_autocast = bool(use_autocast and str(self.device).startswith("cuda"))
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Try to infer output dim for convenience
        self.embed_dim = getattr(self.model, "embed_dim", None) or \
                         getattr(getattr(self.model, "visual", None), "output_dim", None)

    @staticmethod
    def _to_pil(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        assert isinstance(image, np.ndarray), "image must be PIL.Image or numpy array"
        if image.ndim == 3 and image.shape[2] == 3:
            # Assume OpenCV BGR; convert to RGB
            arr = image[:, :, ::-1]
        else:
            raise ValueError("Expected HxWx3 numpy array")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def _ensure_bool_masks(
        masks: Union[np.ndarray, torch.Tensor], size_hw: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Ensure masks is BoolTensor [M,H,W] at the requested size_hw.
        """
        H, W = size_hw
        if isinstance(masks, np.ndarray):
            m = torch.from_numpy(masks)
        else:
            m = masks
        if m.dtype != torch.bool:
            m = (m > 0.5)

        if m.ndim == 2:
            m = m.unsqueeze(0)  # [1,H,W]
        assert m.ndim == 3, "masks must be [M,H,W] or [H,W]"

        # Resize if needed
        Mh, Mw = m.shape[-2:]
        if (Mh, Mw) != (H, W):
            m = F.interpolate(
                m.float().unsqueeze(1), size=(H, W), mode="nearest"
            ).squeeze(1).bool()
        return m  # [M,H,W] bool

    @staticmethod
    def _mask_to_bbox(m: np.ndarray) -> Tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) for a boolean mask; if empty, return a degenerate box."""
        ys, xs = np.where(m)
        if xs.size == 0 or ys.size == 0:
            return 0, 0, 1, 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return x0, y0, x1, y1

    @staticmethod
    def _pad_box(x0: int, y0: int, x1: int, y1: int,
                 pad_frac: float, W: int, H: int,
                 min_box: int = 1) -> Tuple[int, int, int, int]:
        """Pad the box by pad_frac of its size, clamp to image; enforce a minimum box size."""
        w, h = x1 - x0, y1 - y0
        pw, ph = int(round(w * pad_frac)), int(round(h * pad_frac))
        x0 = max(0, x0 - pw);  y0 = max(0, y0 - ph)
        x1 = min(W, x1 + pw);  y1 = min(H, y1 + ph)
        # Enforce minimum size to avoid zero-area crops
        if (x1 - x0) < min_box:
            cx = (x0 + x1) // 2
            x0 = max(0, cx - min_box // 2)
            x1 = min(W, x0 + min_box)
        if (y1 - y0) < min_box:
            cy = (y0 + y1) // 2
            y0 = max(0, cy - min_box // 2)
            y1 = min(H, y0 + min_box)
        return x0, y0, x1, y1

    @torch.inference_mode()
    def embed_masks(
        self,
        image: Union[Image.Image, np.ndarray],
        masks: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
        background_fill: str = "mean",        # used when crop_mode is masked_full or bbox_masked
        return_pixel_index_map: bool = True,
        crop_mode: str = "bbox_masked",       # "masked_full" | "bbox" | "bbox_masked"
        bbox_pad_frac: float = 0.10,          # 10% context around the object
        min_box_size: int = 16,               # avoid ultra-tiny crops after padding
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute one embedding per instance region.

        crop_mode:
          - "masked_full": keep full image size; fill background outside each mask (original behavior).
          - "bbox":        crop to the tight bbox (+ padding); no masking inside the crop.
          - "bbox_masked": crop to bbox (+ padding) and fill outside-mask pixels inside the crop.
        """
        img_pil = self._to_pil(image)
        W, H = img_pil.size
        masks_bool = self._ensure_bool_masks(masks, (H, W))  # [M,H,W]
        M = masks_bool.shape[0]
        if M == 0:
            raise ValueError("No masks provided")

        np_img = np.array(img_pil, dtype=np.uint8)  # H,W,3 RGB

        # Prepare background fill color (used by masked_full and bbox_masked)
        if background_fill == "mean":
            bg_color = np_img.mean(axis=(0, 1), keepdims=True).astype(np.uint8)  # 1,1,3
        elif background_fill == "zero":
            bg_color = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            raise ValueError("background_fill must be 'mean' or 'zero'")

        crops_or_masked: List[Image.Image] = []
        for i in range(M):
            m = masks_bool[i].cpu().numpy().astype(bool)  # H,W

            if crop_mode == "masked_full":
                m3 = np.repeat(m[:, :, None], 3, axis=2)
                img_i = np.where(m3, np_img, bg_color)
                crops_or_masked.append(Image.fromarray(img_i, mode="RGB"))

            else:
                # Compute padded bbox and crop
                x0, y0, x1, y1 = self._mask_to_bbox(m)
                x0, y0, x1, y1 = self._pad_box(x0, y0, x1, y1,
                                               pad_frac=bbox_pad_frac, W=W, H=H,
                                               min_box=min_box_size)
                crop = np_img[y0:y1, x0:x1, :]  # h',w',3
                if crop_mode == "bbox":
                    crops_or_masked.append(Image.fromarray(crop, mode="RGB"))
                elif crop_mode == "bbox_masked":
                    m_crop = m[y0:y1, x0:x1]
                    m3 = np.repeat(m_crop[:, :, None], 3, axis=2)
                    crop_masked = np.where(m3, crop, bg_color)
                    crops_or_masked.append(Image.fromarray(crop_masked, mode="RGB"))
                else:
                    raise ValueError("crop_mode must be 'masked_full', 'bbox', or 'bbox_masked'")

        # Batch preprocess & encode
        embeds = []
        self.model = self.model.to(self.device)
        for start in range(0, M, batch_size):
            chunk = crops_or_masked[start:start + batch_size]
            batch = torch.stack([self.preprocess(im) for im in chunk], dim=0).to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_autocast):
                feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeds.append(feats)
        mask_embeds = torch.cat(embeds, dim=0)  # [M,D]

        if not return_pixel_index_map:
            return mask_embeds, None

        # Index map unchanged (still refers to the original mask layout)
        index_map = torch.full((masks_bool.shape[-2], masks_bool.shape[-1]), -1, dtype=torch.long)
        for i in range(M):
            index_map[masks_bool[i]] = i

        return mask_embeds, index_map
    
    @staticmethod
    def materialize_pixel_embeddings(
        mask_embeds: torch.Tensor,  # [M,D]
        index_map: torch.Tensor,    # [H,W], values in [-1..M-1]
        background_strategy: str = "zeros"  # or "mean"
    ) -> torch.Tensor:
        """
        Create the full pixel-wise tensor [H,W,D]. This can be very large!

        background_strategy:
          - "zeros": background gets a 0-vector
          - "mean":  background gets the mean of mask embeddings
        """
        assert index_map.ndim == 2 and mask_embeds.ndim == 2
        H, W = index_map.shape
        M, D = mask_embeds.shape
        flat = index_map.flatten().to(torch.long)  # [H*W]

        # Build a lookup table with background row first
        if background_strategy == "zeros":
            bg = torch.zeros((1, D), dtype=mask_embeds.dtype, device=mask_embeds.device)
        elif background_strategy == "mean":
            bg = mask_embeds.mean(dim=0, keepdim=True)
        else:
            raise ValueError("background_strategy must be 'zeros' or 'mean'")

        lut = torch.cat([bg, mask_embeds], dim=0)  # [M+1, D]
        # Shift indices by +1; -1 becomes 0 (background)
        flat_shifted = (flat + 1).clamp(min=0)
        out = lut[flat_shifted]  # [H*W,D]
        return out.view(H, W, D)

    @torch.inference_mode()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """
        Convenience wrapper: tokenize and encode text with the current model.
        Returns L2-normalized features [P, D].
        """
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.cuda.amp.autocast(enabled=self.use_autocast):
            feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)


"""
clip_sam_demo.py

End-to-end demo:
- Runs Ultralytics SAM 2.1 to get instance masks.
- Uses the ClipEmbedding helper to compute one MobileCLIP embedding per mask.
- Encodes a prompt with MobileCLIP and finds the mask with the highest cosine similarity.
- Saves an overlay highlighting the best mask.

Usage:
  python clip_sam_demo.py \
    --image /data/000000.jpg \
    --prompt "a dog" \
    --mobileclip-weights weights/mobileclip2_s4.pt \
    --out best_mask_overlay.png
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from ultralytics import SAM


def overlay_mask(img_rgb: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Alpha-blend a single mask onto an RGB image (red overlay)."""
    out = img_rgb.astype(np.float32).copy()
    overlay = out.copy()
    overlay[mask_bool] = np.array([255, 0, 0], dtype=np.float32)
    blended = alpha * overlay + (1.0 - alpha) * out
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_caption(img_rgb: np.ndarray, text: str) -> np.ndarray:
    """Draw a small caption box (top-left) with PIL."""
    pil = Image.fromarray(img_rgb)  # RGB
    draw = ImageDraw.Draw(pil)
    pad = 8
    w = int(draw.textlength(text))
    # Use opaque RGB fill to avoid RGBA-on-RGB errors
    draw.rectangle([(8, 8), (8 + w + 2 * pad, 40)], fill=(0, 0, 0))
    draw.text((8 + pad, 12), text, fill=(255, 255, 255))
    return np.array(pil)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Text prompt (or comma-separated)")
    parser.add_argument("--mobileclip-weights", default="weights/mobileclip2_s4.pt")
    parser.add_argument("--model-name", default="MobileCLIP2-S4")
    parser.add_argument("--sam-weights", default="sam2.1_l.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bg-fill", choices=["mean", "zero"], default="mean")
    parser.add_argument("--min-pixels", type=int, default=64, help="Drop tiny masks below this area")
    parser.add_argument("--topk", type=int, default=10, help="Print top-K mask scores")
    parser.add_argument("--out", default="best_mask_overlay.png")
    args = parser.parse_args()

    DEVICE = args.device
    print(f"[Info] Using device: {DEVICE}")

    # 2) Instantiate the ClipEmbedding helper
    clipper = ClipEmbedding(
        model_name=args.model_name,
        weights_path=args.mobileclip_weights,
        device=DEVICE,
    )

    # 3) Run SAM 2.1 to get masks
    sam = SAM(args.sam_weights)
    results = sam(args.image, device=DEVICE)
    res0 = results[0]

    if res0.masks is None or res0.masks.data is None:
        raise RuntimeError("SAM did not return any masks for this image.")

    orig_bgr = res0.orig_img                            # (H, W, 3) BGR (Ultralytics convention)
    H, W = orig_bgr.shape[:2]
    orig_rgb = orig_bgr[:, :, ::-1]                     # convert for visualization

    masks_small = res0.masks.data                       # [M, mh, mw] float/bool in [0,1]
    masks = F.interpolate(
        masks_small.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(1) > 0.5                                  # [M, H, W] bool
    M = masks.shape[0]
    print(f"[Info] SAM proposed {M} masks")

    # Optional: remove tiny masks (helps noisy images)
    if args.min_pixels > 0:
        keep = [i for i in range(M) if masks[i].sum().item() >= args.min_pixels]
        if len(keep) == 0:
            raise RuntimeError("All masks were filtered out by min-pixels threshold.")
        masks = masks[keep]
        print(f"[Info] Kept {len(keep)} masks after filtering small ones")

    # 4) Compute one CLIP embedding per mask
    # NOTE: pass the original BGR frame; ClipEmbedding converts BGR->RGB internally for PIL.
    mask_embeds, index_map = clipper.embed_masks(
        image=orig_bgr,
        masks=masks,
        batch_size=args.batch_size,
        background_fill=args.bg_fill,
        crop_mode="bbox_masked",     # try also "bbox" for a bit more context
        bbox_pad_frac=0.12,          # ~10–15% context often works well
        min_box_size=24,
        return_pixel_index_map=True,
    )
    print(f"[Info] mask_embeds: {tuple(mask_embeds.shape)}")

    # 5) Encode prompt(s) and compute cosine similarities
    prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    with torch.inference_mode():
        # Use the helper or call the model directly:
        text_feats = clipper.encode_text(prompts)  # [P, D]

    # Both image and text features are L2-normalized (cosine == dot)
    # sims: [M, P]  (masks x prompts)
    sims = mask_embeds @ text_feats.T
    max_per_mask, best_prompt_idx = torch.max(sims, dim=1)      # [M], [M]
    best_mask_idx = int(torch.argmax(max_per_mask).item())
    best_score = float(max_per_mask[best_mask_idx].item())
    best_prompt = prompts[int(best_prompt_idx[best_mask_idx].item())]

    print(f"[Result] Best mask: {best_mask_idx}  |  prompt: '{best_prompt}'  |  sim: {best_score:.4f}")

    # (Optional) Show top-K masks by similarity (w.r.t their own best prompt)
    K = min(args.topk, masks.shape[0])
    order = torch.argsort(max_per_mask, descending=True)[:K].tolist()
    print("\nTop masks:")
    for rank, mi in enumerate(order, 1):
        pr = prompts[int(best_prompt_idx[mi].item())]
        sc = float(max_per_mask[mi].item())
        print(f"{rank:2d}. mask #{mi:<3d}  sim={sc:.4f}  (prompt='{pr}')")

    # 6) Visualize the best mask
    best_mask = masks[best_mask_idx].cpu().numpy()
    overlay = overlay_mask(orig_rgb, best_mask, alpha=0.55)
    caption = f"'{best_prompt}' | mask #{best_mask_idx} | sim={best_score:.3f}"
    overlay = draw_caption(overlay, caption)

    Image.fromarray(overlay).save(args.out)
    print(f"\n[Saved] {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
