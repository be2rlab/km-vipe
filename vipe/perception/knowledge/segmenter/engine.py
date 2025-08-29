import torch
import numpy as np
import torch.nn.functional as F
from torch2trt import TRTModule
from typing import Tuple
from vipe.perception.knowledge.segmenter.utils import (
    SamResize,
    resize_longest_image_size,
    get_preprocess_shape,
    apply_boxes,
)
import tensorrt as trt
import sys

sys.path.append("/workspace/perception/algorithms")
from mobilesamv2 import ObjectAwareModel  # noqa: E402


class MobileSAMv2:
    """
    Optimized engine for segmentation using TensorRT.
    """

    def __init__(
        self,
        encoder_path: str = None,
        decoder_path: str = None,
        detector_path: str = None,
        device: str = "cuda",
        img_size: int = 1024,
        max_object_size: int = 256,
        iou_threshold: float = 0.3,
    ):
        super().__init__()
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.detector_path = detector_path
        self.iou_threshold = iou_threshold
        self.max_object_size = max_object_size
        self.encoder = None
        self.decoder = None
        self.ObjAwareModel = None
        self.img_size = img_size
        self.device = device
        self.pixel_mean = (
            torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1) / 255
        )
        self.pixel_std = (
            torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1) / 255
        )

    def load_model(self) -> None:
        """Load and optimize TensorRT models."""
        # Load encoder
        with open(self.encoder_path, "rb") as f:
            engine_bytes = f.read()
        self.encoder = TRTModule(
            engine=trt.Runtime(trt.Logger()).deserialize_cuda_engine(engine_bytes),
            input_names=["input_image"],
            output_names=["image_embeddings"],
        )

        # Load decoder
        with open(self.decoder_path, "rb") as f:
            engine_bytes = f.read()
        self.decoder = TRTModule(
            engine=trt.Runtime(trt.Logger()).deserialize_cuda_engine(engine_bytes),
            input_names=["image_embeddings", "point_coords", "point_labels"],
            output_names=["masks", "iou_predictions"],
        )

        self.ObjAwareModel = ObjectAwareModel(self.detector_path)
        self.ObjAwareModel.init()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized GPU preprocessing pipeline."""
        x = x.to(self.device)
        x = SamResize(self.img_size)(x).float() / 255
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h), value=0)
        return x.unsqueeze(0)

    def postprocess(
        self, masks: torch.Tensor, orig_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Batch postprocessing optimized for GPU."""
        orig_h, orig_w = orig_size

        masks = F.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        prepadded_size = resize_longest_image_size(
            torch.tensor([orig_h, orig_w], device=masks.device), self.img_size
        )
        # print(prepadded_size)
        masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]
        # print(masks.shape)
        return F.interpolate(
            masks, size=(orig_h, orig_w), mode="bilinear", align_corners=True
        )

    @torch.no_grad()
    def __call__(self, raw_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is None or self.decoder is None:
            self.load_model()

        # Convert input image to GPU tensor if not already
        if isinstance(raw_image, np.ndarray):
            raw_image = torch.from_numpy(raw_image).to(self.device)
        orig_h, orig_w = raw_image.shape[0], raw_image.shape[1]
        origin_image_size = (orig_h, orig_w)
        image = raw_image.float()
        preprocessed = self.preprocess(image)
        input_size = get_preprocess_shape(orig_h, orig_w, self.img_size)

        obj_results = self.ObjAwareModel(
            raw_image.detach().cpu().numpy(),
            device=self.device,
            retina_masks=True,
            imgsz=640,
            conf=0.7,
            iou=0.8,
            verbose=False,
        )
        input_boxes = obj_results[0].boxes.xyxy
        boxes = input_boxes.cpu().numpy()
        boxes = apply_boxes(boxes, origin_image_size, input_size).astype(np.float32)
        boxes = torch.from_numpy(boxes).to(self.device)

        n = boxes.shape[0]
        batch_size = 128
        num_batches = n // batch_size + (1 if n % batch_size != 0 else 0)

        embeddings = self.encoder(preprocessed)[0].reshape(1, 256, 64, 64)

        result_mask = None
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, boxes.shape[0])
            batch = boxes[start_idx:end_idx]

            box_label = torch.tensor(
                [[2, 3] for _ in range(batch.shape[0])],
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1, 2)

            point_coords = batch
            point_labels = box_label

            low_res_masks, confs = self.decoder(embeddings, point_coords, point_labels)
            low_res_masks = low_res_masks.reshape(1, -1, 256, 256)
            masks = self.postprocess(low_res_masks, origin_image_size)[0]

            masks = masks > 0.0

            if result_mask is None:
                result_mask = masks
            else:
                result_mask = torch.cat((result_mask, masks))

        crops, BBs = self.get_mask_crops(
            raw_image, result_mask, max_size=self.max_object_size
        )
        return crops, BBs, masks, confs

    def get_mask_crops(
        self,
        raw_image: torch.Tensor,  # Shape: [H, W, 3]
        masks: torch.Tensor,  # Shape: [N, H, W]
        padding: float = 0.0,
        max_size: int = 256,
    ) -> Tuple[
        torch.Tensor, torch.Tensor
    ]:  # Returns (crops [N, 3, max_size, max_size], BBs [N, 4])
        crops = []
        image_height, image_width = raw_image.shape[:2]
        BBs = []

        # Ensure raw_image is on the correct device
        raw_image = raw_image.to(self.device)

        for mask in masks:
            # Find the bounding box of the mask using GPU operations
            rows = torch.any(mask, dim=1)
            cols = torch.any(mask, dim=0)
            if not torch.any(rows) or not torch.any(cols):
                continue

            rmin, rmax = torch.where(rows)[0][[0, -1]]
            cmin, cmax = torch.where(cols)[0][[0, -1]]

            # Convert to integers for indexing
            rmin, rmax = rmin.item(), rmax.item()
            cmin, cmax = cmin.item(), cmax.item()

            # Calculate padding
            height = rmax - rmin
            width = cmax - cmin
            pad_h = int(height * padding)
            pad_w = int(width * padding)

            # Apply padding with bounds checking
            rmin = max(0, rmin - pad_h)
            rmax = min(image_height, rmax + pad_h)
            cmin = max(0, cmin - pad_w)
            cmax = min(image_width, cmax + pad_w)

            # Extract the crop and convert to [C, H, W] format
            crop = raw_image[rmin:rmax, cmin:cmax].clone()  # [H, W, 3]
            crop = crop.permute(2, 0, 1)  # [3, H, W]
            # Convert to float32 for interpolation
            crop = crop.float()

            # Calculate resize dimensions maintaining aspect ratio
            c, h, w = crop.shape
            if h > w:
                new_h = max_size
                new_w = int(w * max_size / h)
            else:
                new_w = max_size
                new_h = int(h * max_size / w)

            # Resize the crop
            if h != new_h or w != new_w:
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0),  # Add batch dimension for interpolate
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)  # Remove batch dimension

            # Create padding for centered crop
            pad_h = max_size - new_h
            pad_w = max_size - new_w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            # Pad to square
            crop = torch.nn.functional.pad(
                crop,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )

            crops.append(crop)
            BBs.append([cmin, rmin, cmax, rmax])  # [rmin, cmin, rmax, cmax]

        if len(crops) == 0:
            return torch.empty(
                0, 3, max_size, max_size, device=self.device
            ), torch.empty(0, 4, device=self.device)

        # Stack all crops and BBs
        crops = torch.stack(crops)  # [N, 3, max_size, max_size]
        BBs = torch.tensor(BBs, device=self.device)  # [N, 4]

        return crops, BBs
