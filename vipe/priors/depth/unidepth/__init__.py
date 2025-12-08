# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .models.unidepthv2.unidepthv2 import Pinhole, UniDepthV2, BatchCamera
from .models.unidepthv2.unidepthv2 import (
    get_paddings,
    get_resize_factor,
    _postprocess,
)
IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)


from torch2trt import TRTModule
import tensorrt as trt

import matplotlib.pyplot as plt


class UniDepth2Model(DepthEstimationModel):
    def __init__(self, type: Literal["s", "b", "l"] = "l") -> None:
        super().__init__()
        self.model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vit{type}14")
        self.model.interpolation_mode = "bilinear"
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"

        focal_length: float = unpack_optional(src.focal_length)

        # print(f"focal_length: {focal_length}")

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb = torch.clamp(rgb.moveaxis(-1, 1) * 255.0, max=255.0).byte()
        K = torch.tensor(
            [
                [focal_length, 0, rgb.shape[-1] / 2],
                [0, focal_length, rgb.shape[-2] / 2],
                [0, 0, 1],
            ],
            device=rgb.device,
        ).float()
        camera = Pinhole(K=K[None].repeat(rgb.shape[0], 1, 1))

        predictions = self.model.infer(rgb, camera)
        # predictions = self.model.infer(rgb)
        pred_depth = predictions["depth"].squeeze(1)
        confidence = predictions["confidence"].squeeze(1)

        if not batch_dim:
            pred_depth, confidence = pred_depth[0], confidence[0]

        # fig = plt.imshow(pred_depth.squeeze(0).cpu().numpy(), cmap='viridis')
        # plt.savefig("torchdepth.png", dpi=300)

        return DepthEstimationResult(
            metric_depth=pred_depth,
            confidence=confidence,
        )

class UnidepthTRTModel(DepthEstimationModel):
    def __init__(self, engine_path = "/home/user/km-vipe/weights/unidepthv2-l-c-336-602.engine"):
        print(f"Attempting to load TRT model from {engine_path}")
        self.engine_path = engine_path
        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()
        
        engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(engine_bytes)
        model = TRTModule(
            engine=engine,
            input_names=["rgbs", "rays"],
            output_names=["pts_3d", "confidence", "intrinsics"],
        )
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)  # (H,W,3) or (B,H,W,3)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        B, H, W, C = rgb.shape

        focal_length: float = unpack_optional(src.focal_length)
        
        # From Unidepth config
        ratio_bounds = [0.5, 2.5] 
        pixels_bounds = (200000, 600000)

        paddings, (padded_H, padded_W) = get_paddings((H, W), ratio_bounds)
        (pad_left, pad_right, pad_top, pad_bottom) = paddings
        resize_factor, (new_H, new_W) = get_resize_factor((padded_H, padded_W), pixels_bounds)

        rgb = rgb.permute(0, 3, 1, 2).float()

        K = torch.tensor(
            [
                [focal_length, 0, rgb.shape[-1] / 2],
                [0, focal_length, rgb.shape[-2] / 2],
                [0, 0, 1],
            ],
            device=rgb.device,
        ).float()

        camera = Pinhole(K=K[None].repeat(rgb.shape[0], 1, 1))
        camera = BatchCamera.from_camera(camera)

        rgb = F.pad(rgb, paddings, value=0.0)
        rgb = F.interpolate(rgb, size=(new_H, new_W), mode="bilinear", align_corners=False)
        rgb = rgb.to(self.device)

        camera = camera.crop(left=-pad_left, top=-pad_top, right=-pad_right, bottom=-pad_bottom)
        camera = camera.resize(resize_factor).to(self.device)

        # print(camera.params)
        # print(f"trt: {camera.params.shape}")

        rays = camera.get_rays(shapes=(B, new_H, new_W))

        # print(rays.shape)

        with torch.no_grad():
            pts_3d, confidence, intrinsics = self.model(rgb, rays)

        pred_depth = _postprocess(
            pts_3d[:, -1:],
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode="bilinear",
        )
        confidence = _postprocess(
            confidence,
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode="bilinear",
        )

        pred_depth = pred_depth[..., :H, :W].squeeze(1)
        confidence = confidence[..., :H, :W].squeeze(1)

        return DepthEstimationResult(metric_depth=pred_depth, confidence=confidence, rays=rays)

    

