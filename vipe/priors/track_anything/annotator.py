import argparse
import numpy as np
import random

import torch

from PIL import Image
from .ram.ram.models import ram_plus
from .ram.ram import inference_ram as inference
from .ram.ram import get_transform
import torchvision.transforms.functional as F



class Annotator:
    def __init__(self,ram_args):
        self.device = ram_args['device']
        self.image_size = ram_args['image_size']
        self.model_path = ram_args['checkpoint_path']
        
        self.transform = get_transform(self.image_size)
        #######load model
        self.model = ram_plus(pretrained=self.model_path,
                                image_size=self.image_size,
                                vit='swin_l')
        self.model.eval()
        self.model = self.model.to(self.device)

    def annotate(self,frame):
        image = self.transform_frame(frame)
        res = inference(image, self.model)
        words = res[0].split("|")
        for i in range(len(words)):
            if words[i][0]==' ':
                words[i] = words[i][1:]
            if words[i][-1] == ' ':
                words[i] = words[i][:-1]
        return words


    def transform_frame(self, frame):
        """
        Transform a numpy image array (H, W, 3) with uint8 [0, 255] values
        into a normalized torch tensor of shape (3, H, W) with float32 values in [0, 1],
        resized to (self.image_size, self.image_size) and normalized with ImageNet stats.
        
        Args:
            frame (np.ndarray): Input image of shape (H, W, 3), dtype uint8, range [0, 255]
        
        Returns:
            torch.Tensor: Normalized tensor of shape (3, self.image_size, self.image_size)
        """
        # Convert numpy (H, W, 3) uint8 [0,255] -> torch (3, H, W) float32 [0,1]
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Resize to (self.image_size, self.image_size)
        img_resized = F.resize(img_tensor, [self.image_size, self.image_size], antialias=True)

        # Normalize with ImageNet mean and std
        img_normalized = F.normalize(
            img_resized,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return img_normalized.unsqueeze(0).to(self.device)
        
