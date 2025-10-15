import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


class UniDepthPreprocessor:
    """
    Preprocessor for UniDepth models to match TensorRT input shape (672, 1190)
    """
    def __init__(self, target_size=(672, 1190), device="cuda"):
        self.target_height, self.target_width = target_size
        self.device = device
        
        # ImageNet normalization (standard for ViT backbones)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def preprocess_pil(self, pil_image):
        """
        Preprocess PIL Image to tensor format for UniDepth TRT model
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            preprocessed: (1, 3, 672, 1190) tensor ready for TRT inference
            scale_info: dict with scaling information for post-processing
        """
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        original_width, original_height = pil_image.size
        
        # Convert to tensor and normalize to [0, 1]
        tensor = T.ToTensor()(pil_image)  # (3, H, W)
        
        # Resize to target shape
        resized = F.interpolate(
            tensor.unsqueeze(0),  # Add batch dim: (1, 3, H, W)
            size=(self.target_height, self.target_width),
            mode='bilinear',
            align_corners=False
        )
        
        # Apply ImageNet normalization
        normalized = self.normalize(resized.squeeze(0)).unsqueeze(0)
        
        # Move to device
        preprocessed = normalized.to(self.device)
        
        # Store scaling info for later use
        scale_info = {
            'original_height': original_height,
            'original_width': original_width,
            'target_height': self.target_height,
            'target_width': self.target_width,
            'scale_y': original_height / self.target_height,
            'scale_x': original_width / self.target_width
        }
        
        return preprocessed, scale_info
    
    def preprocess_numpy(self, np_image):
        """
        Preprocess numpy array (H, W, 3) to tensor format
        
        Args:
            np_image: numpy array in (H, W, 3) format, values in [0, 255]
            
        Returns:
            preprocessed: (1, 3, 672, 1190) tensor
            scale_info: dict with scaling information
        """
        # Ensure uint8 and correct shape
        if np_image.dtype != np.uint8:
            np_image = np_image.astype(np.uint8)
        
        if len(np_image.shape) != 3 or np_image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got {np_image.shape}")
        
        # Convert to PIL and use PIL preprocessing
        pil_image = Image.fromarray(np_image)
        return self.preprocess_pil(pil_image)
    
    def preprocess_cv2(self, cv2_image):
        """
        Preprocess OpenCV image (BGR format) to tensor format
        
        Args:
            cv2_image: OpenCV image in BGR format
            
        Returns:
            preprocessed: (1, 3, 672, 1190) tensor
            scale_info: dict with scaling information
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return self.preprocess_numpy(rgb_image)
    
    def preprocess_tensor(self, tensor_image):
        """
        Preprocess tensor image to target format
        
        Args:
            tensor_image: torch.Tensor in various formats:
                         - (H, W, 3) or (3, H, W) - single image
                         - (B, H, W, 3) or (B, 3, H, W) - batch of images
                         Values should be in [0, 1] range
        
        Returns:
            preprocessed: (B, 3, 672, 1190) tensor
            scale_info: dict or list of dicts with scaling information
        """
        # Handle different input formats
        if len(tensor_image.shape) == 3:
            if tensor_image.shape[0] == 3:  # (3, H, W)
                tensor_image = tensor_image.unsqueeze(0)  # (1, 3, H, W)
            else:  # (H, W, 3)
                tensor_image = tensor_image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        elif len(tensor_image.shape) == 4:
            if tensor_image.shape[1] != 3:  # (B, H, W, 3) -> (B, 3, H, W)
                tensor_image = tensor_image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor_image.shape}")
        
        batch_size, _, orig_h, orig_w = tensor_image.shape
        
        # Resize to target shape
        resized = F.interpolate(
            tensor_image,
            size=(self.target_height, self.target_width),
            mode='bilinear',
            align_corners=False
        )
        
        # Apply normalization
        normalized = self.normalize(resized)
        
        # Move to device
        preprocessed = normalized.to(self.device)
        
        # Create scale info for each image in batch
        scale_info = {
            'original_height': orig_h,
            'original_width': orig_w,
            'target_height': self.target_height,
            'target_width': self.target_width,
            'scale_y': orig_h / self.target_height,
            'scale_x': orig_w / self.target_width,
            'batch_size': batch_size
        }
        
        return preprocessed, scale_info


def postprocess(depth_output, scale_info, original_size=None):
    """
    Postprocess depth output back to original image dimensions
    
    Args:
        depth_output: (B, H, W) or (B, 1, H, W) depth tensor from TRT model
        scale_info: scaling information from preprocessing
        original_size: optional (height, width) to resize to specific size
        
    Returns:
        resized_depth: depth map resized to original/specified dimensions
    """
    assert len(depth_output.shape) == 4, f"Expected 4D tensor, got shape {depth_output.shape}"
    
    # Determine target size
    if original_size is not None:
        target_h, target_w = original_size
    else:
        target_h = scale_info['original_height']
        target_w = scale_info['original_width']
    
    # Resize back to original dimensions
    resized_depth = F.interpolate(
        depth_output,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
    
    return resized_depth