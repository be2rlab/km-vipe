import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import huggingface_hub
import json
import os
from typing import Tuple, Dict, Any

from image_preprocessing import UniDepthPreprocessor, postprocess_depth

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch2trt import TRTModule
import tensorrt as trt


class ModelComparison:
    """Compare TensorRT and PyTorch UniDepth models"""
    
    def __init__(self, 
                 engine_path: str = "/weights/unidepthv2-l-672-1190.engine",
                 image_path: str = "frame000000.jpg",
                 target_size: Tuple[int, int] = (336, 602)):
        
        self.engine_path = engine_path
        self.image_path = image_path
        self.target_size = target_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize preprocessor
        self.preprocessor = UniDepthPreprocessor(target_size, device=self.device)
        
        # Load models
        self.trt_model = self._load_trt_model()
        self.torch_model = self._load_torch_model()
        
        # Load and preprocess image
        self.image, self.preprocessed_image, self.scale_info = self._load_image()
        
    def _load_trt_model(self):
        """Load TensorRT model"""
        print("Loading TensorRT model...")

        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()
        
        engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(engine_bytes)
        model = TRTModule(
            engine=engine,
            input_names=["rgbs"],
            output_names=["pts_3d", "confidence", "intrinsics"],
        )
        
        print("✅ TensorRT model loaded")
        return model
    
    def _load_torch_model(self):
        """Load original PyTorch model"""
        print("Loading PyTorch model...")
        
        from vipe.priors.depth.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2

        model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval().cuda()

        print("✅ PyTorch model loaded")
        return model
    
    def _load_image(self):
        """Load and preprocess image"""
        print(f"Loading image: {self.image_path}")
        
        # Load original image
        pil_image = Image.open(self.image_path).convert('RGB')
        
        # Preprocess for models
        preprocessed, scale_info = self.preprocessor.preprocess_pil(pil_image)
        
        return pil_image, preprocessed, scale_info
    
    def run_trt_inference(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run TensorRT inference"""
        print("Running TensorRT inference...")

        print(self.preprocessed_image)
        
        with torch.no_grad():
            pts_3d, confidence, intrinsics = self.trt_model(self.preprocessed_image)

        return postprocess_depth(pts_3d, self.scale_info), confidence, intrinsics
    
    def run_torch_inference(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run PyTorch inference"""
        print("Running PyTorch inference...")
        
        with torch.no_grad():
            # Get intermediate outputs for comparison
            B, _, H, W = self.preprocessed_image.shape
            features, tokens = self.torch_model.pixel_encoder(self.preprocessed_image)

            inputs = {}
            inputs["image"] = self.preprocessed_image
            inputs["features"] = [
                self.torch_model.stacking_fn(features[i:j]).contiguous()
                for i, j in self.torch_model.slices_encoder_range
            ]
            inputs["tokens"] = [
                self.torch_model.stacking_fn(tokens[i:j]).contiguous()
                for i, j in self.torch_model.slices_encoder_range
            ]
            
            outputs = self.torch_model.pixel_decoder(inputs, [])
            outputs["rays"] = outputs["rays"].permute(0, 2, 1).reshape(B, 3, H, W)
            pts_3d = outputs["rays"] * outputs["radius"]
            
            confidence = outputs["confidence"]
            intrinsics = outputs["intrinsics"]
        
        return postprocess_depth(pts_3d, self.scale_info), confidence, intrinsics
    
    def compute_metrics(self, 
                       trt_outputs: Tuple[torch.Tensor, ...], 
                       torch_outputs: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Compute comparison metrics"""
        
        trt_pts3d, trt_conf, trt_intrinsics = trt_outputs
        torch_pts3d, torch_conf, torch_intrinsics = torch_outputs
        
        metrics = {}
        
        # Depth comparison (Z component)
        trt_depth = trt_pts3d[:, 2:3, :, :]  # Z component
        torch_depth = torch_pts3d[:, 2:3, :, :]
        depth_diff = torch.abs(trt_depth - torch_depth)
        metrics['depth_mae'] = depth_diff.mean().item()
        metrics['depth_max_error'] = depth_diff.max().item()
        metrics['depth_mse'] = torch.nn.functional.mse_loss(trt_depth, torch_depth).item()
        
        # Confidence comparison
        conf_diff = torch.abs(trt_conf - torch_conf)
        metrics['confidence_mae'] = conf_diff.mean().item()
        metrics['confidence_max_error'] = conf_diff.max().item()
        
        # Intrinsics comparison
        intrinsics_diff = torch.abs(trt_intrinsics - torch_intrinsics)
        metrics['intrinsics_mae'] = intrinsics_diff.mean().item()
        metrics['intrinsics_max_error'] = intrinsics_diff.max().item()
        
        return metrics
    
    def visualize_results(self, 
                         trt_outputs: Tuple[torch.Tensor, ...], 
                         torch_outputs: Tuple[torch.Tensor, ...],
                         save_dir: str = "comparison_results"):
        """Create visualizations comparing outputs"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        trt_pts3d, trt_conf, trt_intrinsics = trt_outputs
        torch_pts3d, torch_conf, torch_intrinsics = torch_outputs
        
        # Extract depth (Z component)
        trt_depth = trt_pts3d[0, 2].cpu().numpy()  # First batch, Z component
        torch_depth = torch_pts3d[0, 2].cpu().numpy()
        
        # Extract confidence
        trt_conf_np = trt_conf[0, 0].cpu().numpy()
        torch_conf_np = torch_conf[0, 0].cpu().numpy()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 15))
        
        # Row 1: Original image and depth maps
        # Original image
        axes[0, 0].imshow(self.image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # TensorRT depth
        im1 = axes[0, 1].imshow(trt_depth, cmap='viridis')
        axes[0, 1].set_title('TensorRT Depth')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # PyTorch depth
        im2 = axes[0, 2].imshow(torch_depth, cmap='viridis')
        axes[0, 2].set_title('PyTorch Depth')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Depth difference
        depth_diff = np.abs(trt_depth - torch_depth)
        im3 = axes[0, 3].imshow(depth_diff, cmap='hot')
        axes[0, 3].set_title('Depth Difference (Abs)')
        axes[0, 3].axis('off')
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Row 3: 3D Points (X, Y components and histogram)
        trt_x = trt_pts3d[0, 0].cpu().numpy()
        torch_x = torch_pts3d[0, 0].cpu().numpy()
        trt_y = trt_pts3d[0, 1].cpu().numpy()
        torch_y = torch_pts3d[0, 1].cpu().numpy()
        
        # X component difference
        x_diff = np.abs(trt_x - torch_x)
        im7 = axes[1, 0].imshow(x_diff, cmap='hot')
        axes[1, 0].set_title('X Component Difference')
        axes[1, 0].axis('off')
        plt.colorbar(im7, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Y component difference
        y_diff = np.abs(trt_y - torch_y)
        im8 = axes[1, 1].imshow(y_diff, cmap='hot')
        axes[1, 1].set_title('Y Component Difference')
        axes[1, 1].axis('off')
        plt.colorbar(im8, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Error histograms
        axes[1, 2].hist(depth_diff.flatten(), bins=50, alpha=0.7, color='red', label='Depth')
        # axes[2, 2].hist(conf_diff.flatten(), bins=50, alpha=0.7, color='blue', label='Confidence')
        axes[1, 2].set_xlabel('Absolute Error')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Error Distribution')
        axes[1, 2].legend()
        axes[1, 2].set_yscale('log')
        
        # Scatter plot: TRT vs PyTorch depth values (sampled)
        # Sample every 10th pixel to avoid overcrowding
        sample_mask = np.zeros_like(trt_depth, dtype=bool)
        sample_mask[::10, ::10] = True
        trt_sample = trt_depth[sample_mask]
        torch_sample = torch_depth[sample_mask]
        
        axes[1, 3].scatter(torch_sample, trt_sample, alpha=0.5, s=1)
        min_val = min(torch_sample.min(), trt_sample.min())
        max_val = max(torch_sample.max(), trt_sample.max())
        axes[1, 3].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Match')
        axes[1, 3].set_xlabel('PyTorch Depth')
        axes[1, 3].set_ylabel('TensorRT Depth')
        axes[1, 3].set_title('Depth Correlation')
        axes[1, 3].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/comparison_visualization.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization: {save_dir}/comparison_visualization.png")
        
        # Save individual depth maps as images
        self._save_depth_image(trt_depth, f"{save_dir}/trt_depth.png")
        self._save_depth_image(torch_depth, f"{save_dir}/torch_depth.png")
        self._save_depth_image(depth_diff, f"{save_dir}/depth_difference.png", cmap='hot')
        
        plt.show()
        
    def _save_depth_image(self, depth_array: np.ndarray, filename: str, cmap: str = 'viridis'):
        """Save depth array as image"""
        plt.figure(figsize=(12, 8))
        plt.imshow(depth_array, cmap=cmap)
        plt.colorbar()
        plt.axis('off')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comparison(self, save_dir: str = "comparison_results") -> Dict[str, float]:
        """Run full comparison pipeline"""
        print("=" * 60)
        print("TENSORRT vs PYTORCH MODEL COMPARISON")
        print("=" * 60)
        
        # Run inference on both models
        trt_outputs = self.run_trt_inference()
        torch_outputs = self.run_torch_inference()
        
        # Compute metrics
        metrics = self.compute_metrics(trt_outputs, torch_outputs)
        
        # Print metrics
        print("\n📊 COMPARISON METRICS:")
        print("-" * 40)
        print(f"Depth MAE:            {metrics['depth_mae']:.6f}")
        print(f"Depth Max Error:      {metrics['depth_max_error']:.6f}")
        print(f"Depth MSE:            {metrics['depth_mse']:.6f}")
        print("-" * 40)
        print(f"Confidence MAE:       {metrics['confidence_mae']:.6f}")
        print(f"Confidence Max Error: {metrics['confidence_max_error']:.6f}")
        print("-" * 40)
        print(f"Intrinsics MAE:       {metrics['intrinsics_mae']:.6f}")
        print(f"Intrinsics Max Error: {metrics['intrinsics_max_error']:.6f}")
        
        # Determine if models are close enough
        depth_mae = metrics['depth_mae']
        if depth_mae < 0.001:
            print("\n✅ EXCELLENT: Models are very close (MAE < 0.001)")
        elif depth_mae < 0.01:
            print("\n✅ GOOD: Models are reasonably close (MAE < 0.01)")
        elif depth_mae < 0.1:
            print("\n⚠️  WARNING: Models have noticeable differences (MAE < 0.1)")
        else:
            print("\n❌ ERROR: Models have significant differences (MAE >= 0.1)")
        
        # Create visualizations
        self.visualize_results(trt_outputs, torch_outputs, save_dir)
        
        # Save metrics to file
        metrics_file = f"{save_dir}/metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("TensorRT vs PyTorch Model Comparison\n")
            f.write("=" * 50 + "\n")
            f.write(f"Image: {self.image_path}\n")
            f.write(f"Engine: {self.engine_path}\n")
            f.write(f"Target Size: {self.target_size}\n")
            f.write("\nMetrics:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.8f}\n")
        
        print(f"📁 Results saved to: {save_dir}")
        
        return metrics


def main():
    """Main comparison function"""
    
    # Configure paths
    engine_path = "weights/unidepthv2-336-602.engine" 
    image_path = "frame000000.jpg"        
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please update the image_path variable")
        return
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        print("Please update the engine_path variable")
        return
    
    try:
        # Run comparison
        comparator = ModelComparison(
            engine_path=engine_path,
            image_path=image_path,
            target_size=(336, 602)
        )
        
        metrics = comparator.run_comparison(save_dir="comparison_results")
        
        print("\n🎉 Comparison completed successfully!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()