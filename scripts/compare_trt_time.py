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
import time

from image_preprocessing import UniDepthPreprocessor, postprocess_depth

from torch2trt import TRTModule
import tensorrt as trt


class ModelComparison:
    """Compare TensorRT and PyTorch UniDepth models"""
    
    def __init__(self, 
                 engine_path: str = "unidepthv2-672-1190.engine",
                 image_path: str = "frame000000.jpg",
                 target_size: Tuple[int, int] = (672, 1190)):
        
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

        model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitb14").eval().cuda()

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
        # print("Running TensorRT inference...")
        
        with torch.no_grad():
            pts_3d, confidence, intrinsics = self.trt_model(self.preprocessed_image)
        
        return pts_3d, confidence, intrinsics
    
    def run_torch_inference(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run PyTorch inference"""
        # print("Running PyTorch inference...")
        
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
        
        return pts_3d, confidence, intrinsics
    
    def compute_metrics(self, 
                       trt_outputs: Tuple[torch.Tensor, ...], 
                       torch_outputs: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Compute comparison metrics"""
        
        trt_pts3d, trt_conf, trt_intrinsics = trt_outputs
        torch_pts3d, torch_conf, torch_intrinsics = torch_outputs
        
        metrics = {}
        
        # 3D Points comparison
        pts3d_diff = torch.abs(trt_pts3d - torch_pts3d)
        metrics['pts3d_mae'] = pts3d_diff.mean().item()
        metrics['pts3d_max_error'] = pts3d_diff.max().item()
        metrics['pts3d_mse'] = torch.nn.functional.mse_loss(trt_pts3d, torch_pts3d).item()
        
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
        print(f"3D Points MAE:        {metrics['pts3d_mae']:.6f}")
        print(f"3D Points Max Error:  {metrics['pts3d_max_error']:.6f}")
        print(f"3D Points MSE:        {metrics['pts3d_mse']:.6f}")
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
    
    def benchmark_inference(self, num_runs: int = 100, warmup_runs: int = 10, save_dir: str = "comparison_results"):
        """Benchmark inference time for both models."""
        print("\n" + "=" * 60)
        print("🚀 RUNNING INFERENCE BENCHMARK")
        print("=" * 60)
        print(f"Warmup runs: {warmup_runs}, Timed runs: {num_runs}")

        # --- Warmup ---
        print("Warming up models...")
        for _ in range(warmup_runs):
            _ = self.run_trt_inference()
            _ = self.run_torch_inference()
        torch.cuda.synchronize()

        # --- TensorRT Benchmark ---
        print("Benchmarking TensorRT model...")
        trt_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = self.run_trt_inference()
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            trt_times.append((end_time - start_time) * 1000)

        # --- PyTorch Benchmark ---
        print("Benchmarking PyTorch model...")
        torch_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = self.run_torch_inference()
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            torch_times.append((end_time - start_time) * 1000)

        # --- Calculate and Print Stats ---
        trt_stats = self._calculate_time_stats(trt_times)
        torch_stats = self._calculate_time_stats(torch_times)
        speedup = torch_stats['mean'] / trt_stats['mean'] if trt_stats['mean'] > 0 else float('inf')

        print("\n📊 BENCHMARK RESULTS (ms per inference):")
        print("-" * 65)
        print(f"{'Metric':<15} | {'TensorRT':<22} | {'PyTorch':<22}")
        print("-" * 65)
        print(f"{'Mean':<15} | {trt_stats['mean']:.3f} ms ({trt_stats['fps']:.2f} FPS)   | {torch_stats['mean']:.3f} ms ({torch_stats['fps']:.2f} FPS)")
        print(f"{'Median':<15} | {trt_stats['median']:.3f} ms                  | {torch_stats['median']:.3f} ms")
        print(f"{'Std Dev':<15} | {trt_stats['std']:.3f} ms                  | {torch_stats['std']:.3f} ms")
        print(f"{'Min':<15} | {trt_stats['min']:.3f} ms                  | {torch_stats['min']:.3f} ms")
        print(f"{'Max':<15} | {trt_stats['max']:.3f} ms                  | {torch_stats['max']:.3f} ms")
        print("-" * 65)
        print(f"🚀 Speedup (PyTorch Mean / TensorRT Mean): {speedup:.2f}x")
        print("-" * 65)

        self._visualize_timings(trt_times, torch_times, save_dir)

        return {
            "trt_stats": trt_stats,
            "torch_stats": torch_stats,
            "speedup": speedup
        }

    def _calculate_time_stats(self, timings: list) -> Dict[str, float]:
        """Helper to calculate timing statistics."""
        timings_np = np.array(timings)
        mean_time = np.mean(timings_np)
        return {
            "mean": mean_time,
            "median": np.median(timings_np),
            "std": np.std(timings_np),
            "min": np.min(timings_np),
            "max": np.max(timings_np),
            "fps": 1000.0 / mean_time if mean_time > 0 else 0
        }

    def _visualize_timings(self, trt_times: list, torch_times: list, save_dir: str):
        """Create a plot to visualize inference time distributions."""
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        box = plt.boxplot([trt_times, torch_times], labels=['TensorRT', 'PyTorch'], showfliers=False, patch_artist=True)
        colors = ['#77DD77', '#AEC6CF']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            
        plt.title('Inference Time Distribution', fontsize=16)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        trt_mean = np.mean(trt_times)
        torch_mean = np.mean(torch_times)
        plt.text(1, trt_mean, f'Mean: {trt_mean:.2f} ms', ha='center', va='bottom', color='darkgreen', weight='bold')
        plt.text(2, torch_mean, f'Mean: {torch_mean:.2f} ms', ha='center', va='bottom', color='darkblue', weight='bold')

        save_path = f"{save_dir}/benchmark_timing_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved timing visualization: {save_path}")
        plt.show()

    def run_comparison(self, save_dir: str = "comparison_results", benchmark_runs: int = 100):
        """Run full comparison pipeline including accuracy and performance."""
        print("=" * 60)
        print("TENSORRT vs PYTORCH MODEL COMPARISON")
        print("=" * 60)
        
        # Run inference on both models
        trt_outputs = self.run_trt_inference()
        torch_outputs = self.run_torch_inference()
        
        # Compute accuracy metrics
        metrics = self.compute_metrics(trt_outputs, torch_outputs)
        
        print("\n📊 ACCURACY METRICS:")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title():<22}: {value:.6f}")
        
        depth_mae = metrics['depth_mae']
        if depth_mae < 0.001:
            print("\n✅ ACCURACY: EXCELLENT - Models are numerically very close.")
        elif depth_mae < 0.01:
            print("\n✅ ACCURACY: GOOD - Models are reasonably close.")
        else:
            print("\n❌ ACCURACY: WARNING - Models have significant differences.")

        # Run performance benchmarking
        benchmark_results = {}
        if benchmark_runs > 0:
            benchmark_results = self.benchmark_inference(num_runs=benchmark_runs, save_dir=save_dir)
        
        # Save summary report to file
        os.makedirs(save_dir, exist_ok=True)
        summary_file = f"{save_dir}/results_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("TensorRT vs PyTorch Model Comparison Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Image: {self.image_path}\n")
            f.write(f"Engine: {self.engine_path}\n\n")
            f.write("Accuracy Metrics:\n")
            for key, value in metrics.items():
                f.write(f"- {key}: {value:.8f}\n")
            
            if benchmark_results:
                f.write("\n" + "=" * 50 + "\n")
                f.write("Inference Benchmark Results\n")
                f.write("=" * 50 + "\n")
                stats = benchmark_results['trt_stats']
                f.write(f"- TensorRT Mean Time: {stats['mean']:.3f} ms ({stats['fps']:.2f} FPS)\n")
                stats = benchmark_results['torch_stats']
                f.write(f"- PyTorch Mean Time:  {stats['mean']:.3f} ms ({stats['fps']:.2f} FPS)\n")
                f.write(f"- Speedup:            {benchmark_results['speedup']:.2f}x\n")
        
        print(f"\n📁 All results saved to: {save_dir}")
        return metrics, benchmark_results


def main():
    """Main comparison function."""
    
    engine_path = "unidepthv2-672-1190-op17.engine" 
    image_path = "frame000000.jpg"        
    
    if not all(os.path.exists(p) for p in [image_path, engine_path]):
        print(f"❌ Error: Ensure '{image_path}' and '{engine_path}' exist.")
        return
    
    try:
        comparator = ModelComparison(
            engine_path=engine_path,
            image_path=image_path,
            target_size=(672, 1190)
        )
        
        comparator.run_comparison(
            save_dir="comparison_results",
            benchmark_runs=100
        )
        
        print("\n🎉 Comparison completed successfully!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()