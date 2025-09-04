import torch as th
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use("dark_background")

from base_matcher import BaseMatcher
from copy import deepcopy
from EfficientLoFTR.src.utils.plotting import (
    make_matching_figure,
)
from EfficientLoFTR.src.loftr import (
    LoFTR,
    full_default_cfg,
    reparameter,
)
import time


class LoftrMatcher(BaseMatcher):

    def __init__(
        self,
        weights_path: str = "./weights/eloftr_outdoor.ckpt",
        return_vis: bool = False,
        use_magsac: bool = True,
        magsac_threshold: float = 1.0,
        magsac_confidence: float = 0.99,
        magsac_max_iters: int = 20000,
        estimation_method: str = "fundamental",  # "fundamental" or "homography"
        device: str = "cuda",  # Add device parameter,
    ) -> None:
        super().__init__()
        self.wpath = weights_path
        self.rvis = return_vis
        self.use_magsac = use_magsac
        self.magsac_threshold = magsac_threshold
        self.magsac_confidence = magsac_confidence
        self.magsac_max_iters = magsac_max_iters
        self.estimation_method = estimation_method

        # Set device with fallback to CPU
        self.device = device if th.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        _default_cfg = deepcopy(full_default_cfg)
        self.model = LoFTR(config=_default_cfg)

        # Load pretrained weights
        checkpoint = th.load(self.wpath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = reparameter(self.model)  # Essential for good performance
        self.model = self.model.eval().to(self.device)  # Move to GPU/device
        print(f"Model loaded on {self.device}")

    def _filter_matches_magsac(self, pts1, pts2):
        """
        Filter matches using MAGSAC robust estimator.

        Args:
            pts1: numpy array of shape (N, 2) - points in first image
            pts2: numpy array of shape (N, 2) - points in second image

        Returns:
            filtered_pts1: numpy array of inlier points in first image
            filtered_pts2: numpy array of inlier points in second image
            inlier_mask: boolean mask indicating which matches are inliers
        """
        if len(pts1) < 8:  # Need at least 8 points for fundamental matrix
            print(f"Warning: Only {len(pts1)} matches found, skipping MAGSAC filtering")
            return pts1, pts2, np.ones(len(pts1), dtype=bool)

        try:
            if self.estimation_method == "fundamental":
                # Use MAGSAC for fundamental matrix estimation
                _, inlier_mask = cv2.findFundamentalMat(
                    pts1,
                    pts2,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=self.magsac_threshold,
                    confidence=self.magsac_confidence,
                    maxIters=self.magsac_max_iters,
                )
            else:
                raise ValueError(f"Unknown estimation method: {self.estimation_method}")

            if inlier_mask is None:
                print("Warning: MAGSAC failed to find a model, keeping all matches")
                return pts1, pts2, np.ones(len(pts1), dtype=bool)

            # Convert to boolean mask if needed
            inlier_mask = inlier_mask.ravel().astype(bool)

            filtered_pts1 = pts1[inlier_mask]
            filtered_pts2 = pts2[inlier_mask]

            print(
                f"MAGSAC filtering: {len(pts1)} -> {len(filtered_pts1)} matches "
                f"({len(filtered_pts1)/len(pts1)*100:.1f}% inliers)"
            )

            return filtered_pts1, filtered_pts2, inlier_mask

        except Exception as e:
            print(f"Warning: MAGSAC filtering failed with error: {e}")
            print("Keeping all matches without filtering")
            return pts1, pts2, np.ones(len(pts1), dtype=bool)

    def match(self, data_item: dict) -> dict:
        """
        Match keypoints between agent and ego images.

        Args:
            data_item: {
                "agent": Union[th.Tensor[C, H, W], str],  # agent image
                "ego": Union[th.Tensor[C, H, W], str]     # ego image
            }

        Returns:
            {
                "keypoints_agent": List[Tuple[float, float]],  # matched points in agent image
                "keypoints_ego": List[Tuple[float, float]],    # matched points in ego image
                "num_raw_matches": int,                        # number of matches before filtering
                "num_filtered_matches": int,                   # number of matches after filtering
                "inlier_ratio": float                          # ratio of inliers to total matches
            }
        """
        try:
            # Convert input format from ORB style to LoFTR style
            item0 = data_item["agent"]
            item1 = data_item["ego"]
            loftr_input = {"view1_img": item0, "view2_img": item1}

            imgs = []
            imgs_raw = []
            for view in loftr_input:
                img = loftr_input[view]
                if isinstance(img, str):
                    img_raw = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(
                        img_raw,
                        (img_raw.shape[1] // 32 * 32, img_raw.shape[0] // 32 * 32),
                    )
                    img = th.from_numpy(img)[None][None] / 255.0
                    imgs_raw.append(img_raw)

                elif isinstance(img, th.Tensor):
                    # Handle tensor input - ensure it's grayscale and properly formatted
                    if img.dim() == 4:
                        img = img.squeeze(0)
                    if img.shape[0] == 3:  # RGB to grayscale
                        img_np = (
                            img.cpu().numpy().transpose(1, 2, 0)
                        )  # Move to CPU first
                        img_np = (img_np * 255).astype("uint8")
                        img_raw = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    else:
                        img_raw = (img.squeeze().cpu().numpy() * 255).astype("uint8")

                    # Resize to be divisible by 32
                    img_raw = cv2.resize(
                        img_raw,
                        (img_raw.shape[1] // 32 * 32, img_raw.shape[0] // 32 * 32),
                    )
                    img = th.from_numpy(img_raw)[None][None] / 255.0
                    imgs_raw.append(img_raw)

                else:
                    raise ValueError(
                        f"Provided data_item must contain th.Tensor or str input -- you provided {type(img)}"
                    )
                imgs.append(img)

            # Create batch and move to device
            batch = {
                f"image{idx}": img.to(self.device) for (idx, img) in enumerate(imgs)
            }

            # GPU inference
            with th.no_grad():
                start = time.time()
                self.model(batch)
                finish = time.time()
                times = finish - start

            # Convert output format to match ORB matcher
            # Move results back to CPU for numpy operations
            if "mkpts0_f" in batch and "mkpts1_f" in batch:
                # Get raw matches (move to CPU)
                pts_agent_raw = batch["mkpts0_f"].cpu().numpy()
                pts_ego_raw = batch["mkpts1_f"].cpu().numpy()

                num_raw_matches = len(pts_agent_raw)

                # Apply MAGSAC filtering if enabled and we have enough matches
                if self.use_magsac and num_raw_matches > 0:
                    pts_agent_filtered, pts_ego_filtered, inlier_mask = (
                        self._filter_matches_magsac(pts_agent_raw, pts_ego_raw)
                    )

                    # Convert to list of tuples
                    pts_agent = [
                        (float(pt[0]), float(pt[1])) for pt in pts_agent_filtered
                    ]
                    pts_ego = [(float(pt[0]), float(pt[1])) for pt in pts_ego_filtered]

                    num_filtered_matches = len(pts_agent)
                    inlier_ratio = (
                        num_filtered_matches / num_raw_matches
                        if num_raw_matches > 0
                        else 0.0
                    )

                    # Store filtered data for visualization
                    if self.rvis and len(pts_agent) > 0:
                        # Filter confidence scores as well (move to CPU)
                        filtered_conf = (
                            batch["mconf"].cpu().numpy()[inlier_mask]
                            if len(inlier_mask) == len(batch["mconf"])
                            else batch["mconf"].cpu().numpy()
                        )
                        self._filtered_pts_agent = pts_agent_filtered
                        self._filtered_pts_ego = pts_ego_filtered
                        self._filtered_conf = filtered_conf


                else:
                    # No filtering
                    pts_agent = [(float(pt[0]), float(pt[1])) for pt in pts_agent_raw]
                    pts_ego = [(float(pt[0]), float(pt[1])) for pt in pts_ego_raw]
                    num_filtered_matches = num_raw_matches
                    inlier_ratio = 1.0

                    # Store unfiltered data for visualization
                    if self.rvis and len(pts_agent) > 0:
                        self._filtered_pts_agent = pts_agent_raw
                        self._filtered_pts_ego = pts_ego_raw
                        self._filtered_conf = batch["mconf"].cpu().numpy()

            else:
                # No matches found
                pts_agent = []
                pts_ego = []
                num_raw_matches = 0
                num_filtered_matches = 0
                inlier_ratio = 0.0

            result = {
                "keypoints_agent": pts_agent,
                "keypoints_ego": pts_ego,
                "num_raw_matches": num_raw_matches,
                "num_filtered_matches": num_filtered_matches,
                "inlier_ratio": inlier_ratio,
                "times":times,
            }

            # Store visualization data if requested (can be accessed separately)
            if self.rvis and len(pts_agent) > 0:
                color = cm.jet(self._filtered_conf)
                self._last_fig = make_matching_figure(
                    imgs_raw[0],
                    imgs_raw[1],
                    self._filtered_pts_agent,
                    self._filtered_pts_ego,
                    color,
                    text=f"Matches: {num_filtered_matches}/{num_raw_matches} ({inlier_ratio:.2f})",
                )

            return result

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"GPU out of memory: {e}")
                print("Clearing GPU cache...")
                th.cuda.empty_cache()
                # You could implement fallback to CPU here if needed
                raise RuntimeError(
                    "GPU out of memory. Try reducing image size or use CPU."
                )
            else:
                raise e

    def get_last_visualization(self):
        """Get the last generated visualization figure (if return_vis=True)"""
        if hasattr(self, "_last_fig"):
            return self._last_fig
        return None

    def set_magsac_params(
        self, threshold=None, confidence=None, max_iters=None, method=None
    ):
        """Update MAGSAC parameters"""
        if threshold is not None:
            self.magsac_threshold = threshold
        if confidence is not None:
            self.magsac_confidence = confidence
        if max_iters is not None:
            self.magsac_max_iters = max_iters
        if method is not None:
            if method not in ["fundamental", "homography"]:
                raise ValueError("method must be 'fundamental' or 'homography'")
            self.estimation_method = method

    def to_device(self, device):
        """Move model to different device"""
        self.device = device
        self.model = self.model.to(device)
        print(f"Model moved to {device}")

    def get_gpu_memory_info(self):
        """Get GPU memory usage info"""
        if th.cuda.is_available() and self.device.startswith("cuda"):
            allocated = th.cuda.memory_allocated(self.device) / 1024**3
            cached = th.cuda.memory_reserved(self.device) / 1024**3
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        return "Not using GPU"
