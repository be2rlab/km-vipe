import torch
from torch import Tensor
import numpy as np
import cv2
from dataclasses import dataclass

from lightglue import LightGlue, SuperPoint, match_pair, viz2d
from lightglue.utils import rbd, load_image
from vipe.streams.base import VideoFrame
from . import SparseTracks
import matplotlib
matplotlib.use("agg")


@dataclass
class Track:
    kp_xys: Tensor # [2]
    kp_feats: Tensor # [dim]
    last_seen: int
class LightGlueTracks(SparseTracks):
    def __init__(
        self,
        n_views: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        visualization = False,
        ransac_threshold: float = 1.0,
        ransac_confidence: float = 0.99,
        ransac_max_iters: int = 2000,
        min_inliers: int = 10,
        max_track_age = 10
    ):
        super().__init__(n_views)
        self.device = device
        self.extractor = None
        self.matcher = None
        self.visualization = visualization

        # RANSAC parameters
        self.ransac_threshold = ransac_threshold  # pixel threshold for inliers
        self.ransac_confidence = ransac_confidence 
        self.ransac_max_iters = ransac_max_iters  
        self.min_inliers = min_inliers

        # tracker
        self.tracks: dict[int: Track] = {}  # Stores all active tracks {track_id: track_data}
        self.track_id_counter = 0  # To generate unique track IDs
        self.active_track_ids: list[int] = []
        self.max_track_age = max_track_age

        # bookkeeping
        self.frame_idx = 0
        self.prev_image = None

    def _build_model(self):
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def _remove_stale_tracks(self):
        """Remove tracks that haven't been seen for too long."""
        tracks_to_remove = []
        for track_id in self.active_track_ids:
            if self.frame_idx - self.tracks[track_id].last_seen > self.max_track_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.active_track_ids.remove(track_id)
            del self.tracks[track_id]
        
        if tracks_to_remove:
            print(f"Frame {self.frame_idx}: Removed {len(tracks_to_remove)} stale tracks")

    def track_image(self, frame_data_list: list[VideoFrame]):
        assert len(frame_data_list) == 1, (
            "Only single-camera supported for now. Mainly due to rig transformations not properly set."
        )
        if self.matcher is None or self.extractor is None:
            self._build_model()
        assert self.matcher is not None, "LightGlue matcher not initialized"
        assert self.extractor is not None, "SuperPoint extractor not initialized"

        current_frame_observations = {}
    
        curr_frame = frame_data_list[0] # len(frame_data_list) == 1
        curr_image = self._to_grayscale(curr_frame.rgb).unsqueeze(0) # [1, H, W] grayscale
        # Extract features from the current frame
        feats_curr = self.extractor.extract(curr_image)
        kpts_curr = feats_curr['keypoints'][0]  # [N, 2]
        desc_curr = feats_curr['descriptors'][0]  # [N, dim]
        
        if self.frame_idx == 0:
            # --- First Frame: Initialize all keypoints as new tracks ---
            for i in range(len(kpts_curr)):
                new_id = self.track_id_counter
                kp_xys = kpts_curr[i]
                kp_feats = desc_curr[i]

                # Create and store the new track
                track = Track(kp_xys, kp_feats, last_seen=self.frame_idx)
                self.tracks[new_id] = track
                self.active_track_ids.append(new_id)
                
                # Record the observation for this frame
                current_frame_observations[new_id] = kp_xys.cpu().numpy()
                self.track_id_counter += 1

        else:
            self._remove_stale_tracks()
            active_track_ids_copy = self.active_track_ids.copy()
            if not active_track_ids_copy:
                print(f"Frame {self.frame_idx}: No active tracks to match.")
                # Treat as a new initialization if all tracks were lost
                kpts_prev = torch.empty((0, 2), device=self.device)
                desc_prev = torch.empty((0, desc_curr.shape[1]), device=self.device)
            else:
                kpts_prev = torch.stack([self.tracks[tid].kp_xys for tid in active_track_ids_copy])
                desc_prev = torch.stack([self.tracks[tid].kp_feats for tid in active_track_ids_copy])
            # Match current frame with all active features
            data0 = {'keypoints': kpts_prev.unsqueeze(0), 'descriptors': desc_prev.unsqueeze(0)}
            data1 = {'keypoints': kpts_curr.unsqueeze(0), 'descriptors': desc_curr.unsqueeze(0)}
                
            # Match current frame features with previous active tracks
            matches = self.matcher({'image0': data0, 'image1': data1})

            # print(matches.keys())
            
            # Use rbd to remove the batch dimension from the outputs
            matches01 = rbd(matches)
            matches0 = matches01['matches'][..., 0]  # Indices in active_track_ids_copy
            matches1 = matches01['matches'][..., 1]  # Indices in kpts_curr

            # Apply RANSAC
            inlier_mask, model = self._ransac_filter(
                kpts_prev=kpts_prev,   # torch.Tensor [N_prev,2]
                kpts_curr=kpts_curr,   # torch.Tensor [N_curr,2]
                matches0=matches0,
                matches1=matches1,
            )
            
            print(f"Frame {self.frame_idx}: Found {len(matches0)} "
                  f"matches between {len(kpts_prev)} active tracks and {len(kpts_curr)} current keypoints, "
                  f"with {inlier_mask.sum()} inliers")
            
            # Process matched keypoints - extend existing tracks
            matched_curr_indices = set()
            for i in range(len(matches0)):
                if not inlier_mask[i]:
                    continue
                prev_idx = matches0[i].item()
                curr_idx = matches1[i].item()
                
                track_id = active_track_ids_copy[prev_idx]
                track = self.tracks[track_id]
                
                # Update track with new observation
                track.kp_xys = kpts_curr[curr_idx]
                track.kp_feats = desc_curr[curr_idx]
                track.last_seen = self.frame_idx
                
                # Add to observations for this frame
                current_frame_observations[track_id] = kpts_curr[curr_idx].cpu().numpy()
                matched_curr_indices.add(curr_idx)

            # Initialize new tracks for unmatched keypoints in the current frame
            all_curr_indices = set(range(len(kpts_curr)))
            unmatched_curr_indices = all_curr_indices - matched_curr_indices

            for idx in unmatched_curr_indices:
                new_id = self.track_id_counter
                track = Track(
                    kp_xys=kpts_curr[idx],
                    kp_feats=desc_curr[idx],
                    last_seen=self.frame_idx,
                )
                self.tracks[new_id] = track
                self.active_track_ids.append(new_id)
                current_frame_observations[new_id] = kpts_curr[idx].cpu().numpy()
                self.track_id_counter += 1

            print(f"Frame {self.frame_idx}: Extended {len(matched_curr_indices)} tracks, "
                  f"created {len(unmatched_curr_indices)} new tracks.")

            # Visualization
            if self.visualization and self.prev_image is not None and len(matches0) > 0:
                prev_rgb = (self.prev_image.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                curr_rgb = (curr_frame.rgb.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                
                # Get matched keypoints for plotting
                kpts_prev_matched = kpts_prev[matches0].cpu().numpy()
                kpts_curr_matched = kpts_curr[matches1].cpu().numpy()

                viz2d.plot_images([prev_rgb, curr_rgb])
                viz2d.plot_matches(kpts_prev_matched, kpts_curr_matched, color="lime", lw=0.2)
                viz2d.save_plot(path=f"tracker_viz/matches_{self.frame_idx}.png")
                matplotlib.pyplot.close()

        # Update state for the next frame
        self.prev_image = curr_frame.rgb  # Save for next frame's visualization
        
        # Add the current frame's observations to the main list
        self.observations[0].append(current_frame_observations)
        
        print(f"Frame {self.frame_idx}: {len(self.active_track_ids)} total active tracks")
        
        self.frame_idx += 1


    def _to_grayscale(self, image: torch.Tensor):
        '''
        Convert an RGB image tensor to grayscale.
        
        Args:
            image (torch.Tensor): Input image tensor of shape [H, W, C] (C=3 for RGB).
        
        Returns:
            torch.Tensor: Grayscale image tensor of shape [H, W].
        '''
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Input tensor must have shape [H, W, 3] for an RGB image.")
        
        # Standard luminance-preserving grayscale conversion coefficients
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        
        # Apply dot product across channels
        image_gray = torch.tensordot(image, weights, dims=([-1], [0]))
        
        return image_gray
    
    def _ransac_filter(
        self,
        kpts_prev,
        kpts_curr,
        matches0,
        matches1,
    ):
        """
        Geometrically verify matches with RANSAC.

        Inputs:
        kpts_prev : torch.Tensor or np.ndarray, shape [N_prev, 2]
        kpts_curr : torch.Tensor or np.ndarray, shape [N_curr, 2]
        matches0  : 1D torch.Tensor or np.ndarray, length M, indices into kpts_prev (or -1)
        matches1  : 1D torch.Tensor or np.ndarray, length M, indices into kpts_curr (or -1)

        Returns:
        inlier_mask_full : np.ndarray(bool), shape (M,)
            Boolean mask over the input matches arrays: True for inlier matches.
        model : np.ndarray or None
            The estimated homography (3x3) or fundamental matrix (3x3), or None if estimation failed.
        """

        # Helper to convert tensors to numpy
        def _to_np(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.array(x)

        kpts_prev_np = _to_np(kpts_prev).astype(np.float32)
        kpts_curr_np = _to_np(kpts_curr).astype(np.float32)
        matches0_np = _to_np(matches0).ravel().astype(np.int32)
        matches1_np = _to_np(matches1).ravel().astype(np.int32)

        M = matches0_np.shape[0]
        inlier_mask_full = np.zeros((M,), dtype=np.bool_)

        # Filter out invalid matches (-1)
        valid_mask = (matches0_np >= 0) & (matches1_np >= 0)
        if valid_mask.sum() < 4:
            # Not enough correspondences for homography; return all False
            return inlier_mask_full, None

        matched_prev_idx = matches0_np[valid_mask]
        matched_curr_idx = matches1_np[valid_mask]

        pts_prev = kpts_prev_np[matched_prev_idx]  # (K,2)
        pts_curr = kpts_curr_np[matched_curr_idx]  # (K,2)

        # Try homography estimation first (suitable for mostly planar / camera motion)
        model = None
        inliers_valid = None
        try:
            # Prefer passing confidence/maxIters if available in this OpenCV build.
            # Fall back to the simpler signature if that raises.
            try:
                H, mask = cv2.findHomography(
                    pts_prev,
                    pts_curr,
                    cv2.RANSAC,
                    float(self.ransac_threshold),
                    None,
                    int(self.ransac_max_iters),
                    float(self.ransac_confidence),
                )
            except TypeError:
                # older OpenCV builds may not accept maxIters/confidence as kwargs
                H, mask = cv2.findHomography(
                    pts_prev,
                    pts_curr,
                    cv2.RANSAC,
                    float(self.ransac_threshold),
                )

            if mask is not None:
                inliers_valid = mask.ravel().astype(bool)
                model = H
        except Exception:
            # homography estimation failed; we'll try fundamental matrix below
            inliers_valid = None
            model = None

        # If homography produced too few inliers (or failed), try fundamental matrix as fallback
        if inliers_valid is None or inliers_valid.sum() < self.min_inliers:
            try:
                # findFundamentalMat returns (F, mask)
                F, maskF = cv2.findFundamentalMat(
                    pts_prev, pts_curr, cv2.FM_RANSAC, float(self.ransac_threshold),
                )
                if maskF is not None:
                    inliers_valid = maskF.ravel().astype(bool)
                    model = F
            except Exception:
                inliers_valid = None
                model = None

        # If still no inliers, return empty mask
        if inliers_valid is None:
            return inlier_mask_full, model

        # Map the valid inlier_mask back to the full matches array
        inlier_mask_full[valid_mask] = inliers_valid.astype(bool)

        return inlier_mask_full, model




# orig_image_np_rgb = current_image.cpu().numpy()
# orig_image_np_rgb = (orig_image_np_rgb * 255).astype(np.uint8)
# orig_image_np_bgr = cv2.cvtColor(orig_image_np_rgb, cv2.COLOR_RGB2BGR)
# gray_image_np = cv2.cvtColor(orig_image_np_bgr, cv2.COLOR_BGR2GRAY)
# grayname = "gray_image.png"
# origname = "orig_image.png"
# cv2.imwrite(grayname, gray_image_np)
# cv2.imwrite(origname, orig_image_np_bgr)