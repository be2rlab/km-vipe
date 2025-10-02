import torch
import numpy as np
import cv2

from lightglue import LightGlue, SuperPoint, match_pair, viz2d
from lightglue.utils import rbd, load_image
from vipe.streams.base import VideoFrame
from . import SparseTracks
import matplotlib


class LightGlueTracks(SparseTracks):
    def __init__(
        self,
        n_views: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(n_views)
        self.device = device
        self.extractor = None
        self.matcher = None

        # tracker
        self.tracks = {}  # Stores all active tracks {track_id: track_data}
        self.track_id_counter = 0  # To generate unique track IDs
        self.prev_kp_idx_to_track_id = {} # Maps prev frame's keypoint index to a track_id

        # bookkeeping
        self.frame_idx = 0
        self.prev_image = None
        self.prev_feats = None
        


    def _build_model(self):
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

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
        curr_feats = self.extractor.extract(curr_image)
        # print(f"current global frame index: {current_frame.raw_frame_idx}")
        if self.frame_idx == 0:
            kpts_curr = curr_feats['keypoints'].cpu().numpy()[0] # [N, 2]
            for i in range(len(kpts_curr)):
                new_id = self.track_id_counter
                # Create a new track with its first observation
                self.tracks[new_id] = {'last_seen': self.frame_idx, 'points': [kpts_curr[i]]}
                # Map the keypoint's index to this new track ID
                self.prev_kp_idx_to_track_id[i] = new_id
                # Record the observation for this frame
                current_frame_observations[new_id] = kpts_curr[i]
                self.track_id_counter += 1

        else:
            matches = self.matcher({'image0': self.prev_feats, 'image1': curr_feats})
            
            # Use rbd to remove the batch dimension from the outputs
            all_kpts_prev, all_kpts_curr, matches01 = rbd(self.prev_feats), rbd(curr_feats), rbd(matches)
            all_kpts_prev = all_kpts_prev['keypoints']
            all_kpts_curr = all_kpts_curr['keypoints']
            matches0 = matches01['matches'][..., 0]
            matches1 = matches01['matches'][..., 1]
            kpts_prev = all_kpts_prev[matches0]
            kpts_curr = all_kpts_curr[matches1]
            # print(f"Number of matches: {matches01['matches'].shape}")
            # print(f"Number of keypoints in previous frame: {kpts_prev.shape}")
            # print(f"Number of keypoints in current frame: {kpts_curr.shape}")
            
            # Get the actual match indices and corresponding keypoints
            kp_indices_prev = matches0
            kp_indices_curr = matches1
            # print(f"kp_indices_prev: {kp_indices_prev}")
            # print(f"kp_indices_curr: {kp_indices_curr}")

            # This will map the CURRENT frame's keypoint indices to track IDs
            curr_kp_idx_to_track_id = {}

            # Extend existing tracks
            for matched_prev_kp_id, matched_curr_kp_id in zip(kp_indices_prev, kp_indices_curr): # loop over pairs of matched kps

                if matched_prev_kp_id in self.prev_kp_idx_to_track_id:
                    # retrive id in tracker
                    track_id = self.prev_kp_idx_to_track_id[matched_prev_kp_id]
                    
                    # retrive current keypoint representation
                    point = all_kpts_curr[matched_curr_kp_id].cpu().numpy()
                    
                    self.tracks[track_id]['points'].append(point)
                    self.tracks[track_id]['last_seen'] = self.frame_idx
                    
                    curr_kp_idx_to_track_id[matched_curr_kp_id] = track_id
                    current_frame_observations[track_id] = point
                if matched_prev_kp_id in self.prev_kp_idx_to_track_id:
                    track_id = self.prev_kp_idx_to_track_id[matched_prev_kp_id]
                    
                    # Add the new point to the track
                    point = all_kpts_curr[matched_curr_kp_id].cpu().numpy()
                    self.tracks[track_id]['points'].append(point)
                    self.tracks[track_id]['last_seen'] = self.frame_idx
                    
                    # Update the mapping for the next frame
                    curr_kp_idx_to_track_id[matched_curr_kp_id] = track_id
                    
                    # Record the observation for this frame
                    current_frame_observations[track_id] = point

            # Initialize new tracks for unmatched keypoints in the current frame
            all_curr_indices = set(range(len(all_kpts_curr)))
            matched_curr_indices = set(kp_indices_curr.cpu().numpy())
            unmatched_curr_indices = all_curr_indices - matched_curr_indices

            for idx in range(len(all_kpts_curr)):
                if idx in unmatched_curr_indices:
                    new_id = self.track_id_counter
                    point = all_kpts_curr[idx].cpu().numpy()

                    self.tracks[new_id] = {'last_seen': self.frame_idx, 'points': [point]}
                    curr_kp_idx_to_track_id[idx] = new_id
                    
                    current_frame_observations[new_id] = point
                    self.track_id_counter += 1

            # Prune old tracks
            # A track is pruned if it hasn't been seen for a few frames
            lost_thresh = 3 
            lost_ids = []
            for track_id, data in self.tracks.items():
                if self.frame_idx - data['last_seen'] > lost_thresh:
                    lost_ids.append(track_id)
            
            for track_id in lost_ids:
                del self.tracks[track_id]

            # Update the previous-to-current mapping for the next iteration
            self.prev_kp_idx_to_track_id = curr_kp_idx_to_track_id

            prev_rgb = (self.prev_image.cpu().numpy() * 255).astype(np.uint8)
            curr_rgb = (curr_frame.rgb.cpu().numpy() * 255).astype(np.uint8) 
            m_kpts0 = kpts_prev.cpu().numpy()
            m_kpts1 = kpts_curr.cpu().numpy()

            viz2d.plot_images([prev_rgb, curr_rgb])
            viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
            viz2d.save_plot(path = f"tracker_viz/matches_{self.frame_idx}.png")
            matplotlib.pyplot.close()

        # Update state for the next frame
        self.prev_feats = curr_feats
        self.prev_image = curr_frame.rgb
        
        # Add the current frame's observations to the list
        self.observations[0].append(current_frame_observations)
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



# orig_image_np_rgb = current_image.cpu().numpy()
# orig_image_np_rgb = (orig_image_np_rgb * 255).astype(np.uint8)
# orig_image_np_bgr = cv2.cvtColor(orig_image_np_rgb, cv2.COLOR_RGB2BGR)
# gray_image_np = cv2.cvtColor(orig_image_np_bgr, cv2.COLOR_BGR2GRAY)
# grayname = "gray_image.png"
# origname = "orig_image.png"
# cv2.imwrite(grayname, gray_image_np)
# cv2.imwrite(origname, orig_image_np_bgr)