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
        visualization = False,
    ):
        super().__init__(n_views)
        self.device = device
        self.extractor = None
        self.matcher = None
        self.visualization = visualization

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
        
        if self.frame_idx == 0:
            # Initialize tracks for the first frame
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
            print(f"Frame {self.frame_idx}: Initialized {len(kpts_curr)} tracks")

        else:
            # Match current frame with previous frame
            matches = self.matcher({'image0': self.prev_feats, 'image1': curr_feats})
            
            # Use rbd to remove the batch dimension from the outputs
            all_kpts_prev, all_kpts_curr, matches01 = rbd(self.prev_feats), rbd(curr_feats), rbd(matches)
            all_kpts_prev = all_kpts_prev['keypoints']
            all_kpts_curr = all_kpts_curr['keypoints']
            matches0 = matches01['matches'][..., 0]  # indices in prev frame
            matches1 = matches01['matches'][..., 1]  # indices in curr frame
            
            print(f"Frame {self.frame_idx}: Found {len(matches0)} matches between {len(all_kpts_prev)} and {len(all_kpts_curr)} keypoints")
            
            # This will map the current frame's keypoint indices to track IDs
            curr_kp_idx_to_track_id = {}
            
            # Process matched keypoints - extend existing tracks
            for matched_prev_idx, matched_curr_idx in zip(matches0, matches1):
                matched_prev_idx = int(matched_prev_idx.cpu().numpy())
                matched_curr_idx = int(matched_curr_idx.cpu().numpy())
                
                if matched_prev_idx in self.prev_kp_idx_to_track_id:
                    # This matched keypoint belongs to an existing track
                    track_id = self.prev_kp_idx_to_track_id[matched_prev_idx]
                    
                    # Get the current keypoint position
                    point = all_kpts_curr[matched_curr_idx].cpu().numpy()
                    
                    # Update the track
                    self.tracks[track_id]['points'].append(point)
                    self.tracks[track_id]['last_seen'] = self.frame_idx
                    
                    # Map current keypoint index to track ID for next frame
                    curr_kp_idx_to_track_id[matched_curr_idx] = track_id
                    
                    # Record observation for this frame
                    current_frame_observations[track_id] = point

            # Initialize new tracks for unmatched keypoints in the current frame
            matched_curr_indices = set(int(idx.cpu().numpy()) for idx in matches1)
            all_curr_indices = set(range(len(all_kpts_curr)))
            unmatched_curr_indices = all_curr_indices - matched_curr_indices

            print(f"Frame {self.frame_idx}: Creating {len(unmatched_curr_indices)} new tracks for unmatched keypoints")
            
            for idx in unmatched_curr_indices:
                new_id = self.track_id_counter
                point = all_kpts_curr[idx].cpu().numpy()

                # Create new track
                self.tracks[new_id] = {'last_seen': self.frame_idx, 'points': [point]}
                curr_kp_idx_to_track_id[idx] = new_id
                current_frame_observations[new_id] = point
                self.track_id_counter += 1

            # Prune old tracks
            lost_thresh = 10
            lost_ids = []
            for track_id, data in self.tracks.items():
                if self.frame_idx - data['last_seen'] > lost_thresh:
                    lost_ids.append(track_id)
            
            if lost_ids:
                print(f"Frame {self.frame_idx}: Pruning {len(lost_ids)} old tracks")
            
            for track_id in lost_ids:
                del self.tracks[track_id]

            # Update the previous-to-current mapping for the next iteration
            self.prev_kp_idx_to_track_id = curr_kp_idx_to_track_id

            # Visualization code
            if self.visualization:
                prev_rgb = (self.prev_image.cpu().numpy() * 255).astype(np.uint8)
                curr_rgb = (curr_frame.rgb.cpu().numpy() * 255).astype(np.uint8) 
                
                # Get matched keypoints for visualization
                kpts_prev = all_kpts_prev[matches0].cpu().numpy()
                kpts_curr = all_kpts_curr[matches1].cpu().numpy()

                viz2d.plot_images([prev_rgb, curr_rgb])
                viz2d.plot_matches(kpts_prev, kpts_curr, color="lime", lw=0.2)
                viz2d.save_plot(path = f"tracker_viz/matches_{self.frame_idx}.png")
                matplotlib.pyplot.close()

        # Update state for the next frame
        self.prev_feats = curr_feats
        self.prev_image = curr_frame.rgb
        
        # Add the current frame's observations to the list
        self.observations[0].append(current_frame_observations)
        
        # Debug information
        print(f"Frame {self.frame_idx}: {len(current_frame_observations)} observations")
        print(f"Frame {self.frame_idx}: {len(self.tracks)} total active tracks")
        # print(f"Frame {self.frame_idx}: Track IDs in current frame: {sorted(current_frame_observations.keys())}")
        
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