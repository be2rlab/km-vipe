from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np

from vipe.streams.base import VideoFrame
from . import SparseTracks

class LightGlueTracks(SparseTracks):
    def __init__(
        self,
        n_views: int,
        model_id: str = "ETH-CVG/lightglue_superpoint", 
        device: str = "cuda"
    ):
        super().__init__(n_views)
        # config
        self.processor = None
        self.tracker = None
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Internal state management
        self.frame_idx = 0
        self.next_id = 0
        
        # Buffer to store the previous frame for pairwise matching
        self.prev_frame: VideoFrame | None = None

    def _build_model(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.tracker = AutoModel.from_pretrained(self.model_id).to(self.device)

    def track_image(self, frame_data_list: list[VideoFrame]):
        if self.tracker is None:
            self._build_model()
        assert self.tracker is not None, "LightGlue Tracker not initialized"

        # The framework passes a list with one new frame per call.
        assert len(frame_data_list) == 1, "LightGlueTracks expects one frame at a time."
        curr_frame = frame_data_list[0]

        # Add a new, empty dictionary for the current frame's observations.
        self.observations[0].append({})

        # Tracking can only begin from the second frame onwards.
        if self.prev_frame is not None:
            prev_obs = self.observations[0][self.frame_idx - 1]
            curr_obs = self.observations[0][self.frame_idx]

            # Perform matching between the stored previous frame and the new current frame.
            prev_image = self.prev_frame.rgb
            curr_image = curr_frame.rgb

            images = [prev_image, curr_image]
            inputs = self.processor(images, return_tensors="pt", do_rescale=False).to(self.device)
            with torch.inference_mode():
                outputs = self.tracker(**inputs)

            keypoints0 = outputs.keypoints[0, 0].cpu()  # Keypoints in prev_image
            keypoints1 = outputs.keypoints[0, 1].cpu()  # Keypoints in curr_image
            matches = outputs.matches[0].cpu()  # shape [2, num_keypoints]
            matches_prev = matches[0] 
            matches_curr = matches[1] 

            # Keep track of which keypoints in the current frame are assigned to a track
            assigned_keypoint_indices_in_curr_frame = set()

            assigned_keypoint_indices_in_curr_frame = set()

            for i0, i1 in zip(matches_prev.tolist(), matches_curr.tolist()):
                if i1 == -1:  # unmatched keypoints
                    continue

                uv0 = keypoints0[i0].numpy()
                uv1 = keypoints1[i1].numpy()

                # propagate track if exists
                kid = None
                for k, v in prev_obs.items():
                    if np.allclose(v, uv0, atol=2.0):
                        kid = k
                        break
                
                if kid is not None:
                    curr_obs[kid] = uv1
                    assigned_keypoint_indices_in_curr_frame.add(i1)


            all_keypoint_indices_in_curr_frame = set(range(len(keypoints1)))
            unassigned_keypoints = all_keypoint_indices_in_curr_frame - assigned_keypoint_indices_in_curr_frame

            for i1 in unassigned_keypoints:
                i1 = int(i1)
                uv1 = keypoints1[i1].numpy()
                new_kid = self.next_id
                self.next_id += 1
                curr_obs[new_kid] = uv1

        # Update the buffer with the current frame for the next iteration.
        self.prev_frame = curr_frame
        
        # Increment the frame counter for the next call.
        self.frame_idx += 1