from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
import cv2

from vipe.streams.base import VideoFrame
from . import SparseTracks


class LightGlueTracks(SparseTracks):
    def __init__(
        self,
        n_views: int,
        model_id: str = "ETH-CVG/lightglue_superpoint", 
        device: str = "cuda",
        output_video_path: str = "tracks_output.mp4",
        fps: int = 20
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
        self.prev_frame: VideoFrame | None = None

        # Track histories: kid -> list of (frame_idx, uv)
        self.tracks: dict[int, list[tuple[int, np.ndarray]]] = {}

        # RANSAC config
        self.ransac_reproj_threshold = 3.0  # pixels
        self.ransac_confidence = 0.99

        # visualization
        self.video_writer = None
        self.output_video_path = output_video_path
        self.fps = fps
        self.track_colors: dict[int, tuple[int, int, int]] = {}

    def _build_model(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.tracker = AutoModel.from_pretrained(self.model_id).to(self.device)

    def _run_ransac_filter(self, keypoints0, keypoints1, matches_prev, matches_curr):
        pts0 = []
        pts1 = []
        order = []
        for i0, i1 in zip(matches_prev.tolist(), matches_curr.tolist()):
            if i1 == -1:
                continue
            pts0.append(keypoints0[i0].tolist())
            pts1.append(keypoints1[i1].tolist())
            order.append((i0, i1))

        if len(pts0) < 8:
            return order, np.ones(len(order), dtype=bool)

        pts0 = np.asarray(pts0, dtype=np.float32)
        pts1 = np.asarray(pts1, dtype=np.float32)

        F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, self.ransac_reproj_threshold, self.ransac_confidence)
        if mask is None:
            mask = np.ones((len(pts0), 1), dtype=np.uint8)

        mask = mask.ravel().astype(bool)
        return order, mask
    
    def _release_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def track_image(self, frame_data_list: list[VideoFrame]):
        if self.tracker is None:
            self._build_model()
        assert self.tracker is not None, "LightGlue Tracker not initialized"

        assert len(frame_data_list) == 1, "LightGlueTracks expects one frame at a time."
        curr_frame = frame_data_list[0]

        self.observations[0].append({})

        if self.prev_frame is not None:
            prev_obs = self.observations[0][self.frame_idx - 1]
            curr_obs = self.observations[0][self.frame_idx]

            prev_image = self.prev_frame.rgb # cuda tensor
            curr_image = curr_frame.rgb # cuda tensor

            images = [prev_image, curr_image]
            inputs = self.processor(images, return_tensors="pt", do_rescale=False).to(self.device)
            with torch.inference_mode():
                outputs = self.tracker(**inputs)

            keypoints0 = outputs.keypoints[0, 0].cpu().numpy()
            keypoints1 = outputs.keypoints[0, 1].cpu().numpy()
            matches = outputs.matches[0].cpu()
            matches_prev = matches[0]
            matches_curr = matches[1]

            order, inlier_mask = self._run_ransac_filter(keypoints0, keypoints1, matches_prev, matches_curr)
        
            vis = self._draw_matches(prev_image, curr_image, keypoints0, keypoints1, order, inlier_mask)

            if self.video_writer is None:
                h, w = vis.shape[:2]
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (w, h)
                )
            self.video_writer.write(vis)

            assigned_keypoint_indices_in_curr_frame = set()

            for (i0, i1), inlier in zip(order, inlier_mask):
                if not inlier:
                    continue
                uv0 = keypoints0[i0]
                uv1 = keypoints1[i1]
                kid = None
                for k, v in prev_obs.items():
                    if np.allclose(v, uv0, atol=2.0):
                        kid = k
                        break

                if kid is not None:
                    curr_obs[kid] = uv1
                    assigned_keypoint_indices_in_curr_frame.add(i1)
                    if kid in self.tracks:
                        self.tracks[kid].append((self.frame_idx, uv1.copy()))
                    else:
                        self.tracks[kid] = [(self.frame_idx, uv1.copy())]

            all_keypoint_indices_in_curr_frame = set(range(len(keypoints1)))
            unassigned_keypoints = all_keypoint_indices_in_curr_frame - assigned_keypoint_indices_in_curr_frame

            for i1 in unassigned_keypoints:
                uv1 = keypoints1[int(i1)].copy()
                new_kid = self.next_id
                self.next_id += 1
                curr_obs[new_kid] = uv1
                self.tracks[new_kid] = [(self.frame_idx, uv1.copy())]

        self.prev_frame = curr_frame
        self.frame_idx += 1

        if len(self.observations[0]) >= 300 and self.video_writer is not None:
            self._release_writer()
            print(f"Video written to {self.output_video_path}")


    def _draw_matches(self, prev_image: np.ndarray, curr_image: np.ndarray, 
                    keypoints0: np.ndarray, keypoints1: np.ndarray, 
                    matches: list[tuple[int, int]], mask: np.ndarray):
        """
        Draw keypoints and matches between two frames using cv2.
        prev_image, curr_image: numpy arrays (H, W, 3), BGR format
        keypoints0, keypoints1: arrays of shape (N, 2)
        matches: list of (i0, i1) index pairs
        mask: boolean array marking which matches are inliers
        """

        # Ensure inputs are uint8 images
        prev_vis = (prev_image.cpu().numpy() * 255).astype(np.uint8) if torch.is_tensor(prev_image) else prev_image.copy()
        curr_vis = (curr_image.cpu().numpy() * 255).astype(np.uint8) if torch.is_tensor(curr_image) else curr_image.copy()

        if prev_vis.shape[2] == 3:
            prev_vis = cv2.cvtColor(prev_vis, cv2.COLOR_RGB2BGR)
        if curr_vis.shape[2] == 3:
            curr_vis = cv2.cvtColor(curr_vis, cv2.COLOR_RGB2BGR)

        # Concatenate images horizontally
        h1, w1 = prev_vis.shape[:2]
        h2, w2 = curr_vis.shape[:2]
        H = max(h1, h2)
        W = w1
        canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1, :] = prev_vis
        canvas[:h2, w1:w1 + w2, :] = curr_vis

        # Draw matches
        for (i, (i0, i1)) in enumerate(matches):
            if not mask[i]:
                continue
            pt0 = [int(keypoints0[i0][0] * W), int(keypoints0[i0][1] * H)]
            pt1 = [int(keypoints1[i1][0] * W), int(keypoints1[i1][1] * H)]
            pt1_shifted = (pt1[0] + w1, pt1[1])  # shift x for right image

            color = [0, 255, 0]
            cv2.circle(canvas, pt0, 1, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt1_shifted, 1, color, -1, cv2.LINE_AA)
            cv2.line(canvas, pt0, pt1_shifted, color, 1, cv2.LINE_AA)

        return canvas

    # def visualize_observations(self):
    #     H, W = self.prev_frame.rgb.shape[0:2]
    #     view_idx = 0  

    #     frames = len(self.observations[view_idx])
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (W, H))

    #     # Assign random colors for each track id
    #     track_colors = {}

    #     # Store last positions for drawing paths
    #     last_positions = {}

    #     for f_idx in range(frames):
    #         frame = np.zeros((H, W, 3), dtype=np.uint8)  # black background

    #         for kid, uv in self.observations[view_idx][f_idx].items():
    #             # Get color for this track
    #             if kid not in track_colors:
    #                 track_colors[kid] = tuple(np.random.randint(0, 255, 3).tolist())

    #             color = track_colors[kid]

    #             # Denormalize uv to pixel coords
    #             x = int(uv[0] * W)
    #             y = int(uv[1] * H)

    #             # Draw trajectory line if previous position exists
    #             if kid in last_positions:
    #                 cv2.line(frame, last_positions[kid], (x, y), color, 2)

    #             # Draw current point
    #             cv2.circle(frame, (x, y), 4, color, -1)

    #             last_positions[kid] = (x, y)

    #         writer.write(frame)

    #     writer.release()
    #     print(f"Video written to {self.output_video_path}")


