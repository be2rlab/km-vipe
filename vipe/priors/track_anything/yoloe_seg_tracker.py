# This file is adapted from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See THIRD_PARTY_LICENSES.md for details.

import numpy as np
from typing import Dict, List, Tuple, Optional

from .aot_tracker import get_aot
from .yoloe_detector import YOLOEDetector
from .segmentor import Segmentor


class YOLOESegTracker:
    def __init__(self, segtracker_args, sam_args, aot_args, yoloe_model_path="yoloe-11l-seg-pf.pt") -> None:
        """
        Initialize SAM, AOT, and YOLOE detector.
        
        Args:
            segtracker_args: Dictionary with segtracker configuration
            sam_args: Dictionary with SAM configuration  
            aot_args: Dictionary with AOT tracker configuration
            yoloe_model_path: Path to YOLOE model weights
        """
        self.sam = Segmentor(sam_args)
        self.tracker = get_aot(aot_args)
        self.detector = YOLOEDetector(yoloe_model_path, device=self.sam.device)
        
        # SegTracker parameters
        self.sam_gap = segtracker_args["sam_gap"]
        self.min_area = segtracker_args["min_area"]
        self.max_obj_num = segtracker_args["max_obj_num"]
        self.min_new_obj_iou = segtracker_args["min_new_obj_iou"]
        
        # YOLOE specific parameters
        self.conf_threshold = segtracker_args.get("conf_threshold", 0.25)
        self.iou_threshold = segtracker_args.get("iou_threshold", 0.45)
        self.box_size_threshold = segtracker_args.get("box_size_threshold", 0.01)  # Minimum box size as fraction of image
        
        # State variables
        self.reference_objs_list = []
        self.object_idx = 1
        self.curr_idx = 1
        self.origin_merged_mask = None
        self.first_frame_mask = None
        
        # Object information storage
        self.object_classes = {}  # Maps object_id -> class_name
        self.object_confidences = {}  # Maps object_id -> confidence
        
        # Debug
        self.everything_points = []
        self.everything_labels = []

    def update_origin_merged_mask(self, updated_merged_mask):
        """Update the origin merged mask."""
        self.origin_merged_mask = updated_merged_mask

    def reset_origin_merged_mask(self, mask, id):
        """Reset the origin merged mask and current index."""
        self.origin_merged_mask = mask
        self.curr_idx = id

    def add_reference(self, frame, mask, frame_step=0):
        """
        Add objects in a mask for tracking.
        
        Args:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
            frame_step: frame step for tracker
        """
        self.reference_objs_list.append(np.unique(mask))
        self.curr_idx = self.get_obj_num()
        self.tracker.add_reference_frame(frame, mask, self.curr_idx, frame_step)
        self.curr_idx += 1

    def track(self, frame, update_memory=False):
        """
        Track all known objects.
        
        Args:
            frame: numpy array (h,w,3)
            update_memory: whether to update tracker memory
            
        Returns:
            numpy array (h,w): tracking mask
        """
        pred_mask = self.tracker.track(frame)
        if update_memory:
            self.tracker.update_memory(pred_mask)
        return pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    def get_tracking_objs(self):
        """Get list of currently tracked object IDs."""
        objs = set()
        for ref in self.reference_objs_list:
            objs.update(set(ref))
        objs = list(sorted(list(objs)))
        objs = [i for i in objs if i != 0]
        return objs

    def get_obj_num(self):
        """Get the maximum object ID currently being tracked."""
        objs = self.get_tracking_objs()
        if len(objs) == 0:
            return 0
        return int(max(objs))

    def find_new_objs(self, track_mask, seg_mask):
        """
        Compare tracked results from AOT with segmented results from SAM.
        Select objects from background if they are not tracked.
        
        Args:
            track_mask: numpy array (h,w) - current tracking results
            seg_mask: numpy array (h,w) - new segmentation results
            
        Returns:
            tuple: (new_obj_mask, seg_to_new_mapping)
        """
        new_obj_mask = (track_mask == 0) * seg_mask
        new_obj_ids = np.unique(new_obj_mask)
        new_obj_ids = new_obj_ids[new_obj_ids != 0]
        seg_to_new_mapping = {}
        obj_num = self.curr_idx
        
        for idx in new_obj_ids:
            new_obj_area = np.sum(new_obj_mask == idx)
            obj_area = np.sum(seg_mask == idx)
            
            if (
                new_obj_area / obj_area < self.min_new_obj_iou
                or new_obj_area < self.min_area
                or obj_num > self.max_obj_num
            ):
                new_obj_mask[new_obj_mask == idx] = 0
            else:
                new_obj_mask[new_obj_mask == idx] = obj_num
                seg_to_new_mapping[idx] = obj_num
                obj_num += 1
                
        return new_obj_mask, seg_to_new_mapping

    def restart_tracker(self):
        """Restart the tracker."""
        self.tracker.restart()

    def add_mask(self, interactive_mask: np.ndarray):
        """
        Merge interactive mask with self.origin_merged_mask
        
        Args:
            interactive_mask: numpy array (h, w)
            
        Returns:
            numpy array (h, w): refined merged mask
        """
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape, dtype=np.uint8)

        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.curr_idx

        return refined_merged_mask

    def detect_and_seg(
        self,
        origin_frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        box_size_threshold: Optional[float] = None,
        reset_image: bool = False,
        class_filter: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int], Dict[int, Dict[str, any]]]:
        """
        Using YOLOE to detect all objects and segment them with SAM
        
        Args:
            origin_frame: numpy array (h, w, 3)
            conf_threshold: confidence threshold for detection (overrides default)
            iou_threshold: IoU threshold for NMS (overrides default) 
            box_size_threshold: minimum box size as fraction of image area
            reset_image: whether to reset SAM image
            class_filter: list of class names to filter detections (None means all classes)
            
        Returns:
            tuple: (refined_merged_mask, annotated_frame_shape, seg_info)
                - refined_merged_mask: numpy array (h, w)
                - annotated_frame_shape: (height, width)
                - seg_info: dict mapping object_id -> {'class': class_name, 'confidence': conf}
        """
        # Use provided thresholds or defaults
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold
        box_size_thresh = box_size_threshold if box_size_threshold is not None else self.box_size_threshold
        
        # Backup current state
        bc_id = self.curr_idx
        bc_mask = self.origin_merged_mask
        seg_info = {}

        # Run YOLOE detection
        image_shape, boxes, masks, class_names, confidences = self.detector.run_detection(
            origin_frame, conf_thresh, iou_thresh
        )
        
        annotated_frame_shape = image_shape
        refined_merged_mask = np.zeros(annotated_frame_shape, dtype=np.uint8)
        
        # Process each detection
        for i in range(len(boxes)):
            bbox = boxes[i]
            class_name = class_names[i]
            confidence = confidences[i]
            
            # Apply class filter if specified
            if class_filter is not None and class_name not in class_filter:
                continue
            
            # Check box size threshold
            box_area = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
            image_area = annotated_frame_shape[0] * annotated_frame_shape[1]
            
            if box_area < image_area * box_size_thresh:
                continue
            
            # Use YOLOE mask if available, otherwise use SAM
            if masks and masks[i] is not None and hasattr(masks[i], 'shape'):
                # Use YOLOE segmentation mask directly
                yoloe_mask = masks[i]
                if yoloe_mask.dtype != bool:
                    yoloe_mask = yoloe_mask > 0.5
                interactive_mask = yoloe_mask.astype(np.uint8)
            else:
                # Fallback to SAM segmentation
                interactive_mask = self.sam.segment_with_box(origin_frame, bbox, reset_image)[0]
            
            # Add mask to merged result
            refined_merged_mask = self.add_mask(interactive_mask)
            
            # Store object information
            seg_info[self.curr_idx] = {
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            }
            self.object_classes[self.curr_idx] = class_name
            self.object_confidences[self.curr_idx] = confidence
            
            # Update state
            self.update_origin_merged_mask(refined_merged_mask)
            self.curr_idx += 1

        # Reset origin mask to backup state
        self.reset_origin_merged_mask(bc_mask, bc_id)

        return refined_merged_mask, annotated_frame_shape, seg_info

    def detect_and_seg_all_classes(
        self,
        origin_frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        reset_image: bool = False
    ) -> Tuple[np.ndarray, Tuple[int, int], Dict[int, Dict[str, any]]]:
        """
        Convenience method to detect and segment all object classes without filtering
        
        Args:
            origin_frame: numpy array (h, w, 3)
            conf_threshold: confidence threshold for detection
            reset_image: whether to reset SAM image
            
        Returns:
            Same as detect_and_seg()
        """
        return self.detect_and_seg(
            origin_frame,
            conf_threshold=conf_threshold,
            reset_image=reset_image,
            class_filter=None
        )

    def detect_and_seg_by_classes(
        self,
        origin_frame: np.ndarray,
        target_classes: List[str],
        conf_threshold: Optional[float] = None,
        reset_image: bool = False
    ) -> Tuple[np.ndarray, Tuple[int, int], Dict[int, Dict[str, any]]]:
        """
        Detect and segment only specific object classes
        
        Args:
            origin_frame: numpy array (h, w, 3)
            target_classes: list of class names to detect (e.g., ['person', 'car', 'bicycle'])
            conf_threshold: confidence threshold for detection
            reset_image: whether to reset SAM image
            
        Returns:
            Same as detect_and_seg()
        """
        return self.detect_and_seg(
            origin_frame,
            conf_threshold=conf_threshold,
            reset_image=reset_image,
            class_filter=target_classes
        )

    def get_object_info(self, object_id: int) -> Optional[Dict[str, any]]:
        """
        Get information about a tracked object
        
        Args:
            object_id: ID of the object
            
        Returns:
            Dictionary with object info or None if not found
        """
        if object_id in self.object_classes:
            return {
                'class': self.object_classes[object_id],
                'confidence': self.object_confidences.get(object_id, 0.0)
            }
        return None

    def get_all_object_info(self) -> Dict[int, Dict[str, any]]:
        """
        Get information about all tracked objects
        
        Returns:
            Dictionary mapping object_id -> object_info
        """
        result = {}
        for obj_id in self.object_classes:
            result[obj_id] = {
                'class': self.object_classes[obj_id],
                'confidence': self.object_confidences.get(obj_id, 0.0)
            }
        return result

    def filter_objects_by_class(self, mask: np.ndarray, target_classes: List[str]) -> np.ndarray:
        """
        Filter a mask to only include objects of specified classes
        
        Args:
            mask: numpy array (h, w) with object IDs
            target_classes: list of class names to keep
            
        Returns:
            Filtered mask
        """
        filtered_mask = np.zeros_like(mask)
        for obj_id in np.unique(mask):
            if obj_id == 0:
                continue
            if obj_id in self.object_classes and self.object_classes[obj_id] in target_classes:
                filtered_mask[mask == obj_id] = obj_id
        return filtered_mask
