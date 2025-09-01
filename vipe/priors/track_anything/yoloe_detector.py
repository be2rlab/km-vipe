import numpy as np
import PIL
import torch
import cv2
from ultralytics import YOLOE


class YOLOEDetector:
    def __init__(self, model_path="yoloe-11l-seg-pf.pt", device="cpu"):
        """
        Initialize YOLOE detector
        
        Args:
            model_path: Path to YOLOE model weights
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.device = device
        self.model = YOLOE(model_path)
        
        # Move model to specified device if CUDA is available
        if device == "cuda" and torch.cuda.is_available():
            self.model.to(device)
    
    def run_detection(self, origin_frame, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run YOLOE detection on frame to get all objects with bounding boxes and masks
        
        Args:
            origin_frame: Input image as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            tuple: (image_shape, bounding_boxes, masks, class_names, confidences)
                - image_shape: (height, width) of original image
                - bounding_boxes: numpy array of shape [N, 4] with format [[x1, y1], [x2, y2]]
                - masks: list of mask arrays or None if no segmentation
                - class_names: list of detected class names
                - confidences: list of confidence scores
        """
        height, width = origin_frame.shape[:2]
        
        # Convert numpy array to PIL Image if needed
        if isinstance(origin_frame, np.ndarray):
            img_pil = PIL.Image.fromarray(origin_frame)
        else:
            img_pil = origin_frame
        
        # Run prediction
        print("====================================")
        print(f"conf_threshold: {conf_threshold}, iou_threshold: {iou_threshold}")
        
        results = self.model.predict(
            img_pil,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Extract results from first image (since we're processing one image)
        result = results[0]
        
        # Get bounding boxes
        boxes = []
        class_names = []
        confidences = []
        masks = []
        
        if result.boxes is not None:
            # Convert boxes to required format [[x1, y1], [x2, y2]]
            xyxy_boxes = result.boxes.xyxy.cpu().numpy()
            for box in xyxy_boxes:
                x1, y1, x2, y2 = box
                boxes.append([[int(x1), int(y1)], [int(x2), int(y2)]])
            
            # Get class names and confidences
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for cls, conf in zip(classes, confs):
                class_names.append(self.model.names[int(cls)])
                confidences.append(float(conf))
        
        # Get segmentation masks if available
        if hasattr(result, 'masks') and result.masks is not None:
            mask_data = result.masks.data.cpu().numpy()
            for mask in mask_data:
                # Resize mask to original image size
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                masks.append(mask_resized)
        else:
            masks = [None] * len(boxes)
        
        return (height, width), np.array(boxes), masks, class_names, confidences
    
    def annotate_image(self, origin_frame, boxes, masks, class_names, confidences, 
                      box_color=(0, 255, 0), text_color=(255, 255, 255), mask_alpha=0.3):
        """
        Annotate image with bounding boxes, masks, and labels
        
        Args:
            origin_frame: Original image as numpy array
            boxes: Bounding boxes in format [[x1, y1], [x2, y2]]
            masks: List of mask arrays
            class_names: List of class names
            confidences: List of confidence scores
            box_color: Color for bounding boxes (B, G, R)
            text_color: Color for text labels (B, G, R)
            mask_alpha: Transparency for masks
            
        Returns:
            Annotated image as numpy array
        """
        annotated_frame = origin_frame.copy()
        
        # Draw masks first (so they appear behind boxes)
        if masks and masks[0] is not None:
            for i, mask in enumerate(masks):
                if mask is not None:
                    # Create colored mask
                    colored_mask = np.zeros_like(annotated_frame)
                    colored_mask[mask > 0.5] = box_color
                    
                    # Blend with original image
                    annotated_frame = cv2.addWeighted(
                        annotated_frame, 1 - mask_alpha,
                        colored_mask, mask_alpha, 0
                    )
        
        # Draw bounding boxes and labels
        for i, (box, class_name, confidence) in enumerate(zip(boxes, class_names, confidences)):
            x1, y1 = box[0]
            x2, y2 = box[1]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                box_color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )
        
        return annotated_frame
    
    def detect_and_annotate(self, image_path_or_array, conf_threshold=0.25, 
                           iou_threshold=0.45, save_path=None):
        """
        Complete detection pipeline: load image, detect objects, and annotate
        
        Args:
            image_path_or_array: Path to image file or numpy array
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            save_path: Path to save annotated image (optional)
            
        Returns:
            tuple: (annotated_image, detection_results)
        """
        # Load image
        if isinstance(image_path_or_array, str):
            origin_frame = cv2.imread(image_path_or_array)
            origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
        else:
            origin_frame = image_path_or_array
        
        # Run detection
        image_shape, boxes, masks, class_names, confidences = self.run_detection(
            origin_frame, conf_threshold, iou_threshold
        )
        
        # Annotate image
        annotated_frame = self.annotate_image(
            origin_frame, boxes, masks, class_names, confidences
        )
        
        # Save if requested
        if save_path:
            # Convert RGB back to BGR for saving
            save_image = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_image)
        
        detection_results = {
            'image_shape': image_shape,
            'boxes': boxes,
            'masks': masks,
            'class_names': class_names,
            'confidences': confidences
        }
        
        return annotated_frame, detection_results
