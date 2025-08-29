from __future__ import annotations
from dataclasses import dataclass
# from mapper.slam_classes import MapObjectList

from vipe.perception.common.structures import Objects
from vipe.perception.knowledge.segmenter.engine import MobileSAMv2 as Segmentor
from vipe.perception.knowledge.vlm.engine import Engine as VLM
from vipe.perception.knowledge.segmenter.utils import show_image_grid, search_query


@dataclass
class ProcessorConfig:
    """Configuration for the Grounder"""

    segmentation_encoder_path: str = (
        "/workspace/perception/algorithms/models/engines/xl0_encoder.engine"
    )
    segmentation_decoder_path: str = (
        "/workspace/perception/algorithms/models/engines/xl0_decoder.engine"
    )
    detector_path: str = (
        "/workspace/perception/algorithms/models/engines/ObjectAwareModel.engine"
    )
    device: str = "cuda"
    max_object_size: int = 256
    iou_threshold: float = 0.7
    img_size: int = 1024
    # clip_model_path: str = "/models/mobileclip_s2.pt"
    # clip_model_name: str = "mobileclip_s2"
    clip_model_name: str = "siglip2-base-patch16-256"
    clip_model_path: str = "google/siglip2-base-patch16-256"


class Grounder:
    """
    Processes images using segmentation and CLIP embedding engines to create object knowledge.
    """

    def __init__(self, config: ProcessorConfig = ProcessorConfig()):
        self.config = config

        # Initialize segmentation engine
        self.segmentation_engine = Segmentor(
            encoder_path=config.segmentation_encoder_path,
            decoder_path=config.segmentation_decoder_path,
            detector_path=config.detector_path,
            device=config.device,
            max_object_size=config.max_object_size,
            iou_threshold=config.iou_threshold,
            img_size=config.img_size,
        )

        # Initialize CLIP engine
        self.clip_engine = VLM(
            model_path=config.clip_model_path,
            model_name=config.clip_model_name,
            device=config.device,
        )

        self.segmentation_engine.load_model()
        self.clip_engine.load_model()

    def process_image(
        self,
        image,
        visualize: bool = False,
    ) -> Objects:
        image_size = (image.shape[0], image.shape[1])
        segmented_objects, BBs, masks, confs = self.segmentation_engine(image)
        embeddings = self.clip_engine(segmented_objects)
        if visualize:
            show_image_grid(segmented_objects)
        results = {
            "segmented_objects": segmented_objects,
            "masks": masks,
            "embeddings": embeddings,
            "BBs": BBs,
            "confs": confs,
            "image_size": image_size,
        }
        return results

    def process_query(
        self,
        text,
        objects,
    ) -> int:
        text_query_ft = self.clip_engine.encode_text(text)
        object_idx = search_query(text_query_ft, objects)
        return object_idx
