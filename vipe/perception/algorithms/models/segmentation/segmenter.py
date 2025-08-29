from vipe.perception.algorithms.models.segmentation import (
    SEGMENTATION_MODELS_REGISTRY,
)
from typing import Dict, Any


def create_segmenter_model(
    sam: Dict[str, Any] = None,
    device: str = "cuda",
    weights_path: str = "weights",
    **kwargs,
):
    """Create the Segmenter model
    Args:
        sam (Dict): segmenter model config
        device (torch.device): device to run the model
        weights_path (str): Path to model weights folder
    Returns:
        Segmenter model
    """
    model_name = sam["sam_type"]
    assert (
        model_name in SEGMENTATION_MODELS_REGISTRY
    ), f"model {model_name} is not supported. Supported models: {list(SEGMENTATION_MODELS_REGISTRY.keys())}"
    model = SEGMENTATION_MODELS_REGISTRY[model_name]
    model = model(sam, device, weights_path, **kwargs)
    return model


class Segmenter(object):
    """Class for segmenting 2D masks from images"""

    def __init__(
        self,
        sam: Dict[str, Any] = None,
        device: str = "cuda",
        weights_path: str = "weights",
        **kwargs,
    ):
        """Initializes the Segmenter class
        Args:
            sam (Dict): segmenter model config
            device (torch.device): device to run the model
            weights_path (str): Path to model weights folder
        """
        self.model_name = sam["sam_type"]
        self.model_version = sam["sam_version"]
        self.mode = sam.get("sam_mode", "predictor")
        self.device = device
        self.weights_path = weights_path

        self.model = create_segmenter_model(
            sam,
            self.device,
            self.weights_path,
            **kwargs,
        )

    def generate(self, image):
        return self.model.generate(image)

    def predict(self, image, bboxes):
        return self.model.predict(image, bboxes)

    def process_image(self, image):
        pass
