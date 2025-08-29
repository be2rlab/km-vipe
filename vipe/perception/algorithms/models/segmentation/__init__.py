from vipe.perception.algorithms.models.segmentation.sam import (
    SAMModel,
)


__all__ = ["SAM"]

SEGMENTATION_MODELS_REGISTRY = {
    "SAM": SAMModel,
}
