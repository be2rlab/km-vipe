from typing import Dict, Any


class SAMModel:
    def __init__(
        self,
        sam: Dict[str, Any] = None,
        device: str = "cuda",
        weights_path: str = "weights",
        **kwargs,
    ) -> None:
        raise NotImplementedError("SAMModel is not implemented yet")

    def generate(self, image):
        pass

    def predict(self, image, bboxes):
        pass
