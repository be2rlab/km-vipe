from __future__ import annotations
import torch
import mobileclip
from typing import Optional, Dict
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class Engine:
    """
    Engine for generating CLIP embeddings.
    """

    def __init__(
        self,
        model_path: str = "/models/mobileclip_s0.pt",
        model_name: str = "mobileclip_s0",
        device: str = "cuda",
    ):
        super().__init__()
        self.model_path = model_path
        self.model_name = model_name
        self.device = device

    def load_model(self) -> None:
        """Load MobileCLIP model."""
        if self.model_name.split("_")[0] == "mobileclip":
            self._model, _, _ = mobileclip.create_model_and_transforms(
                self.model_name, pretrained=self.model_path, device=self.device
            )
            self.tokenizer = mobileclip.get_tokenizer(self.model_name)
        else:
            ckpt = self.model_path
            self.bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            self._model = (
                AutoModel.from_pretrained(
                    ckpt,
                    # device_map="auto",
                    # attn_implementation="sdpa",
                    # quantization_config=self.bnb_config,
                )
                .eval()
                .to(self.device)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def encode_images(self, objects) -> torch.Tensor:
        """Encode image inputs to embeddings."""
        if self._model is None:
            self.load_model()

        torch.cuda.empty_cache()
        objects = objects.to(self.device)
        max_batch_size = 64
        features_list = []
        num_sub_batches = (objects.size(0) + max_batch_size - 1) // max_batch_size

        for i in range(num_sub_batches):
            start_idx = i * max_batch_size
            end_idx = (i + 1) * max_batch_size
            sub_objects = objects[start_idx:end_idx] / 255

            if self.model_name.split("_")[0] == "mobileclip":
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    sub_features = self._model.encode_image(sub_objects)
                    sub_features /= sub_features.norm(dim=-1, keepdim=True)
            else:
                with torch.no_grad():
                    sub_features = self._model.get_image_features(sub_objects)
                    sub_features /= sub_features.norm(dim=-1, keepdim=True)

            features_list.append(sub_features)
            del sub_features
            torch.cuda.empty_cache()

        image_features = torch.cat(features_list, dim=0)

        return image_features

    def encode_text(self, text) -> torch.Tensor:
        """Encode text input to embeddings."""
        if self._model is None:
            self.load_model()
        if self.model_name.split("_")[0] == "mobileclip":
            text = self.tokenizer([text]).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = self._model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
        else:
            text = self.tokenizer(
                [text],  # padding="max_length", max_length=64, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                text_features = self._model.get_text_features(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def __call__(
        self, images: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate CLIP embeddings for images and/or texts.
        Returns dict with keys 'image_embeddings' and/or 'text_embeddings'.
        """

        return self.encode_images(images)
