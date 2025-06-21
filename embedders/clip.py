import torch

from transformers import CLIPModel, AutoTokenizer, AutoProcessor

from typing import Any
from PIL import Image


class ClipEmbedder:
    """
    Provide support for creating embeddings for lists of sentences and images to support an
    in memory vector DB with CLIP embeddings using the Hugging Face stack.
    """

    def __init__(self) -> None:
        # Provides the Embeddings support for text or images.
        self.model: CLIPModel = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        # Provides the toekenizer for text to feed into the embedder.
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.processor: Any = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

    def textEmbeddings(self, text: list[str]) -> torch.FloatTensor:
        inputs: Any = self.tokenizer(
            text,
            padding=True,
            return_tensors="pt",
        )

        return self.model.get_text_features(**inputs)

    def imageEmbeddings(self, images: list[Image.Image]) -> torch.FloatTensor:
        # Since images come in varying shapes we use the processor to
        # scale the images to meet the shape of the expected inputs.
        inputs: dict = self.processor(images=images, return_tensors="pt")

        return self.model.get_image_features(**inputs)

    def getModelLogitScale(self) -> torch.nn.Parameter:
        return self.model.logit_scale
