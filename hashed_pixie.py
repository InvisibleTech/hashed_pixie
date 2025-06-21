import requests
import torch

from typing import Any
from PIL import Image

from embedders.clip import ClipEmbedder

embedder = ClipEmbedder()

text_embeddings: torch.FloatTensor = embedder.textEmbeddings(
    [
        "a photo of a cat",
        "a photo of a dog",
        "Shrimp dumplings with scalloion is pretty good. Right?",
    ]
)

url: str = "http://images.cocodataset.org/val2017/000000039769.jpg"
image: Image = Image.open(requests.get(url, stream=True).raw)

image_embeddings: torch.FloatTensor = embedder.imageEmbeddings([image])

# torch.nn.functional.cosine_similarity takes care of its own normalization but the
# concern is that scale differences in the logits could make that computation less
# stable.
#
# Additionally, looking at the CLIPModel(CLIPPreTrainedModel) forward function as per
#
#  https://codeandlife.com/2023/01/26/mastering-the-huggingface-clip-model-how-to-extract-embeddings-and-calculate-similarity-for-text-and-images/
#
# we can clearly see they choose to normalize, most likely based on their own research
# showing them that across the scope of datasets they used, that noralizing is goodness.
#
print(f"Shape of  image_features {image_embeddings.shape}")
print(f"Shape of  text_features {text_embeddings.shape}")

image_features: torch.FloatTensor = image_embeddings / torch.linalg.norm(
    image_embeddings, dim=1, keepdims=True
)
text_features: torch.FloatTensor = text_embeddings / torch.linalg.norm(
    text_embeddings, dim=1, keepdims=True
)

logit_scale = embedder.getModelLogitScale().exp()
similarity: torch.Tensor = (
    torch.nn.functional.cosine_similarity(text_features, image_features) * logit_scale
)

print(f"cosine similarity using torch on logit features {similarity}")
print(f"cosine similarity as probabilities using softmax {similarity.softmax(dim=0)}")
