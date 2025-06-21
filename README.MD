# Hashed Pixie

## What Is Pixie?

Pixie is the name of an in memory vector database [from this blog post](https://dev.to/vivekalhat/building-a-tiny-vector-store-from-scratch-59ep). It is a pretty basic implementation and was fun to get type in and get working.

## Why Python and Hugging Face?

I chose Python for my sanity, because getting a ViT model to work as an embeddings API in [Deep Java Library](https://djl.ai/) was a more involved choice that leveraged a Python server and various other layers I did not wish learn right now. As you know from [my other post](https://aninvisiblefriend.com/zero-shot-object-detection-using-java-python-and-hugging-face-28bd7f298c48) I found [JEP](https://github.com/ninia/jep) to be a better choice for research and rapid exploration. In that model, you use a lot more Python and Java is more like the front end for interacting with the Python features.

It was easy to be inspired to use [Hugging Face hosted openai / clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) after reading about using [CLIP Embeddings For Text and Image Similarity](https://codeandlife.com/2023/01/26/mastering-the-huggingface-clip-model-how-to-extract-embeddings-and-calculate-similarity-for-text-and-images/). It was familiar ground on a tech stack I had used before and so could reduce some friction for me.

## What is the Game Plan?

I want to do a couple of things here, I want to use a ViT model like the [Hugging Face hosted openai / clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) and I don't want to use the ChatGPT interface (used in the Pixie tutorial). Instead I want to move towards using cosine similarity search for text and images as a light weight in memory database.

After that I want to leverage a [Locality Sensitive Hash](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/) to improve performance. Given the model I am using produces embedding vectors with 768 columns and cosine similarity uses a good bit of math... and the problem space of running a query against all vectors is O(n), well it seems obvious we would benefit from a fast method for binning to improve finding approximate matches when they exists. That is not to say a deeper, more expensive index based on proper k-Means clustering isn't something to consider later but I'd like to see if we can do a layered approach of some kind for LSH.
