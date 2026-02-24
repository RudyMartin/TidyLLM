# 🚀 embedding_helper.py - Embedding Helper for Amazon Titan, Cohere, and Claude

import boto3
import numpy as np
import json
import logging
import hashlib
from functools import lru_cache
from configuration import CONFIG, MODEL_OPTIONS

# **🔹 Initialize AWS Bedrock Client**
bedrock_client = boto3.client("bedrock-runtime")

# **🔹 Logging Setup**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EmbeddingVectorizer:
    """Handles text embedding using Amazon Titan, Cohere, and Claude models via AWS Bedrock."""

    def __init__(self, model_choice: str = "titan_v1"):
        """Initializes the vectorizer with a specific embedding model."""
        self.model_choice = model_choice
        self.model_id = MODEL_OPTIONS[model_choice]["id"]
        self.dimension = MODEL_OPTIONS[model_choice]["dimensions"] or CONFIG["embedding_dimension"]

        logging.info(f"✅ EmbeddingVectorizer initialized with model: {self.model_id} (Dims: {self.dimension})")

    @staticmethod
    def _cache_key(text: str, model_id: str) -> str:
        """Generates a unique cache key based on the input text and model."""
        return hashlib.sha256((model_id + text).encode()).hexdigest()

    @lru_cache(maxsize=500)
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Calls AWS Bedrock to generate an embedding for the given text, with caching for optimization.

        Args:
            text (str): Input text to embed.

        Returns:
            np.ndarray: Normalized embedding vector.
        """
        cache_key = self._cache_key(text, self.model_id)

        try:
            payload = {"inputText": text}
            response = bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload)
            )
            embedding_data = json.loads(response["body"].read().decode("utf-8"))

            if "embedding" not in embedding_data:
                raise ValueError("No embedding data found in response")

            embedding = np.array(embedding_data["embedding"], dtype=np.float32)

            # Normalize the embedding to unit length
            embedding /= np.linalg.norm(embedding)

            logging.info(f"✅ Successfully generated embedding (Model: {self.model_id}) for text: {text[:30]}...")
            return embedding

        except Exception as e:
            logging.error(f"❌ Error generating embedding for model {self.model_id}: {e}")
            return np.zeros(self.dimension)  # Return zero-vector if embedding fails
