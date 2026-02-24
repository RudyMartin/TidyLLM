import json
import os
import datetime
import boto3
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonEmbeddingVectorizer:
    """
    A class that generates embeddings using Amazon Bedrock models and stores them uniquely in JSON.
    """

    MODEL_OPTIONS = {
        "titan_v1": {"id": "amazon.titan-embed-text-v1", "dimensions": 768},
        "titan_v2": {"id": "amazon.titan-embed-text-v2:0", "dimensions": 1024},
        "cohere": {"id": "cohere.embed-english-v3", "dimensions": None},
        "anthropic": {"id": "anthropic.claude-v2", "dimensions": None}
    }

    DIMENSION_DEFAULTS = {
        "amazon.titan-embed-text-v1": 768,
        "amazon.titan-embed-text-v2:0": 1024,
        "cohere.embed-english-v3": None,
        "anthropic.claude-v2": None
    }

    def __init__(self, model_id=None, dimensions=None, region_name="us-east-1", boto3_client=None):
        """
        Initialize the vectorizer with a specific model or pick one randomly.

        Args:
            model_id (str, optional): AWS Bedrock model ID.
            dimensions (int, optional): Embedding dimensions.
            region_name (str, optional): AWS region for Bedrock API.
            boto3_client (boto3.client, optional): Pre-configured Boto3 client.
        """
        self.bedrock_boto3 = boto3_client or boto3.client("bedrock-runtime", region_name=region_name)

        if model_id is None:
            model_id = random.choice(list(self.MODEL_OPTIONS.values()))
        
        if model_id not in self.MODEL_OPTIONS.values():
            raise ValueError(f"Invalid model ID: {model_id}. Choose from {list(self.MODEL_OPTIONS.values())}")

        self.model_id = model_id
        self.dimensions = dimensions or self.DIMENSION_DEFAULTS.get(model_id)

        logger.info(f"✅ Using Amazon Bedrock model: {self.model_id} with dimensions={self.dimensions}")

        # Caching to avoid redundant API calls
        self.query_cache = {}

    def generate_embedding(self, text, normalize=True):
        """
        Generate embeddings using the selected Bedrock model.

        Args:
            text (str): The input text to vectorize.
            normalize (bool, optional): Normalize the embedding.

        Returns:
            List[float]: The embedding vector.
        """
        cache_key = (text, self.model_id, self.dimensions, normalize)
        if cache_key in self.query_cache:
            logger.info(f"⚡ Using cached embedding for '{text}' ({self.model_id}, {self.dimensions}D)")
            return self.query_cache[cache_key]

        body = {"inputText": text, "normalize": normalize}
        if self.dimensions and "titan" in self.model_id:
            body["dimensions"] = self.dimensions  

        response = self.bedrock_boto3.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        response_body = json.loads(response["Body"].read().decode("utf-8"))
        embedding = response_body.get("embedding", [])

        self.query_cache[cache_key] = embedding
        return embedding

    def update_json_with_embeddings(self, json_path, exclude_keys=None):
        """
        Reads an existing JSON file, generates embeddings for the full body (excluding embeddings), 
        and stores them uniquely.

        Args:
            json_path (str): The path to the JSON file.
            exclude_keys (list, optional): Keys to exclude (default: 'embeddings').

        Returns:
            dict: Updated JSON data.
        """
        # Ensure the file exists
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file '{json_path}' not found. No new content should be added.")

        # Load existing JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract full body excluding 'embeddings'
        text_to_encode = self.get_text_to_encode(data, exclude_keys)

        # Generate embedding
        embedding_vector = self.generate_embedding(text_to_encode)

        # Generate a unique key using model_id and dimensions (if applicable)
        model_key = f"{self.model_id}_{self.dimensions}" if self.dimensions else self.model_id

        # Store embedding under its unique key, replacing existing one if present
        if "embeddings" not in data:
            data["embeddings"] = {}

        data["embeddings"][model_key] = {
            "embed_date": datetime.datetime.utcnow().isoformat() + "Z",
            "dimensions": self.dimensions,
            "vector": embedding_vector
        }

        # Write back to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info(f"✅ Successfully stored embeddings for {self.model_id} ({self.dimensions}D) in {json_path}")
        return data

    def get_text_to_encode(self, data, exclude_keys=None):
        """
        Extracts metadata + content while excluding 'embeddings'.

        Args:
            data (dict): JSON dictionary.
            exclude_keys (list, optional): List of keys to exclude from encoding.

        Returns:
            str: Formatted metadata + content for embedding.
        """
        if exclude_keys is None:
            exclude_keys = ["embeddings"]  # Default exclusion

        # Create a filtered version of the JSON excluding unwanted keys
        data_filtered = {k: v for k, v in data.items() if k not in exclude_keys}

        # Convert to a formatted JSON string
        return json.dumps(data_filtered, indent=2)
