"""
Embedding Validation and Model Configuration
============================================

Centralized truth for embedding models and validation logic.
Prevents dimension mismatches and ensures clean vector operations.
"""

import math
import json
import numpy as np
from typing import Dict, Tuple, Optional, List, Any


# ==================== MODEL CONFIGURATION ====================

MODEL_DETAILS = {
    "titan_v1": {
        "model_id": "amazon.titan-embed-text-v1",
        "dims": 768,
        "max_input_length": 8000
    },
    "titan_v2": {
        "model_id": "amazon.titan-embed-text-v2:0",
        "dims": 1024,
        "max_input_length": 8000
    },
    "cohere_v3": {
        "model_id": "cohere.embed-english-v3",
        "dims": 1024,
        "max_input_length": 2000
    },
    "minilm_local": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dims": 384,
        "max_input_length": 512
    }
}


def get_model_details(model_key: str) -> Tuple[str, int, int]:
    """Get model ID, dimensions, and max input length."""
    md = MODEL_DETAILS.get(model_key)
    if not md:
        # Check if it's a model ID directly
        for key, details in MODEL_DETAILS.items():
            if details["model_id"] == model_key:
                return details["model_id"], details["dims"], details["max_input_length"]
        raise ValueError(f"Unknown model_key: {model_key}")
    return md["model_id"], md["dims"], md["max_input_length"]


# ==================== VALIDATION ====================

class EmbeddingValidationError(Exception):
    """Raised when embedding validation fails."""
    pass


def validate_embedding(vec: Any, expected_dim: int, model_key: str = None) -> np.ndarray:
    """
    Validate and normalize an embedding vector.

    Args:
        vec: Raw embedding (list or array)
        expected_dim: Expected dimension count
        model_key: Optional model key for better error messages

    Returns:
        Validated numpy array of float32

    Raises:
        EmbeddingValidationError: If validation fails
    """
    model_info = f" for model {model_key}" if model_key else ""

    if vec is None:
        raise EmbeddingValidationError(f"Embedding is None{model_info}")

    # Convert to numpy array
    if isinstance(vec, list):
        arr = np.asarray(vec, dtype=np.float32)
    elif isinstance(vec, np.ndarray):
        arr = vec.astype(np.float32, copy=False)
    else:
        raise EmbeddingValidationError(
            f"Unsupported type {type(vec)}{model_info}"
        )

    # Check dimensions
    if arr.ndim != 1:
        raise EmbeddingValidationError(
            f"Expected 1D vector, got shape {arr.shape}{model_info}"
        )

    if arr.shape[0] != expected_dim:
        raise EmbeddingValidationError(
            f"Dimension mismatch{model_info}: got {arr.shape[0]}, expected {expected_dim}"
        )

    # Check for NaN/Inf
    if not np.isfinite(arr).all():
        bad_indices = np.where(~np.isfinite(arr))[0][:5]
        raise EmbeddingValidationError(
            f"NaN/Inf at positions {bad_indices.tolist()}{model_info}"
        )

    return arr


def normalize_embedding(vec: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize an embedding vector.

    Args:
        vec: Validated embedding vector
        method: Normalization method ("l2" or "none")

    Returns:
        Normalized vector
    """
    if method == "l2":
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
    return vec


# ==================== BEDROCK ADAPTERS ====================

def titan_embed_request(text: str, normalize: bool = True) -> Dict:
    """Create Titan embedding request body."""
    return {
        "inputText": text,
        "normalize": normalize  # Titan v2 supports built-in normalization
    }


def parse_titan_response(response_data: Dict) -> List[float]:
    """Parse Titan embedding response."""
    # Try different field names used by various SDK versions
    vec = response_data.get("embedding")
    if vec is None:
        vec = response_data.get("vector")
    if vec is None:
        vec = response_data.get("embeddings", [None])[0]

    if vec is None:
        raise RuntimeError(
            f"No embedding field in Titan response. Keys: {list(response_data.keys())}"
        )

    return vec


def cohere_embed_request(texts: List[str], input_type: str = "search_document") -> Dict:
    """Create Cohere embedding request body."""
    return {
        "texts": texts,
        "input_type": input_type,  # "search_document" or "search_query"
        "truncate": "END"  # Truncate from end if too long
    }


def parse_cohere_response(response_data: Dict) -> List[List[float]]:
    """Parse Cohere embedding response."""
    embeddings = response_data.get("embeddings", [])
    if not embeddings:
        raise RuntimeError(
            f"No embeddings in Cohere response. Keys: {list(response_data.keys())}"
        )
    return embeddings


# ==================== FAISS INDEX MANAGEMENT ====================

def get_index_name(model_key: str, dims: int = None) -> str:
    """Generate consistent FAISS index name."""
    if dims is None:
        _, dims, _ = get_model_details(model_key)
    return f"faiss_{model_key}_{dims}d.index"


def validate_index_compatibility(index_dims: int, model_dims: int, model_key: str):
    """Ensure FAISS index matches model dimensions."""
    if index_dims != model_dims:
        raise RuntimeError(
            f"FAISS index dim={index_dims} != {model_key} model dim={model_dims}. "
            f"Index needs rebuild."
        )


# ==================== END-TO-END EMBEDDING ====================

def embed_text_safely(
    model_key: str,
    text: str,
    bedrock_runtime,
    normalize: bool = True,
    telemetry_logger=None
) -> np.ndarray:
    """
    Safe end-to-end text embedding with validation.

    Args:
        model_key: Model identifier from MODEL_DETAILS
        text: Text to embed
        bedrock_runtime: Boto3 Bedrock runtime client
        normalize: Whether to L2 normalize
        telemetry_logger: Optional logger for metrics

    Returns:
        Validated, normalized embedding vector
    """
    import time
    start_time = time.time()

    # Get model configuration
    model_id, expected_dims, max_length = get_model_details(model_key)

    # Truncate if needed
    original_length = len(text)
    if original_length > max_length:
        text = text[:max_length]
        if telemetry_logger:
            telemetry_logger.warning(
                f"Truncated input from {original_length} to {max_length} chars"
            )

    try:
        # Prepare request based on model type
        if "titan" in model_id.lower():
            body = titan_embed_request(text, normalize=normalize)
        elif "cohere" in model_id.lower():
            body = cohere_embed_request([text])
        else:
            raise ValueError(f"Unsupported model type: {model_id}")

        # Invoke model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body).encode("utf-8")
        )

        # Parse response
        response_data = json.loads(response["body"].read().decode("utf-8"))

        if "titan" in model_id.lower():
            raw_vec = parse_titan_response(response_data)
        elif "cohere" in model_id.lower():
            raw_vec = parse_cohere_response(response_data)[0]
        else:
            raw_vec = response_data.get("embedding", [])

        # Validate
        vec = validate_embedding(raw_vec, expected_dims, model_key)

        # Normalize if requested
        if normalize:
            vec = normalize_embedding(vec, method="l2")

        # Log telemetry
        if telemetry_logger:
            elapsed_ms = (time.time() - start_time) * 1000
            telemetry_logger.info({
                "provider": "bedrock",
                "model_id": model_id,
                "dims": expected_dims,
                "input_len": original_length,
                "truncated": original_length > max_length,
                "latency_ms": elapsed_ms,
                "status": "success"
            })

        return vec

    except Exception as e:
        if telemetry_logger:
            elapsed_ms = (time.time() - start_time) * 1000
            telemetry_logger.error({
                "provider": "bedrock",
                "model_id": model_id,
                "error": str(e),
                "latency_ms": elapsed_ms,
                "status": "failed"
            })
        raise


# ==================== TESTING ====================

def test_dimension_enforcement():
    """Test that dimensions are properly enforced."""
    print("Testing dimension enforcement...")

    for model_key, details in MODEL_DETAILS.items():
        model_id, dims, max_len = get_model_details(model_key)
        print(f"  {model_key}: {model_id} -> {dims}D (max {max_len} chars)")

        # Test validation catches wrong dimensions
        wrong_vec = [0.1] * (dims + 1)  # Wrong size
        try:
            validate_embedding(wrong_vec, dims, model_key)
            print(f"    ERROR: Should have caught wrong dimension!")
        except EmbeddingValidationError as e:
            print(f"    [OK] Caught dimension error: {e}")

        # Test validation passes correct dimensions
        correct_vec = [0.1] * dims
        try:
            result = validate_embedding(correct_vec, dims, model_key)
            print(f"    [OK] Valid vector accepted: shape {result.shape}")
        except EmbeddingValidationError as e:
            print(f"    ERROR: Rejected valid vector: {e}")

        # Test NaN/Inf detection
        bad_vec = [0.1] * dims
        bad_vec[0] = float('nan')
        try:
            validate_embedding(bad_vec, dims, model_key)
            print(f"    ERROR: Should have caught NaN!")
        except EmbeddingValidationError as e:
            print(f"    [OK] Caught NaN: {e}")

    print("Dimension enforcement tests complete!")


if __name__ == "__main__":
    test_dimension_enforcement()