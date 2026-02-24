"""
Titan Embeddings Adapter
========================

Handles AWS Bedrock Titan embedding models with proper validation.
Fixes common issues: dimension mismatches, wrong request format, NaN/Inf values.
"""

import json
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TitanModel:
    """Titan model configuration."""
    model_id: str
    dimensions: int
    max_input_length: int


# Centralized model registry - SINGLE SOURCE OF TRUTH
TITAN_MODELS = {
    "titan_v1": TitanModel(
        model_id="amazon.titan-embed-text-v1",
        dimensions=768,
        max_input_length=8000
    ),
    "titan_v2": TitanModel(
        model_id="amazon.titan-embed-text-v2:0",
        dimensions=1024,
        max_input_length=8000
    ),
    "titan_g1": TitanModel(
        model_id="amazon.titan-embed-g1-text-02",
        dimensions=1536,
        max_input_length=8000
    )
}


class EmbeddingValidationError(Exception):
    """Raised when embedding validation fails."""
    pass


def get_titan_model(model_key: str) -> TitanModel:
    """Get Titan model configuration by key."""
    if model_key not in TITAN_MODELS:
        available = list(TITAN_MODELS.keys())
        raise ValueError(f"Unknown model_key: {model_key}. Available: {available}")
    return TITAN_MODELS[model_key]


def validate_embedding(vec: Any, expected_dim: int, model_key: str = None) -> np.ndarray:
    """
    Validate embedding vector with hard constraints.

    Args:
        vec: Raw embedding vector
        expected_dim: Expected dimensions
        model_key: Optional model key for better error messages

    Returns:
        Validated numpy array

    Raises:
        EmbeddingValidationError: If validation fails
    """
    model_info = f" for {model_key}" if model_key else ""

    if vec is None:
        raise EmbeddingValidationError(f"Embedding is None{model_info}")

    # Convert to numpy array
    if isinstance(vec, list):
        arr = np.asarray(vec, dtype=np.float32)
    elif isinstance(vec, np.ndarray):
        arr = vec.astype(np.float32, copy=False)
    else:
        raise EmbeddingValidationError(f"Unsupported type: {type(vec)}{model_info}")

    # Check shape
    if arr.ndim != 1:
        raise EmbeddingValidationError(
            f"Expected 1D vector, got shape {arr.shape}{model_info}"
        )

    # Check dimensions - CRITICAL CHECK
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


def create_titan_request(text: str, normalize: bool = True) -> Dict[str, Any]:
    """
    Create Titan embedding request body.

    CRITICAL: Titan expects 'inputText', not 'text'!
    """
    request = {
        "inputText": text  # NOT "text" - this is critical!
    }

    # Titan v2 supports normalize parameter
    if normalize:
        request["normalize"] = True

    return request


def parse_titan_response(response_data: Dict[str, Any], model_key: str) -> List[float]:
    """
    Parse Titan embedding response.

    Handles different field names across SDK versions.
    """
    # Try different field names
    vec = response_data.get("embedding")
    if vec is None:
        vec = response_data.get("vector")
    if vec is None:
        vec = response_data.get("embeddings", [None])[0]

    if vec is None:
        keys = list(response_data.keys())
        raise RuntimeError(
            f"No embedding field in Titan response for {model_key}. Keys: {keys}"
        )

    return vec


def titan_embed(
    bedrock_runtime,
    model_key: str,
    text: str,
    normalize: bool = True,
    validate: bool = True
) -> np.ndarray:
    """
    Generate Titan embedding with full validation.

    Args:
        bedrock_runtime: Boto3 Bedrock runtime client
        model_key: Model key (titan_v1, titan_v2, etc.)
        text: Text to embed
        normalize: Whether to L2 normalize
        validate: Whether to validate embedding

    Returns:
        Validated embedding vector
    """
    import time
    start_time = time.time()

    # Get model configuration
    model = get_titan_model(model_key)

    # Truncate if needed
    original_length = len(text)
    if original_length > model.max_input_length:
        text = text[:model.max_input_length]
        logger.warning(
            f"Truncated input from {original_length} to {model.max_input_length} chars"
        )

    try:
        # Create request body - USES inputText!
        body = create_titan_request(text, normalize)

        # Invoke model
        response = bedrock_runtime.invoke_model(
            modelId=model.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body).encode("utf-8")
        )

        # Parse response
        response_data = json.loads(response["body"].read().decode("utf-8"))
        raw_vec = parse_titan_response(response_data, model_key)

        # Validate if requested
        if validate:
            vec = validate_embedding(raw_vec, model.dimensions, model_key)
        else:
            vec = np.asarray(raw_vec, dtype=np.float32)

        # Optional L2 normalization (if not done by Titan)
        if normalize and not body.get("normalize"):
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        # Log telemetry
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Titan embed: model={model_key}, dims={model.dimensions}, "
            f"input_len={original_length}, truncated={original_length > model.max_input_length}, "
            f"latency_ms={elapsed_ms:.1f}, status=success"
        )

        return vec

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Titan embed failed: model={model_key}, error={str(e)}, "
            f"latency_ms={elapsed_ms:.1f}, status=failed"
        )
        raise


def get_faiss_index_name(model_key: str) -> str:
    """
    Generate FAISS index name with model and dimensions.

    CRITICAL: Each model+dimension combo needs its own index!
    """
    model = get_titan_model(model_key)
    return f"faiss_{model_key}_{model.dimensions}d.index"


def validate_faiss_index(faiss_index, model_key: str):
    """
    Validate FAISS index matches model dimensions.

    Args:
        faiss_index: FAISS index object
        model_key: Model key to validate against

    Raises:
        RuntimeError: If dimensions don't match
    """
    model = get_titan_model(model_key)
    index_dims = faiss_index.d  # FAISS stores dimensions as 'd'

    if index_dims != model.dimensions:
        raise RuntimeError(
            f"FAISS index dimension mismatch for {model_key}: "
            f"index has {index_dims}d, model expects {model.dimensions}d. "
            f"Index needs rebuild!"
        )


def test_titan_dimensions():
    """Test that Titan dimensions are enforced correctly."""
    print("Testing Titan dimension enforcement...")

    for model_key, model in TITAN_MODELS.items():
        print(f"\n{model_key}:")
        print(f"  Model ID: {model.model_id}")
        print(f"  Dimensions: {model.dimensions}")
        print(f"  Max input: {model.max_input_length}")

        # Test validation catches wrong dimensions
        wrong_vec = [0.1] * (model.dimensions + 1)
        try:
            validate_embedding(wrong_vec, model.dimensions, model_key)
            print(f"  ERROR: Should have caught wrong dimension!")
        except EmbeddingValidationError as e:
            print(f"  OK: Caught dimension error")

        # Test validation passes correct dimensions
        correct_vec = [0.1] * model.dimensions
        try:
            result = validate_embedding(correct_vec, model.dimensions, model_key)
            print(f"  OK: Valid vector accepted: shape {result.shape}")
        except EmbeddingValidationError as e:
            print(f"  ERROR: Rejected valid vector: {e}")

        # Test NaN detection
        bad_vec = [0.1] * model.dimensions
        bad_vec[0] = float('nan')
        try:
            validate_embedding(bad_vec, model.dimensions, model_key)
            print(f"  ERROR: Should have caught NaN!")
        except EmbeddingValidationError as e:
            print(f"  OK: Caught NaN")

        # Test index naming
        index_name = get_faiss_index_name(model_key)
        print(f"  FAISS index: {index_name}")

    print("\nDimension enforcement tests complete!")


if __name__ == "__main__":
    test_titan_dimensions()