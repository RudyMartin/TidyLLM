"""
Embedding Configuration for Multi-Model Support
==============================================

Standardizes embedding dimensions across different models to enable
unified vector storage in pgvector or FAISS.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelEmbeddingConfig:
    """Configuration for a specific embedding model"""
    model_id: str
    native_dimension: int
    display_name: str
    provider: str = "bedrock"
    
@dataclass 
class EmbeddingConfig:
    """Global embedding configuration"""
    target_dimension: int = 1024  # Standardized dimension for all vectors
    padding_strategy: str = "zeros"  # "zeros", "random", or "repeat"
    
# Model configurations with their native dimensions
EMBEDDING_MODELS = {
    "titan_v1": ModelEmbeddingConfig(
        model_id="amazon.titan-embed-text-v1",
        native_dimension=1536,
        display_name="Titan Embed v1",
        provider="bedrock"
    ),
    "titan_v2": ModelEmbeddingConfig(
        model_id="amazon.titan-embed-text-v2:0", 
        native_dimension=1024,
        display_name="Titan Embed v2",
        provider="bedrock"
    ),
    "titan_v2_1024": ModelEmbeddingConfig(
        model_id="amazon.titan-embed-text-v2:0",
        native_dimension=1024,
        display_name="Titan Embed v2 (1024d)",
        provider="bedrock"
    ),
    "titan_v2_256": ModelEmbeddingConfig(
        model_id="amazon.titan-embed-text-v2:0",
        native_dimension=256,  # Titan v2 supports multiple dimensions
        display_name="Titan Embed v2 (256)",
        provider="bedrock"
    ),
    "titan_v2_512": ModelEmbeddingConfig(
        model_id="amazon.titan-embed-text-v2:0",
        native_dimension=512,  # Titan v2 supports multiple dimensions
        display_name="Titan Embed v2 (512)",
        provider="bedrock"
    ),
    "cohere_english": ModelEmbeddingConfig(
        model_id="cohere.embed-english-v3",
        native_dimension=384,
        display_name="Cohere English v3",
        provider="bedrock"
    ),
    "cohere_multilingual": ModelEmbeddingConfig(
        model_id="cohere.embed-multilingual-v3",
        native_dimension=384,
        display_name="Cohere Multilingual v3",
        provider="bedrock"
    ),
    "openai_small": ModelEmbeddingConfig(
        model_id="text-embedding-3-small",
        native_dimension=1536,
        display_name="OpenAI Small",
        provider="openai"
    ),
    "openai_large": ModelEmbeddingConfig(
        model_id="text-embedding-3-large",
        native_dimension=3072,
        display_name="OpenAI Large",
        provider="openai"
    )
}

class EmbeddingStandardizer:
    """Standardizes embeddings to a target dimension"""
    
    def __init__(self, target_dimension: int = 1024, padding_strategy: str = "zeros"):
        """
        Initialize embedding standardizer
        
        Args:
            target_dimension: Target dimension for all embeddings (default: 1024)
            padding_strategy: How to pad smaller embeddings:
                - "zeros": Pad with zeros
                - "random": Pad with small random values
                - "repeat": Repeat the vector cyclically
        """
        self.target_dimension = target_dimension
        self.padding_strategy = padding_strategy
    
    def standardize(self, embedding: List[float], model_key: str) -> List[float]:
        """
        Standardize embedding to target dimension
        
        Args:
            embedding: Raw embedding from model
            model_key: Key identifying the model used
            
        Returns:
            Standardized embedding with target dimension
        """
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        
        model_config = EMBEDDING_MODELS[model_key]
        current_dim = len(embedding)
        
        # If already at target dimension, return as-is
        if current_dim == self.target_dimension:
            return embedding
        
        # If larger than target, truncate
        if current_dim > self.target_dimension:
            return embedding[:self.target_dimension]
        
        # If smaller than target, pad
        return self._pad_embedding(embedding, self.target_dimension)
    
    def _pad_embedding(self, embedding: List[float], target_dim: int) -> List[float]:
        """Pad embedding to target dimension"""
        current_dim = len(embedding)
        padding_size = target_dim - current_dim
        
        if self.padding_strategy == "zeros":
            # Pad with zeros (most common)
            return embedding + [0.0] * padding_size
            
        elif self.padding_strategy == "random":
            # Pad with small random values
            import random
            padding = [random.gauss(0, 0.01) for _ in range(padding_size)]
            return embedding + padding
            
        elif self.padding_strategy == "repeat":
            # Repeat the vector cyclically
            result = embedding.copy()
            for i in range(padding_size):
                result.append(embedding[i % current_dim])
            return result
            
        else:
            raise ValueError(f"Unknown padding strategy: {self.padding_strategy}")
    
    def get_model_info(self, model_key: str) -> Dict[str, any]:
        """Get information about a model's embedding configuration"""
        if model_key not in EMBEDDING_MODELS:
            return None
        
        model = EMBEDDING_MODELS[model_key]
        return {
            "model_key": model_key,
            "model_id": model.model_id,
            "native_dimension": model.native_dimension,
            "target_dimension": self.target_dimension,
            "needs_padding": model.native_dimension < self.target_dimension,
            "needs_truncation": model.native_dimension > self.target_dimension,
            "display_name": model.display_name
        }
    
    def get_dimension_summary(self) -> Dict[str, any]:
        """Get summary of all models and their dimension handling"""
        summary = {
            "target_dimension": self.target_dimension,
            "padding_strategy": self.padding_strategy,
            "models": {}
        }
        
        for model_key in EMBEDDING_MODELS:
            info = self.get_model_info(model_key)
            summary["models"][model_key] = {
                "native": info["native_dimension"],
                "action": "pad" if info["needs_padding"] else "truncate" if info["needs_truncation"] else "none",
                "difference": abs(info["native_dimension"] - self.target_dimension)
            }
        
        return summary

# Global instance for easy access
embedding_standardizer = EmbeddingStandardizer(target_dimension=1024)

def standardize_embedding(embedding: List[float], model_key: str) -> List[float]:
    """Convenience function to standardize embeddings"""
    return embedding_standardizer.standardize(embedding, model_key)

def get_target_dimension() -> int:
    """Get the target dimension for standardized embeddings"""
    return embedding_standardizer.target_dimension