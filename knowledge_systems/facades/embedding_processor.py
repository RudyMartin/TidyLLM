"""
EmbeddingProcessor Facade
========================

Simple interface for embedding processing with automatic standardization.
Provides easy API over the complex embedding configuration and model discovery backend.
"""

from typing import List, Dict, Any, Optional, Union
import logging

# TLM is direct numpy replacement - just substitute import!
try:
    import tidyllm.tlm as np  # Direct numpy substitute
    TLM_AVAILABLE = True
except ImportError:
    TLM_AVAILABLE = False

import random

# Removed redundant config_loader - using existing embedding_config

try:
    from ..core.embedding_config import EmbeddingStandardizer
    from ..core.startup_model_discovery import get_startup_discovery
    EMBEDDING_CONFIG_AVAILABLE = True
except ImportError:
    EMBEDDING_CONFIG_AVAILABLE = False

try:
    import tidyllm
    TIDYLLM_AVAILABLE = True
except ImportError:
    TIDYLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Simple facade for embedding processing with automatic standardization"""
    
    def __init__(self, target_dimension: int = 1024, default_provider=None):
        """
        Initialize embedding processor with flow-based model selection
        
        Args:
            target_dimension: Target dimension for all embeddings
            default_provider: TidyLLM Provider object (e.g., bedrock()) - dynamic model selection
        """
        self.target_dimension = target_dimension
        self.default_provider = default_provider
        
        # Use existing TidyLLM configuration - no need to recreate mappings
        logger.info(f"EmbeddingProcessor initialized: {self.target_dimension}d target")
        logger.info("Using flow-based TidyLLM provider model selection")
        
    def embed(self, text: str, provider=None):
        """
        Generate embedding using TidyLLM's flow-based provider system
        
        Args:
            text: Text to embed  
            provider: TidyLLM Provider object (uses default_provider if None)
            
        Returns:
            Embedding vector as Python-native array (list)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        provider = provider or self.default_provider
        
        if not provider:
            raise ValueError("No provider specified. Use TidyLLM Provider objects like bedrock(), claude(), etc.")
        
        try:
            # Use TidyLLM's verb system with flow-based provider selection
            from tidyllm.verbs import embed, llm_message
            
            # Use TidyLLM's flow: message | embed(provider)
            message = llm_message(text)
            result_frame = message | embed(provider)
            
            # Extract embedding from DataTable result
            if hasattr(result_frame, 'to_py'):
                # Convert DataTable to Python data
                data = result_frame.to_py()
                if 'embedding' in data:
                    embedding = data['embedding'][0] if data['embedding'] else []
                    
                    # Ensure it's a Python-native array (list)
                    if not isinstance(embedding, list):
                        embedding = list(embedding)
                    
                    return embedding
            
            raise Exception("Could not extract embedding from TidyLLM result")
            
        except ImportError:
            raise ImportError("TidyLLM verbs not available")
        except Exception as e:
            logger.error(f"Embedding failed with provider {provider}: {e}")
            raise
    
    def embed_batch(self, texts: List[str], provider=None) -> List:
        """Generate embeddings for multiple texts using flow-based provider"""
        return [self.embed(text, provider) for text in texts]


# Convenience function for quick access
def embed_text(text: str, provider=None) -> List[float]:
    """
    Quick function to embed text using TidyLLM's flow-based providers
    
    Args:
        text: Text to embed
        provider: TidyLLM Provider object (e.g., bedrock(), claude())
        
    Returns:
        Embedding vector as Python-native array (list)
    """
    if not provider:
        # Import a default provider if none specified
        try:
            from tidyllm import bedrock
            provider = bedrock()
        except ImportError:
            raise ValueError("No provider specified and default bedrock() not available")
    
    processor = EmbeddingProcessor(default_provider=provider)
    return processor.embed(text)