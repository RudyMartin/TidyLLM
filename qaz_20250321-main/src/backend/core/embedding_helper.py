#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Embedding Helper with Model Tracking and Dimension Management

Provides vector embedding generation and management with comprehensive tracking
of which model generated which embeddings and their original dimensions.
"""

import logging
from .datamart_numpy_substitution import np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetadata:
    """Metadata for embedding generation"""
    model_name: str
    original_dimensions: int
    target_dimensions: int
    padding_method: str  # 'zero_pad', 'truncate', 'none'
    content_hash: str
    generation_timestamp: str


class EmbeddingHelper:
    """Enhanced embedding helper with model tracking and dimension management"""
    
    def __init__(self, target_dimensions: int = 1024):
        self.target_dimensions = target_dimensions
        self.embedding_model = None
        self.model_metadata = {}
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model with fallback options"""
        # Try to use a model that produces close to target dimensions
        model_options = [
            ('all-mpnet-base-v2', 768),  # Closest to 1024
            ('all-MiniLM-L12-v2', 384),
            ('all-MiniLM-L6-v2', 384),
        ]
        
        for model_name, expected_dim in model_options:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                test_embedding = self.embedding_model.encode("test")
                actual_dim = len(test_embedding)
                
                self.model_metadata = {
                    'model_name': model_name,
                    'original_dimensions': actual_dim,
                    'target_dimensions': self.target_dimensions,
                    'needs_padding': actual_dim < self.target_dimensions,
                    'needs_truncation': actual_dim > self.target_dimensions,
                    'padding_dimensions': max(0, self.target_dimensions - actual_dim),
                    'truncation_dimensions': max(0, actual_dim - self.target_dimensions)
                }
                
                logger.info(f"✅ Initialized embedding model: {model_name}")
                logger.info(f"   Original dimensions: {actual_dim}")
                logger.info(f"   Target dimensions: {self.target_dimensions}")
                if self.model_metadata['needs_padding']:
                    logger.info(f"   Will pad with {self.model_metadata['padding_dimensions']} zeros")
                elif self.model_metadata['needs_truncation']:
                    logger.info(f"   Will truncate {self.model_metadata['truncation_dimensions']} dimensions")
                else:
                    logger.info(f"   Perfect match - no adjustment needed")
                
                break
                
            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {e}")
                continue
        
        if not self.embedding_model:
            raise RuntimeError("Failed to initialize any embedding model")
    
    def generate_embedding(self, text: str, content_id: str = None) -> Tuple[List, EmbeddingMetadata]:
        """
        Generate embedding with comprehensive metadata tracking
        
        Args:
            text: Text to embed
            content_id: Optional content identifier for tracking
            
        Returns:
            Tuple of (embedding_array, metadata)
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        # Generate original embedding
        original_embedding = self.embedding_model.encode(text)
        original_dim = len(original_embedding)
        
        # Adjust dimensions to target
        if original_dim < self.target_dimensions:
            # Pad with zeros using DataMart substitution
            adjusted_embedding = np.pad(
                original_embedding, 
                (0, self.target_dimensions - original_dim), 
                mode='constant', 
                constant_values=0.0
            )
            # Convert to list if it's a datatable Frame
            if hasattr(adjusted_embedding, 'to_list'):
                adjusted_embedding = adjusted_embedding.to_list()[0]
            padding_method = 'zero_pad'
        elif original_dim > self.target_dimensions:
            # Truncate
            adjusted_embedding = original_embedding[:self.target_dimensions]
            padding_method = 'truncate'
        else:
            # Perfect match
            adjusted_embedding = original_embedding
            padding_method = 'none'
        
        # Create metadata
        content_hash = hashlib.sha1(text.encode()).hexdigest()
        metadata = EmbeddingMetadata(
            model_name=self.model_metadata['model_name'],
            original_dimensions=original_dim,
            target_dimensions=self.target_dimensions,
            padding_method=padding_method,
            content_hash=content_hash,
            generation_timestamp=str(np.datetime64('now'))
        )
        
        logger.debug(f"Generated embedding for content {content_id}: "
                    f"{original_dim} -> {len(adjusted_embedding)} dimensions "
                    f"({padding_method})")
        
        return adjusted_embedding, metadata
    
    def generate_batch_embeddings(self, texts: List[str], content_ids: List[str] = None) -> Tuple[List[List], List[EmbeddingMetadata]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            content_ids: Optional list of content identifiers
            
        Returns:
            Tuple of (embedding_arrays, metadata_list)
        """
        if content_ids is None:
            content_ids = [f"content_{i}" for i in range(len(texts))]
        
        embeddings = []
        metadata_list = []
        
        for text, content_id in zip(texts, content_ids):
            embedding, metadata = self.generate_embedding(text, content_id)
            embeddings.append(embedding)
            metadata_list.append(metadata)
        
        return embeddings, metadata_list
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        return {
            'model_metadata': self.model_metadata,
            'target_dimensions': self.target_dimensions,
            'model_loaded': self.embedding_model is not None
        }
    
    def validate_embedding_dimensions(self, embedding: List) -> bool:
        """Validate that embedding has correct dimensions"""
        return len(embedding) == self.target_dimensions


class AmazonEmbeddingVectorizer:
    """Placeholder for Amazon Bedrock embedding integration"""
    
    def __init__(self, model_name: str = "titan-embed-text-v1"):
        self.model_name = model_name
        self.target_dimensions = 1024  # Titan v1 produces 1024 dimensions
        logger.info(f"Amazon Bedrock embedding vectorizer initialized for {model_name}")
    
    def generate_embedding(self, text: str) -> List:
        """
        Generate embedding using Amazon Bedrock (placeholder)
        
        In production, this would use boto3 to call Bedrock
        """
        # Placeholder - in production this would call Bedrock API
        logger.warning("Amazon Bedrock embedding not implemented - using placeholder")
        
        # Return a placeholder embedding of correct dimensions using DataMart substitution
        placeholder_embedding = np.random.normal(0, 1, self.target_dimensions)
        # Convert to list if it's a datatable Frame
        if hasattr(placeholder_embedding, 'to_list'):
            placeholder_embedding = placeholder_embedding.to_list()
        # Normalize to unit vector
        norm_value = np.linalg.norm(placeholder_embedding)
        if norm_value > 0:
            placeholder_embedding = [x / norm_value for x in placeholder_embedding]
        
        return placeholder_embedding
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Amazon Bedrock model"""
        return {
            'model_name': self.model_name,
            'target_dimensions': self.target_dimensions,
            'provider': 'amazon_bedrock',
            'status': 'placeholder'
        }


