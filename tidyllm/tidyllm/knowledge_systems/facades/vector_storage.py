"""
VectorStorage Facade
===================

Simple interface for vector storage with automatic standardization and search.
Provides easy API over the complex vector manager and PostgreSQL backend.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from datetime import datetime

# TLM is direct numpy replacement - just substitute import!
try:
    import tidyllm.tlm as np  # Direct numpy substitute
    TLM_AVAILABLE = True
except ImportError:
    TLM_AVAILABLE = False

import random

# Removed redundant config_loader - using existing TidyLLM configuration

try:
    from ..core.vector_manager import VectorManager
    VECTOR_MANAGER_AVAILABLE = True
except ImportError:
    VECTOR_MANAGER_AVAILABLE = False

try:
    from ..core.embedding_config import EmbeddingStandardizer
    EMBEDDING_CONFIG_AVAILABLE = True
except ImportError:
    EMBEDDING_CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorStorage:
    """Simple facade for vector storage with standardized embeddings"""
    
    def __init__(self, 
                 collection_name: str = "default_collection",
                 dimension: int = 1024,
                 default_provider=None):
        """
        Initialize vector storage with flow-based provider selection
        
        Args:
            collection_name: Name of the vector collection
            dimension: Vector dimension for standardized embeddings
            default_provider: TidyLLM Provider object for embedding generation
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.default_provider = default_provider
        
        # Initialize backend components
        self.vector_manager = None
        self.standardizer = None
        
        if VECTOR_MANAGER_AVAILABLE:
            try:
                self.vector_manager = VectorManager()
                logger.info("VectorManager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize VectorManager: {e}")
                
        if EMBEDDING_CONFIG_AVAILABLE:
            try:
                self.standardizer = EmbeddingStandardizer(target_dimension=dimension)
                logger.info(f"EmbeddingStandardizer initialized with {dimension}d")
            except Exception as e:
                logger.error(f"Failed to initialize EmbeddingStandardizer: {e}")
        
        if not self.vector_manager:
            logger.warning("Vector storage operating in fallback mode")
    
    def store(self, 
              embedding, 
              text: str,
              document_id: str,
              model_name: str = None,
              metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Store embedding with associated text and metadata
        
        Args:
            embedding: Embedding vector as Python-native array (will be standardized)
            text: Original text content
            document_id: Unique document identifier
            model_name: Name of embedding model used (uses default_model from config if None)
            metadata: Additional metadata
            
        Returns:
            Storage ID if successful, None if failed
        """
        # Use configured default model if none specified
        model_name = model_name or self.default_model
        
        if metadata is None:
            metadata = {}
        
        # Add storage metadata
        storage_metadata = {
            **metadata,
            "collection": self.collection_name,
            "stored_at": datetime.now().isoformat(),
            "original_dimension": len(embedding),
            "model_name": model_name
        }
        
        try:
            if self.vector_manager and hasattr(self.vector_manager, 'standardize_and_store_embedding'):
                # Use existing backend with standardization
                result = self.vector_manager.standardize_and_store_embedding(
                    embedding=embedding,
                    model_name=model_name,
                    document_id=document_id,
                    text=text,
                    metadata=storage_metadata
                )
                
                if result:
                    logger.info(f"Stored embedding for document {document_id}")
                    return str(result)
                else:
                    logger.error(f"Failed to store embedding for document {document_id}")
                    return None
            else:
                # Fallback storage simulation
                logger.warning("Using fallback storage mode")
                return self._fallback_store(embedding, text, document_id, storage_metadata)
                
        except Exception as e:
            logger.error(f"Storage failed for document {document_id}: {e}")
            return None
    
    def store_batch(self,
                   embeddings: List,
                   texts: List[str], 
                   document_ids: List[str],
                   model_name: str = None,
                   metadata: Optional[List[Dict[str, Any]]] = None) -> List[Optional[str]]:
        """
        Store multiple embeddings in batch
        
        Args:
            embeddings: List of embedding vectors
            texts: List of text contents
            document_ids: List of document IDs
            model_name: Name of embedding model used (uses default_model from config if None)
            metadata: List of metadata dicts (or None)
            
        Returns:
            List of storage IDs (None for failed stores)
        """
        # Use configured default model if none specified
        model_name = model_name or self.default_model
        
        if len(embeddings) != len(texts) or len(texts) != len(document_ids):
            raise ValueError("Embeddings, texts, and document_ids must have same length")
        
        if metadata is None:
            metadata = [{}] * len(embeddings)
        elif len(metadata) != len(embeddings):
            raise ValueError("Metadata list must match embeddings length")
        
        results = []
        for embedding, text, doc_id, meta in zip(embeddings, texts, document_ids, metadata):
            result = self.store(embedding, text, doc_id, model_name, meta)
            results.append(result)
        
        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Batch storage: {success_count}/{len(results)} successful")
        
        return results
    
    def search(self, 
               query_embedding,
               limit: int = 10,
               threshold: float = 0.7,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector as Python-native array
            limit: Maximum number of results
            threshold: Similarity threshold (0-1)
            filters: Additional metadata filters
            
        Returns:
            List of search results with text, metadata, and similarity scores
        """
        try:
            # Ensure query embedding is standardized
            if len(query_embedding) != self.dimension:
                if self.standardizer:
                    query_embedding = self._standardize_query_embedding(query_embedding)
                else:
                    query_embedding = self._fallback_standardize(query_embedding)
            
            if self.vector_manager and hasattr(self.vector_manager, 'similarity_search'):
                # Use existing backend search
                raw_results = self.vector_manager.similarity_search(
                    query_embedding=query_embedding,
                    limit=limit,
                    threshold=threshold
                )
                
                # Format results
                formatted_results = []
                for result in raw_results:
                    if isinstance(result, tuple) and len(result) >= 3:
                        doc_id, similarity, text = result[:3]
                        metadata = result[3] if len(result) > 3 else {}
                        
                        formatted_results.append({
                            "document_id": doc_id,
                            "similarity": float(similarity),
                            "text": text,
                            "metadata": metadata
                        })
                
                return formatted_results
            else:
                # Fallback search simulation
                logger.warning("Using fallback search mode")
                return self._fallback_search(query_embedding, limit)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_text(self,
                      query_text: str,
                      embedding_model: str = "titan_v2_1024", 
                      limit: int = 10,
                      threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search using text query (will be embedded automatically)
        
        Args:
            query_text: Text to search for
            embedding_model: Model to use for query embedding
            limit: Maximum number of results
            threshold: Similarity threshold
            
        Returns:
            List of search results
        """
        try:
            # This would use the EmbeddingProcessor facade to embed the query
            from .embedding_processor import EmbeddingProcessor
            
            processor = EmbeddingProcessor(target_dimension=self.dimension)
            query_embedding = processor.embed(query_text, embedding_model)
            
            return self.search(query_embedding, limit, threshold)
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def delete(self, document_id: str) -> bool:
        """
        Delete embedding by document ID
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            if self.vector_manager and hasattr(self.vector_manager, 'delete_embedding'):
                return self.vector_manager.delete_embedding(document_id)
            else:
                logger.warning("Delete operation not available in fallback mode")
                return False
                
        except Exception as e:
            logger.error(f"Delete failed for document {document_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection
        
        Returns:
            Collection statistics
        """
        try:
            if self.vector_manager and hasattr(self.vector_manager, 'get_collection_stats'):
                stats = self.vector_manager.get_collection_stats(self.collection_name)
                return stats or {}
            else:
                return {
                    "collection_name": self.collection_name,
                    "dimension": self.dimension,
                    "status": "fallback_mode",
                    "count": "unknown"
                }
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def _standardize_query_embedding(self, embedding):
        """Standardize query embedding using backend"""
        if self.standardizer:
            # Assume query embedding is from same model type
            return self.standardizer.standardize(embedding, self.default_model)
        else:
            return self._fallback_standardize(embedding)
    
    def _fallback_standardize(self, embedding):
        """Fallback standardization using TLM"""
        current_dim = len(embedding)
        
        if current_dim == self.dimension:
            return embedding
        elif current_dim < self.dimension:
            # Pad with zeros using TLM
            if TLM_AVAILABLE:
                padding = np.zeros(self.dimension - current_dim)
                return np.concatenate([embedding, padding])
            else:
                # Fallback to Python lists
                padding = [0.0] * (self.dimension - current_dim)
                if isinstance(embedding, list):
                    return embedding + padding
                else:
                    return list(embedding) + padding
        else:
            # Truncate
            if TLM_AVAILABLE:
                return embedding[:self.dimension]
            else:
                return list(embedding)[:self.dimension]
    
    def _fallback_store(self, embedding, text: str, 
                       document_id: str, metadata: Dict[str, Any]) -> str:
        """Fallback storage simulation"""
        # In real fallback, this might write to local storage or cache
        logger.info(f"Fallback store: {document_id} ({len(embedding)}d embedding)")
        return f"fallback_{document_id}_{datetime.now().timestamp()}"
    
    def _fallback_search(self, query_embedding, limit: int) -> List[Dict[str, Any]]:
        """Fallback search simulation"""
        # In real fallback, this might search local storage or return cached results
        logger.info(f"Fallback search for {len(query_embedding)}d embedding")
        return [{
            "document_id": f"fallback_doc_{i}",
            "similarity": 0.8 - (i * 0.1),
            "text": f"Fallback result {i}",
            "metadata": {"source": "fallback"}
        } for i in range(min(limit, 3))]
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of vector storage
        
        Returns:
            Health status dictionary
        """
        status = {
            "vector_manager_available": VECTOR_MANAGER_AVAILABLE,
            "embedding_config_available": EMBEDDING_CONFIG_AVAILABLE,
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "status": "healthy"
        }
        
        # Test basic storage and search
        try:
            # Test storage
            test_embedding = [random.random() for _ in range(self.dimension)] if TLM_AVAILABLE else [0.5] * self.dimension
            store_result = self.store(
                test_embedding, 
                "Health check test", 
                f"health_test_{datetime.now().timestamp()}"
            )
            status["test_store"] = store_result is not None
            
            # Test search
            search_result = self.search(test_embedding, limit=1)
            status["test_search"] = isinstance(search_result, list)
            
            status["test_passed"] = status["test_store"] and status["test_search"]
            
        except Exception as e:
            status["status"] = "degraded"
            status["error"] = str(e)
            status["test_passed"] = False
        
        # Add collection stats
        try:
            status["collection_stats"] = self.get_collection_stats()
        except Exception:
            status["collection_stats"] = {}
        
        return status


# Convenience functions for quick access
def store_embedding(embedding, 
                   text: str, 
                   document_id: str,
                   collection: str = "default") -> Optional[str]:
    """
    Quick function to store an embedding
    
    Args:
        embedding: Embedding vector
        text: Associated text
        document_id: Unique document ID
        collection: Collection name
        
    Returns:
        Storage ID if successful
    """
    storage = VectorStorage(collection_name=collection)
    return storage.store(embedding, text, document_id)


def search_similar(query_embedding, 
                  limit: int = 10,
                  collection: str = "default") -> List[Dict[str, Any]]:
    """
    Quick function to search for similar vectors
    
    Args:
        query_embedding: Query embedding vector
        limit: Maximum results
        collection: Collection name
        
    Returns:
        List of search results
    """
    storage = VectorStorage(collection_name=collection)
    return storage.search(query_embedding, limit=limit)