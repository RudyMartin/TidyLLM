"""
FAISS Vector Manager
====================

Manages FAISS indexes with per-model/dimension isolation.
Prevents dimension mismatches and index pollution.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorManager:
    """
    Manages FAISS indexes with strict per-model isolation.

    CRITICAL: Each model+dimension combination gets its own index!
    """

    def __init__(self, base_path: str = "./faiss_indexes", s3_bucket: Optional[str] = None):
        """
        Initialize FAISS manager.

        Args:
            base_path: Local directory for indexes
            s3_bucket: Optional S3 bucket for persistence
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        self._index_cache = {}

        # Import FAISS lazily
        self._faiss = None

    def _get_faiss(self):
        """Lazy import of FAISS."""
        if self._faiss is None:
            try:
                import faiss
                self._faiss = faiss
            except ImportError:
                raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        return self._faiss

    def get_index_name(self, model_key: str, dimensions: int) -> str:
        """
        Generate index filename with model and dimensions.

        CRITICAL: Each model+dimension combo needs unique name!
        """
        return f"faiss_{model_key}_{dimensions}d.index"

    def get_index_path(self, model_key: str, dimensions: int) -> Path:
        """Get full path to index file."""
        return self.base_path / self.get_index_name(model_key, dimensions)

    def get_or_create_index(
        self,
        model_key: str,
        dimensions: int,
        force_rebuild: bool = False
    ):
        """
        Get or create FAISS index for model.

        Args:
            model_key: Model identifier
            dimensions: Expected dimensions
            force_rebuild: Force rebuild even if exists

        Returns:
            FAISS index object
        """
        faiss = self._get_faiss()
        cache_key = f"{model_key}_{dimensions}"

        # Check cache first
        if not force_rebuild and cache_key in self._index_cache:
            index = self._index_cache[cache_key]
            self._validate_index_dimensions(index, dimensions, model_key)
            return index

        # Check disk
        index_path = self.get_index_path(model_key, dimensions)

        if not force_rebuild and index_path.exists():
            try:
                index = self._load_index(index_path)
                self._validate_index_dimensions(index, dimensions, model_key)
                self._index_cache[cache_key] = index
                logger.info(f"Loaded index from {index_path}")
                return index
            except Exception as e:
                logger.warning(f"Failed to load index from {index_path}: {e}")
                logger.info("Will create new index")

        # Create new index
        index = self._create_index(dimensions)
        self._index_cache[cache_key] = index
        logger.info(f"Created new index for {model_key} with {dimensions}d")

        # Save to disk
        try:
            self._save_index(index, index_path)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

        return index

    def _create_index(self, dimensions: int):
        """Create new FAISS index."""
        faiss = self._get_faiss()

        # Use IndexFlatL2 for exact search (can upgrade to IVF later)
        index = faiss.IndexFlatL2(dimensions)

        # Add ID mapping for document retrieval
        index = faiss.IndexIDMap2(index)

        return index

    def _validate_index_dimensions(self, index, expected_dims: int, model_key: str):
        """
        Validate index dimensions match expected.

        CRITICAL: This prevents dimension mismatches!
        """
        actual_dims = index.d

        if actual_dims != expected_dims:
            raise RuntimeError(
                f"FAISS index dimension mismatch for {model_key}: "
                f"index has {actual_dims}d, expected {expected_dims}d. "
                f"Use force_rebuild=True to rebuild index."
            )

    def _load_index(self, path: Path):
        """Load FAISS index from disk."""
        faiss = self._get_faiss()
        return faiss.read_index(str(path))

    def _save_index(self, index, path: Path):
        """Save FAISS index to disk."""
        faiss = self._get_faiss()
        faiss.write_index(index, str(path))
        logger.info(f"Saved index to {path}")

    def add_embeddings(
        self,
        model_key: str,
        dimensions: int,
        embeddings: np.ndarray,
        ids: Optional[np.ndarray] = None
    ):
        """
        Add embeddings to index.

        Args:
            model_key: Model identifier
            dimensions: Embedding dimensions
            embeddings: Numpy array of embeddings (N x D)
            ids: Optional IDs for embeddings
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

        if embeddings.shape[1] != dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: got {embeddings.shape[1]}, "
                f"expected {dimensions}"
            )

        index = self.get_or_create_index(model_key, dimensions)

        # Generate IDs if not provided
        if ids is None:
            start_id = index.ntotal
            ids = np.arange(start_id, start_id + len(embeddings), dtype=np.int64)
        else:
            ids = ids.astype(np.int64)

        # Add to index
        index.add_with_ids(embeddings.astype(np.float32), ids)

        # Save updated index
        index_path = self.get_index_path(model_key, dimensions)
        self._save_index(index, index_path)

        logger.info(
            f"Added {len(embeddings)} embeddings to {model_key} index. "
            f"Total: {index.ntotal}"
        )

    def search(
        self,
        model_key: str,
        dimensions: int,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings.

        Args:
            model_key: Model identifier
            dimensions: Embedding dimensions
            query_embedding: Query vector
            k: Number of results

        Returns:
            Tuple of (distances, ids)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != dimensions:
            raise ValueError(
                f"Query dimension mismatch: got {query_embedding.shape[1]}, "
                f"expected {dimensions}"
            )

        index = self.get_or_create_index(model_key, dimensions)

        if index.ntotal == 0:
            logger.warning(f"Index for {model_key} is empty")
            return np.array([]), np.array([])

        # Search
        k = min(k, index.ntotal)
        distances, ids = index.search(query_embedding.astype(np.float32), k)

        return distances[0], ids[0]

    def get_index_stats(self, model_key: str, dimensions: int) -> Dict[str, Any]:
        """Get statistics for an index."""
        try:
            index = self.get_or_create_index(model_key, dimensions, force_rebuild=False)
            return {
                "model_key": model_key,
                "dimensions": dimensions,
                "total_vectors": index.ntotal,
                "index_type": type(index).__name__,
                "index_file": str(self.get_index_path(model_key, dimensions))
            }
        except Exception as e:
            return {
                "model_key": model_key,
                "dimensions": dimensions,
                "error": str(e)
            }

    def rebuild_index(
        self,
        model_key: str,
        dimensions: int,
        embeddings: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None
    ):
        """
        Rebuild index from scratch.

        Args:
            model_key: Model identifier
            dimensions: Embedding dimensions
            embeddings: Optional embeddings to add
            ids: Optional IDs for embeddings
        """
        logger.info(f"Rebuilding index for {model_key} with {dimensions}d")

        # Create new index (force rebuild)
        index = self.get_or_create_index(model_key, dimensions, force_rebuild=True)

        # Add embeddings if provided
        if embeddings is not None:
            self.add_embeddings(model_key, dimensions, embeddings, ids)

        logger.info(f"Index rebuilt for {model_key}")

    def cleanup_old_indexes(self, keep_models: List[Tuple[str, int]]):
        """
        Remove indexes not in keep list.

        Args:
            keep_models: List of (model_key, dimensions) tuples to keep
        """
        keep_files = {
            self.get_index_name(model, dims)
            for model, dims in keep_models
        }

        removed = []
        for index_file in self.base_path.glob("faiss_*.index"):
            if index_file.name not in keep_files:
                try:
                    index_file.unlink()
                    removed.append(index_file.name)
                    logger.info(f"Removed old index: {index_file.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {index_file.name}: {e}")

        if removed:
            logger.info(f"Cleaned up {len(removed)} old indexes")

        return removed


def test_vector_manager():
    """Test vector manager functionality."""
    print("Testing FAISS Vector Manager...")

    manager = FAISSVectorManager(base_path="./test_faiss")

    # Test different models with different dimensions
    test_configs = [
        ("titan_v1", 768),
        ("titan_v2", 1024),
        ("cohere_v3", 1024),
    ]

    for model_key, dims in test_configs:
        print(f"\nTesting {model_key} ({dims}d):")

        # Create some test embeddings
        embeddings = np.random.randn(10, dims).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add embeddings
        manager.add_embeddings(model_key, dims, embeddings)
        print(f"  Added 10 embeddings")

        # Search
        query = embeddings[0].reshape(1, -1)
        distances, ids = manager.search(model_key, dims, query, k=5)
        print(f"  Search found {len(ids)} results")
        print(f"  Top distance: {distances[0]:.4f} (should be ~0 for self)")

        # Get stats
        stats = manager.get_index_stats(model_key, dims)
        print(f"  Stats: {stats['total_vectors']} vectors in {stats['index_file']}")

        # Test dimension mismatch detection
        wrong_query = np.random.randn(dims + 1).astype(np.float32)
        try:
            manager.search(model_key, dims, wrong_query)
            print(f"  ERROR: Should have caught dimension mismatch!")
        except ValueError as e:
            print(f"  OK: Caught dimension mismatch")

    print("\nVector manager tests complete!")


if __name__ == "__main__":
    test_vector_manager()