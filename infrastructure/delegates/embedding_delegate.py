#!/usr/bin/env python3
"""
Embedding Delegate - Infrastructure Layer
=========================================

Delegate for embedding/vector operations following hexagonal architecture.
Provides clean interface for adapters without exposing infrastructure details.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingDelegate:
    """
    Delegate for embedding and vector operations.

    Encapsulates all embedding infrastructure access.
    Adapters use this delegate instead of direct imports.
    """

    def __init__(self):
        """Initialize embedding delegate with lazy loading."""
        self._embedder = None
        self._vector_store = None
        self._initialized = False
        self._model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._target_dimension = self._load_target_dimension()

    def _load_target_dimension(self):
        """Load target dimension from config or use default."""
        try:
            from infrastructure.yaml_loader import get_settings_loader
            loader = get_settings_loader()
            # Use get_config_value with explicit default - NO HARDCODING!
            dimensions = loader.get_config_value('credentials.bedrock_llm.embeddings.dimensions', 1024)
            logger.info(f"EmbeddingDelegate using dimensions: {dimensions}")
            return dimensions
        except Exception as e:
            logger.warning(f"Could not load dimension config: {e}, using default 1024")
            # Only used if settings loading completely fails
            return 1024

    def _initialize(self):
        """Lazy initialization of embedding services."""
        if self._initialized:
            return True

        try:
            # Import only when needed (lazy loading)
            from sentence_transformers import SentenceTransformer
            import faiss

            # Initialize embedder
            self._embedder = SentenceTransformer(self._model_name)

            # Initialize vector store (FAISS)
            self._dimension = self._embedder.get_sentence_embedding_dimension()
            self._vector_store = faiss.IndexFlatL2(self._dimension)
            self._documents = []  # Store documents alongside vectors

            self._initialized = True
            logger.info("Embedding delegate initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"Embedding libraries not available: {e}")
            # Fallback to simple implementation
            self._initialized = self._initialize_fallback()
            return self._initialized

    def _initialize_fallback(self):
        """Initialize with fallback implementation."""
        try:
            # Use simple TF-IDF or word embeddings as fallback
            self._embedder = SimpleTFIDFEmbedder()
            self._vector_store = SimpleVectorStore()
            self._documents = []
            logger.info("Using fallback embedding implementation")
            return True
        except Exception as e:
            logger.error(f"Fallback initialization failed: {e}")
            return False

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if not self._initialize():
            return np.zeros(384)  # Default dimension

        try:
            if hasattr(self._embedder, 'encode'):
                # SentenceTransformer
                embedding = self._embedder.encode(text, convert_to_numpy=True)
            else:
                # Fallback embedder
                embedding = self._embedder.embed(text)

            return embedding

        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return np.zeros(384)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts

        Returns:
            Array of embeddings
        """
        if not self._initialize():
            return np.zeros((len(texts), 384))

        try:
            if hasattr(self._embedder, 'encode'):
                # SentenceTransformer
                embeddings = self._embedder.encode(texts, convert_to_numpy=True)
            else:
                # Fallback embedder
                embeddings = np.array([self._embedder.embed(t) for t in texts])

            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return np.zeros((len(texts), 384))

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to vector store.

        Args:
            documents: List of documents with 'content' field

        Returns:
            Success status
        """
        if not self._initialize():
            return False

        try:
            # Extract content
            texts = [doc.get('content', '') for doc in documents]

            # Generate embeddings
            embeddings = self.embed_batch(texts)

            # Add to vector store
            if hasattr(self._vector_store, 'add'):
                # FAISS
                self._vector_store.add(embeddings)
            else:
                # Fallback store
                self._vector_store.add_vectors(embeddings)

            # Store document metadata
            self._documents.extend(documents)

            logger.info(f"Added {len(documents)} documents to vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def search_similar(self, query: str, top_k: int = 5,
                      threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            top_k: Number of results
            threshold: Similarity threshold

        Returns:
            List of similar documents with scores
        """
        if not self._initialize() or not self._documents:
            return []

        try:
            # Embed query
            query_embedding = self.embed_text(query)

            # Search vector store
            if hasattr(self._vector_store, 'search'):
                # FAISS
                distances, indices = self._vector_store.search(
                    query_embedding.reshape(1, -1), top_k
                )
                # Convert distances to similarities
                similarities = 1 / (1 + distances[0])
            else:
                # Fallback store
                indices, similarities = self._vector_store.search(
                    query_embedding, top_k
                )

            # Build results
            results = []
            for idx, similarity in zip(indices, similarities):
                if idx < len(self._documents) and similarity >= threshold:
                    doc = self._documents[idx].copy()
                    doc['similarity_score'] = float(similarity)
                    results.append(doc)

            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        if not self._initialize():
            return 0.0

        try:
            # Get embeddings
            emb1 = self.embed_text(text1)
            emb2 = self.embed_text(text2)

            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            return float(similarity)

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

    def cluster_documents(self, documents: List[Dict[str, Any]],
                         n_clusters: int = 5) -> List[int]:
        """
        Cluster documents into groups.

        Args:
            documents: List of documents
            n_clusters: Number of clusters

        Returns:
            Cluster assignments
        """
        if not self._initialize():
            return list(range(len(documents)))

        try:
            from sklearn.cluster import KMeans

            # Extract content and embed
            texts = [doc.get('content', '') for doc in documents]
            embeddings = self.embed_batch(texts)

            # Perform clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(documents)))
            clusters = kmeans.fit_predict(embeddings)

            return clusters.tolist()

        except ImportError:
            logger.warning("sklearn not available for clustering")
            # Simple clustering fallback
            return [i % n_clusters for i in range(len(documents))]
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return list(range(len(documents)))

    def save_index(self, path: str) -> bool:
        """
        Save vector index to disk.

        Args:
            path: Save path

        Returns:
            Success status
        """
        if not self._initialize():
            return False

        try:
            import pickle

            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save index and documents
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'index': self._vector_store,
                    'documents': self._documents,
                    'model': self._model_name
                }, f)

            logger.info(f"Saved index to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """
        Load vector index from disk.

        Args:
            path: Load path

        Returns:
            Success status
        """
        if not self._initialize():
            return False

        try:
            import pickle

            with open(path, 'rb') as f:
                data = pickle.load(f)

            self._vector_store = data['index']
            self._documents = data['documents']
            self._model_name = data.get('model', self._model_name)

            logger.info(f"Loaded index from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def is_available(self) -> bool:
        """Check if embedding service is available."""
        return self._initialize()


class SimpleTFIDFEmbedder:
    """Simple TF-IDF based embedder as fallback."""

    def __init__(self):
        """Initialize TF-IDF embedder."""
        self.vocabulary = {}
        self.idf = {}
        self.dimension = 384

    def embed(self, text: str) -> np.ndarray:
        """Simple embedding using word frequencies."""
        # Tokenize
        words = text.lower().split()

        # Build vocabulary
        for word in words:
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary)

        # Create sparse vector
        vector = np.zeros(self.dimension)
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary[word] % self.dimension
                vector[idx] += 1

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector


class SimpleVectorStore:
    """Simple vector store as fallback."""

    def __init__(self):
        """Initialize simple store."""
        self.vectors = []

    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to store."""
        self.vectors.extend(vectors.tolist())

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        """Search for similar vectors."""
        if not self.vectors:
            return [], []

        # Compute similarities
        similarities = []
        for vec in self.vectors:
            similarity = np.dot(query_vector, vec) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-10
            )
            similarities.append(similarity)

        # Get top-k
        indices = np.argsort(similarities)[-top_k:][::-1]
        scores = [similarities[i] for i in indices]

        return indices.tolist(), scores


class EmbeddingDelegateFactory:
    """Factory for creating embedding delegates."""

    _instance = None

    @classmethod
    def get_delegate(cls) -> EmbeddingDelegate:
        """Get singleton embedding delegate instance."""
        if cls._instance is None:
            cls._instance = EmbeddingDelegate()
        return cls._instance


def get_embedding_delegate() -> EmbeddingDelegate:
    """Get embedding delegate instance."""
    return EmbeddingDelegateFactory.get_delegate()