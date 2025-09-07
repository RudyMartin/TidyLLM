import math
from typing import List, Tuple, Union

__all__ = ['cosine_similarity', 'euclidean_distance', 'semantic_search', 'normalize_vector']

Vector = List[float]
Matrix = List[Vector]

def dot_product(a: Vector, b: Vector) -> float:
    """Compute dot product of two vectors."""
    assert len(a) == len(b), "Vectors must have same length"
    return sum(x * y for x, y in zip(a, b))

def vector_norm(v: Vector) -> float:
    """Compute L2 norm of vector."""
    return math.sqrt(sum(x * x for x in v))

def normalize_vector(v: Vector) -> Vector:
    """L2 normalize vector to unit length."""
    norm = vector_norm(v)
    if norm == 0:
        return [0.0] * len(v)
    return [x / norm for x in v]

def cosine_similarity(a: Vector, b: Vector) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product(a, b) / (norm_a * norm_b)

def euclidean_distance(a: Vector, b: Vector) -> float:
    """Compute Euclidean distance between two vectors."""
    assert len(a) == len(b), "Vectors must have same length"
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def semantic_search(query_embedding: Vector, corpus_embeddings: Matrix, top_k: int = 5) -> List[Tuple[int, float]]:
    """Find most similar embeddings to query."""
    similarities = []
    
    for i, corpus_emb in enumerate(corpus_embeddings):
        sim = cosine_similarity(query_embedding, corpus_emb)
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]