import math
from typing import List, Dict, Tuple, Optional
from ..utils.tokenize import word_tokenize

Vector = List[float]
Matrix = List[Vector]

__all__ = ['fit', 'transform', 'fit_transform', 'WordAvgModel']

class WordAvgModel:
    """Word averaging sentence embedding model."""
    
    def __init__(self, embedding_dim: int = 100, use_idf: bool = True):
        self.vocabulary: Dict[str, int] = {}
        self.word_embeddings: Matrix = []
        self.idf_scores: Vector = []
        self.embedding_dim = embedding_dim
        self.use_idf = use_idf
        self.vocab_size = 0
    
    def _build_vocabulary(self, sentences: List[str]) -> None:
        """Build vocabulary from sentences."""
        vocab_set = set()
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            vocab_set.update(tokens)
        
        vocab_list = sorted(vocab_set)
        self.vocabulary = {word: i for i, word in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)
    
    def _init_random_embeddings(self, seed: Optional[int] = None) -> None:
        """Initialize random word embeddings."""
        import random
        rng = random.Random(seed)
        
        # Xavier/Glorot initialization
        bound = math.sqrt(6.0 / (self.vocab_size + self.embedding_dim))
        
        self.word_embeddings = []
        for _ in range(self.vocab_size):
            embedding = [rng.uniform(-bound, bound) for _ in range(self.embedding_dim)]
            self.word_embeddings.append(embedding)
    
    def _compute_idf(self, sentences: List[str]) -> None:
        """Compute IDF scores if enabled."""
        if not self.use_idf:
            self.idf_scores = [1.0] * self.vocab_size
            return
            
        n_docs = len(sentences)
        doc_frequencies = [0] * self.vocab_size
        
        # Count document frequencies
        for sentence in sentences:
            tokens = set(word_tokenize(sentence))
            for token in tokens:
                if token in self.vocabulary:
                    doc_frequencies[self.vocabulary[token]] += 1
        
        # Compute IDF scores
        self.idf_scores = []
        for df in doc_frequencies:
            if df == 0:
                idf = 0.0
            else:
                idf = math.log(n_docs / df)
            self.idf_scores.append(idf)
    
    def _sentence_to_embedding(self, sentence: str) -> Vector:
        """Convert sentence to averaged word embedding."""
        tokens = word_tokenize(sentence)
        
        if not tokens:
            return [0.0] * self.embedding_dim
        
        # Collect embeddings for tokens in vocabulary
        embeddings = []
        weights = []
        
        for token in tokens:
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                embeddings.append(self.word_embeddings[idx])
                weights.append(self.idf_scores[idx])
        
        if not embeddings:
            return [0.0] * self.embedding_dim
        
        # Weighted average of embeddings
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(weights)  # Fallback to uniform weights
            total_weight = len(weights)
        
        result = [0.0] * self.embedding_dim
        for embedding, weight in zip(embeddings, weights):
            for i in range(self.embedding_dim):
                result[i] += embedding[i] * weight
        
        # Normalize by total weight
        return [x / total_weight for x in result]

def fit(sentences: List[str], embedding_dim: int = 100, use_idf: bool = True, seed: Optional[int] = None) -> WordAvgModel:
    """Fit word averaging model on sentences."""
    model = WordAvgModel(embedding_dim, use_idf)
    model._build_vocabulary(sentences)
    model._init_random_embeddings(seed)
    model._compute_idf(sentences)
    return model

def transform(sentences: List[str], model: WordAvgModel) -> Matrix:
    """Transform sentences to averaged word embeddings."""
    return [model._sentence_to_embedding(sentence) for sentence in sentences]

def fit_transform(sentences: List[str], embedding_dim: int = 100, use_idf: bool = True, seed: Optional[int] = None) -> Tuple[Matrix, WordAvgModel]:
    """Fit model and transform sentences in one step."""
    model = fit(sentences, embedding_dim, use_idf, seed)
    embeddings = transform(sentences, model)
    return embeddings, model