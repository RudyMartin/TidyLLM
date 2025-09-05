import math
from typing import List, Dict, Tuple, Optional, Union
from ..utils.tokenize import char_ngrams, word_tokenize

Vector = List[float]
Matrix = List[Vector]

__all__ = ['fit', 'transform', 'fit_transform', 'NGramModel']

class NGramModel:
    """N-gram based sentence embedding model."""
    
    def __init__(self, n: int = 3, ngram_type: str = 'char', max_features: Optional[int] = None):
        self.n = n
        self.ngram_type = ngram_type  # 'char' or 'word'
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf_scores: Vector = []
        self.vocab_size = 0
    
    def _extract_ngrams(self, sentence: str) -> List[str]:
        """Extract n-grams from sentence."""
        if self.ngram_type == 'char':
            return char_ngrams(sentence, self.n)
        elif self.ngram_type == 'word':
            tokens = word_tokenize(sentence)
            if len(tokens) < self.n:
                return [' '.join(tokens)]
            
            ngrams = []
            for i in range(len(tokens) - self.n + 1):
                ngrams.append(' '.join(tokens[i:i+self.n]))
            return ngrams
        else:
            raise ValueError("ngram_type must be 'char' or 'word'")
    
    def _build_vocabulary(self, sentences: List[str]) -> None:
        """Build n-gram vocabulary from sentences."""
        ngram_counts = {}
        
        # Count all n-grams
        for sentence in sentences:
            ngrams = self._extract_ngrams(sentence)
            for ngram in ngrams:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        # Sort by frequency (most common first)
        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size if specified
        if self.max_features:
            sorted_ngrams = sorted_ngrams[:self.max_features]
        
        # Create vocabulary mapping
        self.vocabulary = {ngram: i for i, (ngram, _) in enumerate(sorted_ngrams)}
        self.vocab_size = len(self.vocabulary)
    
    def _compute_idf(self, sentences: List[str]) -> None:
        """Compute IDF scores for n-grams."""
        n_docs = len(sentences)
        doc_frequencies = [0] * self.vocab_size
        
        # Count document frequencies
        for sentence in sentences:
            ngrams_in_doc = set(self._extract_ngrams(sentence))
            for ngram in ngrams_in_doc:
                if ngram in self.vocabulary:
                    doc_frequencies[self.vocabulary[ngram]] += 1
        
        # Compute IDF scores
        self.idf_scores = []
        for df in doc_frequencies:
            if df == 0:
                idf = 0.0
            else:
                idf = math.log(n_docs / df)
            self.idf_scores.append(idf)
    
    def _sentence_to_embedding(self, sentence: str) -> Vector:
        """Convert sentence to n-gram TF-IDF embedding."""
        ngrams = self._extract_ngrams(sentence)
        ngram_counts = {}
        
        # Count n-gram frequencies
        for ngram in ngrams:
            if ngram in self.vocabulary:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        # Create TF-IDF vector
        embedding = [0.0] * self.vocab_size
        total_ngrams = len(ngrams)
        
        if total_ngrams == 0:
            return embedding
        
        for ngram, count in ngram_counts.items():
            idx = self.vocabulary[ngram]
            tf = count / total_ngrams
            idf = self.idf_scores[idx]
            embedding[idx] = tf * idf
        
        return embedding

def fit(sentences: List[str], n: int = 3, ngram_type: str = 'char', max_features: Optional[int] = None) -> NGramModel:
    """Fit n-gram embedding model on sentences."""
    model = NGramModel(n, ngram_type, max_features)
    model._build_vocabulary(sentences)
    model._compute_idf(sentences)
    return model

def transform(sentences: List[str], model: NGramModel) -> Matrix:
    """Transform sentences to n-gram embeddings."""
    return [model._sentence_to_embedding(sentence) for sentence in sentences]

def fit_transform(sentences: List[str], n: int = 3, ngram_type: str = 'char', max_features: Optional[int] = None) -> Tuple[Matrix, NGramModel]:
    """Fit model and transform sentences in one step."""
    model = fit(sentences, n, ngram_type, max_features)
    embeddings = transform(sentences, model)
    return embeddings, model