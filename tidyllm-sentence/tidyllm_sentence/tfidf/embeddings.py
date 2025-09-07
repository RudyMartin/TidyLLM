import math
from typing import List, Dict, Tuple, Optional
from ..utils.preprocessing import preprocess_for_embeddings, PreprocessingPipeline, STANDARD_PIPELINE
from ..utils.corpus import enhance_idf_with_corpus

Vector = List[float]
Matrix = List[Vector]

__all__ = ['fit', 'transform', 'fit_transform', 'TFIDFModel']

class TFIDFModel:
    """TF-IDF sentence embedding model with proper preprocessing."""
    
    def __init__(self, preprocessor: Optional[PreprocessingPipeline] = None):
        self.vocabulary: Dict[str, int] = {}
        self.idf_scores: Vector = []
        self.vocab_size: int = 0
        self.preprocessor = preprocessor or STANDARD_PIPELINE
        
    def _build_vocabulary(self, sentences: List[str]) -> None:
        """Build vocabulary from preprocessed sentences."""
        vocab_set = set()
        for sentence in sentences:
            tokens = self.preprocessor.process(sentence)
            vocab_set.update(tokens)
        
        # Sort for consistent ordering
        vocab_list = sorted(vocab_set)
        self.vocabulary = {word: i for i, word in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)
    
    def _compute_idf(self, sentences: List[str]) -> None:
        """Compute enhanced IDF scores using preprocessing and corpus knowledge."""
        n_docs = len(sentences)
        doc_frequencies = {}
        
        # Count document frequencies with preprocessing
        for sentence in sentences:
            tokens = set(self.preprocessor.process(sentence))  # Unique tokens per doc
            for token in tokens:
                if token in self.vocabulary:
                    doc_frequencies[token] = doc_frequencies.get(token, 0) + 1
        
        # Enhance IDF with corpus knowledge
        enhanced_idfs = enhance_idf_with_corpus(doc_frequencies, n_docs)
        
        # Convert to vector format
        self.idf_scores = [0.0] * self.vocab_size
        for word, idf in enhanced_idfs.items():
            if word in self.vocabulary:
                self.idf_scores[self.vocabulary[word]] = idf
    
    def _sentence_to_tfidf(self, sentence: str) -> Vector:
        """Convert single sentence to TF-IDF vector with preprocessing."""
        tokens = self.preprocessor.process(sentence)
        token_counts = {}
        
        # Count term frequencies
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Create TF-IDF vector
        tfidf_vector = [0.0] * self.vocab_size
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return tfidf_vector
        
        for token, count in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = count / total_tokens
                idf = self.idf_scores[idx]
                tfidf_vector[idx] = tf * idf
        
        return tfidf_vector

def fit(sentences: List[str], preprocessor: Optional[PreprocessingPipeline] = None) -> TFIDFModel:
    """Fit TF-IDF model on sentences with preprocessing."""
    model = TFIDFModel(preprocessor)
    model._build_vocabulary(sentences)
    model._compute_idf(sentences)
    return model

def transform(sentences: List[str], model: TFIDFModel) -> Matrix:
    """Transform sentences to TF-IDF embeddings."""
    return [model._sentence_to_tfidf(sentence) for sentence in sentences]

def fit_transform(sentences: List[str], preprocessor: Optional[PreprocessingPipeline] = None) -> Tuple[Matrix, TFIDFModel]:
    """Fit model and transform sentences in one step."""
    model = fit(sentences, preprocessor)
    embeddings = transform(sentences, model)
    return embeddings, model