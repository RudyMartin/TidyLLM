"""
Complete text preprocessing pipeline for embeddings.
Combines tokenization, stop word removal, stemming, and normalization.
"""

import re
from typing import List, Optional, Set
from .tokenize import preprocess_text as basic_preprocess
from .stemmer import simple_stem, porter_stem
from .stopwords import ENGLISH_STOP_WORDS, CORE_STOP_WORDS, remove_stopwords

__all__ = ['preprocess_for_embeddings', 'PreprocessingPipeline', 
           'normalize_text', 'extract_words']

def normalize_text(text: str) -> str:
    """Advanced text normalization."""
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    
    # Replace numbers (optionally preserve them)
    text = re.sub(r'\b\d+\.?\d*\b', '<NUM>', text)
    
    # Expand contractions
    contractions = {
        "don't": "do not", "won't": "will not", "can't": "cannot",
        "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am", "it's": "it is", "that's": "that is",
        "what's": "what is", "here's": "here is", "there's": "there is",
        "where's": "where is", "who's": "who is", "how's": "how is"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_words(text: str) -> List[str]:
    """Extract meaningful words from text."""
    # Normalize first
    text = normalize_text(text)
    
    # Extract words (alphabetic, length > 1)
    words = re.findall(r'\b[a-z]{2,}\b', text)
    
    return words

def preprocess_for_embeddings(
    text: str,
    remove_stops: bool = True,
    stem: str = 'simple',  # 'simple', 'porter', or None
    stop_words: Optional[Set[str]] = None,
    min_length: int = 2,
    max_length: int = 20
) -> List[str]:
    """
    Complete preprocessing pipeline for embeddings.
    
    Args:
        text: Input text
        remove_stops: Remove stop words
        stem: Stemming method ('simple', 'porter', or None)
        stop_words: Custom stop words set (default: ENGLISH_STOP_WORDS)
        min_length: Minimum word length
        max_length: Maximum word length
    
    Returns:
        List of processed tokens
    """
    if not text or not text.strip():
        return []
    
    # Extract words
    words = extract_words(text)
    
    # Length filtering
    words = [w for w in words if min_length <= len(w) <= max_length]
    
    # Remove stop words
    if remove_stops:
        if stop_words is None:
            stop_words = ENGLISH_STOP_WORDS
        words = [w for w in words if w not in stop_words]
    
    # Stemming
    if stem == 'simple':
        words = [simple_stem(w) for w in words]
    elif stem == 'porter':
        words = [porter_stem(w) for w in words]
    
    # Final cleanup - remove very short stems
    words = [w for w in words if len(w) >= 2]
    
    return words

class PreprocessingPipeline:
    """Configurable preprocessing pipeline for consistent processing."""
    
    def __init__(
        self,
        remove_stops: bool = True,
        stem: str = 'simple',
        stop_words: Optional[Set[str]] = None,
        min_length: int = 2,
        max_length: int = 20,
        preserve_case: bool = False
    ):
        self.remove_stops = remove_stops
        self.stem = stem
        self.stop_words = stop_words or ENGLISH_STOP_WORDS
        self.min_length = min_length
        self.max_length = max_length
        self.preserve_case = preserve_case
        
    def process(self, text: str) -> List[str]:
        """Process a single text."""
        return preprocess_for_embeddings(
            text,
            remove_stops=self.remove_stops,
            stem=self.stem,
            stop_words=self.stop_words,
            min_length=self.min_length,
            max_length=self.max_length
        )
    
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """Process multiple texts."""
        return [self.process(text) for text in texts]
    
    def get_vocabulary(self, texts: List[str]) -> Set[str]:
        """Get vocabulary from texts after preprocessing."""
        vocab = set()
        for text in texts:
            tokens = self.process(text)
            vocab.update(tokens)
        return vocab
    
    def __repr__(self):
        return (f"PreprocessingPipeline(remove_stops={self.remove_stops}, "
                f"stem='{self.stem}', min_length={self.min_length})")

# Preset pipelines for common use cases
MINIMAL_PIPELINE = PreprocessingPipeline(
    remove_stops=False,
    stem=None,
    min_length=1
)

STANDARD_PIPELINE = PreprocessingPipeline(
    remove_stops=True,
    stem='simple',
    min_length=2
)

AGGRESSIVE_PIPELINE = PreprocessingPipeline(
    remove_stops=True,
    stem='porter',
    min_length=3,
    max_length=15
)