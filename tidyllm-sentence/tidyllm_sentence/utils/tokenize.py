import re
import string
from typing import List, Tuple

__all__ = ['simple_tokenize', 'word_tokenize', 'char_ngrams', 'word_ngrams', 'preprocess_text']

def preprocess_text(text: str, lowercase: bool = True, remove_punct: bool = True) -> str:
    """Basic text preprocessing."""
    if lowercase:
        text = text.lower()
    
    if remove_punct:
        # Remove punctuation except spaces
        text = ''.join(c if c not in string.punctuation else ' ' for c in text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return preprocess_text(text).split()

def word_tokenize(text: str) -> List[str]:
    """Basic word tokenization with simple rules."""
    text = preprocess_text(text)
    
    # Split on whitespace and common separators
    tokens = re.split(r'[\s\-_]+', text)
    
    # Filter empty tokens
    return [token for token in tokens if token.strip()]

def char_ngrams(text: str, n: int = 3, pad_char: str = ' ') -> List[str]:
    """Generate character n-grams from text."""
    text = preprocess_text(text, remove_punct=False)
    
    # Add padding
    padded = pad_char * (n-1) + text + pad_char * (n-1)
    
    ngrams = []
    for i in range(len(padded) - n + 1):
        ngrams.append(padded[i:i+n])
    
    return ngrams

def word_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generate word n-grams from tokens."""
    if len(tokens) < n:
        return [' '.join(tokens)]
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i+n]))
    
    return ngrams