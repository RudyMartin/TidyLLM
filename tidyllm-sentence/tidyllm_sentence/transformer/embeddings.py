"""
Transformer-enhanced TF-IDF embeddings for tidyllm-sentence.

Combines the speed of TF-IDF with the semantic power of attention.
Still pure Python, still educational, but now context-aware!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'tlm'))

import tlm
from typing import List, Dict, Tuple, Optional
from ..tfidf.embeddings import TFIDFModel, fit as tfidf_fit
from ..utils.preprocessing import PreprocessingPipeline, STANDARD_PIPELINE

Vector = List[float]
Matrix = List[Vector]

class TransformerTFIDFModel:
    """
    TF-IDF + lightweight transformer for semantic similarity.
    
    Architecture:
    1. Standard TF-IDF embeddings (word-level)
    2. Positional encoding  
    3. Self-attention layer
    4. Sentence pooling (mean pooling)
    
    This gives us context-aware sentence embeddings!
    """
    
    def __init__(self, 
                 preprocessor: Optional[PreprocessingPipeline] = None,
                 attention_heads: int = 4,
                 max_seq_len: int = 64):
        self.tfidf_model = TFIDFModel(preprocessor or STANDARD_PIPELINE)
        self.attention_heads = attention_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = 0
        self.d_model = 0  # Will be set based on vocabulary size
        
    def _build_word_embeddings(self, sentences: List[str]) -> None:
        """Build TF-IDF model for word-level embeddings."""
        # First build vocabulary and IDF scores
        self.tfidf_model._build_vocabulary(sentences)
        self.tfidf_model._compute_idf(sentences)
        
        # Set model dimensions
        self.vocab_size = self.tfidf_model.vocab_size
        self.d_model = self.vocab_size  # Use vocabulary size as embedding dimension
        
    def _sentence_to_word_sequence(self, sentence: str) -> Tuple[Matrix, int]:
        """
        Convert sentence to sequence of word embeddings.
        
        Returns:
            (word_embeddings, actual_length)
            word_embeddings: [seq_len, d_model] padded to max_seq_len
        """
        tokens = self.tfidf_model.preprocessor.process(sentence)
        actual_length = len(tokens)
        
        # Create word-level embeddings
        word_embeddings = []
        for token in tokens:
            # Create one-hot-like embedding for this word
            embedding = [0.0] * self.d_model
            if token in self.tfidf_model.vocabulary:
                idx = self.tfidf_model.vocabulary[token]
                # Use IDF score as embedding weight (context-free)
                embedding[idx] = self.tfidf_model.idf_scores[idx]
            word_embeddings.append(embedding)
        
        # Pad or truncate to max_seq_len
        if len(word_embeddings) > self.max_seq_len:
            word_embeddings = word_embeddings[:self.max_seq_len]
            actual_length = self.max_seq_len
        else:
            # Pad with zeros
            while len(word_embeddings) < self.max_seq_len:
                word_embeddings.append([0.0] * self.d_model)
                
        return word_embeddings, actual_length
    
    def _apply_transformer_layer(self, word_embeddings: Matrix, seq_length: int) -> Matrix:
        """
        Apply transformer layer to word embeddings.
        
        Steps:
        1. Add positional encoding
        2. Self-attention  
        3. Residual connection (simplified)
        """
        # Step 1: Add positional encoding
        pos_encoding = tlm.positional_encoding(len(word_embeddings), self.d_model)
        embedded_with_pos = []
        for i, (word_emb, pos_emb) in enumerate(zip(word_embeddings, pos_encoding)):
            if i < seq_length:  # Only for actual words, not padding
                combined = [w + p for w, p in zip(word_emb, pos_emb)]
            else:
                combined = word_emb  # Keep padding as zeros
            embedded_with_pos.append(combined)
        
        # Step 2: Self-attention (Q=K=V=input)
        attended, attention_weights = tlm.scaled_dot_product_attention(
            embedded_with_pos, embedded_with_pos, embedded_with_pos
        )
        
        # Step 3: Simple residual connection (add input back)
        residual_output = []
        for orig, att in zip(embedded_with_pos, attended):
            # residual = orig + attended (simplified, no layer norm)
            residual_row = [o + a for o, a in zip(orig, att)]
            residual_output.append(residual_row)
        
        return residual_output
    
    def _pool_to_sentence_embedding(self, contextualized_words: Matrix, seq_length: int) -> Vector:
        """
        Pool word embeddings to single sentence embedding.
        Using mean pooling over actual tokens (ignore padding).
        """
        if seq_length == 0:
            return [0.0] * self.d_model
            
        # Mean pool over actual sequence length
        sentence_embedding = [0.0] * self.d_model
        for i in range(seq_length):  # Only actual words, not padding
            for j in range(self.d_model):
                sentence_embedding[j] += contextualized_words[i][j]
        
        # Average
        for j in range(self.d_model):
            sentence_embedding[j] /= seq_length
            
        return sentence_embedding
    
    def sentence_to_embedding(self, sentence: str) -> Vector:
        """
        Convert single sentence to transformer-enhanced embedding.
        
        Pipeline:
        sentence -> word_tokens -> word_embeddings -> +pos_encoding 
        -> attention -> pooling -> sentence_embedding
        """
        # Get word-level embeddings
        word_embeddings, seq_length = self._sentence_to_word_sequence(sentence)
        
        if seq_length == 0:
            return [0.0] * self.d_model
        
        # Apply transformer layer
        contextualized = self._apply_transformer_layer(word_embeddings, seq_length)
        
        # Pool to sentence-level
        sentence_embedding = self._pool_to_sentence_embedding(contextualized, seq_length)
        
        return sentence_embedding

def transformer_fit(sentences: List[str], 
                   preprocessor: Optional[PreprocessingPipeline] = None,
                   attention_heads: int = 4,
                   max_seq_len: int = 32) -> TransformerTFIDFModel:
    """Fit transformer-enhanced TF-IDF model."""
    model = TransformerTFIDFModel(preprocessor, attention_heads, max_seq_len)
    model._build_word_embeddings(sentences)
    return model

def transformer_transform(sentences: List[str], model: TransformerTFIDFModel) -> Matrix:
    """Transform sentences using transformer-enhanced model."""
    return [model.sentence_to_embedding(sentence) for sentence in sentences]

def transformer_fit_transform(sentences: List[str],
                             preprocessor: Optional[PreprocessingPipeline] = None,
                             attention_heads: int = 4,
                             max_seq_len: int = 32) -> Tuple[Matrix, TransformerTFIDFModel]:
    """Fit transformer model and transform sentences."""
    model = transformer_fit(sentences, preprocessor, attention_heads, max_seq_len)
    embeddings = transformer_transform(sentences, model)
    return embeddings, model

__all__ = ['TransformerTFIDFModel', 'transformer_fit', 'transformer_transform', 'transformer_fit_transform']