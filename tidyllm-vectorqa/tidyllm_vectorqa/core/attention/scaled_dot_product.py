"""
Pure Python implementation of scaled dot-product attention.
The core mechanism that powers transformer models.

Educational, transparent, and dependency-free.
"""

import math
from typing import List, Tuple, Optional
from ..pure.ops import Matrix, Vector

def softmax(x: Vector) -> Vector:
    """
    Compute softmax activation with numerical stability.
    
    softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
    
    Args:
        x: Input vector
        
    Returns:
        Softmax probabilities (sum to 1.0)
    """
    if not x:
        return []
        
    # Subtract max for numerical stability
    max_val = max(x)
    exp_x = [math.exp(xi - max_val) for xi in x]
    sum_exp = sum(exp_x)
    
    if sum_exp == 0:
        return [1.0 / len(x)] * len(x)  # Uniform if all zeros
        
    return [ei / sum_exp for ei in exp_x]

def scaled_dot_product_attention(
    Q: Matrix, 
    K: Matrix, 
    V: Matrix,
    mask: Optional[Matrix] = None
) -> Tuple[Matrix, Matrix]:
    """
    Scaled dot-product attention: the heart of transformers.
    
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    
    Where:
    - Q (queries): what we're looking for
    - K (keys): what we're looking at  
    - V (values): what we retrieve
    
    Args:
        Q: Query matrix [seq_len, d_k]
        K: Key matrix [seq_len, d_k] 
        V: Value matrix [seq_len, d_v]
        mask: Optional attention mask
        
    Returns:
        (output, attention_weights)
        - output: Attended values [seq_len, d_v]
        - attention_weights: Attention probabilities [seq_len, seq_len]
    """
    if not Q or not K or not V:
        return [], []
        
    seq_len = len(Q)
    d_k = len(K[0]) if K else 0
    d_v = len(V[0]) if V else 0
    
    if seq_len == 0 or d_k == 0 or d_v == 0:
        return [], []
    
    # Step 1: Compute attention scores QK^T / sqrt(d_k)
    scale = 1.0 / math.sqrt(d_k)
    scores = []
    
    for q in Q:  # For each query
        row_scores = []
        for k in K:  # Dot product with each key
            score = sum(q_i * k_i for q_i, k_i in zip(q, k))
            row_scores.append(score * scale)
        scores.append(row_scores)
    
    # Step 2: Apply mask if provided
    if mask:
        for i in range(seq_len):
            for j in range(len(scores[i])):
                if i < len(mask) and j < len(mask[i]) and mask[i][j] == 0:
                    scores[i][j] = float('-inf')
    
    # Step 3: Apply softmax to get attention weights
    attention_weights = [softmax(row) for row in scores]
    
    # Step 4: Apply attention to values: Attention @ V
    output = []
    for weights in attention_weights:
        output_row = [0.0] * d_v
        for i, weight in enumerate(weights):
            if i < len(V):  # Bounds check
                for j in range(d_v):
                    if j < len(V[i]):
                        output_row[j] += weight * V[i][j]
        output.append(output_row)
    
    return output, attention_weights

def multi_head_attention(
    Q: Matrix,
    K: Matrix, 
    V: Matrix,
    num_heads: int = 8,
    d_model: int = None
) -> Tuple[Matrix, List[Matrix]]:
    """
    Multi-head attention: run several attention heads in parallel.
    
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    For simplicity, we'll split dimensions evenly across heads.
    
    Args:
        Q, K, V: Input matrices
        num_heads: Number of attention heads
        d_model: Model dimension (inferred if None)
        
    Returns:
        (output, all_attention_weights)
    """
    if not Q or not K or not V:
        return [], []
        
    seq_len = len(Q)
    if d_model is None:
        d_model = len(Q[0]) if Q else 0
        
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    
    head_dim = d_model // num_heads
    all_outputs = []
    all_attention_weights = []
    
    # Process each head
    for head in range(num_heads):
        start_idx = head * head_dim
        end_idx = start_idx + head_dim
        
        # Split Q, K, V for this head
        Q_head = [row[start_idx:end_idx] for row in Q]
        K_head = [row[start_idx:end_idx] for row in K] 
        V_head = [row[start_idx:end_idx] for row in V]
        
        # Apply attention
        head_output, head_attention = scaled_dot_product_attention(Q_head, K_head, V_head)
        all_outputs.append(head_output)
        all_attention_weights.append(head_attention)
    
    # Concatenate all head outputs
    output = []
    for i in range(seq_len):
        concat_row = []
        for head_output in all_outputs:
            if i < len(head_output):
                concat_row.extend(head_output[i])
        output.append(concat_row)
    
    return output, all_attention_weights

def positional_encoding(seq_len: int, d_model: int) -> Matrix:
    """
    Generate sinusoidal positional encodings for transformer.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        
    Returns:
        Positional encoding matrix [seq_len, d_model]
    """
    pos_encoding = []
    
    for pos in range(seq_len):
        encoding = []
        for i in range(d_model):
            angle = pos / (10000 ** (i / d_model))
            if i % 2 == 0:
                encoding.append(math.sin(angle))
            else:
                encoding.append(math.cos(angle))
        pos_encoding.append(encoding)
    
    return pos_encoding

__all__ = [
    'softmax',
    'scaled_dot_product_attention', 
    'multi_head_attention',
    'positional_encoding'
]