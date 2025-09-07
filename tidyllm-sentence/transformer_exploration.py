#!/usr/bin/env python3
"""
TRANSFORMER EXPLORATION: What is a transformer and can we build one?

A transformer has:
1. Multi-head attention (scaled dot-product attention)
2. Position encoding
3. Layer normalization  
4. Feed-forward networks
5. Residual connections

For semantic similarity, we mainly need the ATTENTION mechanism.
"""

import sys
sys.path.insert(0, '../tlm')
import tlm
import math

def softmax(x):
    """Compute softmax of vector x."""
    exp_x = [math.exp(xi - max(x)) for xi in x]  # Numerical stability
    sum_exp = sum(exp_x)
    return [ei / sum_exp for ei in exp_x]

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Core transformer attention mechanism.
    Q, K, V are matrices (list of lists).
    Returns attention output and attention weights.
    """
    d_k = len(K[0])  # Key dimension
    
    # Compute Q @ K^T
    scores = []
    for q in Q:
        row_scores = []
        for k in K:
            score = sum(q_i * k_i for q_i, k_i in zip(q, k))
            row_scores.append(score / math.sqrt(d_k))  # Scale
        scores.append(row_scores)
    
    # Apply softmax to each row
    attention_weights = [softmax(row) for row in scores]
    
    # Apply attention to values: Attention @ V
    output = []
    for weights in attention_weights:
        output_row = [0.0] * len(V[0])
        for i, weight in enumerate(weights):
            for j in range(len(V[0])):
                output_row[j] += weight * V[i][j]
        output.append(output_row)
    
    return output, attention_weights

def positional_encoding(seq_len, d_model):
    """Generate positional encodings for transformer."""
    pos_encoding = []
    for pos in range(seq_len):
        encoding = []
        for i in range(d_model):
            if i % 2 == 0:
                encoding.append(math.sin(pos / (10000 ** (i / d_model))))
            else:
                encoding.append(math.cos(pos / (10000 ** ((i-1) / d_model))))
        pos_encoding.append(encoding)
    return pos_encoding

def simple_transformer_block(embeddings):
    """
    A simplified transformer block that could enhance semantic similarity.
    Input: word embeddings (seq_len x d_model)
    Output: contextualized embeddings
    """
    seq_len = len(embeddings)
    d_model = len(embeddings[0])
    
    print(f"Input shape: {seq_len} x {d_model}")
    
    # Add positional encoding
    pos_enc = positional_encoding(seq_len, d_model)
    embedded_with_pos = []
    for i in range(seq_len):
        row = [embeddings[i][j] + pos_enc[i][j] for j in range(d_model)]
        embedded_with_pos.append(row)
    
    # Self-attention (Q=K=V=input)
    attended, attention_weights = scaled_dot_product_attention(
        embedded_with_pos, embedded_with_pos, embedded_with_pos
    )
    
    print(f"Attention weights shape: {len(attention_weights)} x {len(attention_weights[0])}")
    
    return attended, attention_weights

print("TRANSFORMER ANATOMY")
print("=" * 50)

# Test with simple embeddings
test_embeddings = [
    [1.0, 0.0, 0.0, 0.5],  # "cat"
    [0.0, 1.0, 0.0, 0.3],  # "sat" 
    [0.0, 0.0, 1.0, 0.8],  # "mat"
]

print("Original embeddings:")
for i, emb in enumerate(test_embeddings):
    print(f"  Word {i+1}: {emb}")

print(f"\nApplying transformer block...")
contextualized, attention = simple_transformer_block(test_embeddings)

print(f"\nContextualized embeddings:")
for i, emb in enumerate(contextualized):
    print(f"  Word {i+1}: {[f'{x:.3f}' for x in emb]}")

print(f"\nAttention weights (who attends to whom):")
for i, weights in enumerate(attention):
    print(f"  Word {i+1}: {[f'{w:.3f}' for w in weights]}")

# Key insight: attention allows words to "look at" other words in context
print(f"\nKEY INSIGHT:")
print(f"- Word 1 attends to [Word1: {attention[0][0]:.3f}, Word2: {attention[0][1]:.3f}, Word3: {attention[0][2]:.3f}]")
print(f"- This creates CONTEXT-AWARE embeddings!")
print(f"- Each word embedding is now influenced by ALL other words")

print(f"\nFOR TIDYLLM-SENTENCE:")
print(f"- We could add a lightweight attention layer to TF-IDF embeddings")
print(f"- This would capture some semantic relationships within sentences") 
print(f"- Much lighter than full transformers (no pre-training needed)")
print(f"- Still educational and transparent!")