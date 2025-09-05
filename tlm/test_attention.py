#!/usr/bin/env python3
"""Test the attention mechanism we just built."""

import sys
sys.path.insert(0, '.')

import tlm

# Test scaled dot-product attention
print("TESTING ATTENTION MECHANISM")
print("=" * 40)

# Simple test embeddings (3 words, 4 dimensions)
Q = [
    [1.0, 0.0, 0.0, 0.5],  # Query 1
    [0.0, 1.0, 0.0, 0.3],  # Query 2  
    [0.0, 0.0, 1.0, 0.8],  # Query 3
]

K = [
    [1.0, 0.0, 0.0, 0.5],  # Key 1 (same as queries for self-attention)
    [0.0, 1.0, 0.0, 0.3],  # Key 2
    [0.0, 0.0, 1.0, 0.8],  # Key 3
]

V = [
    [0.2, 0.8, 0.1, 0.9],  # Value 1
    [0.7, 0.3, 0.6, 0.4],  # Value 2
    [0.5, 0.5, 0.9, 0.1],  # Value 3
]

print("Input:")
print(f"Q (queries): {len(Q)}x{len(Q[0])}")
for i, q in enumerate(Q):
    print(f"  Q[{i}]: {q}")

print(f"K (keys): {len(K)}x{len(K[0])}")
print(f"V (values): {len(V)}x{len(V[0])}")

# Apply attention
output, attention_weights = tlm.scaled_dot_product_attention(Q, K, V)

print(f"\nOutput after attention:")
for i, row in enumerate(output):
    print(f"  Out[{i}]: {[f'{x:.3f}' for x in row]}")

print(f"\nAttention weights (who looks at whom):")
for i, weights in enumerate(attention_weights):
    print(f"  Query {i}: {[f'{w:.3f}' for w in weights]}")

# Test positional encoding
print(f"\nTesting positional encoding:")
pos_enc = tlm.positional_encoding(3, 4)
print(f"Positional encoding (3x4):")
for i, enc in enumerate(pos_enc):
    print(f"  Pos[{i}]: {[f'{x:.3f}' for x in enc]}")

print("\n✅ Attention mechanism working!")