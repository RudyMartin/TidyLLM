#!/usr/bin/env python3
"""
Debug transformer performance on specific benchmark pairs.
"""

import sys
sys.path.insert(0, '../tlm')

import tidyllm_sentence as tls

# Focus on specific problematic pairs
test_cases = [
    ("The cat sat on the mat", "A feline rested on the rug", 0.9),
    ("Dogs love to play fetch", "Canines enjoy playing games", 0.8),
    ("Machine learning algorithms are powerful", "AI techniques solve complex problems", 0.9),
]

print("DEBUGGING TRANSFORMER PERFORMANCE")
print("=" * 50)

for sent1, sent2, expected in test_cases:
    print(f"\nTesting: '{sent1[:30]}...' vs '{sent2[:30]}...'")
    print(f"Expected similarity: {expected}")
    
    # Standard TF-IDF
    standard_embeddings, _ = tls.tfidf_fit_transform([sent1, sent2])
    standard_sim = tls.cosine_similarity(standard_embeddings[0], standard_embeddings[1])
    
    # Transformer-enhanced  
    transformer_embeddings, _ = tls.transformer_fit_transform([sent1, sent2], max_seq_len=16)
    transformer_sim = tls.cosine_similarity(transformer_embeddings[0], transformer_embeddings[1])
    
    print(f"Standard TF-IDF:  {standard_sim:.4f}")
    print(f"Transformer:      {transformer_sim:.4f}")
    print(f"Improvement:      {transformer_sim - standard_sim:+.4f}")
    print(f"Gap to expected:  Standard={expected - standard_sim:+.4f}, Transformer={expected - transformer_sim:+.4f}")

print(f"\nThe issue might be:")
print(f"1. Our transformer helps with SOME pairs but not all")
print(f"2. The benchmark expects very high similarities (0.8-0.9) that are hard to achieve")
print(f"3. We're still learning within-sentence context, not cross-sentence semantic knowledge")
print(f"4. Without pre-training, we can't know that 'cat' ≈ 'feline' semantically")