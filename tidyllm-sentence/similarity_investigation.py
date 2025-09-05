#!/usr/bin/env python3
"""
Deep investigation of why cosine similarities are all zero.
This could explain the poor semantic correlation.
"""

import sys
sys.path.insert(0, '../tlm')

import tidyllm_sentence as tls
import tlm

# Test with more similar sentences
similar_sentences = [
    "The cat sat on the mat",
    "A cat sits on a mat", 
    "The dog runs in the park",
    "A dog is running in a park",
    "Machine learning is powerful",
    "AI and machine learning are strong"
]

print("SIMILARITY INVESTIGATION")
print("=" * 50)

embeddings, model = tls.tfidf_fit_transform(similar_sentences)
normalized = tlm.l2_normalize(embeddings)

print(f"Vocabulary: {list(model.vocabulary.keys())}")
print(f"Vocabulary size: {model.vocab_size}")

print("\nProcessed sentences:")
for i, sentence in enumerate(similar_sentences):
    tokens = model.preprocessor.process(sentence)
    print(f"  {i+1}. '{sentence}' -> {tokens}")

print("\nEmbedding analysis:")
for i, embedding in enumerate(embeddings):
    non_zero_indices = [j for j, val in enumerate(embedding) if val != 0.0]
    non_zero_words = [list(model.vocabulary.keys())[j] for j in non_zero_indices if j < len(model.vocabulary)]
    print(f"  Sentence {i+1}: {non_zero_words}")

print("\nPairwise similarities:")
for i in range(len(normalized)):
    for j in range(i+1, len(normalized)):
        sim = tls.cosine_similarity(normalized[i], normalized[j])
        print(f"  {i+1} vs {j+1}: {sim:.4f}")
        
        # Detailed dot product analysis
        dot_product = sum(a * b for a, b in zip(normalized[i], normalized[j]))
        print(f"    Dot product: {dot_product:.4f}")
        
        # Check for any common non-zero elements
        common_nonzero = 0
        for k in range(len(normalized[i])):
            if normalized[i][k] != 0 and normalized[j][k] != 0:
                common_nonzero += 1
        print(f"    Common non-zero elements: {common_nonzero}")

print("\nTesting manual cosine similarity calculation:")
# Manual cosine similarity for sentences 1 and 2
vec1, vec2 = normalized[0], normalized[1]
dot = sum(a * b for a, b in zip(vec1, vec2))
norm1 = sum(a * a for a in vec1) ** 0.5
norm2 = sum(b * b for b in vec2) ** 0.5

print(f"Vector 1 norm: {norm1:.6f}")
print(f"Vector 2 norm: {norm2:.6f}")
print(f"Dot product: {dot:.6f}")
print(f"Manual cosine: {dot / (norm1 * norm2):.6f}")
print(f"Library cosine: {tls.cosine_similarity(vec1, vec2):.6f}")