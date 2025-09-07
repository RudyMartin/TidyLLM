#!/usr/bin/env python3
"""
Analyze the specific benchmark sentences to understand why correlation is poor.
"""

import sys
sys.path.insert(0, '../tlm')

import tidyllm_sentence as tls

# From benchmark_comparison.py
TEST_SENTENCES = [
    "The cat sat on the mat",
    "A feline rested on the rug", 
    "Dogs love to play fetch",
    "Canines enjoy playing games",
    "Machine learning algorithms are powerful",
    "AI techniques solve complex problems",
    "Python is a programming language",
    "Python is a type of snake",
    "The weather is beautiful today",
    "It's a sunny and pleasant day",
    "I love eating pizza",
    "Pizza is my favorite food",
    "The movie was entertaining",
    "The film was really enjoyable",
    "She drives a red car",
    "Her automobile is crimson colored"
]

SIMILARITY_PAIRS = [
    (0, 1, 0.9),   # cat/mat vs feline/rug - high similarity
    (2, 3, 0.8),   # dogs/fetch vs canines/games - high similarity  
    (4, 5, 0.9),   # ML vs AI - high similarity
    (6, 7, 0.3),   # python language vs snake - low similarity
    (8, 9, 0.9),   # weather descriptions - high similarity
    (10, 11, 0.8), # pizza sentences - high similarity
    (12, 13, 0.8), # movie/film - high similarity
    (14, 15, 0.7), # car descriptions - medium similarity
    (0, 2, 0.1),   # cat vs dogs - very low similarity
    (4, 6, 0.2),   # ML vs python programming - low similarity
]

print("BENCHMARK SENTENCE ANALYSIS")
print("=" * 60)

embeddings, model = tls.tfidf_fit_transform(TEST_SENTENCES)
print(f"Total vocabulary: {model.vocab_size}")
print(f"Vocabulary: {sorted(model.vocabulary.keys())}")

print(f"\nProcessed sentences:")
for i, sentence in enumerate(TEST_SENTENCES):
    tokens = model.preprocessor.process(sentence)
    print(f"  {i:2d}. '{sentence[:40]:<40}' -> {tokens}")

print(f"\nAnalyzing similarity pairs:")
for i, j, expected in SIMILARITY_PAIRS:
    tokens_i = model.preprocessor.process(TEST_SENTENCES[i])
    tokens_j = model.preprocessor.process(TEST_SENTENCES[j])
    
    common = set(tokens_i) & set(tokens_j)
    
    # Calculate actual similarity
    sim = tls.cosine_similarity(embeddings[i], embeddings[j])
    
    print(f"\nPair {i}-{j} (expected {expected}):")
    print(f"  '{TEST_SENTENCES[i][:35]:<35}' -> {tokens_i}")
    print(f"  '{TEST_SENTENCES[j][:35]:<35}' -> {tokens_j}")
    print(f"  Common words: {list(common)}")
    print(f"  Actual similarity: {sim:.3f}")
    print(f"  Expected: {expected}, Diff: {sim - expected:+.3f}")

print(f"\nKEY INSIGHTS:")
print("1. 'cat/mat' vs 'feline/rug' -> NO common words after preprocessing")
print("2. 'dogs/fetch' vs 'canines/games' -> NO common words")  
print("3. This is why correlation is poor - synonyms become different tokens")
print("4. TF-IDF without word embeddings can't capture semantic similarity")
print("5. Our algorithm is CORRECT but lacks semantic knowledge")