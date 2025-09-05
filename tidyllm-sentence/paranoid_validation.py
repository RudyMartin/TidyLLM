#!/usr/bin/env python3
"""
PARANOID VALIDATION: Deep testing of tidyllm-sentence + tlm correctness
Tests every component to ensure speed isn't from silent failures.
"""

import sys
sys.path.insert(0, '../tlm')

import tidyllm_sentence as tls
import tlm
import time

print("=" * 80)
print("PARANOID VALIDATION - DIGGING FOR SILENT FAILURES")
print("=" * 80)

# Test data
test_sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug", 
    "Dogs love to play fetch",
    "Python is a programming language",
    "Machine learning is powerful"
]

print("\n1. PREPROCESSING VALIDATION")
print("-" * 40)

# Test each preprocessing step
for i, sentence in enumerate(test_sentences):
    print(f"\nSentence {i+1}: '{sentence}'")
    
    # Raw tokenization
    tokens = tls.word_tokenize(sentence)
    print(f"  Tokens: {tokens}")
    
    # Stop word removal
    no_stops = tls.remove_stopwords(tokens)
    print(f"  No stops: {no_stops}")
    
    # Stemming each word
    stemmed = [tls.simple_stem(word) for word in no_stops]
    print(f"  Stemmed: {stemmed}")
    
    # Full pipeline
    pipeline_result = tls.STANDARD_PIPELINE.process(sentence)
    print(f"  Pipeline: {pipeline_result}")
    
    # Verify pipeline matches manual steps
    if stemmed != pipeline_result:
        print(f"  ❌ PIPELINE MISMATCH! Manual: {stemmed}, Pipeline: {pipeline_result}")
    else:
        print(f"  ✅ Pipeline matches manual processing")

print("\n2. TF-IDF COMPUTATION VALIDATION")
print("-" * 40)

# Test TF-IDF step by step
embeddings, model = tls.tfidf_fit_transform(test_sentences)

print(f"Vocabulary size: {model.vocab_size}")
print(f"Vocabulary: {list(model.vocabulary.keys())}")
print(f"IDF scores shape: {len(model.idf_scores)}")

# Verify embeddings shape
print(f"Embeddings count: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 'NONE'}")

# Check if embeddings are all zeros (silent failure)
for i, embedding in enumerate(embeddings):
    non_zero_count = sum(1 for x in embedding if x != 0.0)
    total_sum = sum(embedding)
    print(f"  Sentence {i+1}: {non_zero_count}/{len(embedding)} non-zero, sum={total_sum:.4f}")
    
    if non_zero_count == 0:
        print(f"    ❌ WARNING: All zeros in embedding {i+1}!")
    if total_sum == 0:
        print(f"    ❌ WARNING: Zero sum in embedding {i+1}!")

# Manual TF-IDF calculation for verification
print(f"\nManual TF-IDF verification for sentence 1:")
sentence1_tokens = model.preprocessor.process(test_sentences[0])
print(f"Tokens: {sentence1_tokens}")

manual_tfidf = [0.0] * model.vocab_size
token_counts = {}
for token in sentence1_tokens:
    token_counts[token] = token_counts.get(token, 0) + 1

total_tokens = len(sentence1_tokens)
print(f"Token counts: {token_counts}")
print(f"Total tokens: {total_tokens}")

for token, count in token_counts.items():
    if token in model.vocabulary:
        idx = model.vocabulary[token]
        tf = count / total_tokens
        idf = model.idf_scores[idx]
        tfidf = tf * idf
        manual_tfidf[idx] = tfidf
        print(f"  {token}: tf={tf:.4f}, idf={idf:.4f}, tfidf={tfidf:.4f}")

# Compare with computed embedding
computed_embedding = embeddings[0]
differences = [abs(manual_tfidf[i] - computed_embedding[i]) for i in range(len(manual_tfidf))]
max_diff = max(differences)
print(f"Max difference between manual and computed: {max_diff:.8f}")

if max_diff > 1e-6:
    print("❌ SIGNIFICANT DIFFERENCE IN TF-IDF COMPUTATION!")
    for i, diff in enumerate(differences):
        if diff > 1e-6:
            print(f"  Index {i}: manual={manual_tfidf[i]:.8f}, computed={computed_embedding[i]:.8f}, diff={diff:.8f}")
else:
    print("✅ TF-IDF computation matches manual calculation")

print("\n3. TLM OPERATIONS VALIDATION")
print("-" * 40)

# Test normalization
print("Testing L2 normalization:")
normalized = tlm.l2_normalize(embeddings)

for i, (orig, norm) in enumerate(zip(embeddings, normalized)):
    orig_norm = tlm.norm(orig)
    norm_norm = tlm.norm(norm)
    print(f"  Sentence {i+1}: original_norm={orig_norm:.4f}, normalized_norm={norm_norm:.4f}")
    
    if abs(norm_norm - 1.0) > 1e-6 and orig_norm > 1e-6:
        print(f"    ❌ WARNING: Normalized vector doesn't have unit norm!")
    elif orig_norm <= 1e-6:
        print(f"    ⚠️  Original vector was near-zero")

# Test cosine similarity
print(f"\nTesting cosine similarity:")
for i in range(len(normalized)):
    for j in range(i+1, len(normalized)):
        sim = tls.cosine_similarity(normalized[i], normalized[j])
        print(f"  Sentences {i+1} & {j+1}: similarity = {sim:.4f}")
        
        if abs(sim) > 1.0001:  # Allow for small floating point errors
            print(f"    ❌ WARNING: Cosine similarity outside [-1,1] range!")

# Test clustering
print(f"\nTesting K-means clustering:")
try:
    centers, labels, inertia = tlm.kmeans_fit(normalized, k=3, seed=42, max_iters=10)
    print(f"  Centers shape: {len(centers)}x{len(centers[0]) if centers else 0}")
    print(f"  Labels: {labels}")
    print(f"  Inertia: {inertia:.4f}")
    
    # Verify label count
    unique_labels = set(labels)
    print(f"  Unique labels: {unique_labels}")
    
    if len(centers) != 3:
        print(f"    ❌ WARNING: Expected 3 centers, got {len(centers)}")
    if max(labels) >= 3 or min(labels) < 0:
        print(f"    ❌ WARNING: Invalid label range: {min(labels)} to {max(labels)}")
        
    print("  ✅ K-means clustering completed successfully")
    
except Exception as e:
    print(f"    ❌ K-means failed: {e}")
    import traceback
    traceback.print_exc()

# Test classification
print(f"\nTesting logistic regression:")
try:
    # Create simple labels (first half vs second half)
    y = [0] * (len(test_sentences)//2) + [1] * (len(test_sentences) - len(test_sentences)//2)
    print(f"  Labels: {y}")
    
    w, b, hist = tlm.logreg_fit(normalized, y, lr=0.1, epochs=20)
    print(f"  Weights shape: {len(w)}")
    print(f"  Bias: {b:.4f}")
    print(f"  Training history length: {len(hist)}")
    
    predictions = tlm.logreg_predict(normalized, w, b)
    accuracy = tlm.accuracy(y, predictions)
    print(f"  Predictions: {predictions}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Check predictions are valid probabilities/classes
    for i, pred in enumerate(predictions):
        if not (0 <= pred <= 1):
            print(f"    ❌ WARNING: Invalid prediction {pred} for sample {i}")
    
    print("  ✅ Logistic regression completed successfully")
    
except Exception as e:
    print(f"    ❌ Logistic regression failed: {e}")
    import traceback
    traceback.print_exc()

print("\n4. SEMANTIC SEARCH VALIDATION")
print("-" * 40)

# Test semantic search
query = "cat"
query_embedding = model._sentence_to_tfidf(query)
query_normalized = tlm.l2_normalize([query_embedding])[0]

print(f"Query: '{query}'")
print(f"Query tokens: {model.preprocessor.process(query)}")
print(f"Query embedding non-zeros: {sum(1 for x in query_embedding if x != 0.0)}")

results = tls.semantic_search(query_normalized, normalized, top_k=3)
print(f"Search results:")
for rank, (idx, score) in enumerate(results):
    print(f"  {rank+1}. Sentence {idx+1}: '{test_sentences[idx]}' (score: {score:.4f})")
    
    if score < -1.0001 or score > 1.0001:
        print(f"    ❌ WARNING: Invalid similarity score!")

print("\n5. EDGE CASE TESTING")
print("-" * 40)

# Test with empty/problematic inputs
edge_cases = [
    "",  # Empty string
    "   ",  # Only whitespace  
    "the the the",  # Only stop words
    "!@#$%^&*()",  # Only punctuation
    "a",  # Single character
]

print("Testing edge cases:")
for i, case in enumerate(edge_cases):
    print(f"\nEdge case {i+1}: '{case}'")
    try:
        tokens = tls.STANDARD_PIPELINE.process(case)
        print(f"  Tokens: {tokens}")
        
        embedding = model._sentence_to_tfidf(case)
        non_zero = sum(1 for x in embedding if x != 0.0)
        embedding_sum = sum(embedding)
        print(f"  Embedding: {non_zero}/{len(embedding)} non-zero, sum={embedding_sum:.4f}")
        
        if case.strip() and not tokens:
            print(f"    ⚠️  Non-empty input produced empty tokens")
            
        print(f"  ✅ Handled gracefully")
        
    except Exception as e:
        print(f"    ❌ Failed: {e}")

print("\n" + "=" * 80)
print("PARANOID VALIDATION COMPLETE")
print("=" * 80)
print("\nIf you see any ❌ warnings above, the speed advantage might be from")
print("incorrect implementations or silent failures!")