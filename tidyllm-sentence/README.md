# tidyllm-sentence-transformer

**Educational sentence embeddings that compete with industrial systems**

Pure Python implementation of transformer-enhanced embeddings with complete algorithmic transparency. Achieves **65.5% MAP performance** (77.9% of sentence-transformers quality) while using **177x less memory** and maintaining zero external ML dependencies.

## 🎯 Academic Validation

| **Metric** | **tidyllm-sentence-transformer** | **sentence-transformers** | **Advantage** |
|------------|-----------------------------------|---------------------------|---------------|
| **Precision@1** | 1.000 | 1.000 | **Equal** |
| **Mean Average Precision** | 0.655 | 0.841 | **77.9%** quality |
| **Memory Usage** | 0.5MB | 88.7MB | **177x less** |
| **Dependencies** | **Zero** | Many | **Pure Python** |
| **Transparency** | **Complete** | Black box | **Educational** |

*Rigorous academic evaluation - see `EXHIBIT_1_ACADEMIC_BENCHMARK.md`*

## ⚡ Quick Demo

```bash
git clone https://github.com/tidyllm-verse/tidyllm-sentence
cd tidyllm-sentence
git clone https://github.com/tidyllm-verse/tlm ../tlm
python professor_demo_ascii.py
```

**No pip installs, no GPU setup, no model downloads.**

## Install

```bash
pip install tidyllm-sentence
```

## Quick Start

```python
import tidyllm_sentence as tls

sentences = [
    "The cat sat on the mat",
    "Dogs love to play fetch", 
    "Machine learning is fascinating",
    "Natural language processing"
]

# TF-IDF embeddings (most common)
embeddings, model = tls.tfidf_fit_transform(sentences)
print(f"TF-IDF embeddings shape: {len(embeddings)}x{len(embeddings[0])}")

# Find similar sentences
query = "Cats enjoy sleeping"
query_emb, _ = tls.tfidf_fit_transform([query])
results = tls.semantic_search(query_emb[0], embeddings, top_k=2)
print(f"Most similar: {sentences[results[0][0]]} (score: {results[0][1]:.3f})")
```

## Embedding Methods

### 1. TF-IDF Embeddings
Classic information retrieval approach - transparent and interpretable.

```python
# Fit and transform
embeddings, model = tls.tfidf_fit_transform(sentences)

# Or separately
model = tls.tfidf_fit(sentences)
embeddings = tls.tfidf_transform(new_sentences, model)
```

### 2. Word Averaging Embeddings  
Average random word embeddings for sentence representation.

```python
# With IDF weighting (default)
embeddings, model = tls.word_avg_fit_transform(sentences, embedding_dim=100, use_idf=True)

# Without IDF (simple averaging)
embeddings, model = tls.word_avg_fit_transform(sentences, embedding_dim=100, use_idf=False)
```

### 3. N-gram Embeddings
Character or word n-gram based representations.

```python
# Character trigrams (good for typos, morphology)
embeddings, model = tls.ngram_fit_transform(sentences, n=3, ngram_type='char')

# Word bigrams (good for phrases)
embeddings, model = tls.ngram_fit_transform(sentences, n=2, ngram_type='word', max_features=1000)
```

### 4. LSA Embeddings
Latent Semantic Analysis using SVD dimensionality reduction.

```python
# LSA with 50 components
embeddings, model = tls.lsa_fit_transform(sentences, n_components=50)
```

## Utilities

```python
# Text preprocessing and tokenization
tokens = tls.word_tokenize("Hello, world! How are you?")
# ['hello', 'world', 'how', 'are', 'you']

char_grams = tls.char_ngrams("hello", n=3)  
# ['  h', ' he', 'hel', 'ell', 'llo', 'lo ', 'o  ']

# Similarity and search
similarity = tls.cosine_similarity(embedding1, embedding2)
results = tls.semantic_search(query_embedding, corpus_embeddings, top_k=5)
```

## Integration with tlm

Use with the [tlm](https://github.com/RudyMartin/tlm) library for complete ML workflows:

```python
import tlm
import tidyllm_sentence as tls

# Generate sentence embeddings
sentences = ["Text 1", "Text 2", "Text 3"]  
embeddings, _ = tls.tfidf_fit_transform(sentences)

# Use tlm for ML algorithms
embeddings_normalized = tlm.l2_normalize(embeddings)
clusters, labels, _ = tlm.kmeans_fit(embeddings_normalized, k=2)

# Classification with embeddings
labels = [0, 1, 0]  # Example labels
w, b, _ = tlm.logreg_fit(embeddings, labels)
predictions = tlm.logreg_predict(embeddings, w, b)
```

## Performance Comparison

| Method | sentence-transformers | tidyllm-sentence |
|--------|---------------------|------------------|
| Dependencies | PyTorch, transformers, etc. (1GB+) | None (pure Python) |
| Startup time | 10-30 seconds | Instant |
| Memory usage | 500MB-4GB | 10-100MB |
| Installation | Complex (CUDA, etc.) | `pip install` |
| Transparency | Black box | Every line readable |
| Customization | Difficult | Easy modification |
| Educational | No | Primary purpose |

## Design Philosophy

**tidyllm-sentence** follows the **tidyllm verse** principles:

1. **Simplicity as Strategy** - Complex NLP doesn't require complex tools
2. **Transparency First** - Every algorithm step is readable and understandable  
3. **Educational Mission** - Code teaches concepts, not just results
4. **Zero Vendor Lock-in** - No dependence on corporate ML frameworks
5. **Infrastructure Sovereignty** - Complete user control over NLP pipeline

## Use Cases

### Perfect For:
- ✅ **Learning NLP concepts** - understand how embeddings work
- ✅ **Prototyping** - quick experimentation without setup overhead
- ✅ **Edge deployment** - lightweight, runs anywhere
- ✅ **Educational projects** - transparent implementations
- ✅ **Custom modifications** - easy to adapt algorithms

### Consider sentence-transformers for:
- Large-scale production systems requiring state-of-the-art performance
- When you have access to GPUs and don't mind complexity
- Tasks requiring the latest transformer architectures

## Examples

### Document Similarity
```python
import tidyllm_sentence as tls

documents = [
    "Python is a programming language",
    "Machine learning uses algorithms", 
    "Data science involves statistics",
    "Programming languages enable development"
]

# Generate embeddings
embeddings, model = tls.tfidf_fit_transform(documents)

# Find most similar documents
for i, doc in enumerate(documents):
    similarities = []
    for j, other_emb in enumerate(embeddings):
        if i != j:
            sim = tls.cosine_similarity(embeddings[i], other_emb)
            similarities.append((j, sim))
    
    most_similar = max(similarities, key=lambda x: x[1])
    print(f"'{doc}' is most similar to '{documents[most_similar[0]]}' (score: {most_similar[1]:.3f})")
```

### Clustering Documents
```python
import tlm
import tidyllm_sentence as tls

# Generate embeddings
texts = ["Tech doc 1", "Tech doc 2", "Sports doc 1", "Sports doc 2"]
embeddings, _ = tls.tfidf_fit_transform(texts)

# Cluster using tlm
normalized_embs = tlm.l2_normalize(embeddings)
centers, labels, inertia = tlm.kmeans_fit(normalized_embs, k=2)

for text, label in zip(texts, labels):
    print(f"Cluster {label}: {text}")
```

This project is intentionally simple so students can read every loop and understand **NLP + embeddings**.

---

**Built with ❤️ using [CodeBreakers](https://github.com/rudymartin/codebreakers_manifesto) / tidyllm verse**