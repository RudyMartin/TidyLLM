# TidyLLM Ecosystem Technical Reference

## 🏗️ Architecture Overview

The TidyLLM ecosystem follows a modular, composable architecture where each library serves a specific purpose while integrating seamlessly with others.

## 📦 Library Architecture

### **1. Main TidyLLM Package**

```
tidyllm/
├── __init__.py              # Main exports, version, availability flags
├── cli.py                   # Complete CLI interface
├── gateways/                # AI service integrations
│   ├── __init__.py         # Gateway registry and initialization
│   ├── llm_gateway.py      # LLM service integration
│   ├── dspy_gateway.py     # DSPy framework integration
│   └── heiros_gateway.py   # Custom enterprise gateway
├── knowledge_systems/       # Knowledge management
│   ├── core/               # Core knowledge operations
│   └── facades/            # Simplified interfaces
├── admin/                   # Administrative utilities
├── workflows/              # Workflow definitions
└── demos/                  # Demo applications
```

**Key Components:**
- **Gateway System**: Unified interface to AI services (OpenAI, Anthropic, AWS Bedrock)
- **Knowledge Systems**: Document processing and knowledge management
- **CLI Interface**: Professional command-line tools
- **MLflow Integration**: Experiment tracking and model management

### **2. TLM (Transparent Learning Machines)**

```
tlm/
├── __init__.py             # Main API exports
├── pure/                   # Pure Python data operations
│   └── ops.py             # Core list-based operations
├── core/                   # Fundamental ML components
│   ├── activations.py      # sigmoid, relu, softmax
│   ├── losses.py           # mse, cross_entropy
│   └── metrics.py          # accuracy, confusion_matrix
├── linear_models/          # Linear classification/regression
├── cluster/                # K-means clustering
├── svm/                    # Linear SVM implementation
├── decomp/                 # PCA with power iteration
├── mixture/                # Gaussian Mixture Models
├── anomaly/                # Gaussian anomaly detection
├── naive_bayes/            # Multinomial Naive Bayes
├── mf/                     # Matrix factorization
├── model_selection/        # K-fold cross validation
└── tests/                  # Test suite
```

**Design Principles:**
- **Pure Python**: No external dependencies beyond standard library
- **List-based Operations**: All data structures are Python lists
- **Functional API**: Each module exports fit/predict/transform functions
- **Algorithmic Transparency**: Every step is readable and modifiable

### **3. tidyllm-sentence**

```
tidyllm_sentence/
├── __init__.py             # Main API exports
├── tfidf.py                # TF-IDF implementation
├── word_avg.py             # Word averaging embeddings
├── ngram.py                # N-gram based embeddings
├── lsa.py                  # Latent Semantic Analysis
├── similarity.py           # Similarity calculations
├── preprocessing.py        # Text preprocessing utilities
└── utils.py                # Helper functions
```

**Algorithms Available:**
- **TF-IDF**: Classic information retrieval embeddings
- **Word Averaging**: IDF-weighted word vector averaging
- **N-gram**: Character and word n-gram representations
- **LSA**: SVD-based dimensionality reduction
- **Similarity**: Cosine similarity and semantic search

## 🔧 API Reference

### **TidyLLM Main Package**

```python
import tidyllm

# Gateway initialization
gateway_registry = tidyllm.init_gateways()

# Check availability
print(tidyllm.GATEWAYS_AVAILABLE)          # Boolean
print(tidyllm.KNOWLEDGE_SYSTEMS_AVAILABLE)  # Boolean
print(tidyllm.KNOWLEDGE_SERVER_AVAILABLE)   # Boolean

# Version information
print(tidyllm.__version__)  # "1.0.0"
```

### **TLM Pure Python ML**

```python
import tlm

# Linear Models
w, b, history = tlm.logreg_fit(X, y, lr=0.01, epochs=100)
predictions = tlm.logreg_predict(X_test, w, b)

# Clustering  
centers, labels, inertia = tlm.kmeans_fit(X, k=3, max_iters=100, seed=42)
labels = tlm.kmeans_predict(X_test, centers)

# Dimensionality Reduction
components, explained_var = tlm.pca_fit(X, n_components=2)
X_transformed = tlm.pca_transform(X, components)

# Data Operations
normalized = tlm.l2_normalize(X)
transposed = tlm.transpose(matrix)
result = tlm.dot_product(matrix1, matrix2)

# Evaluation
accuracy = tlm.accuracy_score(y_true, y_pred)
conf_matrix = tlm.confusion_matrix(y_true, y_pred, n_classes=3)
```

### **tidyllm-sentence Embeddings**

```python
import tidyllm_sentence as tls

# TF-IDF Embeddings
embeddings, model = tls.tfidf_fit_transform(documents)
new_embeddings = tls.tfidf_transform(new_documents, model)

# Word Averaging Embeddings
embeddings, model = tls.word_avg_fit_transform(
    documents, 
    embedding_dim=100, 
    use_idf=True
)

# N-gram Embeddings
embeddings, model = tls.ngram_fit_transform(
    documents, 
    n=3, 
    ngram_type='char',  # or 'word'
    max_features=1000
)

# LSA Embeddings
embeddings, model = tls.lsa_fit_transform(documents, n_components=50)

# Similarity and Search
similarity = tls.cosine_similarity(embedding1, embedding2)
results = tls.semantic_search(query_embedding, corpus_embeddings, top_k=5)

# Text Processing
tokens = tls.word_tokenize("Hello, world!")
char_grams = tls.char_ngrams("hello", n=3)
```

## ⚙️ Configuration Management

### **Main Configuration (tidyllm_config.yaml)**

```yaml
# TidyLLM Configuration
qa:
  watch_folder: './qa_files'
  output_folder: './qa_reports' 
  config_folder: './qa_config'
  mlflow_enabled: true

model:
  default_provider: 'anthropic'
  default_model: 'claude-3-sonnet'
  experiment_prefix: 'tidyllm'

integrations:
  aws_enabled: true
  mlflow_enabled: true
  database_enabled: false
```

### **Environment Variables**

```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-1"

# MLflow Configuration  
export MLFLOW_TRACKING_URI="file:///path/to/mlruns"

# API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## 🔄 Integration Patterns

### **Cross-Library Integration**

```python
# Complete pipeline using all libraries
import tidyllm
import tlm  
import tidyllm_sentence as tls

def complete_analysis_pipeline(documents):
    # 1. Process documents with TidyLLM
    gateway = tidyllm.init_gateways()
    
    # 2. Generate embeddings with tidyllm-sentence
    embeddings, model = tls.tfidf_fit_transform(documents)
    
    # 3. Cluster with TLM pure Python ML
    normalized = tlm.l2_normalize(embeddings)
    clusters, labels, inertia = tlm.kmeans_fit(normalized, k=3)
    
    # 4. Return comprehensive analysis
    return {
        'documents': len(documents),
        'embeddings': embeddings,
        'clusters': clusters,
        'labels': labels,
        'inertia': inertia
    }
```

### **CLI Integration Pattern**

```python
# tidyllm/cli.py integration pattern
def launch_qa_processor():
    """Launch QA processor with arguments."""
    try:
        import sys
        from pathlib import Path
        
        # Add parent directory to path for imports
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        # Import and run main from qa_processor
        import qa_processor
        qa_processor.main()
        
    except ImportError as e:
        print(f"[ERROR] Could not launch QA processor: {e}")
        sys.exit(1)
```

## 🧪 Testing Framework

### **Test Structure**

```python
# Example test pattern used across ecosystem
import pytest
import tidyllm_sentence as tls

def test_tfidf_basic_functionality():
    """Test TF-IDF embedding generation"""
    documents = ["hello world", "machine learning", "data science"]
    
    embeddings, model = tls.tfidf_fit_transform(documents)
    
    assert len(embeddings) == 3
    assert len(embeddings[0]) > 0
    assert model is not None

def test_similarity_calculation():
    """Test cosine similarity calculation"""
    vec1 = [1, 0, 0]
    vec2 = [0, 1, 0]
    vec3 = [1, 0, 0]
    
    sim_orthogonal = tls.cosine_similarity(vec1, vec2)
    sim_identical = tls.cosine_similarity(vec1, vec3)
    
    assert abs(sim_orthogonal - 0.0) < 1e-10
    assert abs(sim_identical - 1.0) < 1e-10
```

### **Integration Testing**

```bash
# Test complete ecosystem
python -c "
import tidyllm, tlm, tidyllm_sentence
print('All imports successful')
"

# Test CLI integration
tidyllm status
tidyllm qa --debug-config  
tidyllm test --create-samples

# Test functionality
python -c "
import tidyllm_sentence as tls
docs = ['test1', 'test2']
embs, model = tls.tfidf_fit_transform(docs)
print(f'Generated {len(embs)} embeddings')
"
```

## 📊 Performance Characteristics

### **Memory Usage Comparison**

| Library | Memory Usage | Startup Time | Dependencies |
|---------|-------------|--------------|--------------|
| **tidyllm-sentence** | 0.5-10MB | Instant | None |
| **sentence-transformers** | 88-500MB | 10-30s | PyTorch, etc. |
| **TLM** | <1MB | Instant | None |
| **scikit-learn** | 50-100MB | 1-2s | NumPy, SciPy |

### **Algorithm Complexity**

```python
# TF-IDF: O(n * m) where n=docs, m=unique_terms
# Word Averaging: O(n * avg_doc_length)  
# K-means: O(k * n * d * iterations)
# LSA/SVD: O(min(n,m)^3) for n×m matrix
```

### **Scalability Guidelines**

```python
# Document limits for different methods
TFIDF_MAX_DOCS = 10_000      # Before memory issues
WORD_AVG_MAX_DOCS = 50_000   # Lighter memory footprint  
KMEANS_MAX_POINTS = 1_000    # Pure Python implementation limit
LSA_MAX_FEATURES = 5_000     # SVD computational limit
```

## 🔐 Security Considerations

### **Data Handling**

```python
# Safe text processing - no code execution
def safe_tokenize(text):
    """Tokenize without eval() or exec()"""
    # Uses only string methods and regex
    return tls.word_tokenize(text)

# No pickle loading - JSON/YAML only
def safe_model_save(model, path):
    """Save model safely without pickle"""
    import json
    with open(path, 'w') as f:
        json.dump(model, f)
```

### **API Key Management**

```python
import os
from pathlib import Path

def load_api_keys():
    """Load API keys from environment or config file"""
    
    # Prefer environment variables
    keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'aws_access': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret': os.getenv('AWS_SECRET_ACCESS_KEY')
    }
    
    # Fallback to config file (if exists)
    config_file = Path.home() / '.tidyllm' / 'keys.json'
    if config_file.exists():
        import json
        with open(config_file) as f:
            file_keys = json.load(f)
            for key, value in file_keys.items():
                if keys.get(key) is None:
                    keys[key] = value
    
    return keys
```

## 🚀 Deployment Patterns

### **Docker Deployment**

```dockerfile
# Dockerfile for TidyLLM ecosystem
FROM python:3.11-slim

WORKDIR /app

# Copy ecosystem
COPY tidyllm/ ./tidyllm/
COPY tlm/ ./tlm/  
COPY tidyllm-sentence/ ./tidyllm-sentence/
COPY setup.py .
COPY pyproject.toml .

# Install ecosystem
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tidyllm, tlm, tidyllm_sentence; print('healthy')" || exit 1

# Default command
CMD ["tidyllm", "help"]
```

### **Production Configuration**

```python
# production_config.py
import os

class ProductionConfig:
    """Production configuration for TidyLLM ecosystem"""
    
    # MLflow tracking
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'postgresql://...')
    
    # AWS configuration
    AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    # Processing limits
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_BATCH_SIZE = 100
    
    # Cache settings
    EMBEDDING_CACHE_SIZE = 1000
    MODEL_CACHE_TTL = 3600  # 1 hour
    
    # Security
    ALLOWED_FILE_TYPES = ['.pdf', '.txt', '.docx', '.xlsx']
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
```

## 🎯 Extension Points

### **Adding New Embedding Methods**

```python
# tidyllm_sentence/custom_embeddings.py
def custom_embedding_fit_transform(documents, **kwargs):
    """Template for custom embedding implementation"""
    
    # 1. Preprocessing
    processed_docs = [preprocess_document(doc) for doc in documents]
    
    # 2. Feature extraction
    features = extract_features(processed_docs, **kwargs)
    
    # 3. Embedding generation
    embeddings = generate_embeddings(features)
    
    # 4. Model object for transform
    model = {
        'method': 'custom',
        'features': features,
        'params': kwargs
    }
    
    return embeddings, model

def custom_embedding_transform(documents, model):
    """Transform new documents using fitted model"""
    processed_docs = [preprocess_document(doc) for doc in documents]
    return generate_embeddings_from_model(processed_docs, model)
```

### **Adding New CLI Commands**

```python
# tidyllm/cli.py extension pattern
def handle_new_command():
    """Handler for new CLI command"""
    print("[NEW] Custom command executed")
    # Implementation here

# In main() function:
elif command == 'new-command':
    handle_new_command()
```

### **Adding New TLM Algorithms**

```python
# tlm/custom_algorithm.py
def custom_algorithm_fit(X, y, **params):
    """Fit custom algorithm"""
    # Pure Python implementation
    model = train_custom_model(X, y, **params)
    return model

def custom_algorithm_predict(X, model):
    """Make predictions with fitted model"""
    return apply_custom_model(X, model)

# Export in tlm/__init__.py
from .custom_algorithm import custom_algorithm_fit, custom_algorithm_predict
```

## 📋 Migration Guide

### **From Individual Libraries**

```python
# Old way: Manual installation of each library
# pip install sentence-transformers
# pip install scikit-learn  
# import sentence_transformers, sklearn

# New way: Single ecosystem installation
# pip install -e .
import tidyllm, tlm, tidyllm_sentence
```

### **From External ML Libraries**

```python
# From sentence-transformers to tidyllm-sentence
# Old:
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(documents)

# New:
import tidyllm_sentence as tls
embeddings, model = tls.tfidf_fit_transform(documents)
# or tls.word_avg_fit_transform() for semantic similarity

# From scikit-learn to TLM
# Old:
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3)
# labels = kmeans.fit_predict(X)

# New:
import tlm
centers, labels, inertia = tlm.kmeans_fit(X, k=3)
```

## 🎊 Future Extensions

The modular architecture supports easy addition of:

1. **New Embedding Methods**: Custom algorithms in tidyllm-sentence
2. **Additional ML Algorithms**: More pure Python implementations in TLM
3. **Enhanced CLI Commands**: Extended functionality in tidyllm CLI
4. **Integration Adapters**: Connectors to other systems
5. **Specialized Processors**: Domain-specific document handlers

The ecosystem is designed for growth while maintaining the core principles of transparency, educational value, and algorithmic sovereignty.

---

**The TidyLLM ecosystem demonstrates that sophisticated ML can be simple, transparent, and completely under user control.** 🚀