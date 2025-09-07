# TidyLLM Ecosystem Development Guide

## 🚀 Contributing to the TidyLLM Ecosystem

This guide covers development practices, contribution workflows, and extension patterns for the TidyLLM ecosystem.

## 🏗️ Development Environment Setup

### **Complete Development Installation**

```bash
# 1. Clone the complete ecosystem
git clone https://github.com/tidyllm/tidyllm.git
cd tidyllm

# 2. Install in development mode with all features
pip install -e .[dev,all]

# 3. Install pre-commit hooks (if available)
pre-commit install

# 4. Verify installation
tidyllm status
python -c "import tidyllm, tlm, tidyllm_sentence; print('Development environment ready!')"
```

### **Individual Library Development**

```bash
# Work on specific libraries individually
cd tlm && pip install -e .[dev]
cd ../tidyllm-sentence && pip install -e .[dev]
cd .. && pip install -e .[dev]  # Main package
```

### **Required Development Tools**

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy pre-commit

# Or use the [dev] extra
pip install -e .[dev]
```

## 🧪 Testing Framework

### **Running Tests**

```bash
# Test entire ecosystem
pytest

# Test specific library
cd tlm && pytest tests/
cd tidyllm-sentence && pytest  # If tests exist

# Test with coverage
pytest --cov=tidyllm --cov=tlm --cov=tidyllm_sentence

# Test CLI functionality
tidyllm test --all
tidyllm qa --debug-config
```

### **Writing Tests**

```python
# tests/test_integration.py
import pytest
import tidyllm
import tlm
import tidyllm_sentence as tls

def test_ecosystem_imports():
    """Test that all ecosystem libraries can be imported"""
    assert hasattr(tidyllm, '__version__')
    assert hasattr(tlm, '__name__')  # Module exists
    assert hasattr(tls, 'tfidf_fit_transform')

def test_cross_library_integration():
    """Test that libraries work together"""
    # Generate embeddings
    docs = ["test document 1", "test document 2"]
    embeddings, model = tls.tfidf_fit_transform(docs)
    
    # Use with TLM (when l2_normalize is available)
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0

def test_cli_integration():
    """Test CLI functionality"""
    import subprocess
    import sys
    
    # Test help command
    result = subprocess.run([sys.executable, '-m', 'tidyllm.cli', 'help'], 
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert 'TidyLLM' in result.stdout

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing deals with text analysis",
        "Data science combines statistics and programming"
    ]

def test_embedding_pipeline(sample_documents):
    """Test complete embedding pipeline"""
    embeddings, model = tls.tfidf_fit_transform(sample_documents)
    
    assert len(embeddings) == len(sample_documents)
    assert all(len(emb) > 0 for emb in embeddings)
    
    # Test transformation of new documents
    new_docs = ["New document about AI"]
    new_embeddings = tls.tfidf_transform(new_docs, model)
    assert len(new_embeddings) == 1
```

### **Test Organization**

```
tests/
├── unit/                   # Unit tests for individual functions
│   ├── test_tidyllm.py    # Main package tests
│   ├── test_tlm.py        # TLM algorithm tests
│   └── test_sentence.py   # Embedding tests
├── integration/           # Cross-library integration tests
│   ├── test_ecosystem.py  # Full ecosystem tests
│   └── test_cli.py        # CLI integration tests
└── fixtures/              # Test data and fixtures
    ├── sample_docs.txt
    └── test_config.yaml
```

## 🔧 Code Quality Standards

### **Python Code Style**

```python
# Follow PEP 8 with these specific guidelines:

def function_name(param1: str, param2: int) -> dict:
    """
    Brief description of function.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
        
    Returns:
        Description of return value
    """
    result = {
        'param1': param1,
        'param2': param2
    }
    return result

class ClassName:
    """Class docstring following Google style."""
    
    def __init__(self, value: str):
        self.value = value
    
    def method_name(self) -> str:
        """Method description."""
        return self.value.upper()
```

### **Documentation Standards**

```python
# Module-level docstring
"""
tidyllm.module_name - Brief module description

This module provides functionality for...

Example:
    >>> import tidyllm.module_name as mod
    >>> result = mod.function_name("input")
    >>> print(result)
    "output"
"""

# Function docstring with examples
def embedding_function(documents: list, method: str = 'tfidf') -> tuple:
    """
    Generate embeddings for documents.
    
    This function creates vector representations of text documents
    using the specified embedding method.
    
    Args:
        documents: List of strings to embed
        method: Embedding method ('tfidf', 'word_avg', 'lsa')
        
    Returns:
        tuple: (embeddings_list, model_dict) where embeddings_list
               contains the vector representations and model_dict
               contains the fitted model for transforming new documents
               
    Raises:
        ValueError: If method is not supported
        TypeError: If documents is not a list
        
    Example:
        >>> docs = ["Hello world", "Machine learning"]
        >>> embeddings, model = embedding_function(docs)
        >>> len(embeddings)
        2
    """
    pass
```

### **Type Hints**

```python
from typing import List, Dict, Tuple, Optional, Union, Any

def process_documents(
    documents: List[str],
    method: str = 'tfidf',
    max_features: Optional[int] = None
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Function with comprehensive type hints."""
    pass

# For complex types
EmbeddingMatrix = List[List[float]]
ModelDict = Dict[str, Any]
ProcessingResult = Tuple[EmbeddingMatrix, ModelDict]
```

## 🌳 Git Workflow

### **Branch Strategy**

```bash
# Main branches
main          # Production-ready code
develop       # Integration branch for features

# Feature branches
feature/embedding-improvements     # New embedding methods
feature/cli-enhancements          # CLI improvements  
feature/tlm-algorithms            # New ML algorithms
bugfix/memory-optimization        # Bug fixes
docs/api-reference               # Documentation updates
```

### **Commit Standards**

```bash
# Commit message format
<type>(<scope>): <description>

[optional body]

[optional footer]

# Examples:
feat(sentence): add LSA embedding method
fix(cli): resolve Unicode encoding issue on Windows
docs(ecosystem): update integration guide
test(tlm): add unit tests for kmeans clustering
refactor(gateways): improve error handling

# Types:
feat     # New feature
fix      # Bug fix
docs     # Documentation only
test     # Adding tests
refactor # Code restructuring
perf     # Performance improvement
style    # Code style changes
ci       # CI/CD changes
```

### **Pull Request Process**

```bash
# 1. Create feature branch
git checkout -b feature/new-embedding-method

# 2. Make changes and commit
git add .
git commit -m "feat(sentence): implement word2vec-style embeddings"

# 3. Push and create PR
git push origin feature/new-embedding-method

# 4. PR checklist:
- [ ] Tests added/updated
- [ ] Documentation updated  
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] No breaking changes (or clearly documented)
```

## 🏗️ Architecture Patterns

### **Adding New Embedding Methods**

```python
# tidyllm_sentence/new_method.py

def new_method_fit_transform(documents: List[str], **kwargs) -> Tuple[List[List[float]], Dict]:
    """
    Implement new embedding method.
    
    Follow the standard tidyllm-sentence API pattern:
    1. Accept list of documents
    2. Return (embeddings, model) tuple
    3. Model dict should enable transform() of new docs
    """
    
    # 1. Text preprocessing
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    # 2. Feature extraction/vocabulary building  
    vocabulary = build_vocabulary(processed_docs, **kwargs)
    
    # 3. Embedding generation
    embeddings = []
    for doc in processed_docs:
        embedding = generate_embedding(doc, vocabulary, **kwargs)
        embeddings.append(embedding)
    
    # 4. Create model for future transforms
    model = {
        'method': 'new_method',
        'vocabulary': vocabulary,
        'parameters': kwargs
    }
    
    return embeddings, model

def new_method_transform(documents: List[str], model: Dict) -> List[List[float]]:
    """Transform new documents using fitted model."""
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    embeddings = []
    for doc in processed_docs:
        embedding = generate_embedding(doc, model['vocabulary'], **model['parameters'])
        embeddings.append(embedding)
    
    return embeddings

# Add to __init__.py exports
from .new_method import new_method_fit_transform, new_method_transform
```

### **Adding New TLM Algorithms**

```python
# tlm/new_algorithm.py

def new_algorithm_fit(X: List[List[float]], y: List, **params) -> Dict:
    """
    Fit new algorithm following TLM patterns.
    
    Args:
        X: Training data as list of lists
        y: Target values as list
        **params: Algorithm hyperparameters
        
    Returns:
        Dict containing fitted model parameters
    """
    
    # Pure Python implementation - no external dependencies
    model_params = train_algorithm(X, y, **params)
    
    # Return serializable model
    return {
        'algorithm': 'new_algorithm',
        'parameters': model_params,
        'hyperparameters': params,
        'training_info': {
            'n_samples': len(X),
            'n_features': len(X[0]) if X else 0
        }
    }

def new_algorithm_predict(X: List[List[float]], model: Dict) -> List:
    """Make predictions with fitted model."""
    predictions = []
    for sample in X:
        pred = apply_model(sample, model['parameters'])
        predictions.append(pred)
    
    return predictions

# Add to tlm/__init__.py
from .new_algorithm import new_algorithm_fit, new_algorithm_predict
```

### **Adding New CLI Commands**

```python
# tidyllm/cli.py

def handle_new_command():
    """Handle new CLI command."""
    parser = argparse.ArgumentParser(
        prog='tidyllm new-command',
        description='Description of new command'
    )
    parser.add_argument('--option', help='Command option')
    
    # Parse remaining arguments
    args = parser.parse_args(sys.argv[2:])
    
    print(f"[NEW-COMMAND] Executing with option: {args.option}")
    
    # Implementation here
    try:
        result = execute_new_functionality(args)
        print(f"[SUCCESS] Command completed: {result}")
    except Exception as e:
        print(f"[ERROR] Command failed: {e}")
        sys.exit(1)

# Add to main() function
elif command == 'new-command':
    handle_new_command()

# Update help text
help_text = f"""
...
New Commands:
    new-command         Description of new command
...
"""
```

## 📦 Packaging and Distribution

### **Version Management**

```python
# tidyllm/__init__.py
__version__ = "1.1.0"

# tlm/__init__.py  
__version__ = "1.0.1"

# tidyllm_sentence/__init__.py
__version__ = "1.0.1"

# Semantic versioning: MAJOR.MINOR.PATCH
# MAJOR: Breaking changes
# MINOR: New features, backwards compatible
# PATCH: Bug fixes, backwards compatible
```

### **Release Process**

```bash
# 1. Update version numbers
# 2. Update CHANGELOG.md
# 3. Run full test suite
pytest
tidyllm test --all

# 4. Build packages
python -m build  # For main package
cd tlm && python -m build
cd ../tidyllm-sentence && python -m build

# 5. Tag release
git tag v1.1.0
git push origin v1.1.0

# 6. Create GitHub release with notes
```

### **Setup.py Maintenance**

```python
# Keep setup.py files synchronized
# Main package dependencies should include local packages
install_requires=[
    # Core dependencies
    "pyyaml>=6.0",
    # ...
    
    # Local ecosystem packages
    "tlm @ file://./tlm",
    "tidyllm-sentence @ file://./tidyllm-sentence",
]

# Optional dependencies for extended features
extras_require={
    "documents": ["tidyllm-documents @ file://./_archived/tidyllm-documents"],
    "vectorqa": ["tidyllm-vectorqa @ file://./_archived/tidyllm-vectorqa"],
    # ...
}
```

## 🚀 Performance Optimization

### **Memory Optimization**

```python
# Use generators for large datasets
def process_large_dataset(documents):
    """Process documents in memory-efficient way."""
    for batch in batch_generator(documents, batch_size=100):
        yield process_batch(batch)

def batch_generator(items, batch_size):
    """Generate batches of items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# Avoid loading everything into memory
def streaming_processor(file_path):
    """Process file line by line."""
    with open(file_path, 'r') as f:
        for line in f:
            yield process_line(line.strip())
```

### **Algorithm Optimization**

```python
# Optimize hot paths in pure Python
def optimized_dot_product(vec1, vec2):
    """Optimized dot product for pure Python."""
    # Use list comprehension for speed
    return sum(a * b for a, b in zip(vec1, vec2))

def optimized_l2_normalize(vectors):
    """Optimized L2 normalization."""
    normalized = []
    for vector in vectors:
        # Calculate norm once
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            normalized.append([x / norm for x in vector])
        else:
            normalized.append(vector[:])  # Copy zero vector
    return normalized
```

## 📊 Benchmarking and Profiling

### **Performance Testing**

```python
# tests/performance/benchmark.py
import time
import tidyllm_sentence as tls

def benchmark_embedding_methods():
    """Benchmark different embedding methods."""
    documents = generate_test_documents(1000)  # 1K documents
    
    methods = [
        ('tfidf', tls.tfidf_fit_transform),
        ('word_avg', tls.word_avg_fit_transform), 
        ('lsa', tls.lsa_fit_transform)
    ]
    
    results = {}
    for name, method in methods:
        start_time = time.time()
        embeddings, model = method(documents)
        end_time = time.time()
        
        results[name] = {
            'time': end_time - start_time,
            'embeddings': len(embeddings),
            'dimensions': len(embeddings[0])
        }
    
    return results

def memory_usage_test():
    """Test memory usage patterns."""
    import tracemalloc
    
    tracemalloc.start()
    
    # Your code here
    documents = ["test"] * 1000
    embeddings, model = tls.tfidf_fit_transform(documents)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'current': current / 1024 / 1024,  # MB
        'peak': peak / 1024 / 1024         # MB
    }
```

## 🔍 Debugging and Troubleshooting

### **Debug Utilities**

```python
# tidyllm/debug.py
import logging
from typing import Any, Dict

def setup_debug_logging():
    """Setup comprehensive logging for debugging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tidyllm_debug.log')
        ]
    )

def debug_embedding_pipeline(documents: List[str]) -> Dict[str, Any]:
    """Debug embedding generation step by step."""
    debug_info = {}
    
    # Step 1: Preprocessing
    logger = logging.getLogger('tidyllm.debug')
    logger.debug(f"Processing {len(documents)} documents")
    
    processed = []
    for i, doc in enumerate(documents):
        tokens = tls.word_tokenize(doc)
        processed.append(tokens)
        logger.debug(f"Doc {i}: {len(tokens)} tokens")
    
    debug_info['preprocessing'] = {
        'original_docs': len(documents),
        'avg_tokens': sum(len(p) for p in processed) / len(processed)
    }
    
    # Step 2: Embedding generation
    start_time = time.time()
    embeddings, model = tls.tfidf_fit_transform(documents)
    end_time = time.time()
    
    debug_info['embedding'] = {
        'method': 'tfidf',
        'time': end_time - start_time,
        'dimensions': len(embeddings[0]),
        'vocabulary_size': len(model.get('vocabulary', {}))
    }
    
    return debug_info
```

### **Common Issues and Solutions**

```python
# Common debugging patterns

def diagnose_import_issues():
    """Diagnose ecosystem import problems."""
    try:
        import tidyllm
        print("✅ tidyllm imported successfully")
    except ImportError as e:
        print(f"❌ tidyllm import failed: {e}")
        
    try:
        import tlm
        print("✅ tlm imported successfully")
        print(f"   Available functions: {len([f for f in dir(tlm) if callable(getattr(tlm, f))])}")
    except ImportError as e:
        print(f"❌ tlm import failed: {e}")
        
    try:
        import tidyllm_sentence as tls
        print("✅ tidyllm_sentence imported successfully")
        test_docs = ["test1", "test2"]
        embeddings, model = tls.tfidf_fit_transform(test_docs)
        print(f"   Embedding test: {len(embeddings)} embeddings generated")
    except Exception as e:
        print(f"❌ tidyllm_sentence test failed: {e}")

def diagnose_cli_issues():
    """Diagnose CLI problems."""
    import subprocess
    import sys
    
    # Test CLI accessibility
    try:
        result = subprocess.run(['tidyllm', 'help'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CLI accessible via 'tidyllm' command")
        else:
            print(f"❌ CLI command failed: {result.stderr}")
    except FileNotFoundError:
        print("❌ 'tidyllm' command not found in PATH")
        print("   Try: python -m tidyllm.cli help")
    except subprocess.TimeoutExpired:
        print("⚠️  CLI command timed out")
```

## 🎯 Contributing Guidelines

### **Before Contributing**

1. **Read the ecosystem documentation**
2. **Check existing issues** on GitHub
3. **Discuss major changes** in issues before implementing
4. **Follow the coding standards** outlined above
5. **Write tests** for new functionality
6. **Update documentation** as needed

### **Types of Contributions**

1. **Bug fixes** - Always welcome
2. **New embedding methods** - Add to tidyllm-sentence
3. **New ML algorithms** - Add to TLM (pure Python only)
4. **CLI enhancements** - Improve user experience
5. **Documentation** - Help others understand and use the ecosystem
6. **Performance improvements** - Optimize without adding dependencies
7. **Integration examples** - Real-world usage patterns

### **Code Review Process**

```bash
# All contributions go through code review:
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request
5. Address review feedback
6. Merge after approval
```

## 🎊 Ecosystem Philosophy

Remember the core principles when contributing:

1. **Simplicity as Strategy** - Simple solutions for complex problems
2. **Transparency First** - Every algorithm step should be readable
3. **Educational Mission** - Code should teach concepts
4. **Zero Vendor Lock-in** - Avoid external dependencies where possible
5. **Infrastructure Sovereignty** - Users should control their ML pipeline

The TidyLLM ecosystem is built by and for developers who value understanding their tools. Every contribution should advance these goals while maintaining the high quality and reliability users expect.

---

**Welcome to the TidyLLM ecosystem development community!** 🚀🎯