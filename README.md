# TidyLLM - The Great Walled City of Enterprise AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/tidyllm/tidyllm)

**Complete AI/ML ecosystem with automatic dependency management and algorithmic sovereignty**

## 🚀 Quick Start

```bash
# Single command installs entire ecosystem
pip install -e .

# Verify installation
tidyllm status
```

That's it! You now have access to:
- ✅ **TidyLLM**: QA processing, CLI interface, AWS integration
- ✅ **TLM**: Pure Python ML algorithms (kmeans, SVM, PCA, etc.)
- ✅ **tidyllm-sentence**: Educational embeddings (TF-IDF, LSA, word averaging)

## 🎯 What Makes TidyLLM Different

### **Single Install, Complete Ecosystem**
```bash
pip install -e .  # Installs 3+ libraries automatically
```

### **Professional CLI Interface**
```bash
tidyllm help                    # Complete command reference
tidyllm qa --chat-pdf report.pdf  # Interactive PDF analysis
tidyllm test --all             # Run comprehensive tests
tidyllm status                 # System health check
```

### **Educational Transparency**
Every algorithm is readable Python code - no black boxes:
```python
import tidyllm_sentence as tls
import tlm

# Generate embeddings (educational, transparent)
documents = ["AI research", "ML algorithms", "Data science"]
embeddings, model = tls.tfidf_fit_transform(documents)

# Cluster with pure Python ML
normalized = tlm.l2_normalize(embeddings)  # When available
clusters, labels, inertia = tlm.kmeans_fit(normalized, k=2)
```

## 📊 Performance vs. Transparency

| Library | tidyllm-sentence | sentence-transformers |
|---------|------------------|----------------------|
| **Memory Usage** | 0.5MB | 88.7MB |
| **Startup Time** | Instant | 10-30 seconds |
| **Dependencies** | Zero | Many |
| **Transparency** | Complete | Black box |
| **Performance** | 77.9% of SOTA | State-of-the-art |

**Choose your trade-off: Lightning-fast experimentation vs. maximum performance**

## 🏗️ Complete Feature Matrix

### **Core TidyLLM Package**
- 🎯 **QA Processing**: Excel + PDF workflow automation
- 🤖 **AI Integration**: OpenAI, Anthropic, AWS Bedrock
- 📊 **MLflow Tracking**: Automatic experiment management
- 💻 **Professional CLI**: 15+ commands with comprehensive help
- ⚙️ **Configuration**: YAML-based project setup

### **TLM - Pure Python ML**
- 🧠 **Algorithms**: K-means, SVM, PCA, Linear Models, Naive Bayes
- 🔬 **Zero Dependencies**: Only Python standard library
- 📚 **Educational**: Every line readable and modifiable
- ⚡ **Fast**: Optimized pure Python implementations
- 🎯 **Sklearn-like API**: Familiar fit/predict patterns

### **tidyllm-sentence - Educational Embeddings**
- 📝 **Methods**: TF-IDF, Word Averaging, LSA, N-grams
- 🎓 **Academic Grade**: 65.5% MAP (77.9% of sentence-transformers quality)
- 💡 **Memory Efficient**: 177x less memory than alternatives
- 🔍 **Transparent**: Understand how embeddings really work
- 🚀 **Fast Startup**: Instant initialization, no model downloads

## 🎯 Real-World Usage

### **Document Analysis Pipeline**
```python
import tidyllm
import tidyllm_sentence as tls

# 1. Initialize TidyLLM for document processing
gateway = tidyllm.init_gateways()

# 2. Process business documents
documents = [
    "Q4 financial report shows 15% growth...",
    "Contract terms specify 30-day delivery...",
    "Invoice #12345 totaling $5,000..."
]

# 3. Generate semantic embeddings
embeddings, model = tls.tfidf_fit_transform(documents)
print(f"Generated {len(embeddings)} embeddings")

# 4. Find document similarities
query = "Financial performance metrics"
query_emb, _ = tls.tfidf_transform([query], model)
similarities = [tls.cosine_similarity(query_emb[0], emb) for emb in embeddings]
print("Most relevant:", documents[max(enumerate(similarities), key=lambda x: x[1])[0]])
```

### **CLI Workflow**
```bash
# Initialize new project
tidyllm init

# Process documents with experiment tracking
tidyllm qa --batch --experiment "monthly_reports" --tag "department=finance"

# Interactive PDF analysis
tidyllm chat-pdf research_paper.pdf

# Run comprehensive system tests
tidyllm test --all --verbose
```

### **Pure Python ML Pipeline**
```python
import tlm
import tidyllm_sentence as tls

# Educational ML without external frameworks
documents = ["Tech article", "Sports news", "Finance report"]
embeddings, _ = tls.word_avg_fit_transform(documents, embedding_dim=50)

# Cluster with transparent algorithms
normalized = tlm.l2_normalize(embeddings)
centers, labels, inertia = tlm.kmeans_fit(normalized, k=2)

print(f"Clustered {len(documents)} documents into {len(set(labels))} groups")
```

## 📦 Installation Options

### **Core Ecosystem (Recommended)**
```bash
pip install -e .
# Includes: tidyllm + tlm + tidyllm-sentence
```

### **Extended Features**
```bash
pip install -e .[web]          # Streamlit interfaces
pip install -e .[documents]    # Advanced document processing  
pip install -e .[vectorqa]     # Research analysis tools
pip install -e .[all]          # Everything
```

### **Development**
```bash
pip install -e .[dev]          # Testing and development tools
```

## 🌟 Why Choose TidyLLM?

### **For Data Scientists**
- **Learn by Doing**: Understand ML algorithms through readable code
- **No Vendor Lock-in**: Complete control over your ML pipeline
- **Fast Experimentation**: Instant startup, no GPU setup required
- **Production Path**: Scale from experiments to production seamlessly

### **For Developers**
- **Professional Tools**: Complete CLI with comprehensive help
- **Clean Integration**: Single install, automatic dependency management
- **Extensible**: Easy to add custom algorithms and methods
- **Well Documented**: Comprehensive guides and API reference

### **For Enterprises**
- **Algorithmic Sovereignty**: Full control over AI processing pipeline
- **Security**: No external API calls for core ML operations
- **Compliance**: Complete audit trail of all processing steps
- **Cost Effective**: No GPU infrastructure required for many tasks

### **For Educators**
- **Transparent Algorithms**: Every step visible and modifiable
- **Pure Python**: No complex framework dependencies to install
- **Comparative Analysis**: See exactly how different approaches work
- **Hands-on Learning**: Modify algorithms and see immediate results

## 🏛️ The TidyLLM Philosophy

**"Simplicity as Strategy"** - We believe sophisticated AI doesn't require complex tooling.

1. **🔬 Transparency First**: Every algorithm step is readable and understandable
2. **🎓 Educational Mission**: Code should teach concepts, not hide them
3. **🚀 Infrastructure Sovereignty**: Complete independence from Big Tech ML frameworks
4. **⚡ Simplicity as Strategy**: Revealing essential concepts rather than hiding complexity
5. **🌐 Data Liberation**: Maximum portability with zero vendor lock-in

## 📚 Documentation

- **[Installation Guide](INSTALLATION.md)** - Complete setup instructions
- **[CLI Documentation](CLI_DOCUMENTATION.md)** - All CLI commands and examples
- **[Usage Guide](ECOSYSTEM_USAGE_GUIDE.md)** - Real-world workflows and patterns
- **[Technical Reference](ECOSYSTEM_TECHNICAL_REFERENCE.md)** - API reference and architecture
- **[Development Guide](ECOSYSTEM_DEVELOPMENT_GUIDE.md)** - Contributing and extending

## 🎯 Quick Examples

### **Generate Embeddings**
```python
import tidyllm_sentence as tls

sentences = ["Hello world", "Machine learning", "Data science"]
embeddings, model = tls.tfidf_fit_transform(sentences)
print(f"Shape: {len(embeddings)}x{len(embeddings[0])}")
```

### **Pure Python Clustering**
```python
import tlm

data = [[1, 2], [3, 4], [5, 6], [7, 8]]
centers, labels, inertia = tlm.kmeans_fit(data, k=2)
print(f"Cluster labels: {labels}")
```

### **QA Processing**
```bash
tidyllm qa --file document.xlsx --experiment "analysis_2024"
```

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Memory**: 2GB+ recommended (vs 8GB+ for typical ML stacks)
- **Dependencies**: Managed automatically
- **GPU**: Not required (but AWS Bedrock available for heavy lifting)

## 🤝 Contributing

We welcome contributions! The TidyLLM ecosystem is built by developers who value:
- Educational transparency
- Algorithmic sovereignty  
- Clean, readable code
- Practical business applications

See our [Development Guide](ECOSYSTEM_DEVELOPMENT_GUIDE.md) for details.

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🎊 Get Started Today

```bash
# 1. Install the ecosystem
pip install -e .

# 2. Initialize your project
tidyllm init

# 3. Check system health
tidyllm status

# 4. Start processing documents
tidyllm qa --help

# 5. Explore the ecosystem
python -c "import tidyllm, tlm, tidyllm_sentence; print('Ready to explore!')"
```

---

**TidyLLM: Where Enterprise AI meets Algorithmic Sovereignty** 🏛️🚀

*Built with ❤️ by the TidyLLM community*