# TidyLLM v1.0.4 - Enterprise AI Infrastructure Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.4-brightgreen.svg)](https://github.com/tidyllm/tidyllm)

**Complete enterprise AI infrastructure with corporate onboarding, worker architecture, and algorithmic sovereignty**

## 🚀 Quick Start

```bash
# Install the complete ecosystem
pip install -e .

# Corporate onboarding (NEW!)
python onboarding/enhanced_cli_onboarding.py

# Verify installation
tidyllm status
```

You now have access to:
- ✅ **TidyLLM Core**: Simplified API, worker infrastructure, corporate gateways
- ✅ **Infrastructure**: Session management, workers, monitoring dashboards
- ✅ **Corporate Onboarding**: SSO support, universal pre-flight testing
- ✅ **TLM**: Pure Python ML algorithms (kmeans, SVM, PCA, etc.)
- ✅ **tidyllm-sentence**: Educational embeddings (TF-IDF, LSA, word averaging)

## 🏗️ New Architecture Highlights

### **Simplified Import Structure**
```python
# Clean, flattened imports
import tidyllm
from tidyllm.infrastructure import workers
from tidyllm.gateways import corporate_llm_gateway
```

### **Corporate-Ready Onboarding**
```bash
# Interactive corporate setup
streamlit run onboarding/streamlit_app.py

# CLI wizard with SSO support
python onboarding/enhanced_cli_onboarding.py

# Universal pre-flight validation
python onboarding/universal_preflight.py
```

### **Worker Infrastructure**
```python
from tidyllm.infrastructure.workers import (
    AIDropzoneManager,
    ProcessingWorker, 
    EmbeddingWorker,
    FlowRecoveryWorker
)

# Initialize worker coordinator
manager = AIDropzoneManager()
manager.start_monitoring()
```

## 🎯 What Makes TidyLLM Different

### **Enterprise-First Architecture**
- **Corporate Onboarding**: SSO, proxy, temporary credentials
- **Worker Infrastructure**: Scalable processing pipeline
- **Unified Session Management**: AWS, S3, MLflow integration
- **Compliance Ready**: Audit logging, security standards

### **Professional Infrastructure**
```bash
tidyllm help                    # Complete command reference
tidyllm status --infrastructure # System health with worker status
tidyllm test --corporate       # Corporate environment validation
```

### **Educational Transparency**
Every algorithm is readable Python code - no black boxes:
```python
import tidyllm
import tidyllm_sentence as tls
import tlm

# Simple API (NEW simplified interface)
response = tidyllm.chat("Analyze this document")

# Educational embeddings
documents = ["AI research", "ML algorithms", "Data science"]
embeddings, model = tls.tfidf_fit_transform(documents)

# Pure Python clustering
normalized = tlm.l2_normalize(embeddings)
clusters, labels, inertia = tlm.kmeans_fit(normalized, k=2)
```

## 📊 Performance vs. Transparency

| Feature | TidyLLM v1.0.4 | Traditional Platforms |
|---------|--------------|----------------------|
| **Setup Time** | 5 minutes (corporate) | Hours/Days |
| **Import Structure** | Clean & Flat | Nested/Complex |
| **Corporate Support** | Built-in | Add-on |
| **Worker Architecture** | Native | External |
| **Onboarding** | Guided | Manual |
| **Memory Usage** | 0.5MB core | 88.7MB+ |

## 🏗️ Complete Architecture Overview

### **Core Platform**
- 🎯 **Simplified API**: `tidyllm.chat()`, `tidyllm.process_document()`
- 🏢 **Corporate Gateways**: Enhanced with SSO, proxy, compliance
- 📊 **Infrastructure Management**: Workers, sessions, monitoring
- 💻 **Professional CLI**: 20+ commands with corporate features
- ⚙️ **Configuration**: Template-based with corporate defaults

### **Infrastructure Layer**
- 🔧 **Workers**: 10+ specialized processing workers
- 📡 **Session Management**: Unified AWS, S3, database connections
- 🌐 **API Gateway**: Manager endpoints, infrastructure controls
- 📊 **Monitoring**: Streamlit dashboards, system health
- 🔒 **Standards**: Corporate security and compliance

### **Corporate Onboarding**
- 🏢 **SSO Integration**: SAML, temporary credential management
- 🔍 **Universal Pre-flight**: Comprehensive environment validation
- 📋 **Configuration Wizards**: CLI and web-based setup
- 🎯 **Template System**: Safe, credential-free configuration
- ✅ **Validation Framework**: Corporate network, proxy, database testing

### **TLM - Pure Python ML** (Unchanged)
- 🧠 **Algorithms**: K-means, SVM, PCA, Linear Models, Naive Bayes
- 🔬 **Zero Dependencies**: Only Python standard library
- 📚 **Educational**: Every line readable and modifiable
- ⚡ **Fast**: Optimized pure Python implementations

### **tidyllm-sentence - Educational Embeddings** (Unchanged)
- 📝 **Methods**: TF-IDF, Word Averaging, LSA, N-grams
- 🎓 **Academic Grade**: 65.5% MAP (77.9% of sentence-transformers quality)
- 💡 **Memory Efficient**: 177x less memory than alternatives
- 🔍 **Transparent**: Understand how embeddings really work

## 🎯 Real-World Usage (Updated)

### **Corporate Deployment Pipeline**
```python
import tidyllm
from tidyllm.infrastructure.workers import ProcessingWorker
from tidyllm.onboarding import enhanced_session_validator

# 1. Validate corporate environment
validator = enhanced_session_validator.CorporateValidator()
validation_results = validator.validate_corporate_aws_stack()

# 2. Simple document processing
result = tidyllm.process_document("quarterly_report.pdf")

# 3. Initialize worker infrastructure
worker = ProcessingWorker()
worker.start_processing_queue()
```

### **New CLI Workflow**
```bash
# Corporate setup
python onboarding/enhanced_cli_onboarding.py

# Infrastructure monitoring
streamlit run tidyllm/web/ai_dropzone_dashboard.py

# Processing with workers
tidyllm process --workers 4 --infrastructure

# Corporate validation
tidyllm test --corporate --sso
```

### **Worker Architecture Usage**
```python
from tidyllm.infrastructure.workers import (
    CoordinatorWorker,
    EmbeddingWorker, 
    ProcessingWorker,
    AIDropzoneManager
)

# Initialize worker ecosystem
coordinator = CoordinatorWorker()
embedding_worker = EmbeddingWorker()
processor = ProcessingWorker()

# Start coordinated processing
coordinator.orchestrate([embedding_worker, processor])
```

## 📦 Installation Options (Updated)

### **Corporate Ecosystem (Recommended)**
```bash
pip install -e .
# Includes: tidyllm + infrastructure + onboarding + tlm + tidyllm-sentence
```

### **Corporate Onboarding**
```bash
# Interactive web setup
streamlit run onboarding/streamlit_app.py

# CLI wizard
python onboarding/enhanced_cli_onboarding.py

# Validation only
python onboarding/universal_preflight.py
```

### **Extended Features**
```bash
pip install -e .[web]          # Streamlit dashboards + monitoring
pip install -e .[infrastructure] # Full worker architecture
pip install -e .[corporate]    # Enterprise features
pip install -e .[all]          # Complete platform
```

## 🌟 Why Choose TidyLLM v1.0.4?

### **For Corporate IT**
- **Fast Deployment**: 5-minute corporate setup with guided onboarding
- **SSO Integration**: Built-in support for SAML, temporary credentials
- **Infrastructure Ready**: Worker architecture, monitoring, compliance
- **Security First**: Credential-free templates, audit logging

### **For Data Scientists** (Enhanced)
- **Simplified API**: `tidyllm.chat()` and `tidyllm.process_document()`
- **Worker Infrastructure**: Scale processing across multiple workers
- **Educational Transparency**: Understand every algorithm step
- **Corporate Integration**: Seamless enterprise deployment

### **For Developers** (Enhanced)
- **Clean Architecture**: Flattened imports, organized infrastructure
- **Professional Tooling**: Enhanced CLI with infrastructure commands
- **Worker System**: Build scalable processing pipelines
- **Monitoring Dashboards**: Real-time system health and performance

### **For Enterprises** (Major Updates)
- **Corporate Onboarding**: Guided setup for enterprise environments
- **Compliance Ready**: Security standards, audit trails, data governance
- **Infrastructure Sovereignty**: Complete control over processing pipeline
- **Scalable Architecture**: Worker-based processing, unified session management

## 🏛️ Enhanced TidyLLM Philosophy

**"Enterprise Simplicity"** - Sophisticated infrastructure that remains transparent and controllable.

1. **🏢 Corporate First**: Built for enterprise from day one, not retrofitted
2. **🔬 Infrastructure Transparency**: Every component visible and modifiable
3. **🎓 Educational Mission**: Learn enterprise architecture patterns
4. **🚀 Simplified Complexity**: Complex systems with simple interfaces
5. **🌐 Deployment Sovereignty**: Complete independence and control

## 📚 Documentation (Updated Structure)

- **[Corporate Onboarding Guide](onboarding/README.md)** - Enterprise setup and SSO integration
- **[Infrastructure Guide](tidyllm/infrastructure/workers/README.md)** - Worker architecture and scaling
- **[API Documentation](tidyllm/IMPORT_GUIDE.md)** - Simplified import structure
- **[Installation Guide](INSTALLATION.md)** - Complete setup instructions
- **[CLI Documentation](CLI_DOCUMENTATION.md)** - Enhanced command reference
- **[Architecture Overview](ARCHITECTURE.md)** - System design and infrastructure

## 🎯 Quick Examples (Updated)

### **Simple API Usage**
```python
import tidyllm

# New simplified interface
response = tidyllm.chat("Hello world")
result = tidyllm.process_document("report.pdf")
answer = tidyllm.query("What is machine learning?")
```

### **Corporate Setup**
```python
from tidyllm.onboarding import enhanced_session_validator

validator = enhanced_session_validator.CorporateValidator()
results = validator.validate_corporate_aws_stack(
    s3_bucket="corporate-data",
    postgres_host="db.company.com",
    mlflow_uri="http://mlflow.company.com"
)
```

### **Worker Infrastructure**
```python
from tidyllm.infrastructure.workers import AIDropzoneManager

manager = AIDropzoneManager()
manager.start_monitoring()
manager.process_dropzone("financial_analysis")
```

## 🔧 System Requirements (Updated)

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Corporate Features**: SSO/SAML support, proxy configuration
- **Infrastructure**: Worker coordination, session management
- **Memory**: 2GB+ recommended for infrastructure
- **Dependencies**: Managed automatically with corporate packages

## 🤝 Contributing

We welcome contributions to the TidyLLM enterprise platform! Focus areas:
- Corporate integrations and compliance
- Worker architecture enhancements
- Infrastructure monitoring and management
- Educational transparency and documentation

See our [Development Guide](ECOSYSTEM_DEVELOPMENT_GUIDE.md) for details.

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🎊 Get Started Today (Updated)

```bash
# 1. Install the complete ecosystem
pip install -e .

# 2. Corporate onboarding (NEW!)
python onboarding/enhanced_cli_onboarding.py

# 3. Check infrastructure health
tidyllm status --infrastructure

# 4. Start web monitoring
streamlit run tidyllm/web/ai_dropzone_dashboard.py

# 5. Explore the architecture
python -c "import tidyllm; from tidyllm.infrastructure import workers; print('Enterprise ready!')"
```

---

**TidyLLM v1.0.4: Where Enterprise Infrastructure meets Algorithmic Sovereignty** 🏢🚀

*Built with ❤️ for corporate environments by the TidyLLM community*