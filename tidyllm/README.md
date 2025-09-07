# TidyLLM - DSPy-Optimized AI Enterprise Platform

[![PyPI version](https://badge.fury.io/py/tidyllm.svg)](https://badge.fury.io/py/tidyllm)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DSPy Optimized](https://img.shields.io/badge/DSPy-optimized-purple.svg)](https://github.com/stanfordnlp/dspy)

> **DSPy-optimized AI toolkit with transparent implementations - enterprise workflows made readable**  
> Educational, scalable, and production-ready.

## 🎯 Why TidyLLM?

- **🚀 DSPy-Optimized**: Advanced workflow orchestration with Stanford's DSPy framework
- **🏢 Enterprise Ready**: Production-grade gateways, document processing, and knowledge systems
- **🔬 Educational**: Transparent implementations you can learn from and modify
- **🌐 Gateway Architecture**: Unified interface to AI services, databases, and storage
- **📚 Knowledge Systems**: Advanced RAG, document processing, and embedding management

## ⚡ Quick Start

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tidyllm
```

```python
import tidyllm

# Initialize the TidyLLM interface
tidy = tidyllm.TidyLLMInterface()

# Access enterprise gateways
ai_gateway = tidy.get_gateway('ai_processing')
db_gateway = tidy.get_gateway('database')
file_gateway = tidy.get_gateway('file_storage')

# DSPy-optimized workflows
from tidyllm.flow.agreements import CorporateAgreement
agreement = CorporateAgreement.document_processing()
setup = agreement.activate()

# Process documents with advanced RAG
from tidyllm.knowledge_systems import DocumentProcessor, EmbeddingProcessor
doc_processor = DocumentProcessor()
embed_processor = EmbeddingProcessor()

# Extract and embed documents
chunks = doc_processor.process_file("your_document.pdf")
embeddings = embed_processor.batch_embed([chunk.text for chunk in chunks])
```

## 🏗️ Enterprise Architecture

### Gateway System
```python
# AI Processing Gateway
ai_gateway = tidyllm.AIProcessingGateway(
    provider_config={'openai': {'api_key': 'your-key'}},
    fallback_providers=['anthropic', 'cohere']
)

# Database Gateway (supports multiple DBs)
db_gateway = tidyllm.DatabaseGateway({
    'primary': 'postgresql://user:pass@host:5432/db',
    'cache': 'redis://localhost:6379',
    'vector': 'pinecone://your-index'
})

# File Storage Gateway
storage_gateway = tidyllm.FileStorageGateway({
    's3': {'bucket': 'your-bucket', 'region': 'us-east-1'},
    'local': {'path': '/data/storage'}
})
```

### Knowledge Systems
```python
from tidyllm.knowledge_systems import DomainRAG, EnhancedExtraction

# Domain-specific RAG system
rag = DomainRAG(
    embedding_model='sentence-transformers/all-MiniLM-L6-v2',
    vector_store=db_gateway.get_vector_store(),
    chunk_size=512,
    overlap=100
)

# Enhanced document extraction
extractor = EnhancedExtraction()
documents = extractor.process_directory("./documents/", 
                                      formats=['pdf', 'docx', 'txt'])
```

### DSPy Workflow Orchestration
```python
from tidyllm.flow import execute_flow_command
from tidyllm.flow.agreements import DeveloperAgreement

# Predefined DSPy workflows
result = execute_flow_command("[Document Analysis]", {
    'input_dir': './documents',
    'output_format': 'structured_json',
    'include_entities': True
})

# Developer-friendly DSPy experiments
dev_agreement = DeveloperAgreement.ai_experimentation()
dspy_gateway = dev_agreement.get_gateway()
```

## 🧠 What's Inside

### Core Components

| Component | Description | Use Case |
|-----------|-------------|----------|
| **Gateways** | Unified API interfaces | AI services, databases, storage |
| **Knowledge Systems** | RAG and document processing | Enterprise search, QA systems |
| **Flow Agreements** | Pre-configured workflows | Corporate, developer, research use |
| **DSPy Integration** | Advanced prompt optimization | Production AI workflows |
| **CLI Interface** | Command-line tools | Automation and scripting |

### Gateway Types

```python
# Available gateways
gateways = tidyllm.get_global_registry()

# AI Processing
ai_gw = gateways.get('ai_processing')
response = ai_gw.chat("Analyze this document for key insights")

# Corporate LLM (compliance-aware)
corp_gw = gateways.get('corporate_llm') 
filtered_response = corp_gw.safe_completion(prompt, compliance_rules)

# Workflow Optimizer
optimizer = gateways.get('workflow_optimizer')
optimized_flow = optimizer.optimize_pipeline(steps)
```

## 🎓 Educational Examples

### Building a Corporate RAG System
```python
import tidyllm

# Enterprise setup with compliance
from tidyllm.flow.agreements import CorporateAgreement
corp_setup = CorporateAgreement.document_processing("ACME Corp").activate()

# Process confidential documents
processor = tidyllm.knowledge_systems.DocumentProcessor(
    chunk_size=1000,
    security_level='confidential'
)

# Extract with metadata preservation
chunks = processor.process_file("confidential_report.pdf")
embeddings = corp_setup.embed_processor.embed_chunks(chunks)

# Store in secure vector database
corp_setup.vector_store.store_embeddings(embeddings, metadata={
    'classification': 'confidential',
    'department': 'finance'
})
```

### DSPy-Powered Research Workflow
```python
from tidyllm.flow.agreements import DeveloperAgreement

# Set up research environment
research_env = DeveloperAgreement.ai_experimentation().activate()

# DSPy workflow for paper analysis
import dspy
from tidyllm.knowledge_systems import ResearchFramework

# Configure DSPy
dspy.configure(lm=research_env.llm, rm=research_env.retriever)

# Define research signature
class PaperAnalysis(dspy.Signature):
    """Analyze academic paper for key contributions and methodology"""
    paper_text = dspy.InputField()
    contributions = dspy.OutputField(desc="Key contributions")
    methodology = dspy.OutputField(desc="Research methodology")
    limitations = dspy.OutputField(desc="Study limitations")

# Execute with optimization
analyzer = dspy.ChainOfThought(PaperAnalysis)
result = analyzer(paper_text="Your academic paper content...")
```

## 🏭 Production Features

### Enterprise Security
- **Access Control**: Role-based gateway permissions
- **Audit Logging**: Complete request/response tracking  
- **Data Privacy**: GDPR/CCPA compliance tools
- **Encryption**: End-to-end data protection

### Scalability
- **Load Balancing**: Multi-provider fallbacks
- **Caching**: Intelligent response caching
- **Rate Limiting**: API usage management
- **Monitoring**: Performance metrics and alerting

### Integration
- **REST APIs**: HTTP endpoints for all gateways
- **Database Support**: PostgreSQL, MongoDB, Redis, Vector DBs
- **Cloud Storage**: AWS S3, Azure Blob, Google Cloud
- **AI Providers**: OpenAI, Anthropic, Cohere, local models

## 📦 TidyLLM Ecosystem

TidyLLM works seamlessly with the entire ecosystem:

```bash
# Core ML algorithms (zero dependencies)
pip install tlm

# Sentence embeddings  
pip install tidyllm-sentence

# Full enterprise platform
pip install tidyllm
```

**Dependency Chain**: `tlm` → `tidyllm-sentence` → `tidyllm`

## 🎯 Perfect For

- **🏢 Enterprise AI**: Production workflows with compliance
- **🔬 Research**: DSPy experimentation and optimization
- **📚 Education**: Learn enterprise AI architecture patterns
- **🚀 Startups**: Rapid AI prototype to production
- **🏫 Teaching**: Demonstrate real-world AI systems

## 🚀 Getting Started

1. **Install TidyLLM**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tidyllm
   ```

2. **Initialize your environment**:
   ```python
   import tidyllm
   
   # Quick start with defaults
   tidy = tidyllm.TidyLLMInterface()
   
   # Or use pre-configured agreements
   from tidyllm.flow.agreements import CorporateAgreement
   setup = CorporateAgreement.document_processing().activate()
   ```

3. **Explore the CLI**:
   ```bash
   tidyllm --help
   tidyllm init-project my-ai-app
   tidyllm run-workflow document-analysis
   ```

## 🤝 Contributing

Part of the [TidyLLM ecosystem](https://github.com/RudyMartin/TidyLLM). Contributions welcome!

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by the TidyLLM Team**  
*DSPy-optimized workflows, enterprise-ready, educational by design*