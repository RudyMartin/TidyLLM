# RAG2DAG Core Implementation

**Version:** 2.0.0
**Status:** Production Core Layer
**Last Updated:** 2025-09-15

## Overview

This directory contains the **core algorithmic implementation** of RAG2DAG (Retrieval-Augmented Generation to Directed Acyclic Graph) optimization. This is the foundational layer that provides pattern recognition, DAG conversion, and execution capabilities.

## Architecture Position

```
┌─────────────────────────────────────────┐
│ Enterprise Service Layer                │
│ tidyllm/services/rag2dag/               │
│ - Orchestration & Integration           │
│ - Monitoring & Health Management        │
│ - Enterprise Compliance                 │
└─────────────────┬───────────────────────┘
                  │ depends on
┌─────────────────▼───────────────────────┐
│ Core Implementation Layer (THIS)        │
│ tidyllm/rag2dag/                        │
│ - Pattern Definitions & Recognition     │
│ - DAG Conversion Algorithms             │
│ - Core Execution Logic                  │
└─────────────────────────────────────────┘
```

**Separation of Concerns:**
- **This Layer** - Pure domain logic, algorithms, pattern definitions
- **Service Layer** - Enterprise orchestration, monitoring, integration

## Core Components

### `converter.py` - Pattern Recognition & DAG Generation
**Primary Class:** `RAG2DAGConverter`

**Purpose:**
- Defines 7 RAG patterns (multi-source, research synthesis, etc.)
- Converts linear RAG workflows to optimized DAG structures
- Provides intelligent pattern matching and workflow optimization

**Key Classes:**
- `RAGPatternType` - Enumeration of supported patterns
- `RAGPattern` - Pattern definition with templates and optimization hints
- `DAGWorkflowNode` - Individual nodes in the DAG workflow
- `RAG2DAGConverter` - Main conversion engine

**Usage by Service Layer:**
```python
# Service imports and uses converter
from ...rag2dag.converter import RAG2DAGConverter, RAGPatternType
converter = RAG2DAGConverter(config)
nodes = converter.generate_dag_from_pattern(pattern, query, files)
```

### `config.py` - Configuration Management
**Primary Class:** `RAG2DAGConfig`

**Purpose:**
- Centralized configuration for RAG2DAG operations
- Model selection and optimization level settings
- Bedrock model configuration and routing

**Key Classes:**
- `RAG2DAGConfig` - Main configuration container
- `BedrockModelConfig` - AWS Bedrock model specifications
- Optimization level settings (speed, balanced, quality)

### `executor.py` - Core Execution Engine
**Primary Class:** `DAGExecutor`

**Purpose:**
- Executes DAG workflows with dependency management
- Provides core execution logic without enterprise orchestration
- Foundation for service-layer execution coordination

### `cli.py` - Command Line Interface
**Purpose:**
- Direct CLI access to RAG2DAG capabilities
- Power user interface for optimization and analysis
- Testing and debugging utilities

**Usage:**
```bash
python -m tidyllm.rag2dag.cli convert --input query.json --pattern multi_source
```

### `drop_zone_interface.py` - Integration Interface
**Purpose:**
- Integration point for drop zone file processing
- Batch workflow processing capabilities
- File-based workflow triggers

## Pattern Definitions

The core implementation defines 7 RAG optimization patterns:

### 1. MULTI_SOURCE
- **Use Case:** Parallel retrieval from multiple sources
- **Optimization:** Concurrent document access
- **Speedup:** Up to 3.5x

### 2. RESEARCH_SYNTHESIS
- **Use Case:** Extract and synthesize research findings
- **Optimization:** Parallel extraction + synthesis
- **Speedup:** Up to 2.8x

### 3. COMPARATIVE_ANALYSIS
- **Use Case:** Compare across multiple documents
- **Optimization:** Parallel comparison operations
- **Speedup:** Up to 3.2x

### 4. FACT_CHECKING
- **Use Case:** Validate claims against sources
- **Optimization:** Parallel verification
- **Speedup:** Up to 2.5x

### 5. KNOWLEDGE_EXTRACTION
- **Use Case:** Extract structured information
- **Optimization:** Parallel data collection
- **Speedup:** Up to 2.2x

### 6. DOCUMENT_PIPELINE
- **Use Case:** Sequential document processing
- **Optimization:** Pipeline stage optimization
- **Speedup:** Up to 1.8x

### 7. SIMPLE_QA
- **Use Case:** Basic question-answering
- **Optimization:** None (baseline pattern)
- **Speedup:** 1.0x (no optimization)

## Integration with Service Layer

The service layer (`tidyllm/services/rag2dag/`) builds upon this core implementation:

```python
# Service layer imports core components
from ...rag2dag.converter import RAG2DAGConverter, RAGPatternType
from ...rag2dag.config import RAG2DAGConfig
from ...rag2dag.executor import DAGExecutor

# Service provides enterprise features
class RAG2DAGOptimizationService:
    def __init__(self):
        self.converter = RAG2DAGConverter(config)  # Uses core
        self.executor = DAGExecutor(config)        # Uses core
        # + Enterprise monitoring, health checks, etc.
```

## Preservation Notice

**⚠️ IMPORTANT - DO NOT DELETE THIS MODULE**

This core implementation is:

1. **Dependency for Service Layer** - Required by `tidyllm/services/rag2dag/`
2. **Algorithm Repository** - Contains all pattern definitions and conversion logic
3. **CLI Interface** - Provides direct command-line access
4. **Domain Logic** - Pure algorithmic implementation separate from enterprise concerns

**Migration History:**
- **Before v2.0:** Single gateway implementation in `tidyllm/gateways/`
- **v2.0+:** Two-layer architecture with core algorithms (here) + enterprise services

## Usage Examples

### Direct Core Usage (Advanced)
```python
from tidyllm.rag2dag.converter import RAG2DAGConverter, RAGPatternType
from tidyllm.rag2dag.config import RAG2DAGConfig

# Direct core usage
config = RAG2DAGConfig.create_default_config()
converter = RAG2DAGConverter(config)

# Generate DAG from pattern
nodes = converter.generate_dag_from_pattern(
    RAGPatternType.MULTI_SOURCE,
    "Compare quarterly reports",
    ["q1.pdf", "q2.pdf", "q3.pdf"]
)
```

### Recommended Service Usage
```python
# Use service layer for enterprise features
from tidyllm.services.rag2dag import rag2dag_service

result = rag2dag_service.analyze_request_optimization(
    "Compare quarterly reports",
    source_files=["q1.pdf", "q2.pdf", "q3.pdf"]
)
```

## Development Guidelines

### Adding New Patterns
1. Add pattern type to `RAGPatternType` enum in `converter.py`
2. Define pattern template in `RAG2DAGConverter._load_rag_patterns()`
3. Update service layer pattern detection in `tidyllm/services/rag2dag/`

### Configuration Changes
1. Update `RAG2DAGConfig` in `config.py`
2. Ensure service layer passes configuration correctly
3. Update validation in service health checks

### Core Testing
```python
# Test core functionality
from tidyllm.rag2dag.converter import RAG2DAGConverter
from tidyllm.rag2dag.config import RAG2DAGConfig

config = RAG2DAGConfig.create_default_config()
converter = RAG2DAGConverter(config)

# Verify pattern loading
assert len(converter.patterns) == 7
print("Core patterns loaded successfully")
```

## Files and Their Purposes

| File | Purpose | Service Dependencies |
|------|---------|---------------------|
| `converter.py` | Pattern definitions, DAG generation | High - Core conversion logic |
| `config.py` | Configuration management | High - Service configuration |
| `executor.py` | DAG execution engine | Medium - Execution coordination |
| `cli.py` | Command-line interface | Low - Independent CLI access |
| `drop_zone_interface.py` | File processing integration | Low - Batch processing |

## Version Compatibility

**Current Version:** 2.0.0
- **Compatible Service Versions:** 2.0.0+
- **Breaking Changes from v1.x:** Gateway pattern deprecated, service layer added
- **Backwards Compatibility:** Core algorithms unchanged, only access patterns modified

## Support and Maintenance

**Primary Maintainer:** TidyLLM Core Team
**Integration Support:** Service Layer Team
**Related Documentation:**
- [RAG2DAG Service Architecture](../../docs/RAG2DAG_SERVICE_ARCHITECTURE.md)
- [RAG2DAG Implementation Guide](../../docs/RAG2DAG_IMPLEMENTATION_GUIDE.md)

**Issue Reporting:**
- Core algorithm issues: Report against `tidyllm/rag2dag/`
- Service integration issues: Report against `tidyllm/services/rag2dag/`
- Performance issues: May involve both layers

---

**This core implementation is preserved and maintained as the algorithmic foundation for RAG2DAG optimization in the TidyLLM enterprise platform.**