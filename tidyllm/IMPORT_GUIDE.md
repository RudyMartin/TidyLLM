# TidyLLM Import Guide - Clear Calling Patterns
=============================================

## Package Structure & Import Patterns

### Core Modules (Production Use)
```python
# Main API
from tidyllm import chat, query, process_document

# CLI Interface
from tidyllm.cli import main

# API Server
from tidyllm.api import TidyLLMAPI
```

### Infrastructure Components
```python
# Configuration Management
from tidyllm.infrastructure.config import ConfigManager

# Session Management  
from tidyllm.infrastructure.session import UnifiedSessionManager

# Worker Architecture
from tidyllm.infrastructure.workers import (
    BaseWorker,
    ExtractionWorker,
    EmbeddingWorker,
    IndexingWorker,
    ProcessingWorker
)

# AI Managers
from tidyllm.infrastructure.workers import (
    AIDropzoneManager,
    FlowIntegrationManager
)
```

### Enterprise Gateways
```python
# Core Gateways
from tidyllm.gateways import (
    CorporateLLMGateway,      # LLM access control
    DatabaseGateway,          # Database operations
    FileStorageGateway,       # S3/file management
    AIProcessingGateway       # AI workflow orchestration
)

# Workflow Gateway
from tidyllm.gateways import WorkflowOptimizerGateway
```

### Knowledge Systems
```python
# Domain RAG
from tidyllm.knowledge_systems.core import DomainRAG

# Knowledge Manager
from tidyllm.knowledge_systems.core import KnowledgeManager

# Document Processing
from tidyllm.knowledge_systems.facades import (
    DocumentProcessor,
    EmbeddingProcessor,
    VectorStorage
)
```

### Flow System (Bracket Commands)
```python
# Bracket Registry
from tidyllm.flow.examples import BracketRegistry

# Flow Mappings
from tidyllm.flow.examples import get_flow_mappings
```

### Web Dashboard
```python
# Dashboard Components
from tidyllm.web.components import (
    DropzoneMonitor,
    ProcessingStatusMonitor,
    BracketCommandPanel
)

# Dashboard Utilities
from tidyllm.web.utils import (
    DashboardUtils,
    ConfigManager,
    AlertManager
)
```

## File Location Reference

### ✅ Core Files (tidyllm/)
- `__init__.py` - Package initialization, main API
- `api.py` - REST API server implementation
- `cli.py` - Command-line interface

### ✅ Infrastructure (tidyllm/infrastructure/)
- `config/` - Configuration management
- `session/` - AWS S3 and database sessions
- `workers/` - Worker architecture and AI managers
- `api/` - API endpoints and routing
- `cli/` - CLI commands

### ✅ Gateways (tidyllm/gateways/)
- `corporate_llm_gateway.py` - LLM access control
- `database_gateway.py` - Database operations
- `file_storage_gateway.py` - S3/file management
- `ai_processing_gateway.py` - AI orchestration
- `workflow_optimizer_gateway.py` - Workflow optimization

### ✅ Knowledge Systems (tidyllm/knowledge_systems/)
- `core/` - Core RAG and knowledge management
- `facades/` - High-level interfaces
- `interfaces/` - Abstract interfaces

### ✅ Flow System (tidyllm/flow/)
- `examples/` - Bracket commands and registry
- `agreements/` - YAML workflow definitions

### ✅ Web Dashboard (tidyllm/web/)
- `ai_dropzone_dashboard.py` - Main dashboard app
- `components/` - Reusable UI components
- `pages/` - Additional dashboard pages
- `utils/` - Helper utilities

### ✅ Examples (tidyllm/examples/)
- API usage examples
- Demo applications
- Integration examples

### ✅ Tests (tidyllm/tests/)
- Unit tests
- Integration tests
- Quality tests

## Common Import Errors & Solutions

### Error: `No module named 'tidyllm.tidyllm'`
**Solution**: Remove duplicate `tidyllm` - use `from tidyllm.gateways` not `from tidyllm.tidyllm.gateways`

### Error: `No module named 'scripts'`
**Solution**: Scripts imports should use `tidyllm.infrastructure` instead

### Error: `Cannot import UnifiedSessionManager`
**Solution**: Use `from tidyllm.infrastructure.session import UnifiedSessionManager`

## Quick Reference

| What You Want | Import Statement |
|--------------|------------------|
| Chat with LLM | `from tidyllm import chat` |
| Query documents | `from tidyllm import query` |
| Process files | `from tidyllm import process_document` |
| Use gateways | `from tidyllm.gateways import CorporateLLMGateway` |
| Access workers | `from tidyllm.infrastructure.workers import BaseWorker` |
| Run dashboard | `streamlit run tidyllm/web/ai_dropzone_dashboard.py` |
| Use CLI | `tidyllm status` or `from tidyllm.cli import main` |

## Testing Imports

```python
# Test your imports
python -c "from tidyllm import chat; print('✓ Core API works')"
python -c "from tidyllm.gateways import CorporateLLMGateway; print('✓ Gateways work')"
python -c "from tidyllm.infrastructure.workers import BaseWorker; print('✓ Workers work')"
```

---

Last Updated: 2024-01-15
Version: 1.0