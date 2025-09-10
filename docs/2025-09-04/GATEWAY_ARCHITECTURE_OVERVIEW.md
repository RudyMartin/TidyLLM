# TidyLLM Gateway Architecture Overview

## Executive Summary

TidyLLM employs a gateway-based architecture with three specialized processing engines and a knowledge management system. Each gateway serves a distinct purpose in the AI/ML pipeline, providing enterprise-grade capabilities for different aspects of language model operations.

## Core Architecture Components

### 1. **AIProcessingGateway** (`tidyllm/gateways/ai_processing_gateway.py`)
**Purpose:** Multi-Model AI Processing Engine

**Key Features:**
- **Multiple Backend Support:** AUTO, Bedrock, SageMaker, OpenAI, Anthropic, MLFlow, Mock
- **Enterprise Features:** 
  - Response caching with configurable TTL
  - Retry logic with exponential backoff
  - Metrics tracking and performance monitoring
- **Configuration:** AIRequest dataclass with temperature, max_tokens, timeout controls
- **Default Model:** claude-3-sonnet with 2000 token limit

**Use Cases:**
- Flexible AI model selection based on requirements
- Development and testing with mock backends
- Production deployments with enterprise backends

### 2. **WorkflowOptimizerGateway** (`tidyllm/gateways/workflow_optimizer_gateway.py`)
**Purpose:** Workflow Analysis and Optimization Engine

**Key Features:**
- **Workflow Analysis:** Identifies bottlenecks and inefficiencies
- **Optimization Types:**
  - ANALYZE - Workflow issue detection
  - OPTIMIZE - Performance improvements
  - CLEANUP - Fix messy manual workflows
  - VALIDATE - Compliance checking
  - SUGGEST - Improvement recommendations
  - AUTO_FIX - Automatic issue resolution
- **Components:**
  - HierarchicalDAGManager - Manages workflow DAGs
  - FlowAgreementManager - Handles workflow agreements
- **Enterprise Controls:** Compliance mode, audit trails, configurable optimization levels

**Use Cases:**
- Cleaning up user-created workflows
- Ensuring workflow compliance
- Optimizing workflow performance
- Maintaining audit trails

### 3. **CorporateLLMGateway** (`tidyllm/gateways/corporate_llm_gateway.py`)
**Purpose:** Corporate-Controlled Language Model Access

**Key Features:**
- **Zero Direct External Access:** All requests routed through corporate infrastructure
- **MLFlow Integration:** Uses MLFlow Gateway Client for centralized control
- **IT-Managed Controls:**
  - Available providers list (claude, openai-corporate, azure-gpt)
  - Model whitelist per provider
  - Cost controls (max tokens, budget limits)
  - Temperature range restrictions
- **Enterprise Features:**
  - Full audit trails with required reason fields
  - Cost tracking per request/user
  - Multi-tenant access management
  - Graceful fallback mechanisms

**Default Configuration:**
- Providers: claude, openai
- Models: claude-3-5-sonnet, gpt-4o
- Max tokens: 4096 per request
- Max cost: $1.00 USD per request

**Use Cases:**
- Corporate AI deployments
- Regulated environments requiring audit trails
- Cost-controlled AI operations
- Multi-tenant enterprise applications

### 4. **DatabaseGateway** (`tidyllm/gateways/database_gateway.py`)
**Purpose:** Database Operations and Data Persistence

**Key Features:**
- **Database Abstraction:** Unified interface for database operations
- **Connection Management:** Pool management and connection optimization
- **Query Optimization:** Performance monitoring and query analysis
- **Enterprise Features:**
  - Transaction management
  - Security controls and access patterns
  - Audit logging for data operations

**Use Cases:**
- Centralized database access
- Data persistence and retrieval
- Transaction management
- Database performance optimization

### 5. **FileStorageGateway** (`tidyllm/gateways/file_storage_gateway.py`)
**Purpose:** File Storage and Document Management

**Key Features:**
- **S3-First Architecture:** Direct S3 integration for cloud storage
- **Document Processing:** File upload, processing, and metadata extraction
- **Security Controls:** Encryption, access controls, and audit trails
- **Enterprise Features:**
  - Batch processing capabilities
  - Automatic metadata extraction
  - Version management

**Use Cases:**
- Document storage and retrieval
- File processing pipelines
- S3 bucket management
- Document metadata extraction

### 4. **Knowledge Systems** (`tidyllm/knowledge_systems/`)
**Purpose:** Domain-Specific Knowledge Management and RAG

**Core Components:**

#### Core Managers (`/core/`):
- **KnowledgeManager:** Central knowledge orchestration
- **S3Manager:** Document storage and retrieval from S3
- **VectorManager:** Vector database operations for semantic search
- **DomainRAG:** Domain-specific Retrieval Augmented Generation

#### Supporting Systems:
- **Model Discovery:** 
  - `dynamic_model_discovery.py` - Runtime model detection
  - `startup_model_discovery.py` - Initial model setup
  - `model_discovery_scheduler.py` - Scheduled model updates
- **Embedding Configuration:** `embedding_config.py` - Embedding model settings
- **Enhanced Extraction:** `enhanced_extraction.py` - Advanced document processing
- **Workflow Configuration:** `workflow_config.py` - Workflow-specific settings

#### Facades (`/facades/`):
- Simplified interfaces for complex operations
- `embedding_processor.py` - Embedding generation
- `vector_storage.py` - Vector DB abstraction

**Key Features:**
- Domain-specific knowledge areas (Model Validation, Legal, Technical)
- S3-backed document storage with configurable buckets/prefixes
- Vector search with similarity thresholds
- Metadata extraction and schema validation
- Chunking and processing pipelines

**Use Cases:**
- Building domain-specific knowledge bases
- RAG-powered Q&A systems
- Document processing and indexing
- Semantic search across documents

## Gateway Interconnections

### Base Gateway Interface (`base_gateway.py`)
All gateways inherit from `BaseGateway` and implement:
- **GatewayResponse:** Standardized response format with status, data, metadata, errors
- **GatewayStatus:** SUCCESS, FAILURE, PARTIAL, TIMEOUT, RATE_LIMITED
- **GatewayDependencies:** Inter-gateway dependency configuration

### Dependency Chain
```
CorporateLLMGateway (Corporate Control)
    ↓
AIProcessingGateway (Model Selection)
    ↓
WorkflowOptimizerGateway (Workflow Optimization)
    ↓
Knowledge Systems (Domain Knowledge)
    ↓
DatabaseGateway & FileStorageGateway (Persistence)
```

## Directory Structure
```
tidyllm/
├── gateways/
│   ├── __init__.py                    # Gateway registry and imports
│   ├── base_gateway.py                # Base interface and common types
│   ├── ai_processing_gateway.py       # Multi-model AI processing
│   ├── workflow_optimizer_gateway.py  # Workflow optimization
│   ├── corporate_llm_gateway.py       # Corporate LLM access
│   ├── database_gateway.py            # Database operations
│   ├── file_storage_gateway.py        # File storage operations
│   └── gateway_registry.py            # Gateway registration system
└── knowledge_systems/
    ├── core/                 # Core knowledge components
    │   ├── domain_rag.py
    │   ├── knowledge_manager.py
    │   ├── s3_manager.py
    │   └── vector_manager.py
    ├── facades/              # Simplified interfaces
    ├── interfaces/           # Abstract interfaces
    └── implementations/      # Concrete implementations
```

## Integration Points

1. **Gateway Registry:** Centralized gateway access via `get_gateway("gateway_name")`
2. **Unified Response Format:** All gateways return `GatewayResponse` objects
3. **Dependency Injection:** Gateways can declare dependencies on other gateways
4. **Knowledge Integration:** Knowledge systems can leverage any gateway for processing

## Enterprise Features Summary

- **Security:** Zero direct external API access, IT-controlled endpoints
- **Compliance:** Full audit trails, compliance validation, cost controls
- **Performance:** Caching, retry logic, optimization engines
- **Scalability:** Multi-tenant support, distributed processing
- **Flexibility:** Multiple backend support, configurable optimization levels
- **Knowledge Management:** Domain-specific RAGs, S3 integration, vector search

This architecture provides a robust, enterprise-ready foundation for AI/ML operations with clear separation of concerns and comprehensive control mechanisms.