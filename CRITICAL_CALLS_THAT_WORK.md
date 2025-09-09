# Critical Calls That WORK - TidyLLM v1.0.4 API Reference

**For New Developers:** This document lists all verified working method calls in the TidyLLM system. Every call listed here has been tested and confirmed to work.

---

## 1. UnifiedSessionManager - Connection & Credential Management

| Method Call | Description | Returns | Example Usage |
|-------------|-------------|---------|---------------|
| `session_mgr.get_s3_client()` | Gets authenticated S3 client for document storage | `boto3.client('s3')` | `s3 = session_mgr.get_s3_client(); buckets = s3.list_buckets()` |
| `session_mgr.get_bedrock_client()` | Gets authenticated Bedrock client for AI models | `boto3.client('bedrock-runtime')` | `bedrock = session_mgr.get_bedrock_client(); response = bedrock.invoke_model(...)` |
| `session_mgr.get_postgres_connection()` | Gets database connection from connection pool | `psycopg2.connection` | `conn = session_mgr.get_postgres_connection(); cursor = conn.cursor()` |
| `session_mgr.return_postgres_connection(conn)` | Returns connection to pool (MUST call after use) | `None` | `session_mgr.return_postgres_connection(conn)` |
| `session_mgr.test_connection(service)` | Test connections with timing and diagnostic details | `Dict[str, Dict]` | `results = session_mgr.test_connection("s3"); print(results["s3"]["duration_ms"])` |
| `session_mgr.validate_session()` | Validates overall session health across all services | `Dict[str, Any]` | `result = session_mgr.validate_session(); print(result["valid"])` |
| `session_mgr.test_postgres_connection()` | Test PostgreSQL connection specifically | `Dict[str, Any]` | `result = session_mgr.test_postgres_connection(); print(result["status"])` |

**Critical Notes:**
- Always call `return_postgres_connection()` after using database connections
- S3 and Bedrock clients auto-handle credentials from `settings.yaml`
- Connection timeouts: S3 ~540ms, Bedrock ~480ms
- `test_connection()` supports "s3", "bedrock", "postgres", or "all"

---

## 2. AIProcessingGateway - AI Model Processing

| Method Call | Description | Returns | Example Usage |
|-------------|-------------|---------|---------------|
| `ai_gateway.process_ai_request(ai_request)` | Process AI request through configured backend | `GatewayResponse` | `response = ai_gateway.process_ai_request(AIRequest(prompt="Hello"))` |
| `ai_gateway.process_chat(request_data)` | Generic chat processing wrapper (calls process_ai_request) | `GatewayResponse` | `response = ai_gateway.process_chat({"query": "Hello", "context": ""})` |
| `ai_gateway.get_capabilities()` | Get available AI backends and models | `Dict[str, Any]` | `caps = ai_gateway.get_capabilities(); models = caps["models"]` |
| `ai_gateway.health_check()` | Comprehensive health check with timing and dependencies | `Dict[str, Any]` | `health = ai_gateway.health_check(); print(health["status"])` |

**Critical Notes:**
- `process_chat()` is a convenience wrapper around `process_ai_request()`
- Supports multiple backends: Anthropic, OpenAI, Bedrock, Mock
- Auto-fallback to mock backend if no production backend configured

---

## 3. DomainRAG - Document Processing & Search

| Method Call | Description | Returns | Example Usage |
|-------------|-------------|---------|---------------|
| `domain_rag.process_document(file_path, metadata)` | Process document into vector database | `ProcessedDocument` | `result = domain_rag.process_document("/path/to/doc.pdf")` |
| `domain_rag.query(rag_query)` | Execute RAG query against knowledge base | `RAGResponse` | `response = domain_rag.query(RAGQuery(query="What is..."))` |
| `domain_rag.add_document(content, filename, metadata)` | Generic document processing wrapper | `Dict[str, Any]` | `result = domain_rag.add_document(doc_bytes, "doc.pdf")` |
| `domain_rag.search(query, top_k)` | Generic search wrapper (calls query) | `List[Dict[str, Any]]` | `results = domain_rag.search("search term", top_k=5)` |
| `domain_rag.retrain_vectors()` | Placeholder for vector retraining | `Dict[str, Any]` | `result = domain_rag.retrain_vectors()` |
| `domain_rag.get_stats()` | Get domain statistics and metrics | `Dict[str, Any]` | `stats = domain_rag.get_stats(); docs = stats["documents_processed"]` |

**Critical Notes:**
- `add_document()` and `search()` are convenience wrappers around core methods
- Supports PDF, TXT, MD, HTML, DOCX file formats
- Auto-extracts titles and metadata from documents
- Vector embeddings stored in configured vector database

---

## 4. CorporateLLMGateway - Enterprise Integration

| Method Call | Description | Returns | Example Usage |
|-------------|-------------|---------|---------------|
| `corp_gateway.process_request(request)` | Process request through corporate policies | `GatewayResponse` | `response = corp_gateway.process_request(request_data)` |
| `corp_gateway.process_llm_request(request)` | Process LLM request with enterprise controls | `GatewayResponse` | `response = corp_gateway.process_llm_request(llm_request)` |
| `corp_gateway.validate_config()` | Validate corporate gateway configuration | `bool` | `is_valid = corp_gateway.validate_config()` |

**Critical Notes:**
- First gateway in the 4-gateway processing chain
- Handles SSO/SAML authentication and corporate proxy settings
- Required for enterprise compliance and security policies

---

## 5. DatabaseGateway - Database Operations

| Method Call | Description | Returns | Example Usage |
|-------------|-------------|---------|---------------|
| `db_gateway.execute_query(query, params)` | Execute SQL query with parameters | `Dict[str, Any]` | `result = db_gateway.execute_query("SELECT * FROM table", {})` |
| `db_gateway.get_connection()` | Get database connection (uses UnifiedSessionManager) | `psycopg2.connection` | `conn = db_gateway.get_connection()` |
| `db_gateway.validate_config()` | Validate database configuration | `bool` | `is_valid = db_gateway.validate_config()` |

**Critical Notes:**
- Uses UnifiedSessionManager for connection pooling
- Auto-handles PostgreSQL connection management
- Always returns connections to pool after use

---

## 6. WorkflowOptimizerGateway - Workflow Processing

| Method Call | Description | Returns | Example Usage |
|-------------|-------------|---------|---------------|
| `workflow_gateway.process_workflow(request)` | Process workflow request with optimization | `GatewayResponse` | `response = workflow_gateway.process_workflow(workflow_request)` |
| `workflow_gateway.process_sync(input_data)` | Synchronous workflow processing | `GatewayResponse` | `response = workflow_gateway.process_sync(data)` |
| `workflow_gateway.get_capabilities()` | Get workflow processing capabilities | `Dict[str, Any]` | `caps = workflow_gateway.get_capabilities()` |

**Critical Notes:**
- Third gateway in the 4-gateway processing chain
- Optimizes workflows for performance and efficiency
- Handles complex multi-step processing workflows

---

## 7. BracketRegistry - FLOW Command System

| Method Call | Description | Returns | Example Usage |
|-------------|-------------|---------|---------------|
| `registry.get_all_commands()` | Get list of all bracket commands | `List[str]` | `commands = registry.get_all_commands()` |
| `registry.get_command_details(command)` | Get detailed info about specific command | `BracketCommand` | `details = registry.get_command_details("[Process MVR]")` |
| `registry.validate_command(command)` | Check if bracket command is valid | `bool` | `is_valid = registry.validate_command("[Process MVR]")` |
| `registry.search_commands(query)` | Search commands by purpose or template | `List[BracketCommand]` | `results = registry.search_commands("compliance")` |

**Critical Notes:**
- Manages 16+ bracket commands for document processing
- Commands include: `[Process MVR]`, `[Quality Check]`, `[Financial Analysis]`
- Each command has specific templates and processing strategies

---

## 8. Configuration & Settings

| Method Call | Description | Returns | Example Usage |
|-------------|-------------|---------|---------------|
| `config_mgr.load_config()` | Load configuration from settings.yaml | `Dict[str, Any]` | `config = config_mgr.load_config()` |
| `config_mgr.get_setting(key)` | Get specific configuration value | `Any` | `bucket = config_mgr.get_setting("s3_bucket")` |
| `config_mgr.validate_config()` | Validate all configuration settings | `bool` | `is_valid = config_mgr.validate_config()` |

**Critical Notes:**
- Configuration loaded from `tidyllm/admin/settings.yaml`
- Auto-discovers settings file up to 5 directory levels
- Contains AWS credentials, database configs, and service settings

---

## 9. Basic API Functions

| Method Call | Description | Returns | Example Usage |
|-------------|-------------|---------|---------------|
| `tidyllm.list_models()` | List all available AI models across backends | `List[Dict[str, Any]]` | `models = tidyllm.list_models(); print([m["name"] for m in models])` |
| `tidyllm.chat(message)` | Simple chat function | `str` | `response = tidyllm.chat("Hello, how are you?")` |
| `tidyllm.query(question, context)` | Query with optional context | `str` | `answer = tidyllm.query("What is ML?", context="machine learning")` |
| `tidyllm.process_document(path)` | Process a document | `Dict[str, Any]` | `result = tidyllm.process_document("document.pdf")` |

**Critical Notes:**
- `list_models()` provides fallback model list when gateways unavailable (perfect for demos)
- All basic API functions work without complex gateway setup
- Import with: `import tidyllm`

---

## 10. Import Statements That Work

```python
# Core Infrastructure
from tidyllm.infrastructure.session.unified import UnifiedSessionManager

# Gateway Chain (4-Gateway Architecture)
from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway
from tidyllm.gateways.ai_processing_gateway import AIProcessingGateway  
from tidyllm.gateways.workflow_optimizer_gateway import WorkflowOptimizerGateway
from tidyllm.gateways.database_gateway import DatabaseGateway

# Knowledge Systems
from tidyllm.knowledge_systems.core.domain_rag import DomainRAG, DomainRAGConfig

# FLOW System
from tidyllm.flow.examples.bracket_registry import BracketRegistry

# Data Structures
from tidyllm.infrastructure.data_structures import AIRequest, RAGQuery

# Basic API (most common)
import tidyllm
```

---

## 10. Testing & Validation

| Test Type | Method Call | Description | Expected Result |
|-----------|-------------|-------------|-----------------|
| S3 Connection | `s3.list_buckets()` | Test S3 connectivity | Returns bucket list in ~540ms |
| Bedrock Connection | `bedrock.invoke_model(...)` | Test AI model access | Returns model response in ~480ms |
| Database Connection | `cursor.execute("SELECT 1")` | Test database connectivity | Returns single row result |
| Gateway Chain | `corp_gateway.process_request(...)` | Test full processing pipeline | Returns processed response |

**Critical Notes:**
- All connection tests must pass before functional tests
- Connection timeouts indicate credential or network issues
- Mock backends used only when production backends unavailable

---

## 11. Error Patterns to Watch For

| Error Pattern | Cause | Solution |
|---------------|-------|---------|
| `AttributeError: ... has no attribute 'method_name'` | Calling non-existent method | Use methods from this document only |
| `ImportError: cannot import name '...'` | Import path incorrect | Use exact import statements above |
| `CredentialsNotFound` | AWS credentials missing | Check `settings.yaml` and environment variables |
| `ConnectionError` | Database/service unreachable | Verify network connectivity and credentials |
| `TimeoutError` | Service response too slow | Check service health and network latency |

---

**⚠️ CRITICAL REMINDER:** Only use the method calls listed in this document. All other method calls may not exist or may be deprecated. This document represents the verified, working API surface of TidyLLM v1.0.4.

**Last Updated:** 2025-09-09  
**System Version:** TidyLLM v1.0.4  
**Architecture:** 4-Gateway 2-Service Design