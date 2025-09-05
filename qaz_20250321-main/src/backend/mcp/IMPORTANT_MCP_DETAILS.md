# 🚨 SYSTEMS INTEL - DO NOT DELETE 🚨
# IMPORTANT MCP DETAILS

## ⚠️ CRITICAL WARNING
**This file contains SYSTEM INTELLIGENCE about our MCP architecture.**
**DO NOT DELETE, MODIFY, OR MOVE this file without consulting the team.**
**This document is essential for understanding our MCP system architecture.**

**📋 RELATED SYSTEM INTELLIGENCE:**
- **Database Architecture**: See `database/IMPORTANT_DB_CONNECTION.md` for database connection patterns
- **Credential Management**: See `src/backend/config/credential_manager.py` for credential handling

---

## MCP Architecture Overview

The Model Context Protocol (MCP) implements a hierarchical architecture with three main layers:

1. **Planners** - High-level strategic planning and decision-making
2. **Coordinators** - Mid-level coordination and tactical execution  
3. **Workers** - Task-specific execution and processing components

## Component Tables

### Planners

| Name | Location | Functions | Comments |
|------|----------|-----------|----------|
| `EnhancedPlanner` | `src/backend/mcp/planner/enhanced_planner.py` | • Strategic request analysis<br>• Coordination strategy selection<br>• Live context decision making<br>• Result aggregation | Main entry point for MCP system. Routes requests to appropriate coordinators. |

### Coordinators

| Name | Location | Functions | Comments |
|------|----------|-----------|----------|
| `DocumentCoordinator` | `src/backend/mcp/coordinators/document_coordinator.py` | • PDF processing orchestration<br>• Text cleaning coordination<br>• Embedding generation<br>• Table extraction<br>• Live context integration | Core document processing pipeline. Coordinates multiple workers. |
| `SMEContextCoordinator` | `src/backend/mcp/coordinators/sme_context_coordinator.py` | • Subject matter expert context<br>• Domain-specific processing<br>• Expert knowledge integration | Handles domain-specific processing and expert knowledge. |
| `DSPyCoordinator` | `src/backend/mcp/coordinators/dspy_coordinator.py` | • DSPy framework integration<br>• LLM orchestration<br>• Chain-of-thought processing | Manages DSPy-based LLM operations and reasoning chains. |

### Workers

| Name | Location | Functions | Comments |
|------|----------|-----------|----------|
| `BaseWorker` | `src/backend/mcp/workers/base_worker.py` | • Abstract worker interface<br>• Performance metrics<br>• Audit trail management<br>• Message handling | Base class for all workers. Defines common interface. |
| `PDFProcessorWorker` | `src/backend/mcp/workers/document_workers.py` | • PDF text extraction<br>• Page processing<br>• Metadata extraction | Handles PDF document parsing and text extraction. |
| `TextCleanerWorker` | `src/backend/mcp/workers/document_workers.py` | • Text normalization<br>• Chunking<br>• Quality filtering | Cleans and chunks extracted text for processing. |
| `EmbeddingGeneratorWorker` | `src/backend/mcp/workers/document_workers.py` | • Vector embedding generation<br>• Batch processing<br>• Embedding storage | Generates embeddings for text chunks. |
| `TableExtractorWorker` | `src/backend/mcp/workers/document_workers.py` | • Table extraction<br>• Structured data processing<br>• Table validation | Extracts and processes tables from documents. |
| `LiveContextWorker` | `src/backend/mcp/workers/live_context_worker.py` | • Live database queries<br>• Temporal context<br>• Mock data generation | Provides live context from database or mock data. |

### Orchestrators (4 Core)

| Name | Location | Functions | Comments |
|------|----------|-----------|----------|
| `QAOrchestrator` | `src/backend/mcp/orchestrators/qa_orchestrator.py` | • Basic QA processing<br>• Document validation<br>• Simple reports | Basic orchestrator for simple QA tasks. |
| `QAReviewerOrchestrator` | `src/backend/mcp/orchestrators/qa_reviewer_orchestrator.py` | • Expert QA review<br>• Compliance assessment<br>• Professional reports | Advanced orchestrator for expert-level QA. |
| `LLMEnhancedQAOrchestrator` | `src/backend/mcp/orchestrators/llm_enhanced_qa_orchestrator.py` | • LLM-enhanced processing<br>• Document classification<br>• Batch processing | Uses LLMs to enhance QA processing. |
| `RAGQAOrchestrator` | `src/backend/mcp/orchestrators/rag_qa_orchestrator.py` | • RAG processing<br>• Document search<br>• Context-aware QA | Implements RAG (Retrieval-Augmented Generation). |

## Protocol Components

| Name | Location | Functions | Comments |
|------|----------|-----------|----------|
| `MCPMessage` | `src/backend/mcp/protocol/message_protocol.py` | • Message structure<br>• Payload handling<br>• Type safety | Core message structure for MCP communication. |
| `AuditTrail` | `src/backend/mcp/protocol/message_protocol.py` | • Decision tracking<br>• Performance metrics<br>• Debugging support | Tracks decisions and performance for debugging. |
| `MessageType` | `src/backend/mcp/protocol/message_protocol.py` | • Message type definitions<br>• Type validation<br>• Routing logic | Defines different types of MCP messages. |

## CRITICAL SAFEGUARDS

### 1. Import Chain Dependencies

**NEVER** modify these files without updating their corresponding `__init__.py` files:

- `src/backend/mcp/workers/__init__.py` - Must export all worker classes
- `src/backend/mcp/coordinators/__init__.py` - Must export all coordinator classes  
- `src/backend/mcp/planner/__init__.py` - Must export all planner classes
- `src/backend/mcp/orchestrators/__init__.py` - Must export all orchestrator classes
- `src/backend/mcp/protocol/__init__.py` - Must export all protocol classes
- `src/backend/mcp/__init__.py` - Must import all subpackages

### 2. Class Name Consistency

**CRITICAL**: The following class names must match exactly between files:

- `EmbeddingHelper` in `src/backend/core/embedding_helper.py` (used by RAGQAOrchestrator)
- `AmazonEmbeddingVectorizer` in `src/backend/core/embedding_helper.py` (base implementation)
- All worker class names in `document_workers.py` must match exports in `__init__.py`

### 3. Message Protocol Compatibility

**NEVER** change these function signatures without updating all callers:

- `create_planner_to_coordinator_message()`
- `create_coordinator_to_worker_message()`
- `create_worker_to_coordinator_message()`
- `create_coordinator_to_planner_message()`

### 4. Database Schema Dependencies

The following components depend on specific database table structures:

- `LiveContextWorker` - Requires `events_raw`, `events_daily`, `review_findings` tables
- `RAGQAOrchestrator` - Requires `document_chunks`, `chunk_embeddings` tables
- `EmbeddingGeneratorWorker` - Requires vector storage tables

### 5. External Dependencies

**CRITICAL DEPENDENCIES** that can break the system:

- `sentence_transformers` - Used by RAGQAOrchestrator (NumPy version conflicts possible)
- `pymupdf` - Used by PDFProcessorWorker for PDF processing
- `psycopg2` - Used by LiveContextWorker for database connections
- `dspy-ai` - Used by LLMEnhancedQAOrchestrator and RAGQAOrchestrator

### 6. Database Integration Requirements

**CRITICAL**: The MCP system integrates with the database architecture documented in `database/IMPORTANT_DB_CONNECTION.md`:

- **Credential Manager**: All database connections must use `credential_manager.get_database_config()`
- **Live Context**: `LiveContextWorker` requires access to `realtime_context` table
- **Error Tracking**: All components should log errors to `error_tracking` table
- **Performance Metrics**: Store metrics in `batch_processing_status` table
- **Prompt History**: Track all prompts in `prompt_history` table (MLflow integration)

**NEVER** hardcode database URLs or bypass the credential manager system.

### 6. Configuration Files

**REQUIRED** configuration files that must exist or have fallbacks:

- `dev_configs/qa_criteria_full.yaml` - Used by QAReportGenerator
- Database connection strings in environment variables
- Embedding model configurations

### 7. Error Handling Patterns

**ALWAYS** implement these error handling patterns:

```python
# For optional dependencies
try:
    from .some_module import SomeClass
except ImportError as e:
    logging.warning(f"SomeClass not available: {e}")
    SomeClass = None

# For database operations
try:
    # database operation
except Exception as e:
    logger.warning(f"Database operation failed: {e}")
    # fallback behavior

# For file operations
try:
    # file operation
except FileNotFoundError:
    # create default or use fallback
```

### 8. Testing Requirements

**ALWAYS** run these tests after any changes:

```bash
# Test basic imports
python3 -c "from src.backend.mcp.workers import BaseWorker; print('Workers OK')"
python3 -c "from src.backend.mcp.coordinators import DocumentCoordinator; print('Coordinators OK')"
python3 -c "from src.backend.mcp.planner import EnhancedPlanner; print('Planner OK')"
python3 -c "from src.backend.mcp.orchestrators import QAOrchestrator, LLMEnhancedQAOrchestrator, QAReviewerOrchestrator, RAGQAOrchestrator; print('Orchestrators OK')"

# Test MCP hierarchy
python3 tests/test_mcp_hierarchy_simple.py
```

### 9. Breaking Change Checklist

Before making any changes, check:

- [ ] Are all `__init__.py` files updated with new exports?
- [ ] Are class names consistent across all files?
- [ ] Are message protocol signatures unchanged?
- [ ] Are database schema dependencies documented?
- [ ] Are external dependencies listed?
- [ ] Are configuration file dependencies handled?
- [ ] Are error handling patterns implemented?
- [ ] Do all tests pass after changes?

### 10. Recovery Procedures

If the system breaks:

1. **Import Errors**: Check `__init__.py` files for missing exports
2. **Class Not Found**: Verify class names match exactly
3. **Database Errors**: Check table schemas and connection strings
4. **Dependency Errors**: Check external package versions
5. **Configuration Errors**: Verify config files exist or have fallbacks

## File Modification Priority

**SAFE TO MODIFY** (low risk):
- Individual worker implementations
- Coordinator logic (within existing interfaces)
- Planner decision logic

**HIGH RISK** (requires careful testing):
- `__init__.py` files
- Message protocol structures
- Class names and interfaces
- Database schema dependencies

**CRITICAL** (requires full testing):
- Base classes and interfaces
- Message protocol signatures
- Import chains and dependencies

## Current Architecture Summary

**1 Planner + 4 Orchestrators Structure:**
- **EnhancedPlanner**: Main entry point (strategic planning and coordination)
- **QAOrchestrator**: Basic QA processing
- **QAReviewerOrchestrator**: Expert QA with compliance
- **LLMEnhancedQAOrchestrator**: LLM-enhanced processing
- **RAGQAOrchestrator**: RAG-based document querying

**Removed Components:**
- ~~SmartOrchestratorRouter~~ (enhanced into EnhancedPlanner)
- ~~DSPyCoordinator~~ (placeholder removed - DSPy integrated directly in orchestrators)
