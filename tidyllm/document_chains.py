"""
Document Chains - Two-Layer Chaining for Document Operations
============================================================

Implements Option 3: Two-layer chaining that separates complex backend operations
from simple frontend operations, building on existing TidyLLM chaining infrastructure.

Layer 1 (Backend): Complex document pipeline - INGEST → EMBED → INDEX → TRACK → REPORT
Layer 2 (Frontend): Simple user operations - QUERY → SEARCH

This preserves the existing sophisticated chaining while providing a clean user interface.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

# Import existing chaining infrastructure
from .archive.old_src.tidyllm.verbs import chain as legacy_chain
from .knowledge_resource_server import KnowledgeMCPServer, KnowledgeResourceManager

logger = logging.getLogger("document_chains")


class DocumentOperation(Enum):
    """Document lifecycle operations."""
    # Backend operations (Layer 1)
    INGEST = "ingest"
    EMBED = "embed"  
    INDEX = "index"
    TRACK = "track"
    REPORT = "report"
    
    # Frontend operations (Layer 2)
    QUERY = "query"
    SEARCH = "search"


class ChainExecutionMode(Enum):
    """Chain execution modes."""
    SEQUENTIAL = "sequential"  # One after another
    PIPELINE = "pipeline"      # Stream between stages
    PARALLEL = "parallel"      # Where possible
    AUTO = "auto"             # Intelligent decision


@dataclass
class ChainContext:
    """Context passed between chain operations."""
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, stage: str, result: Any):
        """Add result from a stage."""
        self.stage_results[stage] = result
        self.metadata[f"{stage}_completed"] = datetime.now().isoformat()
    
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0


@dataclass 
class ChainConfig:
    """Configuration for chain execution."""
    execution_mode: ChainExecutionMode = ChainExecutionMode.SEQUENTIAL
    stop_on_error: bool = True
    max_parallel: int = 3
    timeout_per_stage: float = 30.0
    retry_attempts: int = 1
    enable_caching: bool = True


class DocumentOperationChain:
    """
    Chainable document operation that can be composed into larger workflows.
    
    Each operation is a link in the chain that can:
    - Process input data
    - Pass results to next operation
    - Handle errors gracefully
    - Provide metrics and logging
    """
    
    def __init__(self, operation: DocumentOperation, handler: Callable, **config):
        """Initialize chain operation."""
        self.operation = operation
        self.handler = handler
        self.config = config
        self.metrics = {"calls": 0, "errors": 0, "total_time": 0.0}
    
    def __call__(self, context: ChainContext) -> ChainContext:
        """Execute this operation in the chain."""
        start_time = datetime.now()
        self.metrics["calls"] += 1
        
        try:
            logger.info(f"Executing {self.operation.value} operation")
            
            # Execute the handler with context
            result = self.handler(context, **self.config)
            
            # Update context with result
            if result is not None:
                context.data = result
            context.add_result(self.operation.value, result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics["total_time"] += processing_time
            
            logger.info(f"Completed {self.operation.value} in {processing_time:.2f}s")
            
        except Exception as e:
            self.metrics["errors"] += 1
            error_msg = f"Error in {self.operation.value}: {str(e)}"
            context.errors.append(error_msg)
            logger.error(error_msg)
        
        return context
    
    def __rshift__(self, next_operation: 'DocumentOperationChain') -> 'DocumentChain':
        """Chain operations using >> operator."""
        return DocumentChain([self, next_operation])


class DocumentChain:
    """
    Composable chain of document operations.
    
    Supports the existing TidyLLM chaining patterns while providing
    a clean interface for document lifecycle management.
    """
    
    def __init__(self, operations: List[DocumentOperationChain] = None):
        """Initialize document chain."""
        self.operations = operations or []
        self.config = ChainConfig()
        self.execution_history: List[Dict[str, Any]] = []
    
    def add(self, operation: DocumentOperationChain) -> 'DocumentChain':
        """Add operation to chain."""
        self.operations.append(operation)
        return self
    
    def __rshift__(self, operation: DocumentOperationChain) -> 'DocumentChain':
        """Add operation using >> operator."""
        return self.add(operation)
    
    def configure(self, **config) -> 'DocumentChain':
        """Configure chain execution."""
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        return self
    
    def execute(self, initial_data: Any = None) -> ChainContext:
        """Execute the complete chain."""
        context = ChainContext(data=initial_data)
        execution_start = datetime.now()
        
        logger.info(f"Executing document chain with {len(self.operations)} operations")
        
        try:
            if self.config.execution_mode == ChainExecutionMode.SEQUENTIAL:
                context = self._execute_sequential(context)
            elif self.config.execution_mode == ChainExecutionMode.PIPELINE:
                context = self._execute_pipeline(context)
            elif self.config.execution_mode == ChainExecutionMode.PARALLEL:
                context = self._execute_parallel(context)
            else:  # AUTO
                context = self._execute_auto(context)
            
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            # Record execution history
            self.execution_history.append({
                "timestamp": execution_start.isoformat(),
                "execution_time": execution_time,
                "operations": [op.operation.value for op in self.operations],
                "success": not context.has_errors(),
                "errors": context.errors.copy()
            })
            
            logger.info(f"Chain execution completed in {execution_time:.2f}s")
            
        except Exception as e:
            context.errors.append(f"Chain execution failed: {str(e)}")
            logger.error(f"Chain execution failed: {e}")
        
        return context
    
    def _execute_sequential(self, context: ChainContext) -> ChainContext:
        """Execute operations sequentially."""
        for operation in self.operations:
            context = operation(context)
            
            # Stop on error if configured
            if self.config.stop_on_error and context.has_errors():
                logger.warning(f"Stopping chain execution due to errors: {context.errors}")
                break
        
        return context
    
    def _execute_pipeline(self, context: ChainContext) -> ChainContext:
        """Execute operations in pipeline mode (streaming)."""
        # For now, same as sequential - could add streaming later
        return self._execute_sequential(context)
    
    def _execute_parallel(self, context: ChainContext) -> ChainContext:
        """Execute operations in parallel where possible."""
        # For now, same as sequential - would need dependency analysis
        return self._execute_sequential(context)
    
    def _execute_auto(self, context: ChainContext) -> ChainContext:
        """Auto-select best execution mode."""
        # Simple heuristic: short chains sequential, long chains might benefit from parallelism
        if len(self.operations) <= 3:
            return self._execute_sequential(context)
        else:
            return self._execute_pipeline(context)


# =============================================================================
# Layer 1: Backend Document Pipeline (Complex Operations)
# =============================================================================

class BackendDocumentPipeline:
    """
    Layer 1: Complex document processing pipeline.
    
    Handles the sophisticated backend operations:
    INGEST → EMBED → INDEX → TRACK → REPORT
    
    This layer is for data engineers, ML engineers, and system administrators
    who need fine-grained control over the document processing pipeline.
    """
    
    def __init__(self, knowledge_server: KnowledgeMCPServer = None):
        """Initialize backend pipeline."""
        self.knowledge_server = knowledge_server or KnowledgeMCPServer()
        self.resource_manager = self.knowledge_server.resources
        self.processing_stats = {
            "documents_processed": 0,
            "embeddings_generated": 0, 
            "indices_created": 0,
            "total_processing_time": 0.0
        }
    
    def create_ingest_operation(self, source_config: Dict[str, Any]) -> DocumentOperationChain:
        """Create document ingestion operation."""
        def ingest_handler(context: ChainContext, **config) -> Dict[str, Any]:
            """Ingest documents from various sources."""
            source_type = source_config.get("type", "local")
            source_path = source_config.get("path", "")
            domain = source_config.get("domain", "default")
            
            if source_type == "s3":
                from .knowledge_resource_server.sources import S3KnowledgeSource
                source = S3KnowledgeSource(
                    bucket=source_config["bucket"],
                    prefix=source_config.get("prefix", "")
                )
            elif source_type == "local":
                from .knowledge_resource_server.sources import LocalKnowledgeSource
                source = LocalKnowledgeSource(directory=source_path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Register domain and initialize source
            self.knowledge_server.register_domain(domain, source)
            
            doc_count = source.get_document_count()
            self.processing_stats["documents_processed"] += doc_count
            
            return {
                "domain": domain,
                "documents_ingested": doc_count,
                "source_type": source_type,
                "source_path": source_path
            }
        
        return DocumentOperationChain(DocumentOperation.INGEST, ingest_handler, **source_config)
    
    def create_embed_operation(self, model: str = "sentence-transformers") -> DocumentOperationChain:
        """Create embedding generation operation."""
        def embed_handler(context: ChainContext, **config) -> Dict[str, Any]:
            """Generate embeddings for ingested documents."""
            domain = context.stage_results.get("ingest", {}).get("domain", "default")
            doc_count = context.stage_results.get("ingest", {}).get("documents_ingested", 0)
            
            # Mock embedding generation - in production would generate real embeddings
            embeddings_generated = doc_count * 100  # Assume 100 chunks per document
            self.processing_stats["embeddings_generated"] += embeddings_generated
            
            return {
                "domain": domain,
                "model_used": model,
                "embeddings_generated": embeddings_generated,
                "embedding_dimension": 384  # Common dimension
            }
        
        return DocumentOperationChain(DocumentOperation.EMBED, embed_handler, model=model)
    
    def create_index_operation(self, index_config: Dict[str, Any] = None) -> DocumentOperationChain:
        """Create indexing operation."""
        def index_handler(context: ChainContext, **config) -> Dict[str, Any]:
            """Create searchable indices."""
            embed_result = context.stage_results.get("embed", {})
            embeddings_count = embed_result.get("embeddings_generated", 0)
            
            index_config = config.get("index_config", {})
            vector_db = index_config.get("vector_db", "in_memory")
            
            # Mock index creation
            indices_created = 1
            self.processing_stats["indices_created"] += indices_created
            
            return {
                "vector_db": vector_db,
                "embeddings_indexed": embeddings_count,
                "indices_created": indices_created,
                "search_ready": True
            }
        
        return DocumentOperationChain(DocumentOperation.INDEX, index_handler, index_config=index_config or {})
    
    def create_track_operation(self) -> DocumentOperationChain:
        """Create tracking/monitoring operation."""
        def track_handler(context: ChainContext, **config) -> Dict[str, Any]:
            """Track processing metrics and performance."""
            return {
                "pipeline_stats": self.processing_stats.copy(),
                "stage_timings": {
                    stage: context.metadata.get(f"{stage}_completed")
                    for stage in ["ingest", "embed", "index"]
                    if f"{stage}_completed" in context.metadata
                },
                "errors_encountered": len(context.errors),
                "overall_success": not context.has_errors()
            }
        
        return DocumentOperationChain(DocumentOperation.TRACK, track_handler)
    
    def create_report_operation(self, report_format: str = "json") -> DocumentOperationChain:
        """Create reporting operation."""
        def report_handler(context: ChainContext, **config) -> Dict[str, Any]:
            """Generate processing report."""
            tracking_data = context.stage_results.get("track", {})
            
            report = {
                "pipeline_summary": {
                    "total_documents": self.processing_stats["documents_processed"],
                    "total_embeddings": self.processing_stats["embeddings_generated"],
                    "indices_created": self.processing_stats["indices_created"],
                    "processing_time": self.processing_stats["total_processing_time"]
                },
                "execution_details": context.stage_results,
                "errors": context.errors,
                "success": not context.has_errors(),
                "report_format": report_format,
                "generated_at": datetime.now().isoformat()
            }
            
            return report
        
        return DocumentOperationChain(DocumentOperation.REPORT, report_handler, format=report_format)
    
    def create_full_pipeline(self, source_config: Dict[str, Any], **pipeline_config) -> DocumentChain:
        """Create complete backend processing pipeline."""
        embed_model = pipeline_config.get("embed_model", "sentence-transformers")
        index_config = pipeline_config.get("index_config", {})
        report_format = pipeline_config.get("report_format", "json")
        
        # Create the full 5-stage pipeline
        pipeline = DocumentChain([
            self.create_ingest_operation(source_config),
            self.create_embed_operation(embed_model),
            self.create_index_operation(index_config),
            self.create_track_operation(),
            self.create_report_operation(report_format)
        ])
        
        # Configure execution
        execution_mode = pipeline_config.get("execution_mode", "sequential")
        pipeline.configure(
            execution_mode=ChainExecutionMode(execution_mode),
            stop_on_error=pipeline_config.get("stop_on_error", True),
            timeout_per_stage=pipeline_config.get("timeout_per_stage", 30.0)
        )
        
        return pipeline


# =============================================================================
# Layer 2: Frontend Document API (Simple Operations) 
# =============================================================================

class FrontendDocumentAPI:
    """
    Layer 2: Simple document query interface.
    
    Provides clean, simple operations for application developers:
    QUERY + SEARCH
    
    This layer hides all the complexity of ingestion, embedding, and indexing
    behind two simple methods that application developers actually use.
    """
    
    def __init__(self, knowledge_server: KnowledgeMCPServer = None):
        """Initialize frontend API."""
        self.knowledge_server = knowledge_server or KnowledgeMCPServer()
    
    def query(self, question: str, domain: str = None, **options) -> Dict[str, Any]:
        """
        Natural language query against knowledge base.
        
        Simple interface that hides all complexity behind one method.
        
        Args:
            question: Natural language question
            domain: Optional domain to search
            **options: Additional query options
            
        Returns:
            Dictionary with answer and supporting context
        """
        try:
            result = self.knowledge_server.handle_mcp_tool_call("query", {
                "question": question,
                "domain": domain,
                "context_length": options.get("context_length", 2000)
            })
            
            if result["success"]:
                return {
                    "answer": result["answer"],
                    "context": result["context"],
                    "question": question,
                    "domain": domain,
                    "success": True
                }
            else:
                return {"error": result["error"], "success": False}
                
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def search(self, keywords: str, domain: str = None, **options) -> List[Dict[str, Any]]:
        """
        Keyword search across documents.
        
        Simple interface for finding relevant documents.
        
        Args:
            keywords: Search keywords
            domain: Optional domain to search  
            **options: Additional search options
            
        Returns:
            List of matching documents with relevance scores
        """
        try:
            result = self.knowledge_server.handle_mcp_tool_call("search", {
                "query": keywords,
                "domain": domain,
                "max_results": options.get("max_results", 5),
                "similarity_threshold": options.get("similarity_threshold", 0.7)
            })
            
            if result["success"]:
                return result["results"]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


# =============================================================================
# Unified Document System (Both Layers)
# =============================================================================

class DocumentSystem:
    """
    Unified document system providing both layers.
    
    - Backend: Complex pipeline operations for data teams
    - Frontend: Simple query/search for application teams
    
    This provides the "Option 3" two-layer chaining approach.
    """
    
    def __init__(self):
        """Initialize document system."""
        self.knowledge_server = KnowledgeMCPServer()
        self.backend = BackendDocumentPipeline(self.knowledge_server)
        self.frontend = FrontendDocumentAPI(self.knowledge_server)
    
    # Backend Layer (Complex Operations)
    def create_pipeline(self, source_config: Dict[str, Any], **config) -> DocumentChain:
        """Create backend processing pipeline."""
        return self.backend.create_full_pipeline(source_config, **config)
    
    def ingest(self, source_config: Dict[str, Any]) -> DocumentOperationChain:
        """Create ingest operation."""
        return self.backend.create_ingest_operation(source_config)
    
    def embed(self, model: str = "sentence-transformers") -> DocumentOperationChain:
        """Create embed operation."""
        return self.backend.create_embed_operation(model)
    
    def index(self, config: Dict[str, Any] = None) -> DocumentOperationChain:
        """Create index operation."""
        return self.backend.create_index_operation(config)
    
    def track(self) -> DocumentOperationChain:
        """Create tracking operation."""
        return self.backend.create_track_operation()
    
    def report(self, format: str = "json") -> DocumentOperationChain:
        """Create report operation."""
        return self.backend.create_report_operation(format)
    
    # Frontend Layer (Simple Operations)
    def query(self, question: str, domain: str = None, **options) -> Dict[str, Any]:
        """Simple natural language query."""
        return self.frontend.query(question, domain, **options)
    
    def search(self, keywords: str, domain: str = None, **options) -> List[Dict[str, Any]]:
        """Simple keyword search."""
        return self.frontend.search(keywords, domain, **options)
    
    # Convenience methods
    def quick_setup(self, source_path: str, domain: str = "default") -> Dict[str, Any]:
        """Quick setup: ingest → embed → index in one call."""
        source_config = {
            "type": "local",
            "path": source_path,
            "domain": domain
        }
        
        pipeline = self.create_pipeline(source_config)
        result = pipeline.execute()
        
        return {
            "success": not result.has_errors(),
            "domain": domain,
            "processing_summary": result.stage_results,
            "errors": result.errors
        }