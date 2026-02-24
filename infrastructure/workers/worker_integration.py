"""
Worker Integration Layer - Gateway and MCP Integration
======================================================

Integration layer that connects the agent worker architecture with existing
gateways and MCP server infrastructure. Provides seamless integration
without disrupting existing APIs.

Capabilities:
- Gateway worker integration
- MCP server worker delegation  
- Worker pool management
- Load balancing and scaling
- Health monitoring integration
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .base_worker import BaseWorker, WorkerStatus, TaskPriority
from .extraction_worker import ExtractionWorker
from .embedding_worker import EmbeddingWorker
from .indexing_worker import IndexingWorker
from .processing_worker import ProcessingWorker

logger = logging.getLogger("worker_integration")


@dataclass
class WorkerPoolConfig:
    """Configuration for worker pool."""
    max_extraction_workers: int = 2
    max_embedding_workers: int = 3
    max_indexing_workers: int = 2
    max_processing_workers: int = 1
    
    # Worker configuration
    worker_timeout: float = 300.0
    max_queue_size: int = 100
    health_check_interval: float = 30.0


class WorkerPool:
    """
    Manages a pool of specialized workers for integration with gateways.
    
    Provides:
    - Worker lifecycle management
    - Load balancing across worker instances
    - Health monitoring and recovery
    - Unified API for gateway integration
    """
    
    def __init__(self, config: WorkerPoolConfig = None):
        """Initialize worker pool."""
        self.config = config or WorkerPoolConfig()
        
        # Worker pools by type
        self.extraction_workers: List[ExtractionWorker] = []
        self.embedding_workers: List[EmbeddingWorker] = []
        self.indexing_workers: List[IndexingWorker] = []
        self.processing_workers: List[ProcessingWorker] = []
        
        # Load balancing state
        self.extraction_round_robin = 0
        self.embedding_round_robin = 0
        self.indexing_round_robin = 0
        self.processing_round_robin = 0
        
        # Pool state
        self.is_initialized = False
        self.health_check_task: Optional[asyncio.Task] = None
        
        logger.info("Worker pool initialized")
    
    async def initialize(self) -> None:
        """Initialize all worker pools."""
        try:
            logger.info("Initializing worker pool...")
            
            # Initialize extraction workers
            for i in range(self.config.max_extraction_workers):
                worker = ExtractionWorker(
                    worker_name=f"extraction_worker_{i}",
                    task_timeout=self.config.worker_timeout,
                    max_queue_size=self.config.max_queue_size
                )
                await worker.initialize()
                await worker.start()
                self.extraction_workers.append(worker)
            
            # Initialize embedding workers  
            for i in range(self.config.max_embedding_workers):
                worker = EmbeddingWorker(
                    worker_name=f"embedding_worker_{i}",
                    task_timeout=self.config.worker_timeout,
                    max_queue_size=self.config.max_queue_size
                )
                await worker.initialize()
                await worker.start()
                self.embedding_workers.append(worker)
            
            # Initialize indexing workers
            for i in range(self.config.max_indexing_workers):
                worker = IndexingWorker(
                    worker_name=f"indexing_worker_{i}",
                    task_timeout=self.config.worker_timeout,
                    max_queue_size=self.config.max_queue_size
                )
                await worker.initialize()
                await worker.start()
                self.indexing_workers.append(worker)
            
            # Initialize processing workers
            for i in range(self.config.max_processing_workers):
                worker = ProcessingWorker(
                    worker_name=f"processing_worker_{i}",
                    task_timeout=self.config.worker_timeout,
                    max_queue_size=self.config.max_queue_size
                )
                await worker.initialize()
                await worker.start()
                self.processing_workers.append(worker)
            
            self.is_initialized = True
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"Worker pool initialized: "
                       f"{len(self.extraction_workers)} extraction, "
                       f"{len(self.embedding_workers)} embedding, "
                       f"{len(self.indexing_workers)} indexing, "
                       f"{len(self.processing_workers)} processing workers")
            
        except Exception as e:
            logger.error(f"Worker pool initialization failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown all workers in the pool."""
        logger.info("Shutting down worker pool...")
        
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Stop all workers
        shutdown_tasks = []
        
        for worker in (self.extraction_workers + self.embedding_workers + 
                      self.indexing_workers + self.processing_workers):
            shutdown_tasks.append(worker.stop(timeout=30.0))
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.is_initialized = False
        logger.info("Worker pool shutdown complete")
    
    def get_extraction_worker(self) -> Optional[ExtractionWorker]:
        """Get next available extraction worker using round-robin."""
        if not self.extraction_workers:
            return None
        
        # Find healthy worker
        for i in range(len(self.extraction_workers)):
            worker_index = (self.extraction_round_robin + i) % len(self.extraction_workers)
            worker = self.extraction_workers[worker_index]
            
            if worker.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]:
                self.extraction_round_robin = (worker_index + 1) % len(self.extraction_workers)
                return worker
        
        return None
    
    def get_embedding_worker(self) -> Optional[EmbeddingWorker]:
        """Get next available embedding worker using round-robin."""
        if not self.embedding_workers:
            return None
        
        for i in range(len(self.embedding_workers)):
            worker_index = (self.embedding_round_robin + i) % len(self.embedding_workers)
            worker = self.embedding_workers[worker_index]
            
            if worker.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]:
                self.embedding_round_robin = (worker_index + 1) % len(self.embedding_workers)
                return worker
        
        return None
    
    def get_indexing_worker(self) -> Optional[IndexingWorker]:
        """Get next available indexing worker using round-robin."""
        if not self.indexing_workers:
            return None
        
        for i in range(len(self.indexing_workers)):
            worker_index = (self.indexing_round_robin + i) % len(self.indexing_workers)
            worker = self.indexing_workers[worker_index]
            
            if worker.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]:
                self.indexing_round_robin = (worker_index + 1) % len(self.indexing_workers)
                return worker
        
        return None
    
    def get_processing_worker(self) -> Optional[ProcessingWorker]:
        """Get next available processing worker using round-robin."""
        if not self.processing_workers:
            return None
        
        for i in range(len(self.processing_workers)):
            worker_index = (self.processing_round_robin + i) % len(self.processing_workers)
            worker = self.processing_workers[worker_index]
            
            if worker.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]:
                self.processing_round_robin = (worker_index + 1) % len(self.processing_workers)
                return worker
        
        return None
    
    async def _health_check_loop(self) -> None:
        """Background health monitoring loop."""
        while self.is_initialized:
            try:
                # Check all workers
                all_workers = (self.extraction_workers + self.embedding_workers + 
                              self.indexing_workers + self.processing_workers)
                
                for worker in all_workers:
                    try:
                        health = await worker.health_check()
                        if health.get("health_status") == "unhealthy":
                            logger.warning(f"Worker '{worker.worker_name}' is unhealthy: {health.get('issues', [])}")
                    except Exception as e:
                        logger.error(f"Health check failed for '{worker.worker_name}': {e}")
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get overall pool status."""
        def get_worker_stats(workers: List[BaseWorker]) -> Dict[str, Any]:
            healthy = sum(1 for w in workers if w.status in [WorkerStatus.IDLE, WorkerStatus.BUSY])
            total_tasks = sum(w.metrics.total_tasks_processed for w in workers)
            avg_success_rate = sum(w.metrics.success_rate for w in workers) / len(workers) if workers else 0
            
            return {
                "total_workers": len(workers),
                "healthy_workers": healthy,
                "total_tasks_processed": total_tasks,
                "average_success_rate": avg_success_rate
            }
        
        return {
            "pool_initialized": self.is_initialized,
            "extraction_workers": get_worker_stats(self.extraction_workers),
            "embedding_workers": get_worker_stats(self.embedding_workers), 
            "indexing_workers": get_worker_stats(self.indexing_workers),
            "processing_workers": get_worker_stats(self.processing_workers),
            "timestamp": datetime.now().isoformat()
        }


class GatewayWorkerIntegration:
    """
    Integration layer between gateways and workers.
    
    Provides gateway methods that delegate to appropriate workers
    while maintaining existing API compatibility.
    """
    
    def __init__(self, worker_pool: WorkerPool):
        """Initialize gateway worker integration."""
        self.worker_pool = worker_pool
        logger.info("Gateway worker integration initialized")
    
    async def extract_document_content(self, 
                                     document_id: str,
                                     content_source: Union[str, bytes],
                                     document_type: str = "pdf",
                                     extraction_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract document content using extraction workers.
        
        Gateway-compatible API that delegates to ExtractionWorker.
        """
        try:
            worker = self.worker_pool.get_extraction_worker()
            if not worker:
                raise RuntimeError("No extraction workers available")
            
            # Submit extraction task
            task_id = await worker.extract_document(
                document_id=document_id,
                content_source=content_source,
                document_type=document_type,
                extraction_options=extraction_options or {},
                priority=TaskPriority.NORMAL
            )
            
            # Wait for completion (gateway expects synchronous result)
            result = await self._wait_for_task_completion(worker, task_id)
            
            return {
                "document_id": document_id,
                "extracted_text": result.extracted_text,
                "chunks": result.chunks,
                "metadata": result.metadata,
                "warnings": result.warnings,
                "processing_time": result.extraction_time
            }
            
        except Exception as e:
            logger.error(f"Gateway document extraction failed: {e}")
            raise
    
    async def generate_embeddings(self,
                                text_data: Union[str, List[str]],
                                model_provider: str = "default",
                                target_dimension: int = 1024) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using embedding workers.
        
        Gateway-compatible API that delegates to EmbeddingWorker.
        """
        try:
            worker = self.worker_pool.get_embedding_worker()
            if not worker:
                raise RuntimeError("No embedding workers available")
            
            if isinstance(text_data, str):
                # Single text
                task_id = await worker.generate_embedding(
                    text_id="single_text",
                    text_content=text_data,
                    model_provider=model_provider,
                    target_dimension=target_dimension,
                    priority=TaskPriority.HIGH
                )
                
                result = await self._wait_for_task_completion(worker, task_id)
                return result.embedding
                
            else:
                # Multiple texts
                texts = [{"id": f"text_{i}", "content": text} for i, text in enumerate(text_data)]
                
                task_id = await worker.generate_batch_embeddings(
                    batch_id="gateway_batch",
                    texts=texts,
                    model_provider=model_provider,
                    target_dimension=target_dimension,
                    priority=TaskPriority.HIGH
                )
                
                result = await self._wait_for_task_completion(worker, task_id)
                return [emb.embedding for emb in result.embeddings]
            
        except Exception as e:
            logger.error(f"Gateway embedding generation failed: {e}")
            raise
    
    async def index_document(self,
                           document_id: str,
                           title: str,
                           content: str,
                           chunks_with_embeddings: List[Dict[str, Any]],
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Index document using indexing workers.
        
        Gateway-compatible API that delegates to IndexingWorker.
        """
        try:
            worker = self.worker_pool.get_indexing_worker()
            if not worker:
                raise RuntimeError("No indexing workers available")
            
            task_id = await worker.index_document(
                document_id=document_id,
                title=title,
                content=content,
                chunks=chunks_with_embeddings,
                metadata=metadata or {},
                priority=TaskPriority.NORMAL
            )
            
            result = await self._wait_for_task_completion(worker, task_id)
            
            return {
                "document_id": document_id,
                "indexed_chunks": result.indexed_chunks,
                "total_chunks": result.total_chunks,
                "index_status": result.index_status,
                "warnings": result.warnings,
                "processing_time": result.index_time
            }
            
        except Exception as e:
            logger.error(f"Gateway document indexing failed: {e}")
            raise
    
    async def search_similar_documents(self,
                                     query_embedding: List[float],
                                     max_results: int = 10,
                                     similarity_threshold: float = 0.7,
                                     filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using indexing workers.
        
        Gateway-compatible API that delegates to IndexingWorker.
        """
        try:
            worker = self.worker_pool.get_indexing_worker()
            if not worker:
                raise RuntimeError("No indexing workers available")
            
            task_id = await worker.search_vectors(
                query_id="gateway_search",
                query_embedding=query_embedding,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                filter_metadata=filter_metadata or {},
                priority=TaskPriority.HIGH
            )
            
            result = await self._wait_for_task_completion(worker, task_id)
            return result.results
            
        except Exception as e:
            logger.error(f"Gateway document search failed: {e}")
            raise
    
    async def process_document_pipeline(self,
                                      document_id: str,
                                      document_source: Union[str, bytes],
                                      document_type: str = "pdf",
                                      title: str = "",
                                      metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process complete document pipeline using processing workers.
        
        Gateway-compatible API that delegates to ProcessingWorker.
        """
        try:
            worker = self.worker_pool.get_processing_worker()
            if not worker:
                raise RuntimeError("No processing workers available")
            
            task_id = await worker.process_document_pipeline(
                pipeline_id=f"gateway_pipeline_{document_id}",
                document_id=document_id,
                document_source=document_source,
                document_type=document_type,
                title=title,
                metadata=metadata or {},
                priority=TaskPriority.NORMAL
            )
            
            result = await self._wait_for_task_completion(worker, task_id)
            
            return {
                "document_id": document_id,
                "processing_stages": result.processing_stages,
                "overall_status": result.overall_status,
                "extracted_text_length": result.extracted_text_length,
                "chunks_created": result.chunks_created,
                "embeddings_generated": result.embeddings_generated,
                "chunks_indexed": result.chunks_indexed,
                "total_processing_time": result.total_processing_time,
                "warnings": result.warnings,
                "errors": result.errors
            }
            
        except Exception as e:
            logger.error(f"Gateway document pipeline failed: {e}")
            raise
    
    async def _wait_for_task_completion(self, worker: BaseWorker, task_id: str, timeout: float = 300.0) -> Any:
        """Wait for worker task completion and return result."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            task_status = worker.get_task_status(task_id)
            
            if task_status:
                if task_status.get("error_message"):
                    raise RuntimeError(f"Worker task failed: {task_status['error_message']}")
                
                if task_status.get("completed_at"):
                    # Find completed task
                    for completed_task in worker.completed_tasks:
                        if completed_task.task_id == task_id:
                            return completed_task.result
                    
                    raise RuntimeError("Task completed but result not found")
            
            await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError(f"Task {task_id} timed out after {timeout}s")


class MCPWorkerIntegration:
    """
    Integration layer between MCP server and workers.
    
    Extends MCP server capabilities with worker delegation for
    compute-intensive operations.
    """
    
    def __init__(self, worker_pool: WorkerPool):
        """Initialize MCP worker integration."""
        self.worker_pool = worker_pool
        logger.info("MCP worker integration initialized")
    
    def get_worker_enhanced_capabilities(self) -> Dict[str, Any]:
        """Get MCP capabilities enhanced with worker functionality."""
        base_capabilities = {
            "tools": [
                {
                    "name": "extract_document_worker",
                    "description": "Extract document content using dedicated extraction workers",
                    "parameters": {
                        "document_id": {"type": "string", "required": True},
                        "document_source": {"type": "string", "required": True},
                        "document_type": {"type": "string", "default": "pdf"}
                    }
                },
                {
                    "name": "generate_embeddings_worker", 
                    "description": "Generate embeddings using dedicated embedding workers",
                    "parameters": {
                        "texts": {"type": "array", "required": True},
                        "model_provider": {"type": "string", "default": "default"},
                        "target_dimension": {"type": "integer", "default": 1024}
                    }
                },
                {
                    "name": "process_document_pipeline_worker",
                    "description": "Process complete document pipeline using orchestration workers",
                    "parameters": {
                        "document_id": {"type": "string", "required": True},
                        "document_source": {"type": "string", "required": True},
                        "document_type": {"type": "string", "default": "pdf"}
                    }
                },
                {
                    "name": "get_worker_pool_status",
                    "description": "Get status and health information for all workers",
                    "parameters": {}
                }
            ]
        }
        
        # Add worker pool information
        pool_status = self.worker_pool.get_pool_status()
        base_capabilities["worker_pool"] = pool_status
        
        return base_capabilities
    
    async def handle_worker_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls that delegate to workers."""
        try:
            if tool_name == "extract_document_worker":
                return await self._handle_extract_document_worker(parameters)
            elif tool_name == "generate_embeddings_worker":
                return await self._handle_generate_embeddings_worker(parameters)
            elif tool_name == "process_document_pipeline_worker":
                return await self._handle_process_document_pipeline_worker(parameters)
            elif tool_name == "get_worker_pool_status":
                return await self._handle_get_worker_pool_status(parameters)
            else:
                return {
                    "success": False,
                    "error": f"Unknown worker tool: {tool_name}"
                }
                
        except Exception as e:
            logger.error(f"Worker tool call '{tool_name}' failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    async def _handle_extract_document_worker(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document extraction via workers."""
        gateway_integration = GatewayWorkerIntegration(self.worker_pool)
        
        result = await gateway_integration.extract_document_content(
            document_id=params["document_id"],
            content_source=params["document_source"], 
            document_type=params.get("document_type", "pdf")
        )
        
        return {
            "success": True,
            "tool": "extract_document_worker",
            "result": result
        }
    
    async def _handle_generate_embeddings_worker(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle embedding generation via workers."""
        gateway_integration = GatewayWorkerIntegration(self.worker_pool)
        
        embeddings = await gateway_integration.generate_embeddings(
            text_data=params["texts"],
            model_provider=params.get("model_provider", "default"),
            target_dimension=params.get("target_dimension", 1024)
        )
        
        return {
            "success": True,
            "tool": "generate_embeddings_worker",
            "embeddings": embeddings,
            "count": len(embeddings)
        }
    
    async def _handle_process_document_pipeline_worker(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document pipeline processing via workers."""
        gateway_integration = GatewayWorkerIntegration(self.worker_pool)
        
        result = await gateway_integration.process_document_pipeline(
            document_id=params["document_id"],
            document_source=params["document_source"],
            document_type=params.get("document_type", "pdf")
        )
        
        return {
            "success": True,
            "tool": "process_document_pipeline_worker",
            "result": result
        }
    
    async def _handle_get_worker_pool_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle worker pool status request."""
        status = self.worker_pool.get_pool_status()
        
        return {
            "success": True,
            "tool": "get_worker_pool_status",
            "pool_status": status
        }


# Global worker pool instance for integration
_global_worker_pool: Optional[WorkerPool] = None
_global_gateway_integration: Optional[GatewayWorkerIntegration] = None
_global_mcp_integration: Optional[MCPWorkerIntegration] = None


async def initialize_worker_integrations(config: WorkerPoolConfig = None) -> Dict[str, Any]:
    """
    Initialize global worker integration components.
    
    Returns:
        Dictionary with integration components
    """
    global _global_worker_pool, _global_gateway_integration, _global_mcp_integration
    
    try:
        logger.info("Initializing global worker integrations...")
        
        # Initialize worker pool
        _global_worker_pool = WorkerPool(config)
        await _global_worker_pool.initialize()
        
        # Initialize gateway integration
        _global_gateway_integration = GatewayWorkerIntegration(_global_worker_pool)
        
        # Initialize MCP integration
        _global_mcp_integration = MCPWorkerIntegration(_global_worker_pool)
        
        logger.info("Global worker integrations initialized successfully")
        
        return {
            "worker_pool": _global_worker_pool,
            "gateway_integration": _global_gateway_integration,
            "mcp_integration": _global_mcp_integration
        }
        
    except Exception as e:
        logger.error(f"Worker integration initialization failed: {e}")
        raise


def get_global_worker_integrations() -> Dict[str, Any]:
    """Get global worker integration components."""
    return {
        "worker_pool": _global_worker_pool,
        "gateway_integration": _global_gateway_integration,
        "mcp_integration": _global_mcp_integration
    }


async def shutdown_worker_integrations() -> None:
    """Shutdown global worker integration components."""
    global _global_worker_pool, _global_gateway_integration, _global_mcp_integration
    
    if _global_worker_pool:
        await _global_worker_pool.shutdown()
        _global_worker_pool = None
    
    _global_gateway_integration = None
    _global_mcp_integration = None
    
    logger.info("Global worker integrations shut down")