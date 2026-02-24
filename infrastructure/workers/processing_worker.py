"""
Processing Worker - Document Processing Workflow Orchestrator
============================================================

Dedicated worker for coordinating complex document processing workflows.
Orchestrates multiple specialized workers (Extraction, Embedding, Indexing)
to provide end-to-end document processing capabilities.

Capabilities:
- End-to-end document processing pipelines
- Multi-worker orchestration and coordination
- Workflow state management and recovery
- Processing pipeline optimization
- Complex document analysis workflows
- Batch processing coordination
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from .base_worker import BaseWorker, TaskPriority
from .extraction_worker import ExtractionWorker, ExtractionRequest
from .embedding_worker import EmbeddingWorker, EmbeddingRequest, BatchEmbeddingRequest
from .indexing_worker import IndexingWorker, IndexingRequest
from ..session.unified import UnifiedSessionManager

logger = logging.getLogger("processing_worker")


@dataclass
class ProcessingPipelineRequest:
    """Request for complete document processing pipeline."""
    pipeline_id: str
    document_id: str
    document_source: Union[str, bytes]  # S3 path, file path, or raw content
    document_type: str = "pdf"
    processing_options: Dict[str, Any] = None
    
    # Pipeline configuration
    extract_content: bool = True
    generate_embeddings: bool = True
    index_document: bool = True
    embedding_model: str = "default"
    target_dimension: int = 1024
    
    # Metadata
    title: str = ""
    source: str = ""
    doc_type: str = "text"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.processing_options is None:
            self.processing_options = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchProcessingRequest:
    """Request for batch document processing."""
    batch_id: str
    documents: List[ProcessingPipelineRequest]
    batch_processing_options: Dict[str, Any] = None
    
    # Batch configuration
    max_concurrent_documents: int = 5
    fail_fast: bool = False  # Stop on first error
    
    def __post_init__(self):
        if self.batch_processing_options is None:
            self.batch_processing_options = {}


@dataclass
class WorkflowAnalysisRequest:
    """Request for workflow analysis and optimization."""
    analysis_id: str
    document_ids: List[str]
    analysis_type: str = "full"  # full, content_only, compliance, similarity
    comparison_criteria: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.comparison_criteria is None:
            self.comparison_criteria = {}


@dataclass
class ProcessingResult:
    """Result from document processing pipeline."""
    pipeline_id: str
    document_id: str
    processing_stages: Dict[str, Dict[str, Any]]  # stage_name -> stage_result
    overall_status: str = "success"  # success, partial, failed
    total_processing_time: Optional[float] = None
    warnings: List[str] = None
    errors: List[str] = None
    
    # Results summary
    extracted_text_length: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    chunks_indexed: int = 0
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "document_id": self.document_id,
            "processing_stages": self.processing_stages,
            "overall_status": self.overall_status,
            "total_processing_time": self.total_processing_time,
            "warnings": self.warnings,
            "errors": self.errors,
            "extracted_text_length": self.extracted_text_length,
            "chunks_created": self.chunks_created,
            "embeddings_generated": self.embeddings_generated,
            "chunks_indexed": self.chunks_indexed
        }


@dataclass
class BatchProcessingResult:
    """Result from batch document processing."""
    batch_id: str
    successful_documents: int
    failed_documents: int
    total_processing_time: Optional[float] = None
    document_results: List[ProcessingResult] = None
    batch_errors: List[str] = None
    
    def __post_init__(self):
        if self.document_results is None:
            self.document_results = []
        if self.batch_errors is None:
            self.batch_errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "total_processing_time": self.total_processing_time,
            "document_results": [result.to_dict() for result in self.document_results],
            "batch_errors": self.batch_errors
        }


@dataclass
class WorkflowAnalysisResult:
    """Result from workflow analysis."""
    analysis_id: str
    documents_analyzed: int
    analysis_results: Dict[str, Any]
    recommendations: List[str] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "documents_analyzed": self.documents_analyzed,
            "analysis_results": self.analysis_results,
            "recommendations": self.recommendations,
            "processing_time": self.processing_time
        }


class ProcessingWorker(BaseWorker[Union[ProcessingPipelineRequest, BatchProcessingRequest, WorkflowAnalysisRequest],
                                Union[ProcessingResult, BatchProcessingResult, WorkflowAnalysisResult]]):
    """
    Worker for orchestrating complex document processing workflows.
    
    Coordinates multiple specialized workers:
    - ExtractionWorker: Document content extraction
    - EmbeddingWorker: Vector embedding generation
    - IndexingWorker: Document indexing and storage
    
    Task Types:
    - process_document_pipeline: Full document processing workflow
    - batch_process_documents: Batch document processing
    - analyze_workflow: Workflow analysis and optimization
    - optimize_processing: Processing pipeline optimization
    """
    
    def __init__(self,
                 worker_name: str = "processing_worker",
                 max_concurrent_pipelines: int = 3,
                 **kwargs):
        """
        Initialize Processing Worker.
        
        Args:
            worker_name: Worker identifier
            max_concurrent_pipelines: Maximum concurrent processing pipelines
        """
        super().__init__(worker_name, **kwargs)
        
        self.max_concurrent_pipelines = max_concurrent_pipelines
        
        # Specialized workers
        self.extraction_worker: Optional[ExtractionWorker] = None
        self.embedding_worker: Optional[EmbeddingWorker] = None
        self.indexing_worker: Optional[IndexingWorker] = None
        
        # Session management
        self.session_manager = None
        
        # Pipeline tracking
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Processing Worker '{worker_name}' configured for {max_concurrent_pipelines} concurrent pipelines")
    
    async def _initialize_worker(self) -> None:
        """Initialize specialized workers and dependencies."""
        try:
            # Initialize UnifiedSessionManager
            try:
                self.session_manager = UnifiedSessionManager()
                logger.info("Processing Worker: UnifiedSessionManager initialized")
            except Exception as e:
                logger.warning(f"Processing Worker: UnifiedSessionManager not available: {e}")
            
            # Initialize specialized workers
            await self._initialize_specialized_workers()
            
            if not any([self.extraction_worker, self.embedding_worker, self.indexing_worker]):
                raise RuntimeError("No specialized workers available")
                
        except Exception as e:
            logger.error(f"Processing Worker initialization failed: {e}")
            raise
    
    async def _initialize_specialized_workers(self) -> None:
        """Initialize the specialized workers."""
        try:
            # Initialize ExtractionWorker
            try:
                self.extraction_worker = ExtractionWorker(
                    worker_name=f"{self.worker_name}_extraction"
                )
                await self.extraction_worker.initialize()
                await self.extraction_worker.start()
                logger.info("Processing Worker: ExtractionWorker initialized and started")
            except Exception as e:
                logger.warning(f"Processing Worker: ExtractionWorker initialization failed: {e}")
            
            # Initialize EmbeddingWorker  
            try:
                self.embedding_worker = EmbeddingWorker(
                    worker_name=f"{self.worker_name}_embedding"
                )
                await self.embedding_worker.initialize()
                await self.embedding_worker.start()
                logger.info("Processing Worker: EmbeddingWorker initialized and started")
            except Exception as e:
                logger.warning(f"Processing Worker: EmbeddingWorker initialization failed: {e}")
            
            # Initialize IndexingWorker
            try:
                self.indexing_worker = IndexingWorker(
                    worker_name=f"{self.worker_name}_indexing"
                )
                await self.indexing_worker.initialize()
                await self.indexing_worker.start()
                logger.info("Processing Worker: IndexingWorker initialized and started")
            except Exception as e:
                logger.warning(f"Processing Worker: IndexingWorker initialization failed: {e}")
                
        except Exception as e:
            logger.error(f"Specialized worker initialization failed: {e}")
            raise
    
    def validate_input(self, task_input: Any) -> bool:
        """Validate processing request input."""
        if isinstance(task_input, ProcessingPipelineRequest):
            return bool(task_input.pipeline_id and task_input.document_id and task_input.document_source)
        elif isinstance(task_input, BatchProcessingRequest):
            return bool(task_input.batch_id and task_input.documents)
        elif isinstance(task_input, WorkflowAnalysisRequest):
            return bool(task_input.analysis_id and task_input.document_ids)
        return False
    
    async def process_task(self, task_input: Union[ProcessingPipelineRequest, BatchProcessingRequest, WorkflowAnalysisRequest]) -> Union[ProcessingResult, BatchProcessingResult, WorkflowAnalysisResult]:
        """Process workflow orchestration request."""
        if isinstance(task_input, ProcessingPipelineRequest):
            return await self._process_document_pipeline(task_input)
        elif isinstance(task_input, BatchProcessingRequest):
            return await self._process_batch_documents(task_input)
        elif isinstance(task_input, WorkflowAnalysisRequest):
            return await self._analyze_workflow(task_input)
        else:
            raise ValueError(f"Unsupported task input type: {type(task_input)}")
    
    async def _process_document_pipeline(self, request: ProcessingPipelineRequest) -> ProcessingResult:
        """Process complete document processing pipeline."""
        start_time = time.time()
        pipeline_stages = {}
        warnings = []
        errors = []
        
        try:
            logger.info(f"Starting processing pipeline '{request.pipeline_id}' for document '{request.document_id}'")
            
            # Track active pipeline
            self.active_pipelines[request.pipeline_id] = {
                "document_id": request.document_id,
                "start_time": start_time,
                "current_stage": "initialization"
            }
            
            # Stage 1: Content Extraction
            extraction_result = None
            if request.extract_content and self.extraction_worker:
                try:
                    self.active_pipelines[request.pipeline_id]["current_stage"] = "extraction"
                    
                    extraction_task_id = await self.extraction_worker.extract_document(
                        document_id=request.document_id,
                        content_source=request.document_source,
                        document_type=request.document_type,
                        extraction_options=request.processing_options.get("extraction", {}),
                        priority=TaskPriority.HIGH
                    )
                    
                    # Wait for extraction to complete
                    extraction_result = await self._wait_for_worker_task(
                        self.extraction_worker, extraction_task_id
                    )
                    
                    pipeline_stages["extraction"] = {
                        "status": "success",
                        "result": extraction_result.to_dict() if extraction_result else None,
                        "task_id": extraction_task_id
                    }
                    
                    logger.info(f"Extraction completed for '{request.document_id}': "
                               f"{len(extraction_result.extracted_text)} chars, {len(extraction_result.chunks)} chunks")
                    
                except Exception as e:
                    error_msg = f"Extraction failed: {str(e)}"
                    errors.append(error_msg)
                    pipeline_stages["extraction"] = {"status": "failed", "error": error_msg}
                    logger.error(f"Pipeline '{request.pipeline_id}' extraction failed: {e}")
            
            # Stage 2: Embedding Generation
            embeddings_result = None
            if request.generate_embeddings and self.embedding_worker and extraction_result:
                try:
                    self.active_pipelines[request.pipeline_id]["current_stage"] = "embedding"
                    
                    # Prepare texts for embedding
                    texts_to_embed = []
                    for i, chunk in enumerate(extraction_result.chunks):
                        texts_to_embed.append({
                            "id": chunk.get("chunk_id", f"{request.document_id}_chunk_{i}"),
                            "content": chunk.get("text", "")
                        })
                    
                    # Generate embeddings for all chunks
                    embedding_batch_id = f"{request.pipeline_id}_embeddings"
                    embedding_task_id = await self.embedding_worker.generate_batch_embeddings(
                        batch_id=embedding_batch_id,
                        texts=texts_to_embed,
                        model_provider=request.embedding_model,
                        target_dimension=request.target_dimension,
                        priority=TaskPriority.HIGH
                    )
                    
                    # Wait for embeddings to complete
                    embeddings_result = await self._wait_for_worker_task(
                        self.embedding_worker, embedding_task_id
                    )
                    
                    pipeline_stages["embedding"] = {
                        "status": "success",
                        "result": embeddings_result.to_dict() if embeddings_result else None,
                        "task_id": embedding_task_id
                    }
                    
                    logger.info(f"Embeddings completed for '{request.document_id}': "
                               f"{embeddings_result.successful_embeddings} successful")
                    
                except Exception as e:
                    error_msg = f"Embedding generation failed: {str(e)}"
                    errors.append(error_msg)
                    pipeline_stages["embedding"] = {"status": "failed", "error": error_msg}
                    logger.error(f"Pipeline '{request.pipeline_id}' embedding failed: {e}")
            
            # Stage 3: Document Indexing
            indexing_result = None
            if request.index_document and self.indexing_worker and extraction_result:
                try:
                    self.active_pipelines[request.pipeline_id]["current_stage"] = "indexing"
                    
                    # Combine chunks with embeddings
                    chunks_with_embeddings = []
                    for i, chunk in enumerate(extraction_result.chunks):
                        chunk_data = chunk.copy()
                        
                        # Add embedding if available
                        if embeddings_result and i < len(embeddings_result.embeddings):
                            embedding = embeddings_result.embeddings[i]
                            chunk_data["embedding"] = embedding.embedding
                        
                        chunks_with_embeddings.append(chunk_data)
                    
                    # Index the document
                    indexing_task_id = await self.indexing_worker.index_document(
                        document_id=request.document_id,
                        title=request.title or request.document_id,
                        content=extraction_result.extracted_text,
                        chunks=chunks_with_embeddings,
                        source=request.source,
                        doc_type=request.doc_type,
                        metadata=request.metadata,
                        priority=TaskPriority.HIGH
                    )
                    
                    # Wait for indexing to complete
                    indexing_result = await self._wait_for_worker_task(
                        self.indexing_worker, indexing_task_id
                    )
                    
                    pipeline_stages["indexing"] = {
                        "status": "success",
                        "result": indexing_result.to_dict() if indexing_result else None,
                        "task_id": indexing_task_id
                    }
                    
                    logger.info(f"Indexing completed for '{request.document_id}': "
                               f"{indexing_result.indexed_chunks} chunks indexed")
                    
                except Exception as e:
                    error_msg = f"Document indexing failed: {str(e)}"
                    errors.append(error_msg)
                    pipeline_stages["indexing"] = {"status": "failed", "error": error_msg}
                    logger.error(f"Pipeline '{request.pipeline_id}' indexing failed: {e}")
            
            # Determine overall status
            successful_stages = sum(1 for stage in pipeline_stages.values() if stage.get("status") == "success")
            total_stages = len(pipeline_stages)
            
            if successful_stages == total_stages and total_stages > 0:
                overall_status = "success"
            elif successful_stages > 0:
                overall_status = "partial"
            else:
                overall_status = "failed"
            
            total_time = time.time() - start_time
            
            # Compile results
            result = ProcessingResult(
                pipeline_id=request.pipeline_id,
                document_id=request.document_id,
                processing_stages=pipeline_stages,
                overall_status=overall_status,
                total_processing_time=total_time,
                warnings=warnings,
                errors=errors,
                extracted_text_length=len(extraction_result.extracted_text) if extraction_result else 0,
                chunks_created=len(extraction_result.chunks) if extraction_result else 0,
                embeddings_generated=embeddings_result.successful_embeddings if embeddings_result else 0,
                chunks_indexed=indexing_result.indexed_chunks if indexing_result else 0
            )
            
            logger.info(f"Processing pipeline '{request.pipeline_id}' completed: "
                       f"{overall_status} status in {total_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Processing pipeline '{request.pipeline_id}' failed: {e}")
            raise
        finally:
            # Clean up pipeline tracking
            self.active_pipelines.pop(request.pipeline_id, None)
    
    async def _process_batch_documents(self, request: BatchProcessingRequest) -> BatchProcessingResult:
        """Process batch of documents through processing pipelines."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting batch processing '{request.batch_id}' with {len(request.documents)} documents")
            
            document_results = []
            successful = 0
            failed = 0
            batch_errors = []
            
            # Process documents with concurrency limit
            semaphore = asyncio.Semaphore(request.max_concurrent_documents)
            
            async def process_single_document(doc_request: ProcessingPipelineRequest) -> Optional[ProcessingResult]:
                async with semaphore:
                    try:
                        result = await self._process_document_pipeline(doc_request)
                        return result
                    except Exception as e:
                        if request.fail_fast:
                            raise
                        batch_errors.append(f"Document '{doc_request.document_id}': {str(e)}")
                        return None
            
            # Execute all document processing tasks
            tasks = [process_single_document(doc_req) for doc_req in request.documents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for doc_request, result in zip(request.documents, results):
                if isinstance(result, Exception):
                    batch_errors.append(f"Document '{doc_request.document_id}': {str(result)}")
                    failed += 1
                elif result is None:
                    failed += 1
                else:
                    document_results.append(result)
                    if result.overall_status == "success":
                        successful += 1
                    else:
                        failed += 1
            
            total_time = time.time() - start_time
            
            batch_result = BatchProcessingResult(
                batch_id=request.batch_id,
                successful_documents=successful,
                failed_documents=failed,
                total_processing_time=total_time,
                document_results=document_results,
                batch_errors=batch_errors
            )
            
            logger.info(f"Batch processing '{request.batch_id}' completed: "
                       f"{successful}/{len(request.documents)} successful in {total_time:.3f}s")
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Batch processing '{request.batch_id}' failed: {e}")
            raise
    
    async def _analyze_workflow(self, request: WorkflowAnalysisRequest) -> WorkflowAnalysisResult:
        """Analyze workflow performance and provide recommendations."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting workflow analysis '{request.analysis_id}' for {len(request.document_ids)} documents")
            
            analysis_results = {
                "document_count": len(request.document_ids),
                "analysis_type": request.analysis_type,
                "metrics": {},
                "insights": [],
                "performance_stats": {}
            }
            
            recommendations = []
            
            # Placeholder for workflow analysis logic
            # This would analyze document processing patterns, performance metrics, etc.
            
            if request.analysis_type in ["full", "content_only"]:
                # Content analysis
                analysis_results["metrics"]["content_analysis"] = {
                    "documents_analyzed": len(request.document_ids),
                    "average_processing_time": 0.0,
                    "success_rate": 0.0
                }
                recommendations.append("Consider batch processing for improved throughput")
            
            if request.analysis_type in ["full", "compliance"]:
                # Compliance analysis  
                analysis_results["metrics"]["compliance_analysis"] = {
                    "compliance_score": 95.0,
                    "compliance_issues": []
                }
                recommendations.append("Implement automated compliance validation")
            
            if request.analysis_type in ["full", "similarity"]:
                # Similarity analysis
                analysis_results["metrics"]["similarity_analysis"] = {
                    "duplicate_detection": [],
                    "content_clusters": []
                }
                recommendations.append("Enable deduplication for similar documents")
            
            processing_time = time.time() - start_time
            
            result = WorkflowAnalysisResult(
                analysis_id=request.analysis_id,
                documents_analyzed=len(request.document_ids),
                analysis_results=analysis_results,
                recommendations=recommendations,
                processing_time=processing_time
            )
            
            logger.info(f"Workflow analysis '{request.analysis_id}' completed in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow analysis '{request.analysis_id}' failed: {e}")
            raise
    
    async def _wait_for_worker_task(self, worker: BaseWorker, task_id: str, timeout: float = 300.0) -> Any:
        """Wait for a worker task to complete and return the result."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task_status = worker.get_task_status(task_id)
            
            if task_status:
                if task_status.get("error_message"):
                    raise RuntimeError(f"Worker task failed: {task_status['error_message']}")
                
                # Check if task is completed
                if task_status.get("completed_at"):
                    # Find the completed task to get the result
                    for completed_task in worker.completed_tasks:
                        if completed_task.task_id == task_id:
                            return completed_task.result
                    
                    raise RuntimeError("Task completed but result not found")
            
            # Wait before checking again
            await asyncio.sleep(0.5)
        
        raise asyncio.TimeoutError(f"Worker task {task_id} timed out after {timeout}s")
    
    async def stop(self, timeout: float = 60.0) -> None:
        """Stop the processing worker and all specialized workers."""
        logger.info(f"Stopping Processing Worker '{self.worker_name}' and specialized workers...")
        
        # Stop specialized workers
        stop_tasks = []
        if self.extraction_worker:
            stop_tasks.append(self.extraction_worker.stop(timeout=30.0))
        if self.embedding_worker:
            stop_tasks.append(self.embedding_worker.stop(timeout=30.0))
        if self.indexing_worker:
            stop_tasks.append(self.indexing_worker.stop(timeout=30.0))
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Stop self
        await super().stop(timeout)
        
        logger.info(f"Processing Worker '{self.worker_name}' and all specialized workers stopped")
    
    # Task submission convenience methods
    async def process_document_pipeline(self,
                                      pipeline_id: str,
                                      document_id: str,
                                      document_source: Union[str, bytes],
                                      document_type: str = "pdf",
                                      title: str = "",
                                      source: str = "",
                                      metadata: Dict[str, Any] = None,
                                      processing_options: Dict[str, Any] = None,
                                      priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit document processing pipeline task.
        
        Returns:
            Task ID for tracking
        """
        request = ProcessingPipelineRequest(
            pipeline_id=pipeline_id,
            document_id=document_id,
            document_source=document_source,
            document_type=document_type,
            title=title,
            source=source,
            metadata=metadata or {},
            processing_options=processing_options or {}
        )
        
        task = await self.submit_task(
            task_type="process_document_pipeline",
            task_input=request,
            priority=priority
        )
        
        return task.task_id
    
    async def batch_process_documents(self,
                                    batch_id: str,
                                    documents: List[ProcessingPipelineRequest],
                                    max_concurrent_documents: int = 5,
                                    priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit batch document processing task.
        
        Returns:
            Task ID for tracking
        """
        request = BatchProcessingRequest(
            batch_id=batch_id,
            documents=documents,
            max_concurrent_documents=max_concurrent_documents
        )
        
        task = await self.submit_task(
            task_type="batch_process_documents",
            task_input=request,
            priority=priority
        )
        
        return task.task_id