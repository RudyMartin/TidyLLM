"""
Indexing Worker - Document Indexing and Vector Storage Agent
===========================================================

Dedicated worker for document indexing and vector storage operations.
Extracts functionality from existing vector/search components and organizes
it as a scalable agent worker.

Capabilities:
- Document indexing with vector embeddings
- Chunk-based vector storage
- Semantic similarity search
- Vector database management
- Index optimization and maintenance
- Batch indexing operations
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import time
import hashlib

from .base_worker import BaseWorker, TaskPriority
from ..session.unified import UnifiedSessionManager

logger = logging.getLogger("indexing_worker")


@dataclass
class IndexingRequest:
    """Request for document indexing."""
    document_id: str
    title: str
    content: str
    chunks: List[Dict[str, Any]]  # Pre-chunked content with embeddings
    source: str = ""
    doc_type: str = "text"
    metadata: Dict[str, Any] = None
    overwrite_existing: bool = False
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchRequest:
    """Request for vector similarity search."""
    query_id: str
    query_text: str
    query_embedding: List[float]
    max_results: int = 10
    similarity_threshold: float = 0.7
    filter_metadata: Dict[str, Any] = None
    search_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.filter_metadata is None:
            self.filter_metadata = {}
        if self.search_metadata is None:
            self.search_metadata = {}


@dataclass
class BatchIndexingRequest:
    """Request for batch document indexing."""
    batch_id: str
    documents: List[IndexingRequest]
    batch_size: int = 10
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class IndexingResult:
    """Result from document indexing."""
    document_id: str
    indexed_chunks: int
    total_chunks: int
    index_time: Optional[float] = None
    index_status: str = "success"  # success, partial, failed
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "indexed_chunks": self.indexed_chunks,
            "total_chunks": self.total_chunks,
            "index_time": self.index_time,
            "index_status": self.index_status,
            "warnings": self.warnings,
            "metadata": self.metadata
        }


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    query_id: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "results": self.results,
            "total_results": self.total_results,
            "search_time": self.search_time,
            "metadata": self.metadata
        }


@dataclass
class BatchIndexingResult:
    """Result from batch document indexing."""
    batch_id: str
    successful_documents: int
    failed_documents: int
    total_chunks_indexed: int
    batch_time: Optional[float] = None
    document_results: List[IndexingResult] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.document_results is None:
            self.document_results = []
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "total_chunks_indexed": self.total_chunks_indexed,
            "batch_time": self.batch_time,
            "document_results": [result.to_dict() for result in self.document_results],
            "errors": self.errors
        }


class IndexingWorker(BaseWorker[Union[IndexingRequest, SearchRequest, BatchIndexingRequest],
                               Union[IndexingResult, SearchResult, BatchIndexingResult]]):
    """
    Worker for document indexing and vector search operations.
    
    Integrates existing vector capabilities:
    - VectorManager for PostgreSQL vector storage
    - VectorConfig for standardized configuration
    - Enhanced semantic search algorithms
    - Batch processing optimization
    
    Task Types:
    - index_document: Index single document with chunks
    - search_vectors: Perform similarity search
    - batch_index_documents: Index multiple documents efficiently
    - delete_document: Remove document from index
    - optimize_index: Optimize vector index performance
    """
    
    def __init__(self,
                 worker_name: str = "indexing_worker",
                 vector_dimension: int = 1024,
                 batch_size: int = 20,
                 **kwargs):
        """
        Initialize Indexing Worker.
        
        Args:
            worker_name: Worker identifier
            vector_dimension: Standard vector dimension
            batch_size: Maximum batch size for batch operations
        """
        super().__init__(worker_name, **kwargs)
        
        self.vector_dimension = vector_dimension
        self.batch_size = batch_size
        
        # Vector storage backends
        self.vector_manager = None
        self.session_manager = None
        
        # Database connections
        self.db_connection = None
        self.vector_config = None
        
        logger.info(f"Indexing Worker '{worker_name}' configured for {vector_dimension}d vectors")
    
    async def _initialize_worker(self) -> None:
        """Initialize vector storage backends."""
        try:
            # Initialize UnifiedSessionManager
            try:
                self.session_manager = UnifiedSessionManager()
                logger.info("Indexing Worker: UnifiedSessionManager initialized")
            except Exception as e:
                logger.warning(f"Indexing Worker: UnifiedSessionManager not available: {e}")
            
            # Initialize VectorManager
            try:
                from ...knowledge_systems.core.vector_manager import VectorManager, VectorConfig
                self.vector_config = VectorConfig(vector_dimension=self.vector_dimension)
                self.vector_manager = VectorManager(config=self.vector_config)
                logger.info("Indexing Worker: VectorManager initialized")
            except ImportError as e:
                logger.warning(f"Indexing Worker: VectorManager not available: {e}")
            
            # Initialize database connection
            await self._initialize_database()
            
            if not (self.vector_manager or self.db_connection):
                raise RuntimeError("No vector storage backends available")
                
        except Exception as e:
            logger.error(f"Indexing Worker initialization failed: {e}")
            raise
    
    async def _initialize_database(self) -> None:
        """Initialize direct database connection for vector operations."""
        try:
            if self.session_manager:
                # Use UnifiedSessionManager for DB connection
                db_session = self.session_manager.get_db_session()
                if hasattr(db_session, 'connection'):
                    self.db_connection = db_session.connection()
                else:
                    self.db_connection = db_session
                logger.info("Indexing Worker: Database connection via UnifiedSessionManager")
            else:
                # Fallback to direct connection
                import psycopg2
                from psycopg2.extras import RealDictCursor
                
                connection_params = {
                    "host": self.vector_config.host if self.vector_config else "localhost",
                    "port": self.vector_config.port if self.vector_config else 5432,
                    "database": self.vector_config.database if self.vector_config else "vectorqa",
                    "user": self.vector_config.user if self.vector_config else "vectorqa_user",
                    "password": self.vector_config.password if self.vector_config else "password"
                }
                
                self.db_connection = psycopg2.connect(**connection_params)
                logger.info("Indexing Worker: Direct database connection established")
                
        except Exception as e:
            logger.warning(f"Indexing Worker: Database connection failed: {e}")
    
    def validate_input(self, task_input: Any) -> bool:
        """Validate indexing request input."""
        if isinstance(task_input, IndexingRequest):
            return bool(task_input.document_id and task_input.content and task_input.chunks)
        elif isinstance(task_input, SearchRequest):
            return bool(task_input.query_id and (task_input.query_text or task_input.query_embedding))
        elif isinstance(task_input, BatchIndexingRequest):
            return bool(task_input.batch_id and task_input.documents)
        return False
    
    async def process_task(self, task_input: Union[IndexingRequest, SearchRequest, BatchIndexingRequest]) -> Union[IndexingResult, SearchResult, BatchIndexingResult]:
        """Process indexing/search request."""
        if isinstance(task_input, IndexingRequest):
            return await self._process_document_indexing(task_input)
        elif isinstance(task_input, SearchRequest):
            return await self._process_vector_search(task_input)
        elif isinstance(task_input, BatchIndexingRequest):
            return await self._process_batch_indexing(task_input)
        else:
            raise ValueError(f"Unsupported task input type: {type(task_input)}")
    
    async def _process_document_indexing(self, request: IndexingRequest) -> IndexingResult:
        """Process single document indexing."""
        start_time = time.time()
        
        try:
            logger.info(f"Indexing document '{request.document_id}' with {len(request.chunks)} chunks")
            
            indexed_chunks = 0
            warnings = []
            
            # Check if document exists and handle overwrite
            if not request.overwrite_existing:
                existing = await self._check_document_exists(request.document_id)
                if existing:
                    warnings.append(f"Document '{request.document_id}' already exists, skipping")
                    return IndexingResult(
                        document_id=request.document_id,
                        indexed_chunks=0,
                        total_chunks=len(request.chunks),
                        index_status="skipped",
                        warnings=warnings
                    )
            else:
                # Remove existing document
                await self._delete_document_index(request.document_id)
            
            # Index document metadata
            await self._index_document_metadata(request)
            
            # Index chunks with embeddings
            for i, chunk in enumerate(request.chunks):
                try:
                    await self._index_chunk(
                        document_id=request.document_id,
                        chunk_data=chunk,
                        chunk_index=i
                    )
                    indexed_chunks += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to index chunk {i} for document '{request.document_id}': {e}")
                    warnings.append(f"Chunk {i} indexing failed: {str(e)}")
            
            index_time = time.time() - start_time
            
            # Determine status
            if indexed_chunks == len(request.chunks):
                status = "success"
            elif indexed_chunks > 0:
                status = "partial"
            else:
                status = "failed"
            
            result = IndexingResult(
                document_id=request.document_id,
                indexed_chunks=indexed_chunks,
                total_chunks=len(request.chunks),
                index_time=index_time,
                index_status=status,
                warnings=warnings,
                metadata={"source": request.source, "doc_type": request.doc_type}
            )
            
            logger.info(f"Document '{request.document_id}' indexed: "
                       f"{indexed_chunks}/{len(request.chunks)} chunks in {index_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Document indexing failed for '{request.document_id}': {e}")
            raise
    
    async def _process_vector_search(self, request: SearchRequest) -> SearchResult:
        """Process vector similarity search."""
        start_time = time.time()
        
        try:
            logger.info(f"Searching for query '{request.query_id}' with threshold {request.similarity_threshold}")
            
            # Perform vector search
            results = await self._search_similar_vectors(
                query_embedding=request.query_embedding,
                max_results=request.max_results,
                similarity_threshold=request.similarity_threshold,
                filter_metadata=request.filter_metadata
            )
            
            search_time = time.time() - start_time
            
            result = SearchResult(
                query_id=request.query_id,
                results=results,
                total_results=len(results),
                search_time=search_time,
                metadata=request.search_metadata
            )
            
            logger.info(f"Search '{request.query_id}' completed: "
                       f"{len(results)} results in {search_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Vector search failed for query '{request.query_id}': {e}")
            raise
    
    async def _process_batch_indexing(self, request: BatchIndexingRequest) -> BatchIndexingResult:
        """Process batch document indexing."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing batch '{request.batch_id}' with {len(request.documents)} documents")
            
            document_results = []
            successful = 0
            failed = 0
            total_chunks = 0
            errors = []
            
            # Process documents in batches
            for i in range(0, len(request.documents), self.batch_size):
                batch_docs = request.documents[i:i + self.batch_size]
                
                # Process batch concurrently
                batch_tasks = [
                    self._process_document_indexing(doc_request)
                    for doc_request in batch_docs
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for doc_request, result in zip(batch_docs, batch_results):
                    if isinstance(result, Exception):
                        errors.append(f"Document '{doc_request.document_id}': {str(result)}")
                        failed += 1
                    else:
                        document_results.append(result)
                        if result.index_status == "success":
                            successful += 1
                        else:
                            failed += 1
                        total_chunks += result.indexed_chunks
            
            batch_time = time.time() - start_time
            
            result = BatchIndexingResult(
                batch_id=request.batch_id,
                successful_documents=successful,
                failed_documents=failed,
                total_chunks_indexed=total_chunks,
                batch_time=batch_time,
                document_results=document_results,
                errors=errors
            )
            
            logger.info(f"Batch '{request.batch_id}' completed: "
                       f"{successful}/{len(request.documents)} documents, "
                       f"{total_chunks} chunks in {batch_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch indexing failed for '{request.batch_id}': {e}")
            raise
    
    async def _check_document_exists(self, document_id: str) -> bool:
        """Check if document already exists in index."""
        if self.vector_manager:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self.vector_manager.document_exists,
                document_id
            )
        
        # Fallback to direct DB query
        if self.db_connection:
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM documents WHERE id = %s LIMIT 1", (document_id,))
                    return cursor.fetchone() is not None
            except Exception as e:
                logger.error(f"Document existence check failed: {e}")
        
        return False
    
    async def _index_document_metadata(self, request: IndexingRequest) -> None:
        """Index document metadata."""
        if self.vector_manager:
            document_data = {
                "id": request.document_id,
                "title": request.title,
                "content": request.content,
                "source": request.source,
                "doc_type": request.doc_type,
                "metadata": request.metadata
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vector_manager.add_document,
                document_data
            )
        else:
            # Fallback to direct DB insert
            await self._insert_document_direct(request)
    
    async def _insert_document_direct(self, request: IndexingRequest) -> None:
        """Insert document directly into database."""
        if not self.db_connection:
            return
        
        try:
            import json
            from datetime import datetime
            
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO documents (id, title, content, source, doc_type, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        doc_type = EXCLUDED.doc_type,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    request.document_id,
                    request.title,
                    request.content,
                    request.source,
                    request.doc_type,
                    json.dumps(request.metadata),
                    datetime.now()
                ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Direct document insert failed: {e}")
            self.db_connection.rollback()
            raise
    
    async def _index_chunk(self, document_id: str, chunk_data: Dict[str, Any], chunk_index: int) -> None:
        """Index a document chunk with embedding."""
        if self.vector_manager:
            # Use VectorManager
            chunk_info = {
                "document_id": document_id,
                "content": chunk_data.get("text", ""),
                "chunk_index": chunk_index,
                "start_char": chunk_data.get("start_offset", 0),
                "end_char": chunk_data.get("end_offset", 0),
                "embedding": chunk_data.get("embedding", []),
                "metadata": chunk_data.get("metadata", {})
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vector_manager.add_chunk,
                chunk_info
            )
        else:
            # Fallback to direct DB insert
            await self._insert_chunk_direct(document_id, chunk_data, chunk_index)
    
    async def _insert_chunk_direct(self, document_id: str, chunk_data: Dict[str, Any], chunk_index: int) -> None:
        """Insert chunk directly into database."""
        if not self.db_connection:
            return
        
        try:
            import json
            
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            embedding = chunk_data.get("embedding", [])
            
            # Convert embedding to PostgreSQL array format
            embedding_array = "{" + ",".join(map(str, embedding)) + "}"
            
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO document_chunks (id, document_id, content, chunk_index, start_char, end_char, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::float[], %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        chunk_index = EXCLUDED.chunk_index,
                        start_char = EXCLUDED.start_char,
                        end_char = EXCLUDED.end_char,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """, (
                    chunk_id,
                    document_id,
                    chunk_data.get("text", ""),
                    chunk_index,
                    chunk_data.get("start_offset", 0),
                    chunk_data.get("end_offset", 0),
                    embedding_array,
                    json.dumps(chunk_data.get("metadata", {}))
                ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Direct chunk insert failed: {e}")
            self.db_connection.rollback()
            raise
    
    async def _search_similar_vectors(self, 
                                    query_embedding: List[float],
                                    max_results: int = 10,
                                    similarity_threshold: float = 0.7,
                                    filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.vector_manager:
            # Use VectorManager
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.vector_manager.search_similar,
                query_embedding,
                max_results,
                similarity_threshold
            )
            
            return [result.__dict__ if hasattr(result, '__dict__') else result for result in results]
        
        # Fallback to direct DB search
        return await self._search_vectors_direct(query_embedding, max_results, similarity_threshold, filter_metadata)
    
    async def _search_vectors_direct(self, 
                                   query_embedding: List[float],
                                   max_results: int,
                                   similarity_threshold: float,
                                   filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search vectors directly in database."""
        if not self.db_connection:
            return []
        
        try:
            # Convert query embedding to PostgreSQL array format
            embedding_array = "{" + ",".join(map(str, query_embedding)) + "}"
            
            with self.db_connection.cursor() as cursor:
                # Use cosine similarity for vector search
                query = """
                    SELECT 
                        dc.id as chunk_id,
                        dc.document_id,
                        dc.content,
                        d.title,
                        d.source,
                        (1 - (dc.embedding <=> %s::float[])) as similarity_score,
                        dc.metadata,
                        d.metadata as doc_metadata
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE (1 - (dc.embedding <=> %s::float[])) >= %s
                    ORDER BY similarity_score DESC
                    LIMIT %s
                """
                
                cursor.execute(query, (embedding_array, embedding_array, similarity_threshold, max_results))
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        "chunk_id": row[0],
                        "document_id": row[1],
                        "content": row[2],
                        "title": row[3],
                        "source": row[4],
                        "score": float(row[5]),
                        "metadata": row[6],
                        "doc_metadata": row[7]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Direct vector search failed: {e}")
            return []
    
    async def _delete_document_index(self, document_id: str) -> None:
        """Delete document from index."""
        if self.vector_manager:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vector_manager.delete_document,
                document_id
            )
        elif self.db_connection:
            try:
                with self.db_connection.cursor() as cursor:
                    # Delete chunks first (foreign key constraint)
                    cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                    # Delete document
                    cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
                
                self.db_connection.commit()
                
            except Exception as e:
                logger.error(f"Document deletion failed: {e}")
                self.db_connection.rollback()
    
    # Task submission convenience methods
    async def index_document(self,
                           document_id: str,
                           title: str,
                           content: str,
                           chunks: List[Dict[str, Any]],
                           source: str = "",
                           doc_type: str = "text",
                           metadata: Dict[str, Any] = None,
                           overwrite_existing: bool = False,
                           priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit document indexing task.
        
        Returns:
            Task ID for tracking
        """
        request = IndexingRequest(
            document_id=document_id,
            title=title,
            content=content,
            chunks=chunks,
            source=source,
            doc_type=doc_type,
            metadata=metadata or {},
            overwrite_existing=overwrite_existing
        )
        
        task = await self.submit_task(
            task_type="index_document",
            task_input=request,
            priority=priority
        )
        
        return task.task_id
    
    async def search_vectors(self,
                           query_id: str,
                           query_embedding: List[float],
                           max_results: int = 10,
                           similarity_threshold: float = 0.7,
                           filter_metadata: Dict[str, Any] = None,
                           priority: TaskPriority = TaskPriority.HIGH) -> str:
        """
        Submit vector search task.
        
        Returns:
            Task ID for tracking
        """
        request = SearchRequest(
            query_id=query_id,
            query_text="",  # Not needed for pure vector search
            query_embedding=query_embedding,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            filter_metadata=filter_metadata or {}
        )
        
        task = await self.submit_task(
            task_type="search_vectors",
            task_input=request,
            priority=priority
        )
        
        return task.task_id
    
    async def batch_index_documents(self,
                                  batch_id: str,
                                  documents: List[IndexingRequest],
                                  batch_size: int = None,
                                  priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit batch indexing task.
        
        Returns:
            Task ID for tracking
        """
        request = BatchIndexingRequest(
            batch_id=batch_id,
            documents=documents,
            batch_size=batch_size or self.batch_size
        )
        
        task = await self.submit_task(
            task_type="batch_index_documents",
            task_input=request,
            priority=priority
        )
        
        return task.task_id