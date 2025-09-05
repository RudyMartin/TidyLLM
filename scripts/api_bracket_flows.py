"""
TidyLLM API Chain Endpoints Solution
====================================

Extends existing REST API to expose the 7 core document chain operations.
Builds on existing api_server.py but adds dedicated chain contract endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import asyncio

# Import existing document chains
from tidyllm.document_chains import (
    BackendDocumentPipeline, FrontendDocumentAPI,
    DocumentOperation, ChainExecutionMode, ChainContext
)
from tidyllm.gateways import get_global_registry

# API Models for Chain Operations
class DocumentIngestRequest(BaseModel):
    source: str = Field(..., description="Source path or S3 URI")
    domain: str = Field(..., description="Knowledge domain")
    bucket: Optional[str] = Field(None, description="Target S3 bucket")
    document_format: str = Field("auto", description="Document format")
    batch_size: int = Field(10, description="Batch processing size")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DocumentEmbedRequest(BaseModel):
    domain: str = Field(..., description="Knowledge domain")
    model: str = Field("tfidf", description="Embedding model (tidyllm-sentence)")
    target_dimension: int = Field(1024, description="Target embedding dimension")
    bucket: Optional[str] = Field(None, description="S3 bucket for embeddings")
    parallel_workers: int = Field(3, description="Parallel processing workers")

class DocumentIndexRequest(BaseModel):
    domain: str = Field(..., description="Knowledge domain")
    vector_store_uri: Optional[str] = Field(None, description="S3 URI for vector storage")
    index_type: str = Field("simple", description="Index type")
    cluster_count: Optional[int] = Field(None, description="Clusters for tlm.kmeans")

class DocumentQueryRequest(BaseModel):
    domain: str = Field(..., description="Knowledge domain")
    question: str = Field(..., description="Natural language question")
    limit: int = Field(5, description="Number of results")
    similarity_threshold: float = Field(0.7, description="Similarity threshold")

class DocumentSearchRequest(BaseModel):
    domain: str = Field(..., description="Knowledge domain")
    keywords: str = Field(..., description="Search keywords")
    limit: int = Field(10, description="Number of results")

class ChainExecutionRequest(BaseModel):
    operations: List[str] = Field(..., description="Chain operations to execute")
    domain: str = Field(..., description="Knowledge domain")
    source: Optional[str] = Field(None, description="Source for ingest")
    question: Optional[str] = Field(None, description="Question for query")
    execution_mode: str = Field("auto", description="Execution mode")
    config: Optional[Dict[str, Any]] = Field(None, description="Operation config")

class ChainOperationResponse(BaseModel):
    operation_id: str = Field(..., description="Unique operation ID")
    status: str = Field(..., description="Operation status")
    operation_type: str = Field(..., description="Type of operation")
    domain: str = Field(..., description="Knowledge domain")
    started_at: datetime = Field(..., description="Operation start time")
    completed_at: Optional[datetime] = Field(None, description="Operation completion time")
    results: Optional[Dict[str, Any]] = Field(None, description="Operation results")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operation metadata")

class QueryResponse(BaseModel):
    query_id: str = Field(..., description="Unique query ID")
    question: str = Field(..., description="Original question")
    matches: List[Dict[str, Any]] = Field(..., description="Query matches")
    processing_time_ms: float = Field(..., description="Processing time")
    domain: str = Field(..., description="Knowledge domain queried")

class ChainStatusResponse(BaseModel):
    domain: Optional[str] = Field(None, description="Domain status")
    active_operations: int = Field(..., description="Number of active operations")
    total_documents: int = Field(..., description="Total documents processed")
    total_queries: int = Field(..., description="Total queries served")
    avg_response_time_ms: float = Field(..., description="Average response time")
    health_status: str = Field(..., description="Overall health status")
    last_updated: datetime = Field(..., description="Last status update")

# Chain API Router
class TidyLLMChainAPI:
    """API router for TidyLLM chain operations."""
    
    def __init__(self):
        self.router = APIRouter(prefix="/chains", tags=["Document Chains"])
        self.backend_pipeline = BackendDocumentPipeline()
        self.frontend_api = FrontendDocumentAPI()
        self.active_operations: Dict[str, Dict] = {}
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes for chain operations."""
        
        # Backend Layer Operations (Complex)
        @self.router.post("/ingest", response_model=ChainOperationResponse)
        async def ingest_documents(
            request: DocumentIngestRequest,
            background_tasks: BackgroundTasks,
            key_data: Dict = Depends(self._get_api_key)
        ):
            """Ingest documents with S3-first processing."""
            if "ingest" not in key_data["permissions"] and "write" not in key_data["permissions"]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            operation_id = str(uuid.uuid4())
            
            try:
                # Start background processing
                background_tasks.add_task(
                    self._execute_ingest,
                    operation_id,
                    request.dict()
                )
                
                # Record operation
                self.active_operations[operation_id] = {
                    "type": "ingest",
                    "status": "started",
                    "domain": request.domain,
                    "started_at": datetime.now(),
                    "request_data": request.dict()
                }
                
                return ChainOperationResponse(
                    operation_id=operation_id,
                    status="started",
                    operation_type="ingest",
                    domain=request.domain,
                    started_at=datetime.now(),
                    metadata={"source": request.source, "batch_size": request.batch_size}
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")
        
        @self.router.post("/embed", response_model=ChainOperationResponse)
        async def embed_documents(
            request: DocumentEmbedRequest,
            background_tasks: BackgroundTasks,
            key_data: Dict = Depends(self._get_api_key)
        ):
            """Generate embeddings using tidyllm-sentence."""
            if "embed" not in key_data["permissions"] and "write" not in key_data["permissions"]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            operation_id = str(uuid.uuid4())
            
            try:
                background_tasks.add_task(
                    self._execute_embed,
                    operation_id,
                    request.dict()
                )
                
                self.active_operations[operation_id] = {
                    "type": "embed",
                    "status": "started", 
                    "domain": request.domain,
                    "started_at": datetime.now(),
                    "request_data": request.dict()
                }
                
                return ChainOperationResponse(
                    operation_id=operation_id,
                    status="started",
                    operation_type="embed",
                    domain=request.domain,
                    started_at=datetime.now(),
                    metadata={"model": request.model, "target_dimension": request.target_dimension}
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Embed failed: {str(e)}")
        
        @self.router.post("/index", response_model=ChainOperationResponse)
        async def index_documents(
            request: DocumentIndexRequest,
            background_tasks: BackgroundTasks,
            key_data: Dict = Depends(self._get_api_key)
        ):
            """Create searchable indices using tlm."""
            if "index" not in key_data["permissions"] and "write" not in key_data["permissions"]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            operation_id = str(uuid.uuid4())
            
            try:
                background_tasks.add_task(
                    self._execute_index,
                    operation_id,
                    request.dict()
                )
                
                self.active_operations[operation_id] = {
                    "type": "index",
                    "status": "started",
                    "domain": request.domain,
                    "started_at": datetime.now(),
                    "request_data": request.dict()
                }
                
                return ChainOperationResponse(
                    operation_id=operation_id,
                    status="started",
                    operation_type="index",
                    domain=request.domain,
                    started_at=datetime.now(),
                    metadata={"index_type": request.index_type}
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Index failed: {str(e)}")
        
        # Frontend Layer Operations (Simple)
        @self.router.post("/query", response_model=QueryResponse)
        async def query_documents(
            request: DocumentQueryRequest,
            key_data: Dict = Depends(self._get_api_key)
        ):
            """Natural language query (simple interface)."""
            if "query" not in key_data["permissions"] and "read" not in key_data["permissions"]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                import time
                start_time = time.time()
                
                # Use frontend API for simple query
                results = await self._execute_query(request.dict())
                
                processing_time = (time.time() - start_time) * 1000
                
                return QueryResponse(
                    query_id=str(uuid.uuid4()),
                    question=request.question,
                    matches=results["matches"],
                    processing_time_ms=processing_time,
                    domain=request.domain
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
        
        @self.router.post("/search", response_model=QueryResponse)
        async def search_documents(
            request: DocumentSearchRequest,
            key_data: Dict = Depends(self._get_api_key)
        ):
            """Keyword search (simple interface)."""
            if "search" not in key_data["permissions"] and "read" not in key_data["permissions"]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                import time
                start_time = time.time()
                
                # Use frontend API for simple search
                results = await self._execute_search(request.dict())
                
                processing_time = (time.time() - start_time) * 1000
                
                return QueryResponse(
                    query_id=str(uuid.uuid4()),
                    question=f"Search: {request.keywords}",
                    matches=results["matches"],
                    processing_time_ms=processing_time,
                    domain=request.domain
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        # Chain Execution
        @self.router.post("/execute", response_model=List[ChainOperationResponse])
        async def execute_chain(
            request: ChainExecutionRequest,
            background_tasks: BackgroundTasks,
            key_data: Dict = Depends(self._get_api_key)
        ):
            """Execute chained operations."""
            if "chain" not in key_data["permissions"] and "write" not in key_data["permissions"]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                chain_id = str(uuid.uuid4())
                responses = []
                
                for operation in request.operations:
                    operation_id = f"{chain_id}_{operation}"
                    
                    # Queue each operation
                    background_tasks.add_task(
                        self._execute_chain_operation,
                        operation_id,
                        operation,
                        request.dict()
                    )
                    
                    # Record operation
                    self.active_operations[operation_id] = {
                        "type": operation,
                        "status": "queued",
                        "domain": request.domain,
                        "started_at": datetime.now(),
                        "chain_id": chain_id,
                        "request_data": request.dict()
                    }
                    
                    responses.append(ChainOperationResponse(
                        operation_id=operation_id,
                        status="queued",
                        operation_type=operation,
                        domain=request.domain,
                        started_at=datetime.now(),
                        metadata={"chain_id": chain_id, "execution_mode": request.execution_mode}
                    ))
                
                return responses
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Chain execution failed: {str(e)}")
        
        # Status and Monitoring
        @self.router.get("/status/{operation_id}", response_model=ChainOperationResponse)
        async def get_operation_status(
            operation_id: str,
            key_data: Dict = Depends(self._get_api_key)
        ):
            """Get status of specific operation."""
            if "read" not in key_data["permissions"]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            if operation_id not in self.active_operations:
                raise HTTPException(status_code=404, detail="Operation not found")
            
            op_data = self.active_operations[operation_id]
            
            return ChainOperationResponse(
                operation_id=operation_id,
                status=op_data["status"],
                operation_type=op_data["type"],
                domain=op_data["domain"],
                started_at=op_data["started_at"],
                completed_at=op_data.get("completed_at"),
                results=op_data.get("results"),
                errors=op_data.get("errors", []),
                metadata=op_data.get("metadata", {})
            )
        
        @self.router.get("/status", response_model=ChainStatusResponse)
        async def get_system_status(
            domain: Optional[str] = None,
            key_data: Dict = Depends(self._get_api_key)
        ):
            """Get overall system status."""
            if "read" not in key_data["permissions"]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                # Get status from workflow optimizer gateway
                registry = get_global_registry()
                optimizer = registry.get("workflow_optimizer")
                
                if optimizer:
                    status_data = await optimizer.get_domain_status(domain or "all")
                else:
                    # Fallback status
                    status_data = {
                        "active_operations": len(self.active_operations),
                        "total_documents": 0,
                        "total_queries": 0,
                        "avg_response_time_ms": 0.0,
                        "health_status": "unknown"
                    }
                
                return ChainStatusResponse(
                    domain=domain,
                    active_operations=status_data["active_operations"],
                    total_documents=status_data["total_documents"],
                    total_queries=status_data["total_queries"], 
                    avg_response_time_ms=status_data["avg_response_time_ms"],
                    health_status=status_data["health_status"],
                    last_updated=datetime.now()
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
    
    async def _execute_ingest(self, operation_id: str, request_data: Dict):
        """Execute ingest operation in background."""
        try:
            self.active_operations[operation_id]["status"] = "running"
            
            # Use backend pipeline for complex ingest
            result = await self.backend_pipeline.ingest_documents(**request_data)
            
            self.active_operations[operation_id].update({
                "status": "completed",
                "completed_at": datetime.now(),
                "results": result
            })
            
        except Exception as e:
            self.active_operations[operation_id].update({
                "status": "failed",
                "completed_at": datetime.now(),
                "errors": [str(e)]
            })
    
    async def _execute_query(self, request_data: Dict) -> Dict[str, Any]:
        """Execute query using frontend API."""
        return await self.frontend_api.query(**request_data)
    
    def _get_api_key(self):
        """Placeholder for API key validation (integrate with existing auth)."""
        # This would integrate with your existing APIKeyManager
        return {"permissions": ["read", "write", "ingest", "query", "search", "chain"]}


# Integration with existing API server
def add_chain_endpoints_to_existing_api(app):
    """Add chain endpoints to existing FastAPI application."""
    chain_api = TidyLLMChainAPI()
    app.include_router(chain_api.router)
    
    return chain_api


# Usage example:
"""
# In your existing api_server.py, add:

from API_CHAIN_ENDPOINTS_SOLUTION import add_chain_endpoints_to_existing_api

class TidyLLMAPIServer:
    def __init__(self, config: Dict[str, Any]):
        # ... existing init code ...
        
        # Add chain endpoints
        self.chain_api = add_chain_endpoints_to_existing_api(self.app)

# Now your API exposes:
# POST /chains/ingest
# POST /chains/embed  
# POST /chains/index
# POST /chains/query
# POST /chains/search
# POST /chains/execute
# GET /chains/status
# GET /chains/status/{operation_id}
"""