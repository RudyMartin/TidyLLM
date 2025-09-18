# AI-Assisted Manager API Endpoints
# CORE ENTERPRISE API - Manager Integration & External Triggers
#
# This component provides:
# - REST API endpoints for external systems to trigger AI Manager
# - Status monitoring and health checks for manager operations
# - Integration with MCP server and existing gateways
# - Real-time processing status and result retrieval
#
# Dependencies:
# - AI-Assisted Manager for orchestration
# - FastAPI for REST endpoints  
# - UnifiedSessionManager for data persistence
# - Existing gateway integrations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback if FastAPI not available
    logging.warning("FastAPI not available - API endpoints will be limited")
    FastAPI = None

from ..workers.ai_dropzone_manager import AIDropzoneManager, AIManagerTask, ProcessingStrategy, DocumentComplexity
from ..session.unified import UnifiedSessionManager

logger = logging.getLogger(__name__)

# Request/Response Models
class ProcessDocumentRequest(BaseModel):
    document_path: str = Field(..., description="Path to document for processing")
    business_priority: str = Field("normal", description="Processing priority: critical, high, normal, low")
    deadline: Optional[datetime] = Field(None, description="Optional deadline for processing completion")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Additional context from requesting user/system")
    preferred_templates: Optional[List[str]] = Field(None, description="Suggested templates to consider")
    processing_strategy: Optional[str] = Field(None, description="Override processing strategy")

class ProcessingStatusResponse(BaseModel):
    processing_id: str
    status: str  # queued, processing, completed, failed
    estimated_completion: Optional[datetime]
    progress_percentage: int
    current_stage: str
    assigned_workers: List[str]
    quality_metrics: Dict[str, float]

class ManagerStatusResponse(BaseModel):
    manager_status: str  # active, busy, maintenance, error
    active_processings: int
    queue_length: int
    worker_pools: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, float]
    system_resources: Dict[str, float]

class ProcessingResultResponse(BaseModel):
    processing_id: str
    document_path: str
    processing_decision: Dict[str, Any]
    synthesis_result: Optional[str]
    final_recommendations: List[str]
    quality_assessment: Dict[str, float]
    processing_time_minutes: float
    templates_used: List[str]

# AI Manager API Integration
class ManagerAPI:
    """
    API Integration for AI-Assisted Manager with external system triggers.
    """
    
    def __init__(self, manager: Optional[AIDropzoneManager] = None):
        self.manager = manager or AIDropzoneManager()
        self.active_processings: Dict[str, Dict[str, Any]] = {}
        self.processing_results: Dict[str, Any] = {}
        self.session_manager = None
        
        # Initialize FastAPI if available
        if FastAPI:
            self.app = FastAPI(
                title="TidyLLM AI-Assisted Manager API",
                description="Intelligent document processing orchestration API",
                version="1.0.0"
            )
            self._setup_routes()
        else:
            self.app = None
    
    async def initialize(self):
        """Initialize the Manager API."""
        await self.manager.initialize()
        
        try:
            self.session_manager = UnifiedSessionManager()
            await self.session_manager.initialize()
        except Exception as e:
            logger.warning(f"Session manager not available: {e}")
    
    def _setup_routes(self):
        """Setup FastAPI routes for manager integration."""
        if not self.app:
            return
        
        # Document processing endpoints
        self.app.post("/api/v1/process", response_model=ProcessingStatusResponse)(self.process_document)
        self.app.get("/api/v1/process/{processing_id}", response_model=ProcessingStatusResponse)(self.get_processing_status)
        self.app.get("/api/v1/process/{processing_id}/result", response_model=ProcessingResultResponse)(self.get_processing_result)
        self.app.delete("/api/v1/process/{processing_id}")(self.cancel_processing)
        
        # Manager status endpoints
        self.app.get("/api/v1/manager/status", response_model=ManagerStatusResponse)(self.get_manager_status)
        self.app.get("/api/v1/manager/health")(self.health_check)
        self.app.post("/api/v1/manager/feedback")(self.submit_quality_feedback)
        
        # Queue management endpoints
        self.app.get("/api/v1/queue")(self.get_queue_status)
        self.app.post("/api/v1/queue/priority")(self.adjust_queue_priority)
        
        # Template and worker management
        self.app.get("/api/v1/templates")(self.list_available_templates)
        self.app.get("/api/v1/workers")(self.get_worker_status)
        
        # Streaming status endpoint
        self.app.get("/api/v1/process/{processing_id}/stream")(self.stream_processing_status)
    
    async def process_document(self, request: ProcessDocumentRequest, background_tasks: BackgroundTasks) -> ProcessingStatusResponse:
        """
        Main endpoint for processing document requests.
        """
        try:
            import uuid
            processing_id = str(uuid.uuid4())
            
            # Validate document path
            if not Path(request.document_path).exists():
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Create AI Manager task
            task = AIManagerTask(
                document_path=request.document_path,
                user_context=request.user_context,
                business_priority=request.business_priority,
                deadline=request.deadline
            )
            
            # Override processing strategy if specified
            if request.processing_strategy:
                try:
                    strategy = ProcessingStrategy(request.processing_strategy)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid processing strategy")
            
            # Track processing
            self.active_processings[processing_id] = {
                "task": task,
                "status": "queued",
                "started_at": datetime.now(),
                "estimated_completion": datetime.now() + timedelta(minutes=15),
                "progress": 0,
                "current_stage": "queued",
                "assigned_workers": []
            }
            
            # Schedule background processing
            background_tasks.add_task(self._process_document_background, processing_id, task)
            
            return ProcessingStatusResponse(
                processing_id=processing_id,
                status="queued",
                estimated_completion=self.active_processings[processing_id]["estimated_completion"],
                progress_percentage=0,
                current_stage="queued",
                assigned_workers=[],
                quality_metrics={}
            )
            
        except Exception as e:
            logger.error(f"Failed to queue document processing: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_document_background(self, processing_id: str, task: AIManagerTask):
        """
        Background task for document processing.
        """
        try:
            # Update status to processing
            self.active_processings[processing_id].update({
                "status": "processing",
                "current_stage": "document_analysis",
                "progress": 10
            })
            
            # Execute AI Manager processing
            result = await self.manager.process_task(task)
            
            # Update progress through stages
            stages = ["document_analysis", "template_selection", "worker_allocation", "processing", "synthesis", "quality_check"]
            for i, stage in enumerate(stages):
                self.active_processings[processing_id].update({
                    "current_stage": stage,
                    "progress": int((i + 1) / len(stages) * 90)  # Reserve 10% for completion
                })
                await asyncio.sleep(1)  # Simulate processing time
            
            if result.success:
                # Store results
                self.processing_results[processing_id] = {
                    "processing_decision": result.processing_decision.__dict__ if result.processing_decision else {},
                    "worker_assignments": result.worker_assignments,
                    "quality_metrics": result.quality_metrics,
                    "estimated_completion": result.estimated_completion,
                    "completed_at": datetime.now()
                }
                
                # Update final status
                self.active_processings[processing_id].update({
                    "status": "completed",
                    "current_stage": "completed",
                    "progress": 100,
                    "assigned_workers": [w["worker_id"] for w in result.worker_assignments]
                })
            else:
                # Handle failure
                self.active_processings[processing_id].update({
                    "status": "failed",
                    "current_stage": "error",
                    "progress": 0,
                    "error": getattr(result, 'error', 'Unknown error')
                })
                
        except Exception as e:
            logger.error(f"Background processing failed for {processing_id}: {e}")
            self.active_processings[processing_id].update({
                "status": "failed",
                "current_stage": "error",
                "progress": 0,
                "error": str(e)
            })
    
    async def get_processing_status(self, processing_id: str) -> ProcessingStatusResponse:
        """
        Get current status of a processing request.
        """
        if processing_id not in self.active_processings:
            raise HTTPException(status_code=404, detail="Processing ID not found")
        
        processing = self.active_processings[processing_id]
        
        return ProcessingStatusResponse(
            processing_id=processing_id,
            status=processing["status"],
            estimated_completion=processing.get("estimated_completion"),
            progress_percentage=processing.get("progress", 0),
            current_stage=processing.get("current_stage", "unknown"),
            assigned_workers=processing.get("assigned_workers", []),
            quality_metrics=processing.get("quality_metrics", {})
        )
    
    async def get_processing_result(self, processing_id: str) -> ProcessingResultResponse:
        """
        Get final results of a completed processing request.
        """
        if processing_id not in self.processing_results:
            if processing_id in self.active_processings:
                status = self.active_processings[processing_id]["status"]
                if status == "processing":
                    raise HTTPException(status_code=202, detail="Processing still in progress")
                elif status == "failed":
                    raise HTTPException(status_code=500, detail="Processing failed")
            
            raise HTTPException(status_code=404, detail="Results not found")
        
        result = self.processing_results[processing_id]
        processing = self.active_processings[processing_id]
        
        # Calculate processing time
        started_at = processing.get("started_at", datetime.now())
        completed_at = result.get("completed_at", datetime.now())
        processing_time = (completed_at - started_at).total_seconds() / 60.0
        
        return ProcessingResultResponse(
            processing_id=processing_id,
            document_path=processing["task"].document_path,
            processing_decision=result["processing_decision"],
            synthesis_result=None,  # Would come from coordinator worker
            final_recommendations=result.get("final_recommendations", []),
            quality_assessment=result["quality_metrics"],
            processing_time_minutes=processing_time,
            templates_used=result["processing_decision"].get("selected_templates", [])
        )
    
    async def cancel_processing(self, processing_id: str):
        """
        Cancel an active processing request.
        """
        if processing_id not in self.active_processings:
            raise HTTPException(status_code=404, detail="Processing ID not found")
        
        processing = self.active_processings[processing_id]
        if processing["status"] == "completed":
            raise HTTPException(status_code=400, detail="Cannot cancel completed processing")
        
        processing.update({
            "status": "cancelled",
            "current_stage": "cancelled",
            "progress": 0
        })
        
        return {"message": f"Processing {processing_id} cancelled"}
    
    async def get_manager_status(self) -> ManagerStatusResponse:
        """
        Get overall AI Manager status and metrics.
        """
        manager_status = await self.manager.get_manager_status()
        
        # Calculate current metrics
        active_count = len([p for p in self.active_processings.values() if p["status"] == "processing"])
        queue_count = len([p for p in self.active_processings.values() if p["status"] == "queued"])
        
        return ManagerStatusResponse(
            manager_status="active",
            active_processings=active_count,
            queue_length=queue_count,
            worker_pools=manager_status.get("worker_load", {}),
            performance_metrics=manager_status.get("quality_thresholds", {}),
            system_resources={"cpu": 0.65, "memory": 0.45}  # Would get from system monitoring
        )
    
    async def health_check(self):
        """
        Health check endpoint for monitoring systems.
        """
        try:
            # Basic health checks
            manager_status = await self.manager.get_manager_status()
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "manager_active": manager_status.get("status") == "active",
                "session_manager_available": self.session_manager is not None,
                "active_processings": len(self.active_processings),
                "template_count": manager_status.get("available_templates", 0)
            }
            
            return health_status
            
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            )
    
    async def submit_quality_feedback(self, processing_id: str, quality_metrics: Dict[str, float]):
        """
        Submit quality feedback for completed processing to improve future decisions.
        """
        if processing_id not in self.processing_results:
            raise HTTPException(status_code=404, detail="Processing results not found")
        
        try:
            # Get the document path from processing
            document_path = self.active_processings[processing_id]["task"].document_path
            
            # Submit feedback to AI Manager
            await self.manager.update_quality_feedback(document_path, quality_metrics)
            
            # Store feedback in results
            self.processing_results[processing_id]["quality_feedback"] = quality_metrics
            
            return {"message": "Quality feedback submitted successfully"}
            
        except Exception as e:
            logger.error(f"Failed to submit quality feedback: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_queue_status(self):
        """
        Get current processing queue status.
        """
        queued = [p for p in self.active_processings.values() if p["status"] == "queued"]
        processing = [p for p in self.active_processings.values() if p["status"] == "processing"]
        
        return {
            "queue_length": len(queued),
            "processing_count": len(processing),
            "average_wait_time_minutes": 5.0,  # Would calculate from historical data
            "estimated_queue_time_minutes": len(queued) * 5,
            "queue_items": [
                {
                    "processing_id": pid,
                    "document_path": p["task"].document_path,
                    "priority": p["task"].business_priority,
                    "queued_at": p["started_at"].isoformat()
                }
                for pid, p in self.active_processings.items() if p["status"] == "queued"
            ][:10]  # First 10 items
        }
    
    async def adjust_queue_priority(self, processing_id: str, new_priority: str):
        """
        Adjust priority of queued processing request.
        """
        if processing_id not in self.active_processings:
            raise HTTPException(status_code=404, detail="Processing ID not found")
        
        processing = self.active_processings[processing_id]
        if processing["status"] != "queued":
            raise HTTPException(status_code=400, detail="Can only adjust priority for queued items")
        
        # Update task priority
        processing["task"].business_priority = new_priority
        
        return {"message": f"Priority updated to {new_priority}"}
    
    async def list_available_templates(self):
        """
        List all available processing templates.
        """
        manager_status = await self.manager.get_manager_status()
        
        # Get template information from manager
        templates = []
        for template_name in self.manager.available_templates:
            template_info = self.manager.available_templates[template_name]
            performance = self.manager.template_performance.get(template_name, {})
            
            templates.append({
                "name": template_name,
                "domain_focus": template_info.get("domain_focus", "general"),
                "complexity": template_info.get("estimated_complexity", "medium"),
                "success_rate": performance.get("success_rate", 0.85),
                "average_processing_time": performance.get("avg_processing_time", 10),
                "usage_count": performance.get("usage_count", 0)
            })
        
        return {
            "templates": templates,
            "total_count": len(templates)
        }
    
    async def get_worker_status(self):
        """
        Get status of all worker pools.
        """
        manager_status = await self.manager.get_manager_status()
        
        return {
            "worker_pools": manager_status.get("worker_load", {}),
            "active_workers": manager_status.get("active_workers", 0),
            "performance_metrics": {
                worker_id: self.manager.worker_performance.get(worker_id, {})
                for worker_id in self.manager.active_workers
            }
        }
    
    async def stream_processing_status(self, processing_id: str):
        """
        Stream real-time processing status updates.
        """
        if processing_id not in self.active_processings:
            raise HTTPException(status_code=404, detail="Processing ID not found")
        
        async def generate_status_stream():
            while True:
                if processing_id in self.active_processings:
                    processing = self.active_processings[processing_id]
                    status_update = {
                        "processing_id": processing_id,
                        "status": processing["status"],
                        "progress": processing.get("progress", 0),
                        "current_stage": processing.get("current_stage", "unknown"),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    yield f"data: {json.dumps(status_update)}\n\n"
                    
                    # Stop streaming if completed or failed
                    if processing["status"] in ["completed", "failed", "cancelled"]:
                        break
                    
                    await asyncio.sleep(2)  # Update every 2 seconds
                else:
                    break
        
        return StreamingResponse(generate_status_stream(), media_type="text/plain")

# Integration with existing MCP server and gateways
async def integrate_with_mcp_server(mcp_server, manager_api: ManagerAPI):
    """
    Integrate AI Manager API with existing MCP server.
    """
    # Add AI Manager tools to MCP server
    mcp_tools = {
        "ai_manager_process": {
            "description": "Process document using AI-Assisted Manager",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "document_path": {"type": "string"},
                    "business_priority": {"type": "string", "default": "normal"},
                    "templates": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["document_path"]
            }
        },
        "ai_manager_status": {
            "description": "Get AI Manager status and queue information",
            "inputSchema": {"type": "object", "properties": {}}
        }
    }
    
    # Register tools with MCP server
    for tool_name, tool_config in mcp_tools.items():
        await mcp_server.register_tool(tool_name, tool_config)
    
    logger.info("AI Manager integrated with MCP server")

# CLI Integration
async def create_manager_cli_commands():
    """
    Create CLI commands for AI Manager operations.
    """
    cli_commands = {
        "ai-manager": {
            "description": "AI-Assisted Manager operations",
            "subcommands": {
                "status": "Get manager status and metrics",
                "process": "Process document with AI Manager",
                "queue": "Show processing queue status",
                "workers": "Show worker pool status",
                "templates": "List available templates"
            }
        }
    }
    
    return cli_commands