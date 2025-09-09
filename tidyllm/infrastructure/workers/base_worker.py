"""
Base Worker Infrastructure - Agent Worker Pattern
=================================================

Base classes and interfaces for implementing agent workers that can be:
- Independently scaled and deployed
- Queued for background processing
- Monitored for health and performance
- Dynamically allocated based on workload

Worker Types:
- ExtractionWorker: Document content extraction and parsing
- EmbeddingWorker: Vector embedding generation and processing
- IndexingWorker: Document indexing and vector storage operations
- ProcessingWorker: Complex document processing workflows
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger("base_worker")

# Type variables for generic worker implementations
TaskInput = TypeVar('TaskInput')
TaskResult = TypeVar('TaskResult')


class WorkerStatus(Enum):
    """Worker lifecycle status."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"
    INITIALIZING = "initializing"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkerTask:
    """Individual task for worker processing."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    input_data: Any = None
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get task processing time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "has_result": self.result is not None,
            "metadata": self.metadata
        }


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time: float = 0.0
    current_queue_size: int = 0
    uptime: float = 0.0
    last_task_completed: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        if self.total_tasks_processed == 0:
            return 0.0
        return (self.successful_tasks / self.total_tasks_processed) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_tasks_processed": self.total_tasks_processed,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.success_rate,
            "average_processing_time": self.average_processing_time,
            "current_queue_size": self.current_queue_size,
            "uptime": self.uptime,
            "last_task_completed": self.last_task_completed.isoformat() if self.last_task_completed else None
        }


class BaseWorker(ABC, Generic[TaskInput, TaskResult]):
    """
    Base class for all agent workers.
    
    Provides common functionality for:
    - Task queue management
    - Performance monitoring
    - Error handling and recovery
    - Health checks
    - Async processing patterns
    """
    
    def __init__(self, 
                 worker_name: str,
                 max_queue_size: int = 100,
                 max_concurrent_tasks: int = 1,
                 task_timeout: float = 300.0):
        """
        Initialize base worker.
        
        Args:
            worker_name: Unique identifier for this worker
            max_queue_size: Maximum number of queued tasks
            max_concurrent_tasks: Maximum concurrent task processing
            task_timeout: Task timeout in seconds
        """
        self.worker_name = worker_name
        self.max_queue_size = max_queue_size
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        
        # Worker state
        self.status = WorkerStatus.INITIALIZING
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, WorkerTask] = {}
        self.completed_tasks: List[WorkerTask] = []
        self.metrics = WorkerMetrics()
        self.started_at = datetime.now()
        
        # Worker management
        self.worker_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        logger.info(f"Worker '{worker_name}' initialized")
    
    @abstractmethod
    async def process_task(self, task_input: TaskInput) -> TaskResult:
        """
        Process a single task.
        
        Args:
            task_input: Input data for the task
            
        Returns:
            Processing result
            
        Raises:
            Exception: If task processing fails
        """
        pass
    
    @abstractmethod
    def validate_input(self, task_input: Any) -> bool:
        """
        Validate task input data.
        
        Args:
            task_input: Input to validate
            
        Returns:
            True if input is valid
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize worker resources."""
        try:
            await self._initialize_worker()
            self.status = WorkerStatus.IDLE
            logger.info(f"Worker '{self.worker_name}' initialized successfully")
        except Exception as e:
            self.status = WorkerStatus.ERROR
            logger.error(f"Worker '{self.worker_name}' initialization failed: {e}")
            raise
    
    async def _initialize_worker(self) -> None:
        """Override in subclasses for specific initialization."""
        pass
    
    async def start(self) -> None:
        """Start the worker task processing loop."""
        if self.status == WorkerStatus.INITIALIZING:
            await self.initialize()
        
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info(f"Worker '{self.worker_name}' started")
    
    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker '{self.worker_name}'...")
        
        self.shutdown_event.set()
        self.status = WorkerStatus.STOPPED
        
        if self.worker_task:
            try:
                await asyncio.wait_for(self.worker_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Worker '{self.worker_name}' did not stop gracefully, cancelling...")
                self.worker_task.cancel()
        
        logger.info(f"Worker '{self.worker_name}' stopped")
    
    async def submit_task(self, 
                         task_type: str,
                         task_input: TaskInput, 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         metadata: Dict[str, Any] = None) -> WorkerTask:
        """
        Submit a task for processing.
        
        Args:
            task_type: Type identifier for the task
            task_input: Input data for processing
            priority: Task priority level
            metadata: Additional task metadata
            
        Returns:
            WorkerTask instance for tracking
            
        Raises:
            ValueError: If queue is full or input is invalid
        """
        if not self.validate_input(task_input):
            raise ValueError(f"Invalid input for task type '{task_type}'")
        
        if self.task_queue.full():
            raise ValueError(f"Worker '{self.worker_name}' queue is full")
        
        task = WorkerTask(
            task_type=task_type,
            input_data=task_input,
            priority=priority,
            metadata=metadata or {}
        )
        
        await self.task_queue.put(task)
        self.metrics.current_queue_size = self.task_queue.qsize()
        
        logger.info(f"Task '{task.task_id}' submitted to worker '{self.worker_name}'")
        return task
    
    async def _worker_loop(self) -> None:
        """Main worker processing loop."""
        logger.info(f"Worker '{self.worker_name}' processing loop started")
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for tasks or shutdown
                try:
                    task = await asyncio.wait_for(
                        self.task_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the task
                await self._process_worker_task(task)
                
            except Exception as e:
                logger.error(f"Worker '{self.worker_name}' loop error: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error
        
        logger.info(f"Worker '{self.worker_name}' processing loop ended")
    
    async def _process_worker_task(self, task: WorkerTask) -> None:
        """Process a single worker task."""
        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task
        self.status = WorkerStatus.BUSY
        
        try:
            logger.info(f"Processing task '{task.task_id}' of type '{task.task_type}'")
            
            # Process with timeout
            result = await asyncio.wait_for(
                self.process_task(task.input_data),
                timeout=self.task_timeout
            )
            
            # Task completed successfully
            task.result = result
            task.completed_at = datetime.now()
            
            self._update_metrics_success(task)
            logger.info(f"Task '{task.task_id}' completed successfully in {task.processing_time:.2f}s")
            
        except asyncio.TimeoutError:
            task.error_message = f"Task timed out after {self.task_timeout}s"
            task.completed_at = datetime.now()
            self._update_metrics_failure(task)
            logger.error(f"Task '{task.task_id}' timed out")
            
        except Exception as e:
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self._update_metrics_failure(task)
            logger.error(f"Task '{task.task_id}' failed: {e}")
        
        finally:
            # Clean up task
            self.active_tasks.pop(task.task_id, None)
            self.completed_tasks.append(task)
            self.metrics.current_queue_size = self.task_queue.qsize()
            
            # Update worker status
            if not self.active_tasks:
                self.status = WorkerStatus.IDLE
    
    def _update_metrics_success(self, task: WorkerTask) -> None:
        """Update metrics for successful task."""
        self.metrics.total_tasks_processed += 1
        self.metrics.successful_tasks += 1
        self.metrics.last_task_completed = task.completed_at
        
        # Update rolling average processing time
        if task.processing_time:
            total_time = (self.metrics.average_processing_time * 
                         (self.metrics.total_tasks_processed - 1) + 
                         task.processing_time)
            self.metrics.average_processing_time = total_time / self.metrics.total_tasks_processed
    
    def _update_metrics_failure(self, task: WorkerTask) -> None:
        """Update metrics for failed task."""
        self.metrics.total_tasks_processed += 1
        self.metrics.failed_tasks += 1
        self.metrics.last_task_completed = task.completed_at
    
    def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        uptime = (datetime.now() - self.started_at).total_seconds()
        self.metrics.uptime = uptime
        
        return {
            "worker_name": self.worker_name,
            "status": self.status.value,
            "uptime": uptime,
            "active_tasks": len(self.active_tasks),
            "metrics": self.metrics.to_dict()
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return task.to_dict()
        
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            health_status = "healthy"
            issues = []
            
            # Check worker status
            if self.status == WorkerStatus.ERROR:
                health_status = "unhealthy"
                issues.append("Worker in error state")
            elif self.status == WorkerStatus.STOPPED:
                health_status = "stopped"
                issues.append("Worker is stopped")
            
            # Check queue utilization
            queue_utilization = (self.task_queue.qsize() / self.max_queue_size) * 100
            if queue_utilization > 90:
                issues.append(f"High queue utilization: {queue_utilization:.1f}%")
            
            # Check success rate
            if self.metrics.success_rate < 90 and self.metrics.total_tasks_processed > 10:
                issues.append(f"Low success rate: {self.metrics.success_rate:.1f}%")
            
            return {
                "worker_name": self.worker_name,
                "health_status": health_status,
                "issues": issues,
                "queue_utilization": queue_utilization,
                "metrics": self.metrics.to_dict()
            }
            
        except Exception as e:
            return {
                "worker_name": self.worker_name,
                "health_status": "unhealthy",
                "error": str(e),
                "issues": ["Health check failed"]
            }