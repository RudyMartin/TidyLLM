#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Handler - Non-Blocking Processing for MCP System

Provides non-blocking message processing with:
- Concurrent task execution
- Real-time response streaming
- Background task management
- Resource optimization
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Coroutine, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import weakref

from ..protocol.message_protocol import MCPMessage, MessageType, TaskType, Priority


class ProcessingMode(Enum):
    """Processing modes for async operations"""
    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"
    BACKGROUND = "background"


@dataclass
class AsyncTask:
    """Async task configuration"""
    task_id: str
    handler: Callable
    message: MCPMessage
    mode: ProcessingMode
    priority: Priority
    created_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: float = 30.0
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingResponse:
    """Streaming response configuration"""
    task_id: str
    stream_id: str
    chunk_size: int = 1000
    max_chunks: int = 100
    include_metadata: bool = True


class AsyncHandler:
    """Non-blocking message processing handler"""
    
    def __init__(self, max_workers: int = 10, max_queue_size: int = 1000):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Threading and async configuration
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task management
        self.active_tasks: Dict[str, AsyncTask] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.failed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Streaming
        self.streaming_responses: Dict[str, StreamingResponse] = {}
        self.stream_queues: Dict[str, Queue] = {}
        
        # Performance metrics
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "active_tasks": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
            "streaming_sessions": 0
        }
        
        # Threading
        self.running = False
        self.task_monitor_thread = None
        self.lock = threading.Lock()
        
        self.logger.info(f"Async Handler initialized with {max_workers} workers")
    
    def process_message(self, 
                       message: MCPMessage, 
                       handler: Callable, 
                       mode: ProcessingMode = ProcessingMode.ASYNC,
                       streaming_config: Optional[StreamingResponse] = None) -> str:
        """Process a message asynchronously"""
        
        task_id = f"task_{message.message_id}_{int(time.time() * 1000)}"
        
        # Create async task
        async_task = AsyncTask(
            task_id=task_id,
            handler=handler,
            message=message,
            mode=mode,
            priority=message.priority
        )
        
        # Store task
        with self.lock:
            self.active_tasks[task_id] = async_task
            self.metrics["total_tasks"] += 1
            self.metrics["active_tasks"] += 1
        
        # Process based on mode
        if mode == ProcessingMode.SYNC:
            self._process_sync(async_task)
        elif mode == ProcessingMode.ASYNC:
            self._process_async(async_task)
        elif mode == ProcessingMode.STREAMING:
            self._process_streaming(async_task, streaming_config)
        elif mode == ProcessingMode.BACKGROUND:
            self._process_background(async_task)
        
        self.logger.debug(f"Started {mode.value} processing for task {task_id}")
        return task_id
    
    def _process_sync(self, task: AsyncTask):
        """Process task synchronously"""
        try:
            start_time = time.time()
            
            # Execute handler
            result = task.handler(task.message)
            
            # Record completion
            processing_time = time.time() - start_time
            self._record_task_completion(task.task_id, result, processing_time, success=True)
            
        except Exception as e:
            self.logger.error(f"Error in sync processing for task {task.task_id}: {e}")
            self._record_task_completion(task.task_id, {"error": str(e)}, 0.0, success=False)
    
    def _process_async(self, task: AsyncTask):
        """Process task asynchronously"""
        def async_worker():
            try:
                start_time = time.time()
                
                # Execute handler
                result = task.handler(task.message)
                
                # Record completion
                processing_time = time.time() - start_time
                self._record_task_completion(task.task_id, result, processing_time, success=True)
                
            except Exception as e:
                self.logger.error(f"Error in async processing for task {task.task_id}: {e}")
                self._record_task_completion(task.task_id, {"error": str(e)}, 0.0, success=False)
        
        # Submit to thread pool
        self.executor.submit(async_worker)
    
    def _process_streaming(self, task: AsyncTask, streaming_config: StreamingResponse):
        """Process task with streaming response"""
        if not streaming_config:
            self.logger.error(f"No streaming config provided for task {task.task_id}")
            return
        
        # Create stream queue
        stream_queue = Queue(maxsize=100)
        self.stream_queues[streaming_config.stream_id] = stream_queue
        
        def streaming_worker():
            try:
                start_time = time.time()
                chunk_count = 0
                
                # Execute handler with streaming
                if hasattr(task.handler, '__call__'):
                    # Check if handler supports streaming
                    if hasattr(task.handler, 'stream_response'):
                        for chunk in task.handler.stream_response(task.message):
                            if chunk_count >= streaming_config.max_chunks:
                                break
                            
                            # Prepare chunk
                            chunk_data = {
                                "task_id": task.task_id,
                                "stream_id": streaming_config.stream_id,
                                "chunk": chunk,
                                "chunk_index": chunk_count,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            if streaming_config.include_metadata:
                                chunk_data["metadata"] = task.metadata
                            
                            # Send chunk
                            try:
                                stream_queue.put_nowait(chunk_data)
                                chunk_count += 1
                            except:
                                self.logger.warning(f"Stream queue full for {streaming_config.stream_id}")
                                break
                    else:
                        # Fallback to regular processing
                        result = task.handler(task.message)
                        stream_queue.put({
                            "task_id": task.task_id,
                            "stream_id": streaming_config.stream_id,
                            "chunk": result,
                            "chunk_index": 0,
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Record completion
                processing_time = time.time() - start_time
                self._record_task_completion(task.task_id, {"chunks_sent": chunk_count}, processing_time, success=True)
                
                # Clean up stream
                self._cleanup_stream(streaming_config.stream_id)
                
            except Exception as e:
                self.logger.error(f"Error in streaming processing for task {task.task_id}: {e}")
                self._record_task_completion(task.task_id, {"error": str(e)}, 0.0, success=False)
                self._cleanup_stream(streaming_config.stream_id)
        
        # Submit to thread pool
        self.executor.submit(streaming_worker)
        
        # Update metrics
        with self.lock:
            self.metrics["streaming_sessions"] += 1
    
    def _process_background(self, task: AsyncTask):
        """Process task in background"""
        def background_worker():
            try:
                start_time = time.time()
                
                # Execute handler
                result = task.handler(task.message)
                
                # Record completion (background tasks don't block)
                processing_time = time.time() - start_time
                self._record_task_completion(task.task_id, result, processing_time, success=True)
                
            except Exception as e:
                self.logger.error(f"Error in background processing for task {task.task_id}: {e}")
                self._record_task_completion(task.task_id, {"error": str(e)}, 0.0, success=False)
        
        # Submit to thread pool with lower priority
        self.executor.submit(background_worker)
    
    def _record_task_completion(self, task_id: str, result: Any, processing_time: float, success: bool):
        """Record task completion"""
        with self.lock:
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                self.metrics["active_tasks"] -= 1
            
            # Update metrics
            self.metrics["total_processing_time"] += processing_time
            self.metrics["average_processing_time"] = self.metrics["total_processing_time"] / max(self.metrics["total_tasks"], 1)
            
            # Record result
            completion_data = {
                "task_id": task_id,
                "result": result,
                "processing_time": processing_time,
                "completed_at": datetime.now().isoformat(),
                "success": success
            }
            
            if success:
                self.completed_tasks[task_id] = completion_data
                self.metrics["completed_tasks"] += 1
            else:
                self.failed_tasks[task_id] = completion_data
                self.metrics["failed_tasks"] += 1
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "active",
                "mode": task.mode.value,
                "priority": task.priority.value,
                "created_at": task.created_at.isoformat(),
                "elapsed_time": (datetime.now() - task.created_at).total_seconds()
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return {
                "task_id": task_id,
                "status": "completed",
                **self.completed_tasks[task_id]
            }
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            return {
                "task_id": task_id,
                "status": "failed",
                **self.failed_tasks[task_id]
            }
        
        return None
    
    def get_streaming_chunk(self, stream_id: str, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get next chunk from streaming response"""
        stream_queue = self.stream_queues.get(stream_id)
        if not stream_queue:
            return None
        
        try:
            return stream_queue.get(timeout=timeout)
        except:
            return None
    
    def _cleanup_stream(self, stream_id: str):
        """Clean up streaming resources"""
        if stream_id in self.stream_queues:
            del self.stream_queues[stream_id]
        if stream_id in self.streaming_responses:
            del self.streaming_responses[stream_id]
    
    def start_monitoring(self):
        """Start task monitoring thread"""
        if self.running:
            self.logger.warning("Async handler is already running")
            return
        
        self.running = True
        self.task_monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.task_monitor_thread.start()
        self.logger.info("Async handler monitoring started")
    
    def stop_monitoring(self):
        """Stop task monitoring thread"""
        self.running = False
        if self.task_monitor_thread:
            self.task_monitor_thread.join(timeout=5.0)
        self.logger.info("Async handler monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Clean up old completed tasks (keep last 1000)
                with self.lock:
                    if len(self.completed_tasks) > 1000:
                        # Remove oldest tasks
                        oldest_tasks = sorted(
                            self.completed_tasks.items(),
                            key=lambda x: x[1]["completed_at"]
                        )[:len(self.completed_tasks) - 1000]
                        
                        for task_id, _ in oldest_tasks:
                            del self.completed_tasks[task_id]
                    
                    if len(self.failed_tasks) > 1000:
                        # Remove oldest failed tasks
                        oldest_failed = sorted(
                            self.failed_tasks.items(),
                            key=lambda x: x[1]["completed_at"]
                        )[:len(self.failed_tasks) - 1000]
                        
                        for task_id, _ in oldest_failed:
                            del self.failed_tasks[task_id]
                
                # Check for timed out tasks
                current_time = datetime.now()
                timed_out_tasks = []
                
                with self.lock:
                    for task_id, task in self.active_tasks.items():
                        if (current_time - task.created_at).total_seconds() > task.timeout_seconds:
                            timed_out_tasks.append(task_id)
                
                # Handle timed out tasks
                for task_id in timed_out_tasks:
                    self.logger.warning(f"Task {task_id} timed out")
                    self._record_task_completion(task_id, {"error": "timeout"}, 0.0, success=False)
                
                # Sleep
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current async handler metrics"""
        with self.lock:
            return {
                "total_tasks": self.metrics["total_tasks"],
                "completed_tasks": self.metrics["completed_tasks"],
                "failed_tasks": self.metrics["failed_tasks"],
                "active_tasks": self.metrics["active_tasks"],
                "success_rate": self.metrics["completed_tasks"] / max(self.metrics["total_tasks"], 1),
                "average_processing_time": self.metrics["average_processing_time"],
                "streaming_sessions": self.metrics["streaming_sessions"],
                "active_streams": len(self.stream_queues),
                "executor_status": {
                    "max_workers": self.max_workers,
                    "active_threads": len(self.executor._threads),
                    "queue_size": self.executor._work_queue.qsize()
                }
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on async handler"""
        return {
            "component": "async_handler",
            "status": "healthy" if self.running else "unhealthy",
            "metrics": self.get_metrics(),
            "active_tasks_count": len(self.active_tasks),
            "completed_tasks_count": len(self.completed_tasks),
            "failed_tasks_count": len(self.failed_tasks),
            "streaming_sessions_count": len(self.stream_queues),
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Shutdown async handler"""
        self.stop_monitoring()
        self.executor.shutdown(wait=True)
        self.logger.info("Async handler shutdown complete")
