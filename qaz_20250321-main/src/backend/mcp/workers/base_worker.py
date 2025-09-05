#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Worker Class

Base class for all workers in the MCP hierarchy. Provides standardized interface
and audit capabilities for worker operations.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from ..protocol.message_protocol import MCPMessage, TaskType, Priority, AuditTrail


class BaseWorker(ABC):
    """Base class for all workers in the MCP hierarchy"""
    
    def __init__(self, worker_name: str, worker_type: str):
        self.worker_name = worker_name
        self.worker_type = worker_type
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.performance_metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        self.audit_log = []
    
    @abstractmethod
    def process_task(self, message: MCPMessage) -> Dict[str, Any]:
        """
        Process a task based on the incoming message.
        
        Args:
            message: MCP message containing task details
            
        Returns:
            Dictionary containing task results
        """
        pass
    
    def execute(self, message: MCPMessage) -> MCPMessage:
        """
        Execute a task and return a response message with audit trail.
        
        Args:
            message: Incoming MCP message
            
        Returns:
            Response MCP message with results and audit trail
        """
        start_time = time.time()
        
        # Add audit entry for task start
        message.add_audit_entry(
            action=f"task_started_{self.worker_name}",
            decision_reasoning=f"Worker {self.worker_name} started processing task",
            confidence_score=1.0
        )
        
        try:
            # Process the task
            result = self.process_task(message)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self._update_metrics(True, processing_time)
            
            # Add audit entry for successful completion
            message.add_audit_entry(
                action=f"task_completed_{self.worker_name}",
                decision_reasoning=f"Worker {self.worker_name} successfully completed task",
                confidence_score=result.get('confidence_score', 1.0),
                performance_metrics={
                    'processing_time': processing_time,
                    'worker_name': self.worker_name,
                    'worker_type': self.worker_type
                }
            )
            
            # Create response message
            response_message = self._create_response_message(message, result, True)
            
            # Log successful execution
            self.logger.info(f"Worker {self.worker_name} completed task in {processing_time:.2f}s")
            
            return response_message
            
        except Exception as e:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self._update_metrics(False, processing_time)
            
            # Add audit entry for failure
            message.add_audit_entry(
                action=f"task_failed_{self.worker_name}",
                decision_reasoning=f"Worker {self.worker_name} failed to process task",
                confidence_score=0.0,
                performance_metrics={
                    'processing_time': processing_time,
                    'worker_name': self.worker_name,
                    'worker_type': self.worker_type
                },
                error_details=str(e)
            )
            
            # Create error response message
            error_result = {
                'success': False,
                'error': str(e),
                'confidence_score': 0.0,
                'processing_time': processing_time
            }
            
            response_message = self._create_response_message(message, error_result, False)
            
            # Log error
            self.logger.error(f"Worker {self.worker_name} failed: {e}")
            
            return response_message
    
    def _create_response_message(self, original_message: MCPMessage, 
                               result: Dict[str, Any], success: bool) -> MCPMessage:
        """Create a response message based on the original message and results"""
        from ..protocol.message_protocol import create_worker_to_coordinator_message, TaskType
        
        # Determine task type based on result
        task_type = TaskType.PROCESSING
        if 'analysis' in result:
            task_type = TaskType.ANALYSIS
        elif 'validation' in result:
            task_type = TaskType.VALIDATION
        elif 'generation' in result:
            task_type = TaskType.GENERATION
        elif 'retrieval' in result:
            task_type = TaskType.RETRIEVAL
        
        # Create response payload
        payload = {
            'worker_name': self.worker_name,
            'worker_type': self.worker_type,
            'success': success,
            'result': result,
            'original_message_id': original_message.message_id
        }
        
        # Create response message
        response_message = create_worker_to_coordinator_message(
            coordinator="coordinator",
            task_type=task_type,
            payload=payload,
            context=original_message.context
        )
        
        # Copy audit trail from original message
        response_message.audit_trail = original_message.audit_trail.copy()
        
        return response_message
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update performance metrics"""
        self.performance_metrics['total_tasks'] += 1
        self.performance_metrics['total_processing_time'] += processing_time
        
        if success:
            self.performance_metrics['successful_tasks'] += 1
        else:
            self.performance_metrics['failed_tasks'] += 1
        
        # Update average processing time
        total_tasks = self.performance_metrics['total_tasks']
        total_time = self.performance_metrics['total_processing_time']
        self.performance_metrics['average_processing_time'] = total_time / total_tasks
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Calculate success rate
        total_tasks = metrics['total_tasks']
        successful_tasks = metrics['successful_tasks']
        metrics['success_rate'] = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
        
        return metrics
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log for this worker"""
        return self.audit_log.copy()
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the worker"""
        return {
            'worker_name': self.worker_name,
            'worker_type': self.worker_type,
            'status': 'healthy',
            'performance_metrics': self.get_performance_metrics(),
            'timestamp': datetime.now().isoformat()
        }
