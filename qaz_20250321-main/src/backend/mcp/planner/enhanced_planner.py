#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Planner

Enhanced planner that orchestrates the MCP hierarchy with proper Planner → Coordinator → Worker flow.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from ..protocol.message_protocol import (
    MCPMessage, TaskType, Priority, MessageType,
    create_planner_to_coordinator_message
)
from ..coordinators.document_coordinator import DocumentCoordinator


class EnhancedPlanner:
    """Enhanced planner for MCP hierarchy orchestration"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize coordinators
        self.document_coordinator = DocumentCoordinator()
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'success_rate': 0.0,
            'coordinator_usage': {
                'document_coordinator': 0,
                'analysis_coordinator': 0,
                'validation_coordinator': 0
            }
        }
        
        # Decision tracking
        self.decision_log = []
    
    def process_request(self, user_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a user request through the MCP hierarchy"""
        start_time = datetime.now()
        
        # Create initial message
        message = self._create_initial_message(user_request)
        
        # Add audit entry for request start
        message.add_audit_entry(
            action="request_processing_started",
            decision_reasoning="Enhanced planner started processing user request",
            confidence_score=1.0
        )
        
        try:
            # Analyze request and determine coordination strategy
            coordination_strategy = self._analyze_request(user_request)
            include_live_context = self._should_include_live_context(user_request)
            
            # Add decision audit entry
            message.add_audit_entry(
                action="coordination_strategy_selected",
                decision_reasoning=f"Selected coordination strategy: {coordination_strategy}. Live context: {'enabled' if include_live_context else 'disabled'}",
                confidence_score=0.9
            )
            
            # Execute coordination strategy
            result = self._execute_coordination_strategy(message, coordination_strategy)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance metrics
            self._update_metrics(True, processing_time, coordination_strategy)
            
            # Add completion audit entry
            message.add_audit_entry(
                action="request_processing_completed",
                decision_reasoning="Request processing completed successfully",
                confidence_score=0.95,
                performance_metrics={
                    'processing_time': processing_time,
                    'coordination_strategy': coordination_strategy
                }
            )
            
            # Create final result
            final_result = {
                'success': True,
                'processing_time': processing_time,
                'coordination_strategy': coordination_strategy,
                'live_context_enabled': include_live_context,
                'result': result,
                'audit_trail': message.audit_trail,
                'performance_metrics': self.get_performance_metrics()
            }
            
            self.logger.info(f"Request processing completed in {processing_time:.2f}s")
            return final_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(False, processing_time, "unknown")
            
            message.add_audit_entry(
                action="request_processing_failed",
                decision_reasoning=f"Request processing failed: {str(e)}",
                confidence_score=0.0,
                error_details=str(e)
            )
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'audit_trail': message.audit_trail
            }
    
    def _create_initial_message(self, user_request: Dict[str, Any]) -> MCPMessage:
        """Create initial MCP message from user request"""
        from ..protocol.message_protocol import MCPMessageBuilder, TaskType, Priority
        
        # Determine task type based on request
        task_type = TaskType.PROCESSING
        if 'analysis' in user_request.get('type', '').lower():
            task_type = TaskType.ANALYSIS
        elif 'validation' in user_request.get('type', '').lower():
            task_type = TaskType.VALIDATION
        elif 'retrieval' in user_request.get('type', '').lower():
            task_type = TaskType.RETRIEVAL
        
        # Create message
        message = (MCPMessageBuilder()
                  .set_message_type(MessageType.PLANNER_TO_COORDINATOR)
                  .set_source("planner")
                  .set_target("coordinator")
                  .set_task_type(task_type)
                  .set_priority(Priority.MEDIUM)
                  .set_payload(user_request)
                  .set_context({
                      'user_id': user_request.get('user_id', 'unknown'),
                      'request_timestamp': datetime.now().isoformat(),
                      'request_type': user_request.get('type', 'general')
                  })
                  .build())
        
        return message
    
    def _analyze_request(self, user_request: Dict[str, Any]) -> str:
        """Analyze user request and determine coordination strategy"""
        request_type = user_request.get('type', '').lower()
        files = user_request.get('files', [])
        user_preferences = user_request.get('user_preferences', {})
        
        # Decision logic
        if 'document' in request_type or files:
            return 'document_coordinator'
        elif 'analysis' in request_type:
            return 'analysis_coordinator'
        elif 'validation' in request_type:
            return 'validation_coordinator'
        else:
            # Default to document coordinator for general processing
            return 'document_coordinator'
    
    def _should_include_live_context(self, user_request: Dict[str, Any]) -> bool:
        """Determine if live context should be included"""
        user_preferences = user_request.get('user_preferences', {})
        
        # Check explicit preference
        if 'include_live_context' in user_preferences:
            return user_preferences['include_live_context']
        
        # Check request type
        request_type = user_request.get('type', '').lower()
        if 'live_context' in request_type or 'temporal' in request_type:
            return True
        
        # Default to False for backward compatibility
        return False
    
    def _execute_coordination_strategy(self, message: MCPMessage, strategy: str) -> Dict[str, Any]:
        """Execute the selected coordination strategy"""
        if strategy == 'document_coordinator':
            return self._execute_document_coordination(message)
        elif strategy == 'analysis_coordinator':
            return self._execute_analysis_coordination(message)
        elif strategy == 'validation_coordinator':
            return self._execute_validation_coordination(message)
        else:
            raise ValueError(f"Unknown coordination strategy: {strategy}")
    
    def _execute_document_coordination(self, message: MCPMessage) -> Dict[str, Any]:
        """Execute document coordination strategy"""
        # Create coordinator message
        coordinator_message = create_planner_to_coordinator_message(
            coordinator="document_coordinator",
            task_type=message.task_type,
            payload=message.payload,
            context=message.context
        )
        coordinator_message.audit_trail = message.audit_trail.copy()
        
        # Execute document coordinator
        response = self.document_coordinator.process_document(coordinator_message)
        
        # Log decision
        self._log_decision("document_coordinator", message.payload, response.payload)
        
        return response.payload
    
    def _execute_analysis_coordination(self, message: MCPMessage) -> Dict[str, Any]:
        """Execute analysis coordination strategy (placeholder)"""
        # TODO: Implement analysis coordinator
        return {
            'success': False,
            'error': 'Analysis coordinator not yet implemented',
            'confidence_score': 0.0
        }
    
    def _execute_validation_coordination(self, message: MCPMessage) -> Dict[str, Any]:
        """Execute validation coordination strategy (placeholder)"""
        # TODO: Implement validation coordinator
        return {
            'success': False,
            'error': 'Validation coordinator not yet implemented',
            'confidence_score': 0.0
        }
    
    def _log_decision(self, coordinator: str, request: Dict[str, Any], result: Dict[str, Any]):
        """Log decision for audit purposes"""
        decision = {
            'timestamp': datetime.now().isoformat(),
            'coordinator': coordinator,
            'request_type': request.get('type', 'unknown'),
            'success': result.get('success', False),
            'confidence_score': result.get('confidence_score', 0.0)
        }
        
        self.decision_log.append(decision)
        
        # Keep only last 100 decisions
        if len(self.decision_log) > 100:
            self.decision_log = self.decision_log[-100:]
    
    def _update_metrics(self, success: bool, processing_time: float, strategy: str):
        """Update performance metrics"""
        self.performance_metrics['total_requests'] += 1
        
        if success:
            self.performance_metrics['successful_requests'] += 1
        else:
            self.performance_metrics['failed_requests'] += 1
        
        # Update success rate
        total_requests = self.performance_metrics['total_requests']
        successful_requests = self.performance_metrics['successful_requests']
        self.performance_metrics['success_rate'] = successful_requests / total_requests
        
        # Update average processing time
        total_time = self.performance_metrics.get('total_processing_time', 0) + processing_time
        self.performance_metrics['total_processing_time'] = total_time
        self.performance_metrics['average_processing_time'] = total_time / total_requests
        
        # Update coordinator usage
        if strategy in self.performance_metrics['coordinator_usage']:
            self.performance_metrics['coordinator_usage'][strategy] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_coordinator_metrics(self) -> Dict[str, Any]:
        """Get metrics from all coordinators"""
        return {
            'document_coordinator': self.document_coordinator.get_performance_metrics(),
            'document_coordinator_workers': self.document_coordinator.get_worker_metrics()
        }
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Get decision log for audit purposes"""
        return self.decision_log.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on planner and coordinators"""
        return {
            'planner': 'enhanced_planner',
            'status': 'healthy',
            'performance_metrics': self.get_performance_metrics(),
            'coordinator_metrics': self.get_coordinator_metrics(),
            'document_coordinator_health': self.document_coordinator.health_check(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit summary for the system"""
        return {
            'total_requests': self.performance_metrics['total_requests'],
            'success_rate': self.performance_metrics['success_rate'],
            'average_processing_time': self.performance_metrics['average_processing_time'],
            'coordinator_usage': self.performance_metrics['coordinator_usage'],
            'recent_decisions': self.decision_log[-10:],  # Last 10 decisions
            'timestamp': datetime.now().isoformat()
        }
