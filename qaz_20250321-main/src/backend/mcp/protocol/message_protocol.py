#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Message Protocol

Standardized message format for communication between Planner, Coordinators, and Workers
in the MCP hierarchy.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class MessageType(Enum):
    """Message types for different layers"""
    PLANNER_TO_COORDINATOR = "planner_to_coordinator"
    COORDINATOR_TO_WORKER = "coordinator_to_worker"
    WORKER_TO_COORDINATOR = "worker_to_coordinator"
    COORDINATOR_TO_PLANNER = "coordinator_to_planner"
    ERROR = "error"
    STATUS = "status"


class TaskType(Enum):
    """Task types for different operations"""
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    GENERATION = "generation"
    PROCESSING = "processing"
    STORAGE = "storage"
    # Document processing specific tasks
    DOCUMENT_PROCESSING = "document_processing"
    TEXT_PROCESSING = "text_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    TABLE_EXTRACTION = "table_extraction"
    # VST vs MVR specific tasks
    VST_MVR_COMPARISON = "vst_mvr_comparison"
    SIMILARITY_CALCULATION = "similarity_calculation"
    GAP_ANALYSIS = "gap_analysis"


class Priority(Enum):
    """Message priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AuditTrail:
    """Audit trail for tracking decisions and actions"""
    message_id: str
    timestamp: str
    source: str
    target: str
    action: str
    decision_reasoning: Optional[str] = None
    confidence_score: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None


@dataclass
class MCPMessage:
    """Standard MCP message format"""
    message_id: str
    timestamp: str
    message_type: MessageType
    source: str
    target: str
    task_type: TaskType
    priority: Priority
    context: Dict[str, Any]
    payload: Dict[str, Any]
    audit_trail: List[AuditTrail]
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['task_type'] = self.task_type.value
        data['priority'] = self.priority.value
        data['audit_trail'] = [asdict(trail) for trail in self.audit_trail]
        return data
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary"""
        # Convert enum values back to enums
        data['message_type'] = MessageType(data['message_type'])
        data['task_type'] = TaskType(data['task_type'])
        data['priority'] = Priority(data['priority'])
        
        # Convert audit trail back to objects
        audit_trail = []
        for trail_data in data.get('audit_trail', []):
            audit_trail.append(AuditTrail(**trail_data))
        data['audit_trail'] = audit_trail
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MCPMessage':
        """Create message from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def create_simple(cls, task_type: TaskType, payload: Dict[str, Any], 
                     source: str = "system", target: str = "worker",
                     priority: Priority = Priority.MEDIUM) -> 'MCPMessage':
        """Create a simple MCP message with minimal required fields"""
        return cls(
            message_id="",
            timestamp="",
            message_type=MessageType.COORDINATOR_TO_WORKER,
            source=source,
            target=target,
            task_type=task_type,
            priority=priority,
            context={},
            payload=payload,
            audit_trail=[]
        )
    
    def add_audit_entry(self, action: str, decision_reasoning: str = None, 
                       confidence_score: float = None, 
                       performance_metrics: Dict[str, Any] = None,
                       error_details: str = None):
        """Add an audit trail entry"""
        audit_entry = AuditTrail(
            message_id=self.message_id,
            timestamp=datetime.now().isoformat(),
            source=self.source,
            target=self.target,
            action=action,
            decision_reasoning=decision_reasoning,
            confidence_score=confidence_score,
            performance_metrics=performance_metrics,
            error_details=error_details
        )
        self.audit_trail.append(audit_entry)


class MCPMessageBuilder:
    """Builder for creating MCP messages"""
    
    def __init__(self):
        self.message_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.message_type = None
        self.source = None
        self.target = None
        self.task_type = None
        self.priority = Priority.MEDIUM
        self.context = {}
        self.payload = {}
        self.audit_trail = []
    
    def set_message_type(self, message_type: MessageType) -> 'MCPMessageBuilder':
        self.message_type = message_type
        return self
    
    def set_source(self, source: str) -> 'MCPMessageBuilder':
        self.source = source
        return self
    
    def set_target(self, target: str) -> 'MCPMessageBuilder':
        self.target = target
        return self
    
    def set_task_type(self, task_type: TaskType) -> 'MCPMessageBuilder':
        self.task_type = task_type
        return self
    
    def set_priority(self, priority: Priority) -> 'MCPMessageBuilder':
        self.priority = priority
        return self
    
    def set_context(self, context: Dict[str, Any]) -> 'MCPMessageBuilder':
        self.context = context
        return self
    
    def set_payload(self, payload: Dict[str, Any]) -> 'MCPMessageBuilder':
        self.payload = payload
        return self
    
    def add_context(self, key: str, value: Any) -> 'MCPMessageBuilder':
        self.context[key] = value
        return self
    
    def add_payload(self, key: str, value: Any) -> 'MCPMessageBuilder':
        self.payload[key] = value
        return self
    
    def build(self) -> MCPMessage:
        """Build the MCP message"""
        if not all([self.message_type, self.source, self.target, self.task_type]):
            raise ValueError("Missing required fields: message_type, source, target, task_type")
        
        return MCPMessage(
            message_id=self.message_id,
            timestamp=self.timestamp,
            message_type=self.message_type,
            source=self.source,
            target=self.target,
            task_type=self.task_type,
            priority=self.priority,
            context=self.context,
            payload=self.payload,
            audit_trail=self.audit_trail
        )


# Convenience functions for common message types
def create_planner_to_coordinator_message(
    coordinator: str,
    task_type: TaskType,
    payload: Dict[str, Any],
    context: Dict[str, Any] = None,
    priority: Priority = Priority.MEDIUM
) -> MCPMessage:
    """Create a message from planner to coordinator"""
    return (MCPMessageBuilder()
            .set_message_type(MessageType.PLANNER_TO_COORDINATOR)
            .set_source("planner")
            .set_target(coordinator)
            .set_task_type(task_type)
            .set_priority(priority)
            .set_payload(payload)
            .set_context(context or {})
            .build())


def create_coordinator_to_worker_message(
    worker: str,
    task_type: TaskType,
    payload: Dict[str, Any],
    context: Dict[str, Any] = None,
    priority: Priority = Priority.MEDIUM
) -> MCPMessage:
    """Create a message from coordinator to worker"""
    return (MCPMessageBuilder()
            .set_message_type(MessageType.COORDINATOR_TO_WORKER)
            .set_source("coordinator")
            .set_target(worker)
            .set_task_type(task_type)
            .set_priority(priority)
            .set_payload(payload)
            .set_context(context or {})
            .build())


def create_worker_to_coordinator_message(
    coordinator: str,
    task_type: TaskType,
    payload: Dict[str, Any],
    context: Dict[str, Any] = None,
    priority: Priority = Priority.MEDIUM
) -> MCPMessage:
    """Create a message from worker to coordinator"""
    return (MCPMessageBuilder()
            .set_message_type(MessageType.WORKER_TO_COORDINATOR)
            .set_source("worker")
            .set_target(coordinator)
            .set_task_type(task_type)
            .set_priority(priority)
            .set_payload(payload)
            .set_context(context or {})
            .build())


def create_coordinator_to_planner_message(
    task_type: TaskType,
    payload: Dict[str, Any],
    context: Dict[str, Any] = None,
    priority: Priority = Priority.MEDIUM
) -> MCPMessage:
    """Create a message from coordinator to planner"""
    return (MCPMessageBuilder()
            .set_message_type(MessageType.COORDINATOR_TO_PLANNER)
            .set_source("coordinator")
            .set_target("planner")
            .set_task_type(task_type)
            .set_priority(priority)
            .set_payload(payload)
            .set_context(context or {})
            .build())
