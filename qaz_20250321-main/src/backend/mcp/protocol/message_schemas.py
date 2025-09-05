"""
MCP Message Schemas

Standardized message schemas for MCP communication between layers.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class MessageType(Enum):
    """Types of MCP messages"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CONTEXT_UPDATE = "context_update"
    RESULT_AGGREGATION = "result_aggregation"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MCPMessageSchema:
    """Base schema for MCP messages"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessageSchema':
        """Create message from dictionary format"""
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType(data["message_type"]),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=MessagePriority(data["priority"]),
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {}),
            correlation_id=data.get("correlation_id")
        )


@dataclass
class TaskRequestSchema:
    """Schema for task request messages"""
    task_id: str
    task_type: str
    task_data: Dict[str, Any]
    context: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    priority: MessagePriority = MessagePriority.NORMAL
    timeout: Optional[int] = None

    def to_payload(self) -> Dict[str, Any]:
        """Convert to message payload"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_data": self.task_data,
            "context": self.context,
            "constraints": self.constraints,
            "priority": self.priority.value,
            "timeout": self.timeout
        }


@dataclass
class TaskResponseSchema:
    """Schema for task response messages"""
    task_id: str
    result: Dict[str, Any]
    status: str  # "success", "error", "partial"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Convert to message payload"""
        return {
            "task_id": self.task_id,
            "result": self.result,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class ContextUpdateSchema:
    """Schema for context update messages"""
    context_id: str
    context_data: Dict[str, Any]
    update_type: str  # "add", "modify", "remove"
    source_layer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Convert to message payload"""
        return {
            "context_id": self.context_id,
            "context_data": self.context_data,
            "update_type": self.update_type,
            "source_layer": self.source_layer,
            "metadata": self.metadata
        }


@dataclass
class ResultAggregationSchema:
    """Schema for result aggregation messages"""
    aggregation_id: str
    results: List[Dict[str, Any]]
    aggregation_strategy: str
    final_result: Dict[str, Any]
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Convert to message payload"""
        return {
            "aggregation_id": self.aggregation_id,
            "results": self.results,
            "aggregation_strategy": self.aggregation_strategy,
            "final_result": self.final_result,
            "quality_score": self.quality_score,
            "metadata": self.metadata
        }
