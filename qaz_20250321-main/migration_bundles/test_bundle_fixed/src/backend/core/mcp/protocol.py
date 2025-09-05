"""
MCP Message Protocol and Communication Framework

This module implements the communication protocol for the Model Context Protocol (MCP)
hierarchical LLM system. It provides message serialization, validation, and queuing
capabilities for inter-node communication within the MCP hierarchy.

The protocol ensures standardized message formats, validation, and error handling
across all MCP nodes (Planner, Coordinators, Workers) for reliable communication.

TODO - Add protocol versioning and backward compatibility
TODO - Add message encryption and security
TODO - Add protocol performance monitoring
TODO - Add message compression for large payloads
TODO - Add protocol debugging and tracing capabilities
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import uuid
import logging

logger = logging.getLogger(__name__)

@dataclass
class MCPMessage:
    """Standardized message format for MCP communication"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 0
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "retry_count": self.retry_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary format"""
        return cls(
            message_id=data.get("message_id", ""),
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=data["message_type"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=data.get("priority", 0),
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {})
        )

class MCPMessageProtocol:
    """Protocol handler for MCP message communication"""
    
    def __init__(self):
        self.message_types = {
            "task_request": {"required": ["task", "context"]},
            "task_response": {"required": ["result", "status"]},
            "error": {"required": ["error_type", "message"]},
            "status_update": {"required": ["status", "details"]},
            "heartbeat": {"required": ["node_id", "timestamp"]}
        }
    
    def create_message(self, 
                      sender_id: str, 
                      receiver_id: str, 
                      message_type: str, 
                      payload: Dict[str, Any],
                      priority: int = 0) -> MCPMessage:
        """Create a new MCP message"""
        return MCPMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            priority=priority
        )
    
    def validate_message(self, message: MCPMessage) -> bool:
        """Validate message format and required fields"""
        if message.message_type not in self.message_types:
            logger.error(f"Unknown message type: {message.message_type}")
            return False
        
        required_fields = self.message_types[message.message_type]["required"]
        for field in required_fields:
            if field not in message.payload:
                logger.error(f"Missing required field '{field}' in {message.message_type}")
                return False
        
        return True
    
    def serialize_message(self, message: MCPMessage) -> str:
        """Serialize message to JSON string"""
        return json.dumps(message.to_dict())
    
    def deserialize_message(self, data: str) -> MCPMessage:
        """Deserialize message from JSON string"""
        try:
            message_dict = json.loads(data)
            return MCPMessage.from_dict(message_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize message: {e}")
            raise ValueError(f"Invalid message format: {e}")

class MCPMessageQueue:
    """Message queue for MCP communication"""
    
    def __init__(self, max_size: int = 1000):
        self.queue: List[MCPMessage] = []
        self.max_size = max_size
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def enqueue(self, message: MCPMessage) -> bool:
        """Add message to queue"""
        if len(self.queue) >= self.max_size:
            self.logger.warning("Message queue is full, dropping oldest message")
            self.queue.pop(0)
        
        self.queue.append(message)
        self.queue.sort(key=lambda x: x.priority, reverse=True)
        return True
    
    def dequeue(self) -> Optional[MCPMessage]:
        """Remove and return next message from queue"""
        if not self.queue:
            return None
        return self.queue.pop(0)
    
    def peek(self) -> Optional[MCPMessage]:
        """View next message without removing it"""
        return self.queue[0] if self.queue else None
    
    def size(self) -> int:
        """Get current queue size"""
        return len(self.queue)
    
    def clear(self):
        """Clear all messages from queue"""
        self.queue.clear()
