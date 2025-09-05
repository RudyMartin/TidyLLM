"""
MCP Protocol Communication

Protocol implementation for MCP message handling and communication.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from .message_schemas import (
    MCPMessageSchema, 
    MessageType, 
    MessagePriority,
    TaskRequestSchema,
    TaskResponseSchema,
    ContextUpdateSchema,
    ResultAggregationSchema
)


class MCPProtocol:
    """Protocol implementation for MCP message handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.message_types = {
            MessageType.TASK_REQUEST: {"required": ["task_id", "task_type", "task_data", "context"]},
            MessageType.TASK_RESPONSE: {"required": ["task_id", "result", "status"]},
            MessageType.STATUS_UPDATE: {"required": ["status", "details"]},
            MessageType.ERROR: {"required": ["error_type", "message"]},
            MessageType.HEARTBEAT: {"required": ["node_id", "timestamp"]},
            MessageType.CONTEXT_UPDATE: {"required": ["context_id", "context_data", "update_type"]},
            MessageType.RESULT_AGGREGATION: {"required": ["aggregation_id", "results", "final_result"]}
        }

    def create_message(self,
                      sender_id: str,
                      receiver_id: str,
                      message_type: MessageType,
                      payload: Dict[str, Any],
                      priority: MessagePriority = MessagePriority.NORMAL,
                      correlation_id: Optional[str] = None) -> MCPMessageSchema:
        """Create a new MCP message"""
        message_id = str(uuid.uuid4())
        
        return MCPMessageSchema(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id
        )

    def validate_message(self, message: MCPMessageSchema) -> bool:
        """Validate message format and required fields"""
        if message.message_type not in self.message_types:
            self.logger.error(f"Unknown message type: {message.message_type}")
            return False

        required_fields = self.message_types[message.message_type]["required"]
        for field in required_fields:
            if field not in message.payload:
                self.logger.error(f"Missing required field '{field}' in {message.message_type}")
                return False

        return True

    def serialize_message(self, message: MCPMessageSchema) -> str:
        """Serialize message to JSON string"""
        try:
            return json.dumps(message.to_dict())
        except Exception as e:
            self.logger.error(f"Failed to serialize message: {e}")
            raise ValueError(f"Message serialization failed: {e}")

    def deserialize_message(self, data: str) -> MCPMessageSchema:
        """Deserialize message from JSON string"""
        try:
            message_dict = json.loads(data)
            return MCPMessageSchema.from_dict(message_dict)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to deserialize message: {e}")
            raise ValueError(f"Invalid message format: {e}")

    def create_task_request(self,
                           sender_id: str,
                           receiver_id: str,
                           task_id: str,
                           task_type: str,
                           task_data: Dict[str, Any],
                           context: Dict[str, Any],
                           constraints: Optional[Dict[str, Any]] = None,
                           priority: MessagePriority = MessagePriority.NORMAL,
                           timeout: Optional[int] = None) -> MCPMessageSchema:
        """Create a task request message"""
        task_request = TaskRequestSchema(
            task_id=task_id,
            task_type=task_type,
            task_data=task_data,
            context=context,
            constraints=constraints,
            priority=priority,
            timeout=timeout
        )
        
        return self.create_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.TASK_REQUEST,
            payload=task_request.to_payload(),
            priority=priority
        )

    def create_task_response(self,
                            sender_id: str,
                            receiver_id: str,
                            task_id: str,
                            result: Dict[str, Any],
                            status: str,
                            error_message: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> MCPMessageSchema:
        """Create a task response message"""
        task_response = TaskResponseSchema(
            task_id=task_id,
            result=result,
            status=status,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        return self.create_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.TASK_RESPONSE,
            payload=task_response.to_payload()
        )

    def create_context_update(self,
                             sender_id: str,
                             receiver_id: str,
                             context_id: str,
                             context_data: Dict[str, Any],
                             update_type: str,
                             source_layer: str,
                             metadata: Optional[Dict[str, Any]] = None) -> MCPMessageSchema:
        """Create a context update message"""
        context_update = ContextUpdateSchema(
            context_id=context_id,
            context_data=context_data,
            update_type=update_type,
            source_layer=source_layer,
            metadata=metadata or {}
        )
        
        return self.create_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.CONTEXT_UPDATE,
            payload=context_update.to_payload()
        )

    def create_result_aggregation(self,
                                 sender_id: str,
                                 receiver_id: str,
                                 aggregation_id: str,
                                 results: List[Dict[str, Any]],
                                 aggregation_strategy: str,
                                 final_result: Dict[str, Any],
                                 quality_score: Optional[float] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> MCPMessageSchema:
        """Create a result aggregation message"""
        result_aggregation = ResultAggregationSchema(
            aggregation_id=aggregation_id,
            results=results,
            aggregation_strategy=aggregation_strategy,
            final_result=final_result,
            quality_score=quality_score,
            metadata=metadata or {}
        )
        
        return self.create_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.RESULT_AGGREGATION,
            payload=result_aggregation.to_payload()
        )

    def extract_task_request(self, message: MCPMessageSchema) -> TaskRequestSchema:
        """Extract task request from message"""
        if message.message_type != MessageType.TASK_REQUEST:
            raise ValueError(f"Message is not a task request: {message.message_type}")
        
        payload = message.payload
        return TaskRequestSchema(
            task_id=payload["task_id"],
            task_type=payload["task_type"],
            task_data=payload["task_data"],
            context=payload["context"],
            constraints=payload.get("constraints"),
            priority=MessagePriority(payload.get("priority", 1)),
            timeout=payload.get("timeout")
        )

    def extract_task_response(self, message: MCPMessageSchema) -> TaskResponseSchema:
        """Extract task response from message"""
        if message.message_type != MessageType.TASK_RESPONSE:
            raise ValueError(f"Message is not a task response: {message.message_type}")
        
        payload = message.payload
        return TaskResponseSchema(
            task_id=payload["task_id"],
            result=payload["result"],
            status=payload["status"],
            error_message=payload.get("error_message"),
            metadata=payload.get("metadata", {})
        )
