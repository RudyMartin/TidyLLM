"""
MCP Base Classes - Model Context Protocol Foundation

This module provides the foundational classes for implementing the MCP framework
in a hierarchical LLM architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import logging

from ..llm_manager import LLMManager

logger = logging.getLogger(__name__)

class MCPRole(Enum):
    """Roles in the MCP hierarchy"""
    PLANNER = "planner"
    COORDINATOR = "coordinator"
    WORKER = "worker"

@dataclass
class MCPContext:
    """Context data for MCP nodes"""
    user_request: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    parent_context: Optional['MCPContext'] = None
    context_id: str = field(default_factory=lambda: f"ctx_{datetime.now().timestamp()}")
    
    def enrich(self, additional_data: Dict[str, Any]) -> 'MCPContext':
        """Create enriched context with additional data"""
        new_data = {**self.session_data, **additional_data}
        return MCPContext(
            user_request=self.user_request,
            constraints=self.constraints,
            session_data=new_data,
            parent_context=self,
            context_id=f"{self.context_id}_enriched"
        )
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get complete context including parent contexts"""
        full_context = {
            "current": self.session_data,
            "constraints": self.constraints,
            "user_request": self.user_request
        }
        
        if self.parent_context:
            full_context["parent"] = self.parent_context.get_full_context()
        
        return full_context

@dataclass
class MCPProtocol:
    """Protocol definition for MCP communication"""
    message_format: str  # "json", "function_call", "dspy_signature"
    validation_schema: Dict[str, Any]
    error_handling: Dict[str, Any]
    retry_policy: Dict[str, Any]
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message against protocol schema"""
        try:
            # Basic validation - can be enhanced with JSON Schema
            required_fields = self.validation_schema.get("required", [])
            for field in required_fields:
                if field not in message:
                    logger.warning(f"Missing required field: {field}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

class MCPNode(ABC):
    """Base class for all MCP nodes (Planner, Coordinator, Worker)"""
    
    def __init__(self, role: MCPRole, model_config: Dict[str, Any]):
        self.role = role
        self.model_config = model_config
        self.protocol = self._define_protocol()
        self.llm_manager = LLMManager()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def _define_protocol(self) -> MCPProtocol:
        """Define the protocol for this node type"""
        pass
    
    @abstractmethod
    def process(self, context: MCPContext, input_data: Any) -> Any:
        """Process input according to MCP protocol"""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input against protocol schema"""
        try:
            if isinstance(input_data, dict):
                return self.protocol.validate_message(input_data)
            return True
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return False
    
    def handle_error(self, error: Exception, context: MCPContext) -> Dict[str, Any]:
        """Handle errors according to protocol"""
        error_config = self.protocol.error_handling
        retry_count = error_config.get("retry_count", 0)
        fallback_strategy = error_config.get("fallback_strategy", "simple_response")
        
        self.logger.error(f"Error in {self.role.value}: {error}")
        
        error_response = {
            "error": str(error),
            "error_type": type(error).__name__,
            "node_role": self.role.value,
            "context_id": context.context_id,
            "timestamp": datetime.now().isoformat()
        }
        
        if fallback_strategy == "simple_response":
            error_response["fallback_response"] = self._generate_fallback_response(context)
        elif fallback_strategy == "worker_fallback":
            error_response["fallback_response"] = self._worker_fallback(context)
        
        return error_response
    
    def _generate_fallback_response(self, context: MCPContext) -> str:
        """Generate a simple fallback response"""
        fallback_prompt = f"""
        Generate a simple response for the following request:
        
        Request: {context.user_request}
        
        Provide a basic, helpful response even if limited information is available.
        """
        
        try:
            return self.llm_manager.generate_response(fallback_prompt)
        except Exception as e:
            self.logger.error(f"Fallback response generation failed: {e}")
            return "I apologize, but I'm unable to process this request at the moment."
    
    def _worker_fallback(self, context: MCPContext) -> Dict[str, Any]:
        """Worker-specific fallback strategy"""
        return {
            "status": "fallback_activated",
            "message": "Using simplified processing due to error",
            "result": self._generate_fallback_response(context)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get node status information"""
        return {
            "role": self.role.value,
            "protocol": {
                "message_format": self.protocol.message_format,
                "retry_policy": self.protocol.retry_policy
            },
            "model_config": self.model_config,
            "status": "healthy"
        }
