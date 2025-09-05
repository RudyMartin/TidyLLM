"""
MCP Message Router

Message routing logic for MCP layer communication.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from ..protocol.message_schemas import MCPMessageSchema


class MCPMessageRouter:
    """Message routing logic for MCP layer communication"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.routes: Dict[str, str] = {}
        self.route_handlers: Dict[str, Callable] = {}
        self.routing_history: List[Dict[str, Any]] = []

    def register_route(self, message_type: str, target_layer: str):
        """Register a route for message type"""
        self.routes[message_type] = target_layer
        self.logger.info(f"Registered route: {message_type} -> {target_layer}")

    def register_handler(self, target_layer: str, handler: Callable):
        """Register a handler for target layer"""
        self.route_handlers[target_layer] = handler
        self.logger.info(f"Registered handler for layer: {target_layer}")

    def route_message(self, message: MCPMessageSchema) -> Optional[str]:
        """Route message to appropriate layer"""
        message_type = message.message_type.value
        target_layer = self.routes.get(message_type)
        
        if target_layer:
            self._record_routing(message, target_layer, "success")
            self.logger.debug(f"Routed message {message.message_id} to {target_layer}")
            return target_layer
        else:
            self._record_routing(message, None, "no_route")
            self.logger.warning(f"No route found for message type: {message_type}")
            return None

    def handle_message(self, message: MCPMessageSchema, target_layer: str) -> bool:
        """Handle message with registered handler"""
        handler = self.route_handlers.get(target_layer)
        
        if handler:
            try:
                handler(message)
                self.logger.debug(f"Handled message {message.message_id} with {target_layer} handler")
                return True
            except Exception as e:
                self.logger.error(f"Error handling message {message.message_id}: {e}")
                return False
        else:
            self.logger.warning(f"No handler registered for layer: {target_layer}")
            return False

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        route_counts = {}
        success_count = 0
        failure_count = 0
        
        for record in self.routing_history:
            route_type = record["message_type"]
            route_counts[route_type] = route_counts.get(route_type, 0) + 1
            
            if record["status"] == "success":
                success_count += 1
            else:
                failure_count += 1

        return {
            "total_routes": len(self.routing_history),
            "successful_routes": success_count,
            "failed_routes": failure_count,
            "route_type_counts": route_counts,
            "registered_routes": list(self.routes.keys()),
            "registered_handlers": list(self.route_handlers.keys())
        }

    def _record_routing(self, message: MCPMessageSchema, target_layer: Optional[str], status: str):
        """Record routing attempt"""
        self.routing_history.append({
            "message_id": message.message_id,
            "message_type": message.message_type.value,
            "target_layer": target_layer,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
