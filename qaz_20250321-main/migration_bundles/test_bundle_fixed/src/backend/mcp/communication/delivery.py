"""
MCP Message Delivery

Message delivery and acknowledgment handling.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from ..protocol.message_schemas import MCPMessageSchema


class MessageDelivery:
    """Message delivery and acknowledgment handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.delivery_handlers: Dict[str, Callable] = {}
        self.delivery_history: List[Dict[str, Any]] = []

    def register_delivery_handler(self, target_layer: str, handler: Callable):
        """Register a delivery handler"""
        self.delivery_handlers[target_layer] = handler
        self.logger.info(f"Registered delivery handler for: {target_layer}")

    def deliver_message(self, message: MCPMessageSchema, target_layer: str) -> bool:
        """Deliver message to target layer"""
        handler = self.delivery_handlers.get(target_layer)
        
        if handler:
            try:
                success = handler(message)
                self._record_delivery(message, target_layer, "success" if success else "failed")
                return success
            except Exception as e:
                self.logger.error(f"Error delivering message {message.message_id}: {e}")
                self._record_delivery(message, target_layer, "error")
                return False
        else:
            self.logger.warning(f"No delivery handler registered for layer: {target_layer}")
            self._record_delivery(message, target_layer, "no_handler")
            return False

    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get delivery statistics"""
        delivery_counts = {}
        success_count = 0
        failure_count = 0
        
        for record in self.delivery_history:
            layer = record["target_layer"]
            delivery_counts[layer] = delivery_counts.get(layer, 0) + 1
            
            if record["status"] == "success":
                success_count += 1
            else:
                failure_count += 1

        return {
            "total_deliveries": len(self.delivery_history),
            "successful_deliveries": success_count,
            "failed_deliveries": failure_count,
            "delivery_counts_by_layer": delivery_counts,
            "registered_handlers": list(self.delivery_handlers.keys())
        }

    def _record_delivery(self, message: MCPMessageSchema, target_layer: str, status: str):
        """Record delivery attempt"""
        self.delivery_history.append({
            "message_id": message.message_id,
            "target_layer": target_layer,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
