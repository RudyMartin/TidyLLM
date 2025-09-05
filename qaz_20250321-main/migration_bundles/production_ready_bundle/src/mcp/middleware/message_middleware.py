"""
MCP Message Middleware

Message-specific middleware functionality.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime


class MessageMiddleware:
    """Message-specific middleware functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.message_handlers: Dict[str, Callable] = {}
        self.message_history: List[Dict[str, Any]] = []

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered message handler for: {message_type}")

    def process_message(self, message_type: str, message: Any) -> Any:
        """Process message of specific type"""
        handler = self.message_handlers.get(message_type)
        
        if not handler:
            self.logger.warning(f"No handler registered for message type: {message_type}")
            return message
        
        try:
            result = handler(message)
            self._record_message_execution(message_type, "success")
            return result
        except Exception as e:
            self.logger.error(f"Error processing message type {message_type}: {e}")
            self._record_message_execution(message_type, "error", str(e))
            raise

    def _record_message_execution(self, message_type: str, status: str, error: Optional[str] = None):
        """Record message execution"""
        self.message_history.append({
            "message_type": message_type,
            "status": status,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_message_statistics(self) -> Dict[str, Any]:
        """Get message statistics"""
        message_stats = {}
        total_executions = len(self.message_history)
        
        for record in self.message_history:
            message_type = record["message_type"]
            if message_type not in message_stats:
                message_stats[message_type] = {"success": 0, "error": 0}
            
            if record["status"] == "success":
                message_stats[message_type]["success"] += 1
            else:
                message_stats[message_type]["error"] += 1

        return {
            "total_executions": total_executions,
            "message_stats": message_stats,
            "registered_message_types": list(self.message_handlers.keys())
        }
