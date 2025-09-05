"""
MCP Layer Middleware

Layer-specific middleware functionality.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime


class LayerMiddleware:
    """Layer-specific middleware functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.layer_handlers: Dict[str, Callable] = {}
        self.layer_history: List[Dict[str, Any]] = []

    def register_layer_handler(self, layer_name: str, handler: Callable):
        """Register a layer handler"""
        self.layer_handlers[layer_name] = handler
        self.logger.info(f"Registered layer handler for: {layer_name}")

    def process_layer_message(self, layer_name: str, message: Any) -> Any:
        """Process message for specific layer"""
        handler = self.layer_handlers.get(layer_name)
        
        if not handler:
            self.logger.warning(f"No handler registered for layer: {layer_name}")
            return message
        
        try:
            result = handler(message)
            self._record_layer_execution(layer_name, "success")
            return result
        except Exception as e:
            self.logger.error(f"Error in layer {layer_name}: {e}")
            self._record_layer_execution(layer_name, "error", str(e))
            raise

    def _record_layer_execution(self, layer_name: str, status: str, error: Optional[str] = None):
        """Record layer execution"""
        self.layer_history.append({
            "layer_name": layer_name,
            "status": status,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_layer_statistics(self) -> Dict[str, Any]:
        """Get layer statistics"""
        layer_stats = {}
        total_executions = len(self.layer_history)
        
        for record in self.layer_history:
            layer_name = record["layer_name"]
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {"success": 0, "error": 0}
            
            if record["status"] == "success":
                layer_stats[layer_name]["success"] += 1
            else:
                layer_stats[layer_name]["error"] += 1

        return {
            "total_executions": total_executions,
            "layer_stats": layer_stats,
            "registered_layers": list(self.layer_handlers.keys())
        }
