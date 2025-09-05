"""
MCP Middleware

Core MCP middleware functionality.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime


class MCPMiddleware:
    """Core MCP middleware functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.middleware_chain: List[Callable] = []
        self.middleware_history: List[Dict[str, Any]] = []

    def add_middleware(self, middleware_func: Callable):
        """Add middleware to the chain"""
        self.middleware_chain.append(middleware_func)
        self.logger.info(f"Added middleware: {middleware_func.__name__}")

    def process_message(self, message: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process message through middleware chain"""
        current_message = message
        context = context or {}

        for middleware in self.middleware_chain:
            try:
                current_message = middleware(current_message, context)
                self._record_middleware_execution(middleware.__name__, "success")
            except Exception as e:
                self.logger.error(f"Error in middleware {middleware.__name__}: {e}")
                self._record_middleware_execution(middleware.__name__, "error", str(e))
                raise

        return current_message

    def _record_middleware_execution(self, middleware_name: str, status: str, error: Optional[str] = None):
        """Record middleware execution"""
        self.middleware_history.append({
            "middleware_name": middleware_name,
            "status": status,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_middleware_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        middleware_stats = {}
        total_executions = len(self.middleware_history)
        
        for record in self.middleware_history:
            middleware_name = record["middleware_name"]
            if middleware_name not in middleware_stats:
                middleware_stats[middleware_name] = {"success": 0, "error": 0}
            
            if record["status"] == "success":
                middleware_stats[middleware_name]["success"] += 1
            else:
                middleware_stats[middleware_name]["error"] += 1

        return {
            "total_executions": total_executions,
            "middleware_stats": middleware_stats,
            "middleware_chain": [m.__name__ for m in self.middleware_chain]
        }
