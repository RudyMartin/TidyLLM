"""
MCP Tool Registry

Tool registration and discovery for MCP layers.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime


class MCPToolRegistry:
    """Tool registration and discovery for MCP layers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_handlers: Dict[str, Callable] = {}

    def register_tool(self, tool_id: str, tool_info: Dict[str, Any], handler: Callable):
        """Register a tool"""
        self.tools[tool_id] = {
            **tool_info,
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }
        self.tool_handlers[tool_id] = handler
        self.logger.info(f"Registered tool: {tool_id}")

    def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool"""
        if tool_id in self.tools:
            del self.tools[tool_id]
            del self.tool_handlers[tool_id]
            self.logger.info(f"Unregistered tool: {tool_id}")
            return True
        return False

    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get tool information"""
        return self.tools.get(tool_id)

    def get_tool_handler(self, tool_id: str) -> Optional[Callable]:
        """Get tool handler"""
        return self.tool_handlers.get(tool_id)

    def list_tools(self, tool_type: Optional[str] = None) -> List[str]:
        """List available tools"""
        if tool_type:
            return [
                tool_id for tool_id, tool_info in self.tools.items()
                if tool_info.get("type") == tool_type
            ]
        return list(self.tools.keys())

    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get tool registry statistics"""
        tool_types = {}
        for tool_info in self.tools.values():
            tool_type = tool_info.get("type", "unknown")
            tool_types[tool_type] = tool_types.get(tool_type, 0) + 1

        return {
            "total_tools": len(self.tools),
            "tool_types": tool_types,
            "registered_tools": list(self.tools.keys())
        }
