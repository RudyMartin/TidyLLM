"""
MCP Tool Executor

Tool execution framework for MCP layers.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime


class MCPToolExecutor:
    """Tool execution framework for MCP layers"""
    
    def __init__(self, tool_registry):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.tool_registry = tool_registry
        self.execution_history: List[Dict[str, Any]] = []

    def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with parameters"""
        tool_info = self.tool_registry.get_tool(tool_id)
        handler = self.tool_registry.get_tool_handler(tool_id)
        
        if not tool_info or not handler:
            self.logger.error(f"Tool {tool_id} not found or no handler registered")
            return {"success": False, "error": "Tool not found"}
        
        try:
            start_time = datetime.now()
            result = handler(parameters)
            end_time = datetime.now()
            
            execution_record = {
                "tool_id": tool_id,
                "parameters": parameters,
                "result": result,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_ms": (end_time - start_time).total_seconds() * 1000,
                "success": True
            }
            
            self.execution_history.append(execution_record)
            self.logger.info(f"Executed tool {tool_id} successfully")
            
            return {"success": True, "result": result}
            
        except Exception as e:
            execution_record = {
                "tool_id": tool_id,
                "parameters": parameters,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            
            self.execution_history.append(execution_record)
            self.logger.error(f"Error executing tool {tool_id}: {e}")
            
            return {"success": False, "error": str(e)}

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        total_executions = len(self.execution_history)
        successful_executions = len([e for e in self.execution_history if e["success"]])
        failed_executions = total_executions - successful_executions
        
        tool_usage = {}
        for record in self.execution_history:
            tool_id = record["tool_id"]
            tool_usage[tool_id] = tool_usage.get(tool_id, 0) + 1

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "tool_usage": tool_usage
        }
