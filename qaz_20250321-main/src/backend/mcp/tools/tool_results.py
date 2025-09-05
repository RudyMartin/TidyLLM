"""
MCP Tool Results

Tool result handling and processing.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class ToolResults:
    """Tool result handling and processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.results_history: List[Dict[str, Any]] = []

    def process_result(self, tool_id: str, result: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and store tool result"""
        processed_result = {
            "tool_id": tool_id,
            "result": result,
            "metadata": metadata or {},
            "processed_at": datetime.now().isoformat(),
            "result_type": type(result).__name__
        }
        
        self.results_history.append(processed_result)
        self.logger.debug(f"Processed result for tool {tool_id}")
        
        return processed_result

    def get_result_statistics(self) -> Dict[str, Any]:
        """Get tool result statistics"""
        result_types = {}
        tool_results = {}
        
        for record in self.results_history:
            result_type = record["result_type"]
            result_types[result_type] = result_types.get(result_type, 0) + 1
            
            tool_id = record["tool_id"]
            tool_results[tool_id] = tool_results.get(tool_id, 0) + 1

        return {
            "total_results": len(self.results_history),
            "result_types": result_types,
            "tool_results": tool_results
        }
