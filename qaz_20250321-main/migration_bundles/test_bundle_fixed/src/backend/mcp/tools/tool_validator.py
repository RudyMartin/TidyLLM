"""
MCP Tool Validator

Tool validation and verification.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class ToolValidator:
    """Tool validation and verification"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self.validation_history: List[Dict[str, Any]] = []

    def validate_tool(self, tool_id: str, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool information"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "tool_id": tool_id,
            "validated_at": datetime.now().isoformat()
        }

        # Check required fields
        required_fields = ["name", "type", "version"]
        for field in required_fields:
            if field not in tool_info:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Required field '{field}' missing")

        # Check tool type
        valid_types = ["embedding", "analysis", "generation", "validation", "utility"]
        if "type" in tool_info and tool_info["type"] not in valid_types:
            validation_result["warnings"].append(f"Tool type '{tool_info['type']}' not in standard types")

        self.validation_history.append(validation_result)
        return validation_result

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total_validations = len(self.validation_history)
        successful_validations = len([v for v in self.validation_history if v["valid"]])
        failed_validations = total_validations - successful_validations

        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0
        }
