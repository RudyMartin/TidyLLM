"""
MCP Context Validator

Validates context data and structure.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .context_manager import MCPContext


class ContextValidator:
    """Validates context data and structure"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self.validation_history: List[Dict[str, Any]] = []

    def validate_context(self, context: MCPContext, validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate context against rules"""
        if not validation_rules:
            validation_rules = self._get_default_rules(context.source_layer)

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "context_id": context.context_id,
            "validated_at": datetime.now().isoformat()
        }

        for field, rule in validation_rules.items():
            field_result = self._validate_field(context, field, rule)
            if not field_result["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(field_result["errors"])
            if field_result["warnings"]:
                validation_result["warnings"].extend(field_result["warnings"])

        # Record validation
        self.validation_history.append(validation_result)
        
        return validation_result

    def _validate_field(self, context: MCPContext, field: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a specific field"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        if field not in context.context_data:
            if rule.get("required", False):
                result["valid"] = False
                result["errors"].append(f"Required field '{field}' missing")
            return result

        value = context.context_data[field]
        
        # Type validation
        if "type" in rule:
            expected_type = rule["type"]
            if not isinstance(value, expected_type):
                result["valid"] = False
                result["errors"].append(f"Field '{field}' should be {expected_type}, got {type(value)}")

        # Value validation
        if "min_value" in rule and value < rule["min_value"]:
            result["valid"] = False
            result["errors"].append(f"Field '{field}' value {value} is below minimum {rule['min_value']}")

        if "max_value" in rule and value > rule["max_value"]:
            result["valid"] = False
            result["errors"].append(f"Field '{field}' value {value} is above maximum {rule['max_value']}")

        # Pattern validation
        if "pattern" in rule and isinstance(value, str):
            import re
            if not re.match(rule["pattern"], value):
                result["valid"] = False
                result["errors"].append(f"Field '{field}' does not match pattern {rule['pattern']}")

        # Custom validation
        if "custom_validator" in rule:
            try:
                validator_func = rule["custom_validator"]
                if not validator_func(value):
                    result["valid"] = False
                    result["errors"].append(f"Field '{field}' failed custom validation")
            except Exception as e:
                result["warnings"].append(f"Custom validation for '{field}' failed: {e}")

        return result

    def _get_default_rules(self, source_layer: str) -> Dict[str, Any]:
        """Get default validation rules for source layer"""
        default_rules = {
            "orchestrator": {
                "workflow_id": {"required": True, "type": str},
                "priority": {"required": False, "type": str, "pattern": r"^(low|normal|high|critical)$"},
                "constraints": {"required": False, "type": dict}
            },
            "coordinator": {
                "task_id": {"required": True, "type": str},
                "task_type": {"required": True, "type": str},
                "worker_capabilities": {"required": False, "type": dict}
            },
            "worker": {
                "execution_id": {"required": True, "type": str},
                "input_data": {"required": True, "type": dict},
                "parameters": {"required": False, "type": dict}
            },
            "tool": {
                "tool_id": {"required": True, "type": str},
                "tool_config": {"required": False, "type": dict},
                "input_format": {"required": False, "type": str}
            }
        }
        return default_rules.get(source_layer, {})

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total_validations = len(self.validation_history)
        successful_validations = len([v for v in self.validation_history if v["valid"]])
        failed_validations = total_validations - successful_validations
        
        error_counts = {}
        warning_counts = {}
        
        for validation in self.validation_history:
            for error in validation["errors"]:
                error_type = error.split(":")[0] if ":" in error else "unknown"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for warning in validation["warnings"]:
                warning_type = warning.split(":")[0] if ":" in warning else "unknown"
                warning_counts[warning_type] = warning_counts.get(warning_type, 0) + 1

        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
            "error_counts": error_counts,
            "warning_counts": warning_counts
        }
