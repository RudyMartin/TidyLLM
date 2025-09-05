"""
MCP Result Validator

Result validation and quality checks.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class MCPResultValidator:
    """Result validation and quality checks"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self.validation_history: List[Dict[str, Any]] = []

    def validate_result(self, result: Dict[str, Any], validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate result against rules"""
        if not validation_rules:
            validation_rules = self._get_default_rules()

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat()
        }

        for field, rule in validation_rules.items():
            if field in result:
                field_result = self._validate_field(result[field], rule)
                if not field_result["valid"]:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(field_result["errors"])
                if field_result["warnings"]:
                    validation_result["warnings"].extend(field_result["warnings"])
            elif rule.get("required", False):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Required field '{field}' missing")

        self.validation_history.append(validation_result)
        return validation_result

    def _validate_field(self, value: Any, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a specific field"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Type validation
        if "type" in rule and not isinstance(value, rule["type"]):
            result["valid"] = False
            result["errors"].append(f"Expected type {rule['type']}, got {type(value)}")

        # Value validation
        if "min_value" in rule and value < rule["min_value"]:
            result["valid"] = False
            result["errors"].append(f"Value {value} is below minimum {rule['min_value']}")

        if "max_value" in rule and value > rule["max_value"]:
            result["valid"] = False
            result["errors"].append(f"Value {value} is above maximum {rule['max_value']}")

        return result

    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default validation rules"""
        return {
            "result": {"required": True, "type": dict},
            "status": {"required": True, "type": str},
            "timestamp": {"required": False, "type": str}
        }

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
