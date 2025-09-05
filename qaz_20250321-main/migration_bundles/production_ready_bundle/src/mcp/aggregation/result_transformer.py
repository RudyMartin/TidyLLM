"""
MCP Result Transformer

Result transformation and formatting.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class MCPResultTransformer:
    """Result transformation and formatting"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.transformation_rules: Dict[str, callable] = {}
        self.transformation_history: List[Dict[str, Any]] = []

    def register_transformation(self, output_format: str, transform_func: callable):
        """Register a transformation rule"""
        self.transformation_rules[output_format] = transform_func
        self.logger.info(f"Registered transformation for format: {output_format}")

    def transform_result(self, result: Dict[str, Any], output_format: str) -> Dict[str, Any]:
        """Transform result to specified format"""
        transform_func = self.transformation_rules.get(output_format)
        
        if not transform_func:
            self.logger.warning(f"No transformation rule for format '{output_format}'")
            return result
        
        try:
            transformed_result = transform_func(result)
            
            transformation_record = {
                "output_format": output_format,
                "original_result": result,
                "transformed_result": transformed_result,
                "timestamp": datetime.now().isoformat()
            }
            
            self.transformation_history.append(transformation_record)
            self.logger.debug(f"Transformed result to format '{output_format}'")
            
            return transformed_result
            
        except Exception as e:
            self.logger.error(f"Error transforming result to format '{output_format}': {e}")
            return result

    def get_transformation_statistics(self) -> Dict[str, Any]:
        """Get transformation statistics"""
        format_usage = {}
        total_transformations = len(self.transformation_history)
        
        for record in self.transformation_history:
            format_name = record["output_format"]
            format_usage[format_name] = format_usage.get(format_name, 0) + 1

        return {
            "total_transformations": total_transformations,
            "format_usage": format_usage,
            "registered_formats": list(self.transformation_rules.keys())
        }
