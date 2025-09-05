"""
MCP Context Enricher

Enriches context as it flows down the MCP hierarchy.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .context_manager import MCPContext


class ContextEnricher:
    """Enriches context as it flows down the MCP hierarchy"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.enrichment_strategies: Dict[str, callable] = {}
        self.enrichment_history: List[Dict[str, Any]] = []

    def register_enrichment_strategy(self, strategy_name: str, strategy_func: callable):
        """Register an enrichment strategy"""
        self.enrichment_strategies[strategy_name] = strategy_func
        self.logger.info(f"Registered enrichment strategy: {strategy_name}")

    def enrich_context(self, 
                      context: MCPContext, 
                      target_layer: str,
                      enrichment_strategies: Optional[List[str]] = None) -> MCPContext:
        """Enrich context for target layer"""
        if not enrichment_strategies:
            enrichment_strategies = self._get_default_strategies(target_layer)

        enriched_data = context.context_data.copy()
        enrichment_metadata = {
            "enriched_at": datetime.now().isoformat(),
            "target_layer": target_layer,
            "strategies_used": enrichment_strategies
        }

        for strategy_name in enrichment_strategies:
            if strategy_name in self.enrichment_strategies:
                try:
                    strategy_result = self.enrichment_strategies[strategy_name](
                        context, target_layer, enriched_data
                    )
                    if strategy_result:
                        enriched_data.update(strategy_result)
                        self.logger.debug(f"Applied enrichment strategy: {strategy_name}")
                except Exception as e:
                    self.logger.error(f"Failed to apply enrichment strategy {strategy_name}: {e}")

        # Update context with enriched data
        context.update(enriched_data, enrichment_metadata)
        
        # Record enrichment
        self.enrichment_history.append({
            "context_id": context.context_id,
            "target_layer": target_layer,
            "strategies_used": enrichment_strategies,
            "timestamp": datetime.now().isoformat()
        })

        return context

    def _get_default_strategies(self, target_layer: str) -> List[str]:
        """Get default enrichment strategies for target layer"""
        default_strategies = {
            "orchestrator": ["add_workflow_context", "add_resource_context"],
            "coordinator": ["add_task_context", "add_worker_context"],
            "worker": ["add_execution_context", "add_tool_context"],
            "tool": ["add_tool_specific_context"]
        }
        return default_strategies.get(target_layer, [])

    def add_workflow_context(self, context: MCPContext, target_layer: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add workflow-specific context"""
        return {
            "workflow_id": current_data.get("workflow_id", "default"),
            "workflow_stage": "orchestration",
            "workflow_priority": current_data.get("priority", "normal")
        }

    def add_resource_context(self, context: MCPContext, target_layer: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add resource allocation context"""
        return {
            "available_resources": current_data.get("available_resources", {}),
            "resource_constraints": current_data.get("resource_constraints", {}),
            "resource_allocation_strategy": "dynamic"
        }

    def add_task_context(self, context: MCPContext, target_layer: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add task-specific context"""
        return {
            "task_decomposition_level": "coordinator",
            "task_dependencies": current_data.get("task_dependencies", []),
            "task_priority": current_data.get("task_priority", "normal")
        }

    def add_worker_context(self, context: MCPContext, target_layer: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add worker assignment context"""
        return {
            "available_workers": current_data.get("available_workers", []),
            "worker_capabilities": current_data.get("worker_capabilities", {}),
            "load_balancing_strategy": "round_robin"
        }

    def add_execution_context(self, context: MCPContext, target_layer: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add execution-specific context"""
        return {
            "execution_mode": "synchronous",
            "timeout_settings": current_data.get("timeout", 300),
            "retry_policy": current_data.get("retry_policy", {"max_retries": 3})
        }

    def add_tool_context(self, context: MCPContext, target_layer: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add tool-specific context"""
        return {
            "tool_configuration": current_data.get("tool_config", {}),
            "tool_parameters": current_data.get("parameters", {}),
            "tool_version": current_data.get("version", "latest")
        }

    def add_tool_specific_context(self, context: MCPContext, target_layer: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add tool-specific context for tool layer"""
        return {
            "tool_type": current_data.get("tool_type", "unknown"),
            "tool_input_format": current_data.get("input_format", "text"),
            "tool_output_format": current_data.get("output_format", "json")
        }

    def get_enrichment_statistics(self) -> Dict[str, Any]:
        """Get enrichment statistics"""
        strategy_usage = {}
        layer_usage = {}
        
        for record in self.enrichment_history:
            # Count strategy usage
            for strategy in record["strategies_used"]:
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            # Count layer usage
            layer = record["target_layer"]
            layer_usage[layer] = layer_usage.get(layer, 0) + 1

        return {
            "total_enrichments": len(self.enrichment_history),
            "strategy_usage": strategy_usage,
            "layer_usage": layer_usage,
            "registered_strategies": list(self.enrichment_strategies.keys())
        }
