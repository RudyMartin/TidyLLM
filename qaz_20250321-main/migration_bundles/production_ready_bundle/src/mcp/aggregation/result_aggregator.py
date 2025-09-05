"""
MCP Result Aggregator

Result aggregation logic for MCP layers.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class MCPResultAggregator:
    """Result aggregation logic for MCP layers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.aggregation_strategies: Dict[str, callable] = {}
        self.aggregation_history: List[Dict[str, Any]] = []

    def register_aggregation_strategy(self, strategy_name: str, strategy_func: callable):
        """Register an aggregation strategy"""
        self.aggregation_strategies[strategy_name] = strategy_func
        self.logger.info(f"Registered aggregation strategy: {strategy_name}")

    def aggregate_results(self, results: List[Dict[str, Any]], strategy: str = "default") -> Dict[str, Any]:
        """Aggregate results using specified strategy"""
        strategy_func = self.aggregation_strategies.get(strategy)
        
        if not strategy_func:
            self.logger.warning(f"No aggregation strategy '{strategy}' found, using default")
            strategy_func = self._default_aggregation
        
        try:
            aggregated_result = strategy_func(results)
            
            aggregation_record = {
                "strategy": strategy,
                "input_count": len(results),
                "result": aggregated_result,
                "timestamp": datetime.now().isoformat()
            }
            
            self.aggregation_history.append(aggregation_record)
            self.logger.info(f"Aggregated {len(results)} results using strategy '{strategy}'")
            
            return aggregated_result
            
        except Exception as e:
            self.logger.error(f"Error aggregating results with strategy '{strategy}': {e}")
            return {"error": str(e)}

    def _default_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default aggregation strategy"""
        if not results:
            return {"aggregated_result": None, "count": 0}
        
        # Simple merge strategy
        aggregated = {}
        for result in results:
            aggregated.update(result)
        
        return {
            "aggregated_result": aggregated,
            "count": len(results),
            "strategy": "default"
        }

    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        strategy_usage = {}
        total_aggregations = len(self.aggregation_history)
        
        for record in self.aggregation_history:
            strategy = record["strategy"]
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

        return {
            "total_aggregations": total_aggregations,
            "strategy_usage": strategy_usage,
            "registered_strategies": list(self.aggregation_strategies.keys())
        }
