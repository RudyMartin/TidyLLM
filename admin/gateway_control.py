"""
Gateway Control Backend - MLFlow Gateway Administration

Backend administration for TidyLLM's enterprise gateway system.
Handles gateway configuration, monitoring, and control operations.
"""

import logging
import time
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelConfiguration:
    """Configuration for individual models"""
    name: str
    cost_per_1k_tokens: float
    performance_score: float
    max_tokens: int = 4096
    enabled: bool = True
    fallback_priority: int = 1


@dataclass
class GatewayStatus:
    """Gateway status information"""
    online: bool
    response_time_ms: float
    active_connections: int
    total_requests: int
    error_rate: float
    last_health_check: str
    uptime_seconds: int


@dataclass
class ModelStats:
    """Statistics for model usage"""
    model_id: str
    total_requests: int
    total_tokens: int
    total_cost: float
    average_response_time: float
    error_count: int
    last_used: str


class GatewayController:
    """Backend controller for TidyLLM gateway management"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self._gateway_config = None
        self._models = {}
        self._load_gateway_config()
    
    def _load_gateway_config(self):
        """Load gateway configuration"""
        if self.config_manager:
            self._gateway_config = self.config_manager.get_module_config("gateway")
        else:
            # Default configuration
            self._gateway_config = self._get_default_gateway_config()
        
        # Load model configurations
        self._load_model_configs()
    
    def _get_default_gateway_config(self) -> Dict[str, Any]:
        """Get default gateway configuration"""
        return {
            "enabled": True,
            "base_url": "http://localhost:5000",
            "model_routing_enabled": True,
            "model_routing_strategy": "cost_optimized",
            "fallback_model": "gpt-3.5-turbo",
            "rate_limiting_enabled": True,
            "requests_per_minute": 60,
            "audit_logging": True
        }
    
    def _load_model_configs(self):
        """Load model configurations"""
        default_models = {
            "gpt-4": ModelConfiguration(
                name="gpt-4",
                cost_per_1k_tokens=0.03,
                performance_score=9.5,
                max_tokens=8192
            ),
            "gpt-3.5-turbo": ModelConfiguration(
                name="gpt-3.5-turbo", 
                cost_per_1k_tokens=0.002,
                performance_score=8.0,
                max_tokens=4096
            ),
            "claude-3-sonnet": ModelConfiguration(
                name="claude-3-sonnet",
                cost_per_1k_tokens=0.015,
                performance_score=9.0,
                max_tokens=200000
            ),
            "claude-3-haiku": ModelConfiguration(
                name="claude-3-haiku",
                cost_per_1k_tokens=0.0025,
                performance_score=8.5,
                max_tokens=200000
            )
        }
        self._models = default_models
    
    def get_gateway_status(self) -> GatewayStatus:
        """Get current gateway status"""
        try:
            # Try to ping gateway
            if self._gateway_config.get("base_url"):
                response = requests.get(
                    f"{self._gateway_config['base_url']}/health",
                    timeout=5
                )
                online = response.status_code == 200
                response_time = response.elapsed.total_seconds() * 1000
            else:
                online = False
                response_time = 0
                
            return GatewayStatus(
                online=online,
                response_time_ms=response_time,
                active_connections=self._get_active_connections(),
                total_requests=self._get_total_requests(),
                error_rate=self._get_error_rate(),
                last_health_check=datetime.now().isoformat(),
                uptime_seconds=self._get_uptime()
            )
            
        except Exception as e:
            logger.error(f"Error getting gateway status: {e}")
            return GatewayStatus(
                online=False,
                response_time_ms=0,
                active_connections=0,
                total_requests=0,
                error_rate=0,
                last_health_check=datetime.now().isoformat(),
                uptime_seconds=0
            )
    
    def _get_active_connections(self) -> int:
        """Get number of active connections"""
        # Mock implementation - would connect to actual gateway
        return 5
    
    def _get_total_requests(self) -> int:
        """Get total number of requests"""
        # Mock implementation - would query gateway metrics
        return 1250
    
    def _get_error_rate(self) -> float:
        """Get current error rate"""
        # Mock implementation - would calculate from gateway logs
        return 0.02  # 2%
    
    def _get_uptime(self) -> int:
        """Get gateway uptime in seconds"""
        # Mock implementation - would get from gateway
        return 86400  # 1 day
    
    def get_model_configurations(self) -> Dict[str, ModelConfiguration]:
        """Get all model configurations"""
        return self._models
    
    def update_model_configuration(self, model_id: str, config: Dict[str, Any]) -> bool:
        """Update configuration for a specific model"""
        try:
            if model_id in self._models:
                model = self._models[model_id]
                for key, value in config.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating model config: {e}")
            return False
    
    def get_model_stats(self) -> List[ModelStats]:
        """Get usage statistics for all models"""
        # Mock implementation - would query actual usage data
        stats = []
        for model_id in self._models:
            stats.append(ModelStats(
                model_id=model_id,
                total_requests=100 + hash(model_id) % 500,
                total_tokens=50000 + hash(model_id) % 100000,
                total_cost=5.50 + (hash(model_id) % 100) / 10,
                average_response_time=1.2 + (hash(model_id) % 30) / 10,
                error_count=hash(model_id) % 5,
                last_used=(datetime.now() - timedelta(minutes=hash(model_id) % 60)).isoformat()
            ))
        return stats
    
    def enable_model(self, model_id: str) -> bool:
        """Enable a specific model"""
        return self.update_model_configuration(model_id, {"enabled": True})
    
    def disable_model(self, model_id: str) -> bool:
        """Disable a specific model"""
        return self.update_model_configuration(model_id, {"enabled": False})
    
    def set_routing_strategy(self, strategy: str) -> bool:
        """Set model routing strategy"""
        valid_strategies = ["cost_optimized", "performance", "balanced", "round_robin"]
        if strategy in valid_strategies:
            self._gateway_config["model_routing_strategy"] = strategy
            return True
        return False
    
    def update_rate_limits(self, requests_per_minute: int) -> bool:
        """Update rate limiting settings"""
        try:
            if requests_per_minute > 0:
                self._gateway_config["requests_per_minute"] = requests_per_minute
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating rate limits: {e}")
            return False
    
    def restart_gateway(self) -> bool:
        """Restart the gateway service"""
        try:
            # Mock implementation - would actually restart gateway
            logger.info("Gateway restart initiated")
            time.sleep(2)  # Simulate restart time
            return True
        except Exception as e:
            logger.error(f"Error restarting gateway: {e}")
            return False


class GatewayMonitor:
    """Gateway monitoring and alerting"""
    
    def __init__(self, controller: GatewayController):
        self.controller = controller
        self.alerts = []
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        status = self.controller.get_gateway_status()
        health_score = self._calculate_health_score(status)
        
        return {
            "status": status,
            "health_score": health_score,
            "alerts": self.get_active_alerts(),
            "recommendations": self._get_recommendations(status)
        }
    
    def _calculate_health_score(self, status: GatewayStatus) -> float:
        """Calculate overall health score (0-100)"""
        score = 100
        
        if not status.online:
            score -= 50
        if status.error_rate > 0.05:  # 5%
            score -= 20
        if status.response_time_ms > 2000:  # 2 seconds
            score -= 15
        if status.active_connections > 100:
            score -= 10
            
        return max(0, score)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        # Mock implementation - would check actual alerts
        return [
            {
                "level": "WARNING",
                "message": "Response time above threshold",
                "timestamp": datetime.now().isoformat(),
                "resolved": False
            }
        ]
    
    def _get_recommendations(self, status: GatewayStatus) -> List[str]:
        """Get performance recommendations"""
        recommendations = []
        
        if status.error_rate > 0.03:
            recommendations.append("Consider enabling additional fallback models")
        if status.response_time_ms > 1500:
            recommendations.append("Check model availability and network latency")
        if not status.online:
            recommendations.append("Gateway is offline - check service status")
            
        return recommendations


# Utility functions
def create_gateway_controller(config_manager=None) -> GatewayController:
    """Create gateway controller instance"""
    return GatewayController(config_manager)


def get_gateway_health() -> Dict[str, Any]:
    """Quick gateway health check"""
    controller = GatewayController()
    monitor = GatewayMonitor(controller)
    return monitor.check_health()