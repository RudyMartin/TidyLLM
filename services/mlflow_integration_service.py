"""
MLflow Integration Service
==========================

Dedicated service for managing MLflow Gateway integration.
Provides clean separation of MLflow concerns from business logic.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("mlflow_integration")

# Try to import MLflow
try:
    import mlflow
    from mlflow.gateway import MlflowGatewayClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("MLflow not installed - service will operate in offline mode")


@dataclass
class MLflowConfig:
    """Configuration for MLflow Integration Service."""
    gateway_uri: str = "http://localhost:5000"
    timeout: int = 30
    retry_count: int = 3
    enable_caching: bool = True
    

class MLflowIntegrationService:
    """
    Manages MLflow Gateway integration for enterprise LLM access.
    
    This service handles:
    - MLflow Gateway client connection
    - Request routing to MLflow
    - Graceful fallback when MLflow unavailable
    - Connection health monitoring
    - Request/response transformation
    """
    
    def __init__(self, config: Optional[MLflowConfig] = None):
        """Initialize MLflow Integration Service."""
        self.config = config or MLflowConfig()
        self.client = None
        self.is_connected = False
        self.last_error = None
        
        # Try to initialize MLflow client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize MLflow Gateway client."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available - cannot initialize client")
            self.last_error = "MLflow library not installed"
            return
            
        try:
            self.client = MlflowGatewayClient(
                gateway_uri=self.config.gateway_uri
            )
            # Test connection
            self._test_connection()
            self.is_connected = True
            logger.info(f"✅ MLflow Gateway connected: {self.config.gateway_uri}")
        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            logger.warning(f"⚠️ MLflow Gateway unavailable: {e}")
            
    def _test_connection(self):
        """Test MLflow Gateway connection."""
        if self.client:
            try:
                # Try to list routes as a connection test
                routes = self.client.list_routes()
                logger.debug(f"MLflow routes available: {len(routes) if routes else 0}")
                return True
            except Exception as e:
                logger.debug(f"MLflow connection test failed: {e}")
                return False
        return False
        
    def query(self, route: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Query MLflow Gateway.
        
        Args:
            route: The MLflow route to query (e.g., "claude", "openai-corporate")
            data: Request data including prompt, max_tokens, temperature, etc.
            
        Returns:
            MLflow response dict or None if unavailable
        """
        if not self.is_connected or not self.client:
            logger.debug(f"MLflow not connected: {self.last_error}")
            return None
            
        try:
            response = self.client.query(
                route=route,
                data=data
            )
            return response
        except Exception as e:
            logger.error(f"MLflow query failed: {e}")
            self.last_error = str(e)
            # Try to reconnect for next request
            self._initialize_client()
            return None
            
    def list_routes(self) -> list:
        """
        List available MLflow Gateway routes.
        
        Returns:
            List of available routes or empty list if unavailable
        """
        if not self.is_connected or not self.client:
            return []
            
        try:
            routes = self.client.list_routes()
            return routes if routes else []
        except Exception as e:
            logger.error(f"Failed to list MLflow routes: {e}")
            return []
            
    def get_route_info(self, route: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific route.
        
        Args:
            route: Route name
            
        Returns:
            Route information or None if unavailable
        """
        if not self.is_connected or not self.client:
            return None
            
        try:
            info = self.client.get_route(route)
            return info
        except Exception as e:
            logger.error(f"Failed to get route info for {route}: {e}")
            return None
            
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MLflow integration.
        
        Returns:
            Health status dictionary
        """
        # Re-test connection
        connection_ok = self._test_connection() if MLFLOW_AVAILABLE else False
        
        return {
            "service": "mlflow_integration",
            "healthy": connection_ok,
            "mlflow_available": MLFLOW_AVAILABLE,
            "connected": self.is_connected,
            "gateway_uri": self.config.gateway_uri,
            "last_error": self.last_error,
            "routes_available": len(self.list_routes()) if connection_ok else 0
        }
        
    def get_status(self) -> str:
        """Get human-readable status."""
        if not MLFLOW_AVAILABLE:
            return "MLflow not installed"
        elif self.is_connected:
            return f"Connected to {self.config.gateway_uri}"
        else:
            return f"Disconnected: {self.last_error or 'Unknown error'}"
            
    def is_available(self) -> bool:
        """Check if service is available for use."""
        return MLFLOW_AVAILABLE and self.is_connected
        
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to MLflow Gateway.
        
        Returns:
            True if reconnection successful
        """
        logger.info("Attempting to reconnect to MLflow Gateway...")
        self._initialize_client()
        return self.is_connected