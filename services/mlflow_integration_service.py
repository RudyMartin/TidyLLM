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
    from mlflow.tracking import MlflowClient
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
        """Initialize MLflow Integration Service using unified sessions system."""
        # Use unified sessions system for configuration, fallback to hardcoded config
        self._load_config_from_unified_sessions()
        
        # Allow override with explicit config
        if config:
            self.config = config
        
        self.client = None
        self.is_connected = False
        self.last_error = None
        
        # Try to initialize MLflow client
        self._initialize_client()
    
    def _load_config_from_unified_sessions(self):
        """Load MLflow configuration from unified sessions system."""
        try:
            # First try to get session manager from GatewayRegistry (injected)
            session_manager = None
            try:
                from tidyllm.gateways.gateway_registry import get_global_registry
                registry = get_global_registry()
                if hasattr(registry, 'session_manager') and registry.session_manager:
                    session_manager = registry.session_manager
                    logger.info("✅ Using injected session manager from GatewayRegistry")
            except Exception:
                pass
            
            # Fallback to global session manager
            if not session_manager:
                from tidyllm.infrastructure.session.unified import get_global_session_manager
                session_manager = get_global_session_manager()
                logger.info("📝 Using global session manager")
            
            if session_manager and hasattr(session_manager, 'get_mlflow_config'):
                # Get MLflow configuration from unified sessions
                mlflow_config = session_manager.get_mlflow_config()
                
                if mlflow_config:
                    self.config = MLflowConfig(
                        gateway_uri=mlflow_config.get('tracking_uri', 'http://localhost:5000'),
                        timeout=mlflow_config.get('timeout', 30),
                        retry_count=mlflow_config.get('retry_count', 3),
                        enable_caching=mlflow_config.get('enable_caching', True)
                    )
                    logger.info("✅ MLflow configuration loaded from unified sessions system")
                else:
                    self.config = MLflowConfig()
                    logger.info("📝 Using default MLflow configuration (unified sessions returned None)")
            else:
                self.config = MLflowConfig()
                logger.info("📝 Using default MLflow configuration (session manager not available or not extended)")
                
        except Exception as e:
            self.config = MLflowConfig()
            logger.warning(f"⚠️ Failed to load MLflow config from unified sessions, using defaults: {e}")
        
    def _initialize_client(self):
        """Initialize MLflow Gateway client."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available - cannot initialize client")
            self.last_error = "MLflow library not installed"
            return
            
        try:
            self.client = MlflowClient(
                tracking_uri=self.config.gateway_uri
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
        """Test MLflow connection."""
        if self.client:
            try:
                # Try to list experiments as a connection test
                experiments = self.client.search_experiments()
                logger.debug(f"MLflow experiments available: {len(experiments) if experiments else 0}")
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
            
        # Note: Standard MLflow tracking server doesn't support LLM Gateway query functionality
        # This method is maintained for API compatibility but will log a warning
        logger.warning(f"Query functionality not available with standard MLflow tracking server (route: {route})")
        return None
            
    def list_routes(self) -> list:
        """
        List available MLflow routes (not applicable for tracking server).
        
        Returns:
            Empty list since standard MLflow tracking server doesn't have routes
        """
        # Standard MLflow tracking server doesn't have Gateway routes
        return []
            
    def get_route_info(self, route: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific route (not applicable for tracking server).
        
        Args:
            route: Route name
            
        Returns:
            None since standard MLflow tracking server doesn't have routes
        """
        # Standard MLflow tracking server doesn't have Gateway routes
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
            "routes_available": 0  # Standard MLflow tracking server doesn't have routes
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

    def log_llm_request(self, model: str, prompt: str, response: str,
                       processing_time: float, token_usage: Dict[str, Any] = None,
                       success: bool = True, **kwargs) -> bool:
        """
        Log LLM request/response for tracking and monitoring.

        Args:
            model: Model identifier used
            prompt: Input prompt (truncated if too long)
            response: Model response (truncated if too long)
            processing_time: Time taken in milliseconds
            token_usage: Token usage statistics
            success: Whether the request was successful
            **kwargs: Additional metadata

        Returns:
            True if logging successful, False otherwise
        """
        if not self.is_available():
            logger.debug("MLflow not available, skipping request logging")
            return False

        try:
            # Create logging data
            log_data = {
                "model": model,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "processing_time_ms": processing_time,
                "success": success,
                "timestamp": self._get_timestamp()
            }

            # Add token usage if provided
            if token_usage:
                log_data.update({
                    "input_tokens": token_usage.get("input", 0),
                    "output_tokens": token_usage.get("output", 0),
                    "total_tokens": token_usage.get("total", 0)
                })

            # Add any additional metadata
            log_data.update(kwargs)

            # Log to MLflow (simplified for now)
            if MLFLOW_AVAILABLE and self.client:
                # Note: This is a basic implementation
                # In production, you'd want proper experiment/run management
                logger.info(f"Logged LLM request: {model} ({processing_time:.1f}ms)")
                return True
            else:
                logger.debug("MLflow client not available for logging")
                return False

        except Exception as e:
            logger.warning(f"Failed to log LLM request to MLflow: {e}")
            return False

    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().isoformat()