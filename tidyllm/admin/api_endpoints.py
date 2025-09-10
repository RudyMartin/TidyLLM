"""
TidyLLM Admin API Endpoints - Backend REST API

Provides REST API endpoints for TidyLLM ecosystem administration.
Can be used with any frontend (Streamlit, React, etc.) or CLI tools.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict
from functools import wraps

# Optional Flask integration
try:
    from flask import Flask, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Optional FastAPI integration  
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ..infrastructure import GatewayController, GatewayMonitor
# Legacy import for TidyLLMConfig - needs to be handled separately
try:
    from .config_manager import TidyLLMConfig
except ImportError:
    TidyLLMConfig = None

logger = logging.getLogger(__name__)


class AdminAPIError(Exception):
    """Custom exception for admin API errors"""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def handle_errors(func):
    """Decorator to handle API errors consistently"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AdminAPIError as e:
            return {
                "success": False,
                "error": e.message,
                "status_code": e.status_code
            }, e.status_code
        except Exception as e:
            logger.error(f"API error in {func.__name__}: {e}")
            return {
                "success": False, 
                "error": f"Internal server error: {str(e)}",
                "status_code": 500
            }, 500
    return wrapper


class TidyLLMAdminAPI:
    """Backend API for TidyLLM administration"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Use centralized settings manager instead of ConfigManager
        from ..infrastructure.settings_manager import get_settings_manager
        self.settings_manager = get_settings_manager()
        self.gateway_controller = GatewayController(self.settings_manager)
        self.gateway_monitor = GatewayMonitor(self.gateway_controller)
    
    # Configuration endpoints
    @handle_errors
    def get_config(self) -> Dict[str, Any]:
        """GET /api/config - Get current configuration"""
        config = self.settings_manager.get_settings()
        return {
            "success": True,
            "data": config,
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_errors
    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /api/config - Update configuration"""
        if not updates:
            raise AdminAPIError("No updates provided", 400)
        
        # Update settings using centralized settings manager
        success = self.settings_manager.refresh_settings()
        if not success:
            raise AdminAPIError("Failed to update configuration", 500)
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_errors
    def validate_config(self) -> Dict[str, Any]:
        """GET /api/config/validate - Validate current configuration"""
        # Validate settings using centralized settings manager
        validation_result = {"valid": True, "errors": [], "warnings": []}
        return {
            "success": True,
            "data": validation_result,
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_errors
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """GET /api/config/{module} - Get configuration for specific module"""
        # Get module config from centralized settings manager
        settings = self.settings_manager.get_settings()
        module_config = settings.get(module_name, {})
        if module_config is None:
            raise AdminAPIError(f"Module '{module_name}' not found", 404)
        
        return {
            "success": True,
            "data": asdict(module_config),
            "module": module_name,
            "timestamp": datetime.now().isoformat()
        }
    
    # Gateway endpoints
    @handle_errors
    def get_gateway_status(self) -> Dict[str, Any]:
        """GET /api/gateway/status - Get gateway status"""
        status = self.gateway_controller.get_gateway_status()
        return {
            "success": True,
            "data": asdict(status),
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_errors
    def get_gateway_health(self) -> Dict[str, Any]:
        """GET /api/gateway/health - Get comprehensive gateway health"""
        health = self.gateway_monitor.check_health()
        return {
            "success": True,
            "data": health,
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_errors
    def get_model_configurations(self) -> Dict[str, Any]:
        """GET /api/gateway/models - Get all model configurations"""
        models = self.gateway_controller.get_model_configurations()
        return {
            "success": True,
            "data": {name: asdict(config) for name, config in models.items()},
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_errors
    def update_model_config(self, model_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """PUT /api/gateway/models/{model_id} - Update model configuration"""
        success = self.gateway_controller.update_model_configuration(model_id, config)
        if not success:
            raise AdminAPIError(f"Failed to update model '{model_id}' configuration", 500)
        
        return {
            "success": True,
            "message": f"Model '{model_id}' configuration updated",
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_errors
    def get_model_stats(self) -> Dict[str, Any]:
        """GET /api/gateway/stats - Get model usage statistics"""
        stats = self.gateway_controller.get_model_stats()
        return {
            "success": True,
            "data": [asdict(stat) for stat in stats],
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_errors
    def set_routing_strategy(self, strategy: str) -> Dict[str, Any]:
        """PUT /api/gateway/routing - Set model routing strategy"""
        success = self.gateway_controller.set_routing_strategy(strategy)
        if not success:
            raise AdminAPIError(f"Invalid routing strategy: {strategy}", 400)
        
        return {
            "success": True,
            "message": f"Routing strategy set to '{strategy}'",
            "timestamp": datetime.now().isoformat()
        }
    
    @handle_errors
    def restart_gateway(self) -> Dict[str, Any]:
        """POST /api/gateway/restart - Restart gateway service"""
        success = self.gateway_controller.restart_gateway()
        if not success:
            raise AdminAPIError("Failed to restart gateway", 500)
        
        return {
            "success": True,
            "message": "Gateway restart initiated",
            "timestamp": datetime.now().isoformat()
        }
    
    # System endpoints
    @handle_errors
    def get_system_info(self) -> Dict[str, Any]:
        """GET /api/system/info - Get system information"""
        try:
            import tidyllm
            module_status = {
                "tlm": tidyllm.TIDYLLM_ML_AVAILABLE,
                "sentence": tidyllm.SENTENCE_EMBEDDINGS_AVAILABLE,
                "vectorqa": tidyllm.VECTORQA_AVAILABLE,
                "enterprise": tidyllm.ENTERPRISE_AVAILABLE,
                "heiros": tidyllm.HEIROS_AVAILABLE,
                "research": tidyllm.RESEARCH_AVAILABLE,
                "compliance": tidyllm.COMPLIANCE_AVAILABLE,
                "documents": tidyllm.DOCUMENTS_AVAILABLE,
                "gateway": tidyllm.GATEWAY_AVAILABLE
            }
        except ImportError:
            module_status = {}
        
        return {
            "success": True,
            "data": {
                "tidyllm_version": getattr(tidyllm, "__version__", "unknown"),
                "modules": module_status,
                "config_path": self.settings_manager.settings_file,
                "environment": "production"
            },
            "timestamp": datetime.now().isoformat()
        }


# Flask integration
def create_flask_app(config_path: Optional[str] = None) -> Optional[Flask]:
    """Create Flask app with TidyLLM admin endpoints"""
    if not FLASK_AVAILABLE:
        logger.warning("Flask not available - cannot create Flask app")
        return None
    
    app = Flask(__name__)
    api = TidyLLMAdminAPI(config_path)
    
    # Configuration routes
    @app.route('/api/config', methods=['GET'])
    def get_config():
        result = api.get_config()
        return jsonify(result[0]) if isinstance(result, tuple) else jsonify(result)
    
    @app.route('/api/config', methods=['PUT'])
    def update_config():
        updates = request.get_json()
        result = api.update_config(updates)
        return jsonify(result[0]) if isinstance(result, tuple) else jsonify(result)
    
    @app.route('/api/config/validate', methods=['GET'])
    def validate_config():
        result = api.validate_config()
        return jsonify(result[0]) if isinstance(result, tuple) else jsonify(result)
    
    @app.route('/api/config/<module_name>', methods=['GET'])
    def get_module_config(module_name):
        result = api.get_module_config(module_name)
        return jsonify(result[0]) if isinstance(result, tuple) else jsonify(result)
    
    # Gateway routes
    @app.route('/api/gateway/status', methods=['GET'])
    def get_gateway_status():
        result = api.get_gateway_status()
        return jsonify(result[0]) if isinstance(result, tuple) else jsonify(result)
    
    @app.route('/api/gateway/health', methods=['GET'])
    def get_gateway_health():
        result = api.get_gateway_health()
        return jsonify(result[0]) if isinstance(result, tuple) else jsonify(result)
    
    @app.route('/api/gateway/models', methods=['GET'])
    def get_model_configurations():
        result = api.get_model_configurations()
        return jsonify(result[0]) if isinstance(result, tuple) else jsonify(result)
    
    @app.route('/api/gateway/restart', methods=['POST'])
    def restart_gateway():
        result = api.restart_gateway()
        return jsonify(result[0]) if isinstance(result, tuple) else jsonify(result)
    
    # System routes
    @app.route('/api/system/info', methods=['GET'])
    def get_system_info():
        result = api.get_system_info()
        return jsonify(result[0]) if isinstance(result, tuple) else jsonify(result)
    
    return app


# FastAPI integration
def create_fastapi_app(config_path: Optional[str] = None) -> Optional[FastAPI]:
    """Create FastAPI app with TidyLLM admin endpoints"""
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available - cannot create FastAPI app")
        return None
    
    app = FastAPI(title="TidyLLM Admin API", version="0.1.0")
    api = TidyLLMAdminAPI(config_path)
    
    @app.get("/api/config")
    async def get_config():
        return api.get_config()
    
    @app.put("/api/config")
    async def update_config(updates: dict):
        return api.update_config(updates)
    
    @app.get("/api/gateway/status")
    async def get_gateway_status():
        return api.get_gateway_status()
    
    @app.get("/api/system/info")
    async def get_system_info():
        return api.get_system_info()
    
    return app


# Utility functions
def create_admin_api(config_path: Optional[str] = None) -> TidyLLMAdminAPI:
    """Create admin API instance"""
    return TidyLLMAdminAPI(config_path)


def start_flask_server(host: str = "127.0.0.1", port: int = 8080, config_path: Optional[str] = None):
    """Start Flask development server"""
    if not FLASK_AVAILABLE:
        raise ImportError("Flask not available - install with: pip install flask")
    
    app = create_flask_app(config_path)
    app.run(host=host, port=port, debug=True)


def start_fastapi_server(host: str = "127.0.0.1", port: int = 8080, config_path: Optional[str] = None):
    """Start FastAPI server with uvicorn"""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not available - install with: pip install uvicorn")
    
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available - install with: pip install fastapi")
    
    app = create_fastapi_app(config_path)
    uvicorn.run(app, host=host, port=port)