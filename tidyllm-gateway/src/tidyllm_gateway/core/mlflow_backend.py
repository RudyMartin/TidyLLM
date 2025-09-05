#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLFlow Gateway Backend - Real LLM Provider Integration

Integrates TidyLLM Gateway with MLFlow Gateway for actual LLM provider access
through corporate-controlled infrastructure. Provides real LLM responses
while maintaining enterprise governance and security controls.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import requests
import json

# MLFlow Gateway integration
try:
    import mlflow
    from mlflow.gateway import MlflowGatewayClient
    from mlflow.gateway.config import Route
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .provider_registry import EnterpriseProviderRegistry, ModelSpec, ProviderConfig

logger = logging.getLogger(__name__)


class MLFlowGatewayBackend:
    """
    MLFlow Gateway Backend for TidyLLM Gateway
    
    Provides real LLM provider access through MLFlow Gateway while maintaining
    enterprise governance. Routes requests through corporate IT infrastructure
    with full audit trails and cost tracking.
    
    Architecture:
    TidyLLM Gateway → MLFlow Gateway → Corporate IT → LLM Providers
    """
    
    def __init__(
        self, 
        gateway_uri: str = "http://localhost:5000",
        provider_registry: Optional[EnterpriseProviderRegistry] = None
    ):
        self.gateway_uri = gateway_uri
        self.provider_registry = provider_registry
        
        # Initialize MLFlow Gateway client
        self.mlflow_client = None
        if MLFLOW_AVAILABLE:
            try:
                self.mlflow_client = MlflowGatewayClient(gateway_uri=gateway_uri)
                logger.info(f"✅ MLFlow Gateway connected: {gateway_uri}")
            except Exception as e:
                logger.warning(f"⚠️ MLFlow Gateway connection failed: {e}")
                logger.info("Falling back to HTTP client")
        
        # Fallback HTTP client for direct API calls
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Route mappings (MLFlow route name → TidyLLM model)
        self.route_mappings = self._initialize_route_mappings()
        
        logger.info("🔗 MLFlow Gateway Backend initialized")
    
    def _initialize_route_mappings(self) -> Dict[str, str]:
        """Initialize MLFlow route to model mappings"""
        
        # Default mappings - can be configured by IT
        mappings = {
            # OpenAI routes
            "openai-gpt4": "gpt-4",
            "openai-gpt4o": "gpt-4o", 
            "openai-gpt4o-mini": "gpt-4o-mini",
            "openai-gpt35-turbo": "gpt-3.5-turbo",
            
            # Anthropic routes  
            "anthropic-claude-3-5-sonnet": "claude-3-5-sonnet",
            "anthropic-claude-3-haiku": "claude-3-haiku",
            "anthropic-claude-3-opus": "claude-3-opus",
            
            # AWS Bedrock routes
            "bedrock-claude-3-5-sonnet": "claude-3-5-sonnet",
            "bedrock-claude-3-haiku": "claude-3-haiku",
            "bedrock-titan-text": "amazon-titan-text",
            
            # Azure OpenAI routes
            "azure-gpt4": "gpt-4",
            "azure-gpt4o": "gpt-4o",
            "azure-gpt35-turbo": "gpt-3.5-turbo",
            
            # Local/Ollama routes
            "ollama-llama3-70b": "llama-3.1-70b",
            "ollama-llama3-8b": "llama-3.1-8b",
            "ollama-codellama": "codellama-34b",
            
            # Embedding routes
            "openai-embeddings": "text-embedding-3-large",
            "azure-embeddings": "text-embedding-ada-002"
        }
        
        return mappings
    
    def execute_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        provider_config: ProviderConfig,
        model_spec: ModelSpec,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute completion request through MLFlow Gateway"""
        
        start_time = datetime.utcnow()
        
        try:
            # Find MLFlow route for model
            route_name = self._find_mlflow_route(model, provider_config)
            if not route_name:
                raise ValueError(f"No MLFlow route configured for model: {model}")
            
            # Prepare request data
            request_data = self._prepare_completion_request(messages, model_spec, **kwargs)
            
            # Execute through MLFlow Gateway
            if self.mlflow_client:
                response = self._execute_mlflow_completion(route_name, request_data)
            else:
                response = self._execute_http_completion(route_name, request_data)
            
            # Standardize response format
            standardized_response = self._standardize_completion_response(
                response, model, provider_config.provider_id, start_time
            )
            
            # Calculate actual cost
            actual_cost = self._calculate_actual_cost(standardized_response, model_spec)
            standardized_response["cost_usd"] = actual_cost
            
            return standardized_response
            
        except Exception as e:
            logger.error(f"MLFlow completion failed for {model}: {e}")
            
            # Return error response with fallback info
            return {
                "error": str(e),
                "model": model,
                "provider": provider_config.provider_id,
                "fallback_available": True,
                "timestamp": datetime.utcnow().isoformat(),
                "cost_usd": 0.0
            }
    
    def execute_embedding(
        self,
        model: str,
        input_text: Union[str, List[str]],
        provider_config: ProviderConfig,
        model_spec: ModelSpec,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute embedding request through MLFlow Gateway"""
        
        start_time = datetime.utcnow()
        
        try:
            # Find embedding route
            route_name = self._find_mlflow_route(model, provider_config, endpoint_type="embeddings")
            if not route_name:
                raise ValueError(f"No embedding route configured for model: {model}")
            
            # Prepare embedding request
            request_data = {
                "input": input_text if isinstance(input_text, str) else input_text,
                **kwargs
            }
            
            # Execute through MLFlow Gateway
            if self.mlflow_client:
                response = self.mlflow_client.query(route=route_name, data=request_data)
            else:
                response = self._execute_http_request(route_name, request_data)
            
            # Standardize embedding response
            return self._standardize_embedding_response(
                response, model, provider_config.provider_id, start_time
            )
            
        except Exception as e:
            logger.error(f"MLFlow embedding failed for {model}: {e}")
            return {
                "error": str(e),
                "model": model,
                "provider": provider_config.provider_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _find_mlflow_route(
        self, 
        model: str, 
        provider_config: ProviderConfig,
        endpoint_type: str = "chat"
    ) -> Optional[str]:
        """Find MLFlow route for model and provider"""
        
        # Try exact model mapping first
        for route, mapped_model in self.route_mappings.items():
            if mapped_model == model and provider_config.provider_type in route:
                return route
        
        # Try provider-based mapping
        provider_prefix = provider_config.provider_type.replace("_", "-")
        model_suffix = model.replace(".", "-").replace("_", "-")
        
        possible_routes = [
            f"{provider_prefix}-{model_suffix}",
            f"{provider_prefix}-{endpoint_type}",
            f"{provider_config.provider_id}-{model_suffix}",
            model  # Sometimes route name matches model name
        ]
        
        # Check which routes exist in MLFlow Gateway
        if self.mlflow_client:
            try:
                available_routes = self.mlflow_client.search_routes()
                available_route_names = [r.name for r in available_routes]
                
                for route in possible_routes:
                    if route in available_route_names:
                        return route
            except Exception as e:
                logger.warning(f"Could not search MLFlow routes: {e}")
        
        # Return first possible route as fallback
        return possible_routes[0] if possible_routes else None
    
    def _prepare_completion_request(
        self, 
        messages: List[Dict[str, str]], 
        model_spec: ModelSpec,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare completion request for MLFlow Gateway"""
        
        request_data = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": min(
                kwargs.get("max_tokens", 1000),
                model_spec.max_output_tokens
            )
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            request_data["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            request_data["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            request_data["presence_penalty"] = kwargs["presence_penalty"]
        if "stop" in kwargs:
            request_data["stop"] = kwargs["stop"]
        
        return request_data
    
    def _execute_mlflow_completion(self, route_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute completion through MLFlow Gateway client"""
        
        logger.debug(f"Executing MLFlow route: {route_name}")
        
        try:
            response = self.mlflow_client.query(route=route_name, data=request_data)
            logger.debug(f"✅ MLFlow response received from {route_name}")
            return response
        except Exception as e:
            logger.error(f"❌ MLFlow client query failed: {e}")
            raise e
    
    def _execute_http_completion(self, route_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute completion through HTTP fallback"""
        
        url = f"{self.gateway_uri}/gateway/{route_name}/invocations"
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TidyLLM-Gateway/1.0"
        }
        
        logger.debug(f"Executing HTTP request to: {url}")
        
        try:
            response = self.session.post(
                url,
                headers=headers,
                json=request_data,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"✅ HTTP response received from {route_name}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"❌ HTTP request failed: {e}")
            raise e
    
    def _execute_http_request(self, route_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic HTTP request execution"""
        return self._execute_http_completion(route_name, request_data)
    
    def _standardize_completion_response(
        self, 
        response: Dict[str, Any], 
        model: str, 
        provider_id: str,
        start_time: datetime
    ) -> Dict[str, Any]:
        """Standardize completion response format"""
        
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Handle different response formats
        if "choices" in response:
            # OpenAI format
            standardized = {
                "id": response.get("id", f"tidyllm-{int(datetime.utcnow().timestamp())}"),
                "object": "chat.completion",
                "created": int(datetime.utcnow().timestamp()),
                "model": model,
                "choices": response["choices"],
                "usage": response.get("usage", {}),
                "provider": provider_id,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
        elif "content" in response:
            # Anthropic format
            content = response["content"]
            if isinstance(content, list) and len(content) > 0:
                content_text = content[0].get("text", str(content))
            else:
                content_text = str(content)
                
            standardized = {
                "id": response.get("id", f"tidyllm-{int(datetime.utcnow().timestamp())}"),
                "object": "chat.completion", 
                "created": int(datetime.utcnow().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content_text
                    },
                    "finish_reason": response.get("stop_reason", "stop")
                }],
                "usage": response.get("usage", {}),
                "provider": provider_id,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Generic format
            standardized = {
                "id": f"tidyllm-{int(datetime.utcnow().timestamp())}",
                "object": "chat.completion",
                "created": int(datetime.utcnow().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant", 
                        "content": str(response.get("response", response))
                    },
                    "finish_reason": "stop"
                }],
                "usage": response.get("usage", {"total_tokens": 0}),
                "provider": provider_id,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return standardized
    
    def _standardize_embedding_response(
        self, 
        response: Dict[str, Any], 
        model: str, 
        provider_id: str,
        start_time: datetime
    ) -> Dict[str, Any]:
        """Standardize embedding response format"""
        
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "object": "list",
            "data": response.get("data", response.get("embeddings", [])),
            "model": model,
            "usage": response.get("usage", {}),
            "provider": provider_id,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_actual_cost(self, response: Dict[str, Any], model_spec: ModelSpec) -> float:
        """Calculate actual cost based on token usage"""
        
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        return model_spec.pricing.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    def get_available_routes(self) -> List[Dict[str, Any]]:
        """Get available MLFlow Gateway routes"""
        
        if not self.mlflow_client:
            return []
        
        try:
            routes = self.mlflow_client.search_routes()
            return [
                {
                    "name": route.name,
                    "route_type": route.route_type,
                    "model": getattr(route, "model", {}).get("name", "unknown"),
                    "provider": getattr(route, "model", {}).get("provider", "unknown")
                }
                for route in routes
            ]
        except Exception as e:
            logger.error(f"Failed to get MLFlow routes: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check MLFlow Gateway health"""
        
        try:
            if self.mlflow_client:
                # Try to get routes as health check
                routes = self.mlflow_client.search_routes()
                return {
                    "status": "healthy",
                    "gateway_uri": self.gateway_uri,
                    "routes_available": len(routes),
                    "mlflow_client": "connected",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Try HTTP health check
                response = self.session.get(f"{self.gateway_uri}/health", timeout=5)
                return {
                    "status": "healthy" if response.status_code == 200 else "degraded",
                    "gateway_uri": self.gateway_uri,
                    "mlflow_client": "not_available",
                    "http_fallback": "available",
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "gateway_uri": self.gateway_uri,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Export main class
__all__ = ['MLFlowGatewayBackend', 'MLFLOW_AVAILABLE']