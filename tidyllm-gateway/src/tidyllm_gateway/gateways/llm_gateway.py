#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Gateway - Corporate-Controlled Language Model Access

Enterprise LLM gateway that routes all language model requests through 
corporate IT infrastructure using MLFlow Gateway. Provides centralized
control, governance, and security for AI/ML applications.

Key Features:
- Zero direct external API access
- IT-controlled provider availability
- Full audit trails and compliance
- Cost tracking and budget controls
- Multi-tenant access management
- Graceful fallback mechanisms
"""

from typing import Dict, Any, Optional, List, Union
import json
from datetime import datetime
from ..core.base_gateway import BaseGateway, GatewayConfig, GatewayResponse

# MLFlow integration for corporate gateway
try:
    import mlflow
    from mlflow.gateway import MlflowGatewayClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class LLMGatewayConfig(GatewayConfig):
    """Configuration specific to LLM Gateway"""
    
    # MLFlow Gateway settings
    mlflow_gateway_uri: str = "http://localhost:5000"
    
    # Provider controls (IT managed)
    available_providers: List[str] = None  # Set by IT: ["claude", "openai-corporate", "azure-gpt"]
    default_provider: str = "claude"
    
    # Model controls (IT managed) 
    provider_models: Dict[str, List[str]] = None  # {"claude": ["claude-3-5-sonnet"], "openai": ["gpt-4o"]}
    default_models: Dict[str, str] = None  # {"claude": "claude-3-5-sonnet", "openai": "gpt-4o"}
    
    # Cost controls
    max_tokens_per_request: int = 4096
    max_cost_per_request_usd: float = 1.0
    budget_limit_daily_usd: Optional[float] = None
    
    # Quality controls
    temperature_limits: tuple = (0.0, 1.0)
    require_audit_reason: bool = True
    
    def __post_init__(self):
        if self.available_providers is None:
            self.available_providers = ["claude", "openai"]
        if self.provider_models is None:
            self.provider_models = {
                "claude": ["claude-3-5-sonnet", "claude-3-haiku"],
                "openai": ["gpt-4o", "gpt-4o-mini"]
            }
        if self.default_models is None:
            self.default_models = {
                "claude": "claude-3-5-sonnet",
                "openai": "gpt-4o"
            }


class LLMGateway(BaseGateway):
    """
    Corporate LLM Gateway - Centralized language model access
    
    Routes all LLM requests through corporate IT infrastructure with:
    - Provider controls (what models are available)
    - Cost controls (budget limits, per-request limits)  
    - Quality controls (temperature, token limits)
    - Audit controls (full request/response logging)
    - Security controls (no direct external access)
    
    Architecture:
    Application → LLM Gateway → MLFlow Gateway → Corporate IT → External Provider
    """
    
    def __init__(self, config: LLMGatewayConfig):
        super().__init__(config)
        self.llm_config = config
        
        # Initialize MLFlow Gateway client
        self.mlflow_client = None
        if MLFLOW_AVAILABLE:
            try:
                self.mlflow_client = MlflowGatewayClient(
                    gateway_uri=config.mlflow_gateway_uri
                )
                logger.info(f"🔗 MLFlow Gateway connected: {config.mlflow_gateway_uri}")
            except Exception as e:
                logger.warning(f"⚠️ MLFlow Gateway unavailable: {e}")
        
        # Cost tracking
        self.daily_spend_usd = 0.0
        self.request_costs: List[float] = []
        
        logger.info("🤖 LLM Gateway initialized")
        logger.info(f"   Available providers: {config.available_providers}")
        logger.info(f"   Default provider: {config.default_provider}")
    
    def _execute_request(self, endpoint: str, data: Dict[str, Any], **kwargs) -> Any:
        """Execute LLM request through corporate gateway"""
        
        # Parse request parameters
        provider = data.get("provider", self.llm_config.default_provider)
        model = data.get("model", self._get_default_model(provider))
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.1)
        max_tokens = data.get("max_tokens", 1000)
        
        # Validate request
        self._validate_llm_request(provider, model, messages, temperature, max_tokens)
        
        # Execute through MLFlow Gateway
        if self.mlflow_client:
            response = self._execute_mlflow_request(provider, model, messages, temperature, max_tokens)
        else:
            response = self._execute_fallback_request(provider, model, messages, temperature, max_tokens)
        
        # Track costs
        self._track_request_cost(response)
        
        return response
    
    def _validate_llm_request(
        self, 
        provider: str, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int
    ):
        """Validate LLM request against corporate policies"""
        
        # Check provider availability
        if provider not in self.llm_config.available_providers:
            raise ValueError(f"Provider '{provider}' not available. IT approved: {self.llm_config.available_providers}")
        
        # Check model availability  
        available_models = self.llm_config.provider_models.get(provider, [])
        if model not in available_models:
            raise ValueError(f"Model '{model}' not available for {provider}. IT approved: {available_models}")
        
        # Check temperature limits
        min_temp, max_temp = self.llm_config.temperature_limits
        if not (min_temp <= temperature <= max_temp):
            raise ValueError(f"Temperature {temperature} outside IT policy limits: {min_temp}-{max_temp}")
        
        # Check token limits
        if max_tokens > self.llm_config.max_tokens_per_request:
            raise ValueError(f"Token limit {max_tokens} exceeds policy maximum: {self.llm_config.max_tokens_per_request}")
        
        # Check message format
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be non-empty list")
        
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
        
        # Check daily budget
        if self.llm_config.budget_limit_daily_usd:
            if self.daily_spend_usd >= self.llm_config.budget_limit_daily_usd:
                raise ValueError(f"Daily budget limit reached: ${self.daily_spend_usd:.2f}")
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider"""
        return self.llm_config.default_models.get(provider, "claude-3-5-sonnet")
    
    def _execute_mlflow_request(
        self, 
        provider: str, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int
    ) -> Dict[str, Any]:
        """Execute request through MLFlow Gateway"""
        
        # Convert to MLFlow route format
        route_name = self._get_mlflow_route_name(provider)
        
        request_data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add model if supported
        if provider == "openai":
            request_data["model"] = model
        
        try:
            response = self.mlflow_client.query(route=route_name, data=request_data)
            
            # Standardize response format
            return self._standardize_response(response, provider, model)
            
        except Exception as e:
            logger.error(f"MLFlow Gateway request failed: {e}")
            return self._create_error_response(str(e), provider, model)
    
    def _execute_fallback_request(
        self, 
        provider: str, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int
    ) -> Dict[str, Any]:
        """Fallback when MLFlow Gateway unavailable"""
        
        logger.warning("🔄 Using fallback response - MLFlow Gateway unavailable")
        
        # Generate fallback response
        user_message = messages[-1].get("content", "") if messages else ""
        
        fallback_content = (
            f"I apologize, but the corporate LLM gateway is temporarily unavailable. "
            f"Your request has been logged for processing. "
            f"Please contact IT support if this issue persists."
        )
        
        return {
            "success": True,
            "provider": provider,
            "model": f"{model}-fallback",
            "content": fallback_content,
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(fallback_content.split()),
                "total_tokens": len(user_message.split()) + len(fallback_content.split())
            },
            "cost_usd": 0.0,
            "fallback": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_mlflow_route_name(self, provider: str) -> str:
        """Map provider to MLFlow route name"""
        
        route_mappings = {
            "claude": "claude-chat",
            "openai": "openai-chat", 
            "azure-openai": "azure-openai-chat",
            "ollama": "ollama-chat",
            "gemini": "gemini-chat"
        }
        
        return route_mappings.get(provider, provider)
    
    def _standardize_response(self, response: Any, provider: str, model: str) -> Dict[str, Any]:
        """Standardize response format across providers"""
        
        if isinstance(response, dict):
            # Extract content based on provider format
            if provider == "claude":
                content = response.get("content", "")
            elif provider in ["openai", "azure-openai"]:
                choices = response.get("choices", [])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
            else:
                content = response.get("content", str(response))
            
            usage = response.get("usage", {})
            
            return {
                "success": True,
                "provider": provider,
                "model": model,
                "content": content,
                "usage": usage,
                "cost_usd": self._calculate_cost(usage, provider, model),
                "raw_response": response,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Handle non-dict responses
        return {
            "success": True,
            "provider": provider,
            "model": model,
            "content": str(response),
            "usage": {"total_tokens": 0},
            "cost_usd": 0.0,
            "raw_response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _create_error_response(self, error: str, provider: str, model: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "provider": provider,
            "model": model,
            "content": f"Request failed: {error}",
            "error": error,
            "usage": {"total_tokens": 0},
            "cost_usd": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_cost(self, usage: Dict[str, int], provider: str, model: str) -> float:
        """Calculate request cost based on usage"""
        
        # Cost per 1K tokens (would be configured by IT)
        cost_per_1k_tokens = {
            "claude": {"claude-3-5-sonnet": 0.003, "claude-3-haiku": 0.00025},
            "openai": {"gpt-4o": 0.005, "gpt-4o-mini": 0.0005}
        }
        
        total_tokens = usage.get("total_tokens", 0)
        rate = cost_per_1k_tokens.get(provider, {}).get(model, 0.001)
        
        return (total_tokens / 1000) * rate
    
    def _track_request_cost(self, response: Dict[str, Any]):
        """Track request cost for budget management"""
        cost = response.get("cost_usd", 0.0)
        
        if cost > 0:
            self.request_costs.append(cost)
            self.daily_spend_usd += cost
            
            # Check per-request limit
            if cost > self.llm_config.max_cost_per_request_usd:
                logger.warning(f"⚠️ High cost request: ${cost:.4f} > ${self.llm_config.max_cost_per_request_usd}")
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        user_id: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        audit_reason: Optional[str] = None,
        **kwargs
    ) -> GatewayResponse:
        """
        Chat completion through corporate gateway
        
        Args:
            messages: Chat messages in OpenAI format
            user_id: User making request (for audit)
            provider: LLM provider (IT controlled)
            model: Model name (IT controlled)
            temperature: Response randomness
            max_tokens: Maximum response tokens
            audit_reason: Reason for request (compliance)
            
        Returns:
            Standardized gateway response
        """
        
        # Validate audit requirement
        if self.llm_config.require_audit_reason and not audit_reason:
            raise ValueError("Audit reason required for LLM requests")
        
        # Prepare request data
        request_data = {
            "provider": provider or self.llm_config.default_provider,
            "model": model,  # Will use default if None
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Execute through base gateway (includes audit logging, rate limiting, etc.)
        return self.execute(
            endpoint="chat",
            data=request_data,
            user_id=user_id,
            audit_reason=audit_reason
        )
    
    def get_available_providers(self) -> List[str]:
        """Get IT-approved available providers"""
        return self.llm_config.available_providers.copy()
    
    def get_available_models(self, provider: str) -> List[str]:
        """Get IT-approved models for provider"""
        return self.llm_config.provider_models.get(provider, [])
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        return {
            "daily_spend_usd": self.daily_spend_usd,
            "daily_limit_usd": self.llm_config.budget_limit_daily_usd,
            "requests_today": len(self.request_costs),
            "avg_cost_per_request": sum(self.request_costs) / len(self.request_costs) if self.request_costs else 0,
            "most_expensive_request": max(self.request_costs) if self.request_costs else 0,
            "budget_remaining_pct": (
                (self.llm_config.budget_limit_daily_usd - self.daily_spend_usd) / 
                self.llm_config.budget_limit_daily_usd * 100
            ) if self.llm_config.budget_limit_daily_usd else None
        }