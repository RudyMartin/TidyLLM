#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM Gateway - Corporate-Safe LiteLLM Clone

Drop-in replacement for LiteLLM with enterprise governance, security controls,
and IT-managed provider routing. Supports 100+ LLM providers through corporate
infrastructure with full audit trails and cost management.

Key Features:
- Unified interface across 100+ LLM providers
- Corporate security and compliance controls
- Enterprise cost tracking and budget management
- Automatic fallbacks with governance
- IT-managed provider configurations
- Complete audit trails for regulatory compliance
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime

# Core gateway components
from .core.base_gateway import BaseGateway, GatewayConfig, GatewayResponse
from .core.provider_registry import (
    EnterpriseProviderRegistry, 
    get_provider_registry,
    ModelSpec,
    ProviderConfig,
    ModelCapability,
    ProviderTier
)
from .core.security import SecurityManager, AuditLogger
from .core.rate_limiter import RateLimiter, QuotaManager
from .core.mlflow_backend import MLFlowGatewayBackend, MLFLOW_AVAILABLE

logger = logging.getLogger(__name__)


class TidyLLMGatewayConfig(GatewayConfig):
    """Configuration for TidyLLM Gateway (LiteLLM Clone)"""
    
    # LiteLLM compatibility settings
    default_model: str = "gpt-4o"
    enable_fallbacks: bool = True
    max_fallback_attempts: int = 3
    
    # Corporate controls
    require_user_attribution: bool = True
    require_audit_reason: bool = True
    enable_cost_tracking: bool = True
    enforce_model_restrictions: bool = True
    
    # Budget and quota settings
    default_budget_usd: float = 1000.0
    cost_alert_threshold: float = 0.8  # 80% of budget
    
    # Performance settings
    default_timeout: int = 30
    enable_streaming: bool = True
    enable_caching: bool = False  # Disabled for compliance


class TidyLLMGateway(BaseGateway):
    """
    Corporate-Safe LiteLLM Clone with Enterprise Governance
    
    Provides unified interface to 100+ LLM providers with:
    - Enterprise security and compliance controls
    - IT-managed provider configurations
    - Automatic fallbacks with governance
    - Comprehensive cost tracking and budgets
    - Complete audit trails for regulatory compliance
    - Zero direct external API access
    
    Compatible with LiteLLM API while adding enterprise features.
    """
    
    def __init__(self, config: Optional[TidyLLMGatewayConfig] = None):
        # Use default config if none provided
        if config is None:
            config = TidyLLMGatewayConfig(
                base_url="https://corporate-gateway.company.com",
                tenant_id="default",
                enable_audit_logging=True
            )
        
        super().__init__(config)
        self.gateway_config = config
        
        # Initialize provider registry
        self.provider_registry = get_provider_registry()
        
        # Initialize MLFlow Gateway backend
        self.mlflow_backend = MLFlowGatewayBackend(
            gateway_uri=config.base_url,
            provider_registry=self.provider_registry
        )
        
        # Cost tracking
        self.session_costs: Dict[str, float] = {}  # user_id -> cumulative cost
        self.model_usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.request_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_requests": 0,
            "avg_latency_ms": 0.0,
            "total_tokens_processed": 0,
            "total_cost_usd": 0.0
        }
        
        logger.info("🚀 TidyLLM Gateway (LiteLLM Clone) initialized")
        logger.info(f"   Provider registry: {len(self.provider_registry.providers)} providers")
        logger.info(f"   Available models: {len(self.provider_registry.models)} models")
        logger.info(f"   MLFlow backend: {'✅ Connected' if MLFLOW_AVAILABLE else '⚠️ Fallback mode'}")
    
    def _execute_request(self, endpoint: str, data: Dict[str, Any], **kwargs) -> Any:
        """Execute LLM request through corporate provider registry"""
        
        # Extract request parameters
        model = data.get("model", self.gateway_config.default_model)
        messages = data.get("messages", [])
        user_id = data.get("user_id", "unknown")
        
        # Route to appropriate handler
        if endpoint == "completion" or endpoint == "chat/completions":
            return self._handle_completion_request(model, messages, user_id, data, **kwargs)
        elif endpoint == "embeddings":
            return self._handle_embedding_request(model, data.get("input", ""), user_id, data, **kwargs)
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint}")
    
    def _handle_completion_request(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        user_id: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Handle chat completion request with fallbacks"""
        
        # Get model specification
        model_spec = self.provider_registry.get_model(model)
        if not model_spec:
            raise ValueError(f"Model '{model}' not found in provider registry")
        
        # Check IT approval
        if self.gateway_config.enforce_model_restrictions and not model_spec.it_approved:
            raise ValueError(f"Model '{model}' not approved by IT")
        
        # Estimate cost
        input_tokens = self._estimate_input_tokens(messages)
        max_tokens = data.get("max_tokens", 1000)
        estimated_cost = model_spec.pricing.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=max_tokens
        )
        
        # Check budget
        self._check_budget(user_id, estimated_cost)
        
        # Try primary model first
        try:
            result = self._execute_model_request(model, messages, data, **kwargs)
            
            # Track successful request
            self._track_request_success(model, user_id, result)
            return result
            
        except Exception as primary_error:
            logger.warning(f"Primary model {model} failed: {primary_error}")
            
            # Try fallbacks if enabled
            if self.gateway_config.enable_fallbacks:
                return self._handle_fallback_request(
                    model, messages, user_id, data, primary_error, **kwargs
                )
            else:
                raise primary_error
    
    def _handle_fallback_request(
        self,
        original_model: str,
        messages: List[Dict[str, str]],
        user_id: str,
        data: Dict[str, Any],
        original_error: Exception,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle fallback request with governance"""
        
        # Get fallback models
        fallback_models = self.provider_registry.get_model_fallbacks(
            original_model,
            max_fallbacks=self.gateway_config.max_fallback_attempts
        )
        
        if not fallback_models:
            raise ValueError(f"No fallback models available for {original_model}")
        
        last_error = original_error
        
        for fallback_model in fallback_models:
            try:
                logger.info(f"Attempting fallback: {original_model} -> {fallback_model}")
                
                # Update data with fallback model
                fallback_data = data.copy()
                fallback_data["model"] = fallback_model
                
                # Execute fallback request
                result = self._execute_model_request(fallback_model, messages, fallback_data, **kwargs)
                
                # Add fallback metadata to response
                result["fallback_used"] = True
                result["original_model"] = original_model
                result["fallback_model"] = fallback_model
                result["fallback_reason"] = str(original_error)
                
                # Track fallback success
                self._track_fallback_success(original_model, fallback_model, user_id, result)
                
                logger.info(f"✅ Fallback successful: {fallback_model}")
                return result
                
            except Exception as fallback_error:
                logger.warning(f"Fallback {fallback_model} failed: {fallback_error}")
                last_error = fallback_error
                continue
        
        # All fallbacks failed
        self._track_request_failure(original_model, user_id, last_error)
        raise Exception(f"All fallbacks failed. Last error: {last_error}")
    
    def _execute_model_request(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute actual model request through provider"""
        
        # Get provider for model
        provider_id = self.provider_registry.find_model_provider(model)
        if not provider_id:
            raise ValueError(f"No provider found for model: {model}")
        
        provider_config = self.provider_registry.get_provider(provider_id)
        if not provider_config:
            raise ValueError(f"Provider configuration not found: {provider_id}")
        
        # Check provider health
        provider_health = self.provider_registry.get_provider_health(provider_id)
        if provider_health.get("circuit_breaker_open", False):
            raise Exception(f"Provider {provider_id} circuit breaker is open")
        
        # Execute through MLFlow Gateway backend
        start_time = time.time()
        
        try:
            # Use MLFlow backend for actual LLM requests
            model_spec = self.provider_registry.get_model(model)
            result = self.mlflow_backend.execute_completion(
                model=model,
                messages=messages,
                provider_config=provider_config,
                model_spec=model_spec,
                **data
            )
            
            # Calculate actual response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Update provider health
            self.provider_registry.update_provider_health(
                provider_id, "healthy", response_time_ms, error_occurred=False
            )
            
            # Add metadata
            result.update({
                "model": model,
                "provider": provider_id,
                "response_time_ms": response_time_ms,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            # Update provider health on error
            response_time_ms = (time.time() - start_time) * 1000
            self.provider_registry.update_provider_health(
                provider_id, "error", response_time_ms, error_occurred=True
            )
            raise e
    
    def _execute_openai_request(
        self, 
        provider: ProviderConfig, 
        model: str, 
        messages: List[Dict[str, str]], 
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute OpenAI-compatible request"""
        
        # Simulate OpenAI API response (in production, this would make actual HTTP request)
        # through corporate-managed endpoints
        
        # Calculate tokens (simplified)
        input_tokens = self._estimate_input_tokens(messages)
        output_tokens = len("This is a simulated response from the corporate OpenAI endpoint.") // 4
        
        # Calculate cost
        model_spec = self.provider_registry.get_model(model)
        cost = model_spec.pricing.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        ) if model_spec else 0.0
        
        return {
            "id": f"chatcmpl-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"This is a simulated response from the corporate {model} endpoint through TidyLLM Gateway. Original request had {len(messages)} messages."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost_usd": cost,
            "provider_endpoint": provider.base_url
        }
    
    def _execute_anthropic_request(
        self, 
        provider: ProviderConfig, 
        model: str, 
        messages: List[Dict[str, str]], 
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute Anthropic-compatible request"""
        
        # Calculate tokens
        input_tokens = self._estimate_input_tokens(messages)
        output_tokens = len("This is a simulated response from the corporate Claude endpoint.") // 4
        
        # Calculate cost
        model_spec = self.provider_registry.get_model(model)
        cost = model_spec.pricing.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        ) if model_spec else 0.0
        
        return {
            "id": f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "type": "message",
            "role": "assistant",
            "content": f"This is a simulated response from the corporate {model} endpoint through TidyLLM Gateway. Processed {len(messages)} messages through AWS Bedrock.",
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost_usd": cost,
            "provider_endpoint": provider.base_url
        }
    
    def _execute_ollama_request(
        self, 
        provider: ProviderConfig, 
        model: str, 
        messages: List[Dict[str, str]], 
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute Ollama-compatible request"""
        
        # Calculate tokens
        input_tokens = self._estimate_input_tokens(messages)
        output_tokens = len("This is a simulated response from the on-premises Llama model.") // 4
        
        # Local models typically have very low cost
        cost = 0.001  # Minimal infrastructure cost
        
        return {
            "model": model,
            "created_at": datetime.utcnow().isoformat(),
            "response": f"This is a simulated response from the on-premises {model} endpoint through TidyLLM Gateway. Processed locally with zero external data transmission.",
            "done": True,
            "context": [],
            "total_duration": 1500000000,  # nanoseconds
            "load_duration": 100000000,
            "prompt_eval_count": input_tokens,
            "prompt_eval_duration": 500000000,
            "eval_count": output_tokens,
            "eval_duration": 900000000,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost_usd": cost,
            "provider_endpoint": provider.base_url
        }
    
    def _execute_generic_request(
        self, 
        provider: ProviderConfig, 
        model: str, 
        messages: List[Dict[str, str]], 
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute generic provider request"""
        
        # Generic response format
        input_tokens = self._estimate_input_tokens(messages)
        output_tokens = 50
        
        return {
            "model": model,
            "provider": provider.provider_id,
            "response": f"Generic response from {provider.display_name}",
            "usage": {
                "input_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost_usd": 0.01,
            "provider_endpoint": provider.base_url
        }
    
    def _estimate_input_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate input tokens (simplified)"""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return max(1, total_chars // 4)  # Rough approximation
    
    def _check_budget(self, user_id: str, estimated_cost: float):
        """Check user budget before processing request"""
        
        current_spend = self.session_costs.get(user_id, 0.0)
        if current_spend + estimated_cost > self.gateway_config.default_budget_usd:
            raise ValueError(f"Budget exceeded. Current: ${current_spend:.2f}, Estimated: ${estimated_cost:.2f}, Limit: ${self.gateway_config.default_budget_usd:.2f}")
    
    def _track_request_success(self, model: str, user_id: str, result: Dict[str, Any]):
        """Track successful request metrics"""
        
        # Update session costs
        cost = result.get("cost_usd", 0.0)
        self.session_costs[user_id] = self.session_costs.get(user_id, 0.0) + cost
        
        # Update model usage stats
        if model not in self.model_usage_stats:
            self.model_usage_stats[model] = {
                "requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_response_time": 0.0
            }
        
        stats = self.model_usage_stats[model]
        stats["requests"] += 1
        stats["total_tokens"] += result.get("usage", {}).get("total_tokens", 0)
        stats["total_cost"] += cost
        
        # Update global metrics
        self.request_metrics["total_requests"] += 1
        self.request_metrics["successful_requests"] += 1
        self.request_metrics["total_cost_usd"] += cost
        self.request_metrics["total_tokens_processed"] += result.get("usage", {}).get("total_tokens", 0)
    
    def _track_fallback_success(self, original_model: str, fallback_model: str, user_id: str, result: Dict[str, Any]):
        """Track successful fallback request"""
        
        self._track_request_success(fallback_model, user_id, result)
        self.request_metrics["fallback_requests"] += 1
        
        logger.info(f"Fallback successful: {original_model} -> {fallback_model} for user {user_id}")
    
    def _track_request_failure(self, model: str, user_id: str, error: Exception):
        """Track failed request metrics"""
        
        self.request_metrics["total_requests"] += 1
        self.request_metrics["failed_requests"] += 1
        
        logger.error(f"Request failed: {model} for user {user_id}: {error}")


# LiteLLM-compatible interface functions
def completion(
    model: str,
    messages: List[Dict[str, str]],
    user_id: Optional[str] = None,
    audit_reason: Optional[str] = None,
    department: Optional[str] = None,
    max_cost_usd: Optional[float] = None,
    fallbacks: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Corporate-safe LiteLLM completion function
    
    Drop-in replacement for litellm.completion() with enterprise governance.
    
    Args:
        model: Model identifier (routed through corporate registry)
        messages: Chat messages in OpenAI format
        user_id: Corporate user ID (required for audit)
        audit_reason: Business reason for request (required for compliance)
        department: Department for cost allocation
        max_cost_usd: Maximum cost limit for request
        fallbacks: Alternative models if primary fails
        **kwargs: Additional model parameters
        
    Returns:
        Standardized completion response with enterprise metadata
    """
    
    # Get or create gateway instance
    gateway = _get_global_gateway()
    
    # Prepare request data
    request_data = {
        "model": model,
        "messages": messages,
        "user_id": user_id or "unknown",
        "audit_reason": audit_reason,
        "department": department,
        "max_cost_usd": max_cost_usd,
        "fallbacks": fallbacks,
        **kwargs
    }
    
    # Execute through enterprise gateway
    response = gateway.execute(
        endpoint="completion",
        data=request_data,
        user_id=user_id or "unknown",
        audit_reason=audit_reason or "LiteLLM compatible request"
    )
    
    if response.success:
        return response.data
    else:
        raise Exception(f"Completion failed: {response.error}")


def embedding(
    model: str,
    input: Union[str, List[str]],
    user_id: Optional[str] = None,
    audit_reason: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Corporate-safe embedding function"""
    
    gateway = _get_global_gateway()
    
    request_data = {
        "model": model,
        "input": input,
        "user_id": user_id or "unknown",
        **kwargs
    }
    
    response = gateway.execute(
        endpoint="embeddings",
        data=request_data,
        user_id=user_id or "unknown",
        audit_reason=audit_reason or "Embedding request"
    )
    
    if response.success:
        return response.data
    else:
        raise Exception(f"Embedding failed: {response.error}")


def completion_cost(completion_response: Dict[str, Any]) -> float:
    """Calculate completion cost (LiteLLM compatible)"""
    return completion_response.get("cost_usd", 0.0)


def token_counter(model: str, messages: List[Dict[str, str]]) -> int:
    """Count tokens for model (LiteLLM compatible)"""
    
    # Simple token estimation (in production, use actual tokenizer)
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    return max(1, total_chars // 4)


def get_supported_models() -> List[str]:
    """Get list of corporate-approved models"""
    
    registry = get_provider_registry()
    return list(registry.models.keys())


def get_model_info(model: str) -> Optional[Dict[str, Any]]:
    """Get model information"""
    
    registry = get_provider_registry()
    model_spec = registry.get_model(model)
    
    if not model_spec:
        return None
    
    return {
        "model_id": model_spec.model_id,
        "display_name": model_spec.display_name,
        "provider": model_spec.provider,
        "capabilities": [cap.value for cap in model_spec.capabilities],
        "max_context_tokens": model_spec.max_context_tokens,
        "max_output_tokens": model_spec.max_output_tokens,
        "it_approved": model_spec.it_approved,
        "security_tier": model_spec.security_tier.value,
        "data_residency": model_spec.data_residency,
        "pricing": {
            "input_per_1k": model_spec.pricing.input_per_1k,
            "output_per_1k": model_spec.pricing.output_per_1k
        }
    }


# Global gateway instance
_global_gateway: Optional[TidyLLMGateway] = None


def _get_global_gateway() -> TidyLLMGateway:
    """Get or create global gateway instance"""
    global _global_gateway
    
    if _global_gateway is None:
        config = TidyLLMGatewayConfig()
        _global_gateway = TidyLLMGateway(config)
    
    return _global_gateway


def set_gateway_config(config: TidyLLMGatewayConfig):
    """Set global gateway configuration"""
    global _global_gateway
    _global_gateway = TidyLLMGateway(config)


# Export LiteLLM-compatible interface
__all__ = [
    # Main interface functions (LiteLLM compatible)
    'completion',
    'embedding', 
    'completion_cost',
    'token_counter',
    'get_supported_models',
    'get_model_info',
    
    # Enterprise gateway classes
    'TidyLLMGateway',
    'TidyLLMGatewayConfig',
    
    # Configuration functions
    'set_gateway_config'
]