#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise Provider Registry - Corporate-Safe LLM Provider Management

Manages 100+ LLM providers with enterprise governance, security controls,
and IT-managed configurations. Provides corporate-safe alternative to
direct LiteLLM provider access.
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class ProviderTier(Enum):
    """Provider security and trust tiers"""
    INTERNAL = "internal"           # On-premises, fully controlled
    CORPORATE = "corporate"         # Corporate cloud, IT managed
    TRUSTED_EXTERNAL = "trusted"    # Approved external providers  
    RESTRICTED = "restricted"       # Limited access, high security
    EXPERIMENTAL = "experimental"   # R&D only, non-production


class ModelCapability(Enum):
    """Model capability types"""
    TEXT_COMPLETION = "text_completion"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDINGS = "embeddings"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    FUNCTION_CALLING = "function_calling"


@dataclass
class TokenPricing:
    """Token pricing structure"""
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    embedding_per_1k: float = 0.0
    image_per_request: float = 0.0
    audio_per_minute: float = 0.0
    
    def calculate_cost(self, input_tokens: int = 0, output_tokens: int = 0, 
                      embeddings: int = 0, images: int = 0, audio_minutes: float = 0) -> float:
        """Calculate total cost for usage"""
        cost = 0.0
        cost += (input_tokens / 1000) * self.input_per_1k
        cost += (output_tokens / 1000) * self.output_per_1k  
        cost += (embeddings / 1000) * self.embedding_per_1k
        cost += images * self.image_per_request
        cost += audio_minutes * self.audio_per_minute
        return cost


@dataclass
class ModelSpec:
    """Model specification and capabilities"""
    model_id: str
    display_name: str
    provider: str
    capabilities: List[ModelCapability]
    max_context_tokens: int
    max_output_tokens: int
    pricing: TokenPricing
    
    # Enterprise controls
    it_approved: bool = False
    security_tier: ProviderTier = ProviderTier.EXPERIMENTAL
    compliance_certifications: List[str] = field(default_factory=list)
    data_residency: str = "unknown"
    
    # Performance characteristics
    avg_latency_ms: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    availability_sla: float = 99.0
    
    # Access controls
    allowed_departments: Optional[List[str]] = None
    allowed_user_roles: Optional[List[str]] = None
    max_requests_per_day: Optional[int] = None


@dataclass
class ProviderConfig:
    """Provider configuration and endpoints"""
    provider_id: str
    display_name: str
    provider_type: str              # openai, anthropic, bedrock, azure, etc.
    
    # Endpoint configuration
    base_url: str
    api_version: Optional[str] = None
    
    # Authentication (IT managed)
    auth_method: str = "bearer"     # bearer, oauth, aws_sig4, certificate
    credential_source: str = "vault" # vault, env, certificate_store
    
    # Enterprise settings
    security_tier: ProviderTier = ProviderTier.EXPERIMENTAL
    it_approved: bool = False
    compliance_certifications: List[str] = field(default_factory=list)
    
    # Rate limiting and performance
    max_concurrent_requests: int = 10
    rate_limit_per_minute: int = 1000
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    # Monitoring and health
    health_check_url: Optional[str] = None
    health_check_interval: int = 300  # seconds
    circuit_breaker_threshold: int = 5
    
    # Models available from this provider
    models: List[ModelSpec] = field(default_factory=list)


class EnterpriseProviderRegistry:
    """
    Enterprise Provider Registry - Corporate-safe LLM provider management
    
    Features:
    - IT-managed provider configurations
    - Security tier enforcement  
    - Model capability mapping
    - Cost tracking and budgets
    - Compliance certification tracking
    - Health monitoring and circuit breaking
    - Automatic failover routing
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.providers: Dict[str, ProviderConfig] = {}
        self.models: Dict[str, ModelSpec] = {}  # model_id -> ModelSpec
        self.provider_health: Dict[str, Dict[str, Any]] = {}
        
        # Load configuration
        if config_file:
            self.load_configuration(config_file)
        else:
            self._initialize_default_providers()
        
        self.logger = logging.getLogger(f"{__name__}.ProviderRegistry")
        self.logger.info(f"🏢 Provider Registry initialized with {len(self.providers)} providers")
    
    def _initialize_default_providers(self):
        """Initialize with corporate-safe default providers"""
        
        # OpenAI Corporate (Azure)
        openai_corporate = ProviderConfig(
            provider_id="openai-corporate",
            display_name="OpenAI Corporate (Azure)",
            provider_type="azure_openai",
            base_url="https://azure-openai.company.com",
            api_version="2024-02-01",
            security_tier=ProviderTier.CORPORATE,
            it_approved=True,
            compliance_certifications=["SOC2", "ISO27001", "FedRAMP"],
            models=[
                ModelSpec(
                    model_id="gpt-4o",
                    display_name="GPT-4o Corporate",
                    provider="openai-corporate",
                    capabilities=[ModelCapability.CHAT_COMPLETION, ModelCapability.FUNCTION_CALLING],
                    max_context_tokens=128000,
                    max_output_tokens=4096,
                    pricing=TokenPricing(input_per_1k=0.005, output_per_1k=0.015),
                    it_approved=True,
                    security_tier=ProviderTier.CORPORATE,
                    compliance_certifications=["SOC2", "FedRAMP"],
                    data_residency="US"
                ),
                ModelSpec(
                    model_id="gpt-4o-mini",
                    display_name="GPT-4o Mini Corporate",
                    provider="openai-corporate", 
                    capabilities=[ModelCapability.CHAT_COMPLETION],
                    max_context_tokens=128000,
                    max_output_tokens=16384,
                    pricing=TokenPricing(input_per_1k=0.0005, output_per_1k=0.0015),
                    it_approved=True,
                    security_tier=ProviderTier.CORPORATE
                )
            ]
        )
        
        # Claude Corporate (Bedrock)
        claude_corporate = ProviderConfig(
            provider_id="claude-corporate",
            display_name="Claude Corporate (AWS Bedrock)",
            provider_type="bedrock",
            base_url="https://bedrock.company.com",
            auth_method="aws_sig4",
            security_tier=ProviderTier.CORPORATE,
            it_approved=True,
            compliance_certifications=["SOC2", "ISO27001", "HIPAA"],
            models=[
                ModelSpec(
                    model_id="claude-3-5-sonnet",
                    display_name="Claude 3.5 Sonnet Corporate",
                    provider="claude-corporate",
                    capabilities=[ModelCapability.CHAT_COMPLETION, ModelCapability.IMAGE_ANALYSIS],
                    max_context_tokens=200000,
                    max_output_tokens=8192,
                    pricing=TokenPricing(input_per_1k=0.003, output_per_1k=0.015),
                    it_approved=True,
                    security_tier=ProviderTier.CORPORATE,
                    compliance_certifications=["SOC2", "HIPAA"],
                    data_residency="US"
                ),
                ModelSpec(
                    model_id="claude-3-haiku",
                    display_name="Claude 3 Haiku Corporate",
                    provider="claude-corporate",
                    capabilities=[ModelCapability.CHAT_COMPLETION],
                    max_context_tokens=200000,
                    max_output_tokens=4096,
                    pricing=TokenPricing(input_per_1k=0.00025, output_per_1k=0.00125),
                    it_approved=True,
                    security_tier=ProviderTier.CORPORATE
                )
            ]
        )
        
        # Local Llama (On-premises)
        local_llama = ProviderConfig(
            provider_id="local-llama",
            display_name="Local Llama (On-Premises)",
            provider_type="ollama",
            base_url="https://local-inference.company.com",
            security_tier=ProviderTier.INTERNAL,
            it_approved=True,
            compliance_certifications=["Internal Security Review"],
            models=[
                ModelSpec(
                    model_id="llama-3.1-70b",
                    display_name="Llama 3.1 70B Local",
                    provider="local-llama",
                    capabilities=[ModelCapability.CHAT_COMPLETION, ModelCapability.CODE_GENERATION],
                    max_context_tokens=32000,
                    max_output_tokens=8192,
                    pricing=TokenPricing(input_per_1k=0.0001, output_per_1k=0.0001),
                    it_approved=True,
                    security_tier=ProviderTier.INTERNAL,
                    data_residency="On-premises"
                ),
                ModelSpec(
                    model_id="llama-3.1-8b", 
                    display_name="Llama 3.1 8B Local",
                    provider="local-llama",
                    capabilities=[ModelCapability.CHAT_COMPLETION],
                    max_context_tokens=32000,
                    max_output_tokens=8192,
                    pricing=TokenPricing(input_per_1k=0.00005, output_per_1k=0.00005),
                    it_approved=True,
                    security_tier=ProviderTier.INTERNAL
                )
            ]
        )
        
        # Register providers
        self.register_provider(openai_corporate)
        self.register_provider(claude_corporate) 
        self.register_provider(local_llama)
    
    def register_provider(self, provider: ProviderConfig):
        """Register a new provider with models"""
        self.providers[provider.provider_id] = provider
        
        # Register all models from this provider
        for model in provider.models:
            self.models[model.model_id] = model
        
        self.logger.info(f"Registered provider: {provider.provider_id} with {len(provider.models)} models")
    
    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get provider configuration"""
        return self.providers.get(provider_id)
    
    def get_model(self, model_id: str) -> Optional[ModelSpec]:
        """Get model specification"""
        return self.models.get(model_id)
    
    def find_model_provider(self, model_id: str) -> Optional[str]:
        """Find which provider offers a model"""
        model = self.get_model(model_id)
        return model.provider if model else None
    
    def get_approved_models(
        self, 
        department: Optional[str] = None,
        user_role: Optional[str] = None,
        security_tier: Optional[ProviderTier] = None,
        capability: Optional[ModelCapability] = None
    ) -> List[ModelSpec]:
        """Get IT-approved models with optional filtering"""
        
        approved_models = []
        
        for model in self.models.values():
            # Must be IT approved
            if not model.it_approved:
                continue
            
            # Department filter
            if department and model.allowed_departments:
                if department not in model.allowed_departments:
                    continue
            
            # User role filter
            if user_role and model.allowed_user_roles:
                if user_role not in model.allowed_user_roles:
                    continue
            
            # Security tier filter
            if security_tier:
                if model.security_tier != security_tier:
                    continue
            
            # Capability filter
            if capability:
                if capability not in model.capabilities:
                    continue
            
            approved_models.append(model)
        
        return approved_models
    
    def get_model_fallbacks(
        self, 
        model_id: str,
        max_fallbacks: int = 3,
        same_capability: bool = True
    ) -> List[str]:
        """Get fallback models for a given model"""
        
        primary_model = self.get_model(model_id)
        if not primary_model:
            return []
        
        fallbacks = []
        
        # Get models with similar capabilities
        candidates = []
        for candidate_model in self.models.values():
            if candidate_model.model_id == model_id:
                continue
            
            if not candidate_model.it_approved:
                continue
            
            # Check capability overlap
            if same_capability:
                if not any(cap in candidate_model.capabilities for cap in primary_model.capabilities):
                    continue
            
            candidates.append(candidate_model)
        
        # Sort by security tier (internal first), then by cost (cheaper first)
        def fallback_priority(model: ModelSpec) -> Tuple[int, float]:
            tier_priority = {
                ProviderTier.INTERNAL: 0,
                ProviderTier.CORPORATE: 1, 
                ProviderTier.TRUSTED_EXTERNAL: 2,
                ProviderTier.RESTRICTED: 3,
                ProviderTier.EXPERIMENTAL: 4
            }
            
            avg_cost = (model.pricing.input_per_1k + model.pricing.output_per_1k) / 2
            return (tier_priority.get(model.security_tier, 5), avg_cost)
        
        candidates.sort(key=fallback_priority)
        
        # Return top fallback candidates
        return [model.model_id for model in candidates[:max_fallbacks]]
    
    def estimate_cost(
        self, 
        model_id: str, 
        input_tokens: int, 
        output_tokens: int = 0,
        **kwargs
    ) -> Optional[float]:
        """Estimate cost for model usage"""
        
        model = self.get_model(model_id)
        if not model:
            return None
        
        return model.pricing.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            **kwargs
        )
    
    def get_provider_health(self, provider_id: str) -> Dict[str, Any]:
        """Get provider health status"""
        
        if provider_id not in self.provider_health:
            self.provider_health[provider_id] = {
                "status": "unknown",
                "last_check": None,
                "response_time_ms": None,
                "error_rate": 0.0,
                "circuit_breaker_open": False
            }
        
        return self.provider_health[provider_id]
    
    def update_provider_health(
        self, 
        provider_id: str, 
        status: str, 
        response_time_ms: Optional[float] = None,
        error_occurred: bool = False
    ):
        """Update provider health metrics"""
        
        health = self.get_provider_health(provider_id)
        
        health["status"] = status
        health["last_check"] = datetime.utcnow()
        
        if response_time_ms is not None:
            health["response_time_ms"] = response_time_ms
        
        if error_occurred:
            health["error_rate"] = min(1.0, health.get("error_rate", 0.0) + 0.1)
        else:
            health["error_rate"] = max(0.0, health.get("error_rate", 0.0) - 0.05)
        
        # Circuit breaker logic
        provider = self.get_provider(provider_id)
        if provider and health["error_rate"] > 0.5:
            health["circuit_breaker_open"] = True
            self.logger.warning(f"Circuit breaker opened for provider: {provider_id}")
        elif health["error_rate"] < 0.1:
            health["circuit_breaker_open"] = False
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get comprehensive registry summary"""
        
        # Count by security tier
        tier_counts = {}
        for model in self.models.values():
            tier = model.security_tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Count by capability
        capability_counts = {}
        for model in self.models.values():
            for cap in model.capabilities:
                cap_name = cap.value
                capability_counts[cap_name] = capability_counts.get(cap_name, 0) + 1
        
        # Provider health summary
        healthy_providers = sum(1 for h in self.provider_health.values() if h["status"] == "healthy")
        total_providers = len(self.providers)
        
        return {
            "providers": {
                "total": total_providers,
                "healthy": healthy_providers,
                "unhealthy": total_providers - healthy_providers
            },
            "models": {
                "total": len(self.models),
                "it_approved": sum(1 for m in self.models.values() if m.it_approved),
                "by_security_tier": tier_counts,
                "by_capability": capability_counts
            },
            "compliance": {
                "certifications": list(set(
                    cert for provider in self.providers.values() 
                    for cert in provider.compliance_certifications
                )),
                "data_residency_options": list(set(
                    model.data_residency for model in self.models.values() 
                    if model.data_residency != "unknown"
                ))
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def load_configuration(self, config_file: str):
        """Load provider configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for provider_data in config_data.get("providers", []):
                # Convert dict to ProviderConfig
                provider = self._dict_to_provider_config(provider_data)
                self.register_provider(provider)
            
            self.logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _dict_to_provider_config(self, data: Dict[str, Any]) -> ProviderConfig:
        """Convert dictionary to ProviderConfig"""
        
        # Convert models
        models = []
        for model_data in data.get("models", []):
            pricing = TokenPricing(**model_data.get("pricing", {}))
            
            model = ModelSpec(
                model_id=model_data["model_id"],
                display_name=model_data["display_name"],
                provider=data["provider_id"],
                capabilities=[ModelCapability(cap) for cap in model_data.get("capabilities", [])],
                max_context_tokens=model_data.get("max_context_tokens", 4096),
                max_output_tokens=model_data.get("max_output_tokens", 2048),
                pricing=pricing,
                it_approved=model_data.get("it_approved", False),
                security_tier=ProviderTier(model_data.get("security_tier", "experimental")),
                compliance_certifications=model_data.get("compliance_certifications", []),
                data_residency=model_data.get("data_residency", "unknown")
            )
            models.append(model)
        
        # Create provider config
        provider = ProviderConfig(
            provider_id=data["provider_id"],
            display_name=data["display_name"],
            provider_type=data["provider_type"],
            base_url=data["base_url"],
            api_version=data.get("api_version"),
            auth_method=data.get("auth_method", "bearer"),
            credential_source=data.get("credential_source", "vault"),
            security_tier=ProviderTier(data.get("security_tier", "experimental")),
            it_approved=data.get("it_approved", False),
            compliance_certifications=data.get("compliance_certifications", []),
            models=models
        )
        
        return provider


# Global registry instance
_global_registry: Optional[EnterpriseProviderRegistry] = None


def get_provider_registry() -> EnterpriseProviderRegistry:
    """Get or create global provider registry"""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = EnterpriseProviderRegistry()
    
    return _global_registry


def set_provider_registry(registry: EnterpriseProviderRegistry):
    """Set global provider registry"""
    global _global_registry
    _global_registry = registry


# Export for convenience
__all__ = [
    'EnterpriseProviderRegistry',
    'ProviderConfig', 
    'ModelSpec',
    'TokenPricing',
    'ProviderTier',
    'ModelCapability',
    'get_provider_registry',
    'set_provider_registry'
]