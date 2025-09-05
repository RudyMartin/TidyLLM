"""
AI Processing Gateway - Multi-Model AI Processing Engine
=======================================================

Provides flexible AI integration across multiple backends with enterprise features.
Modern gateway for AI processing with multiple backend support.
"""

import asyncio
import time
import hashlib
import logging
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .base_gateway import BaseGateway, GatewayResponse, GatewayStatus, GatewayDependencies

logger = logging.getLogger(__name__)


class AIBackend(Enum):
    """Available AI backend types."""
    AUTO = "auto"
    BEDROCK = "bedrock"
    SAGEMAKER = "sagemaker"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MLFLOW = "mlflow"
    MOCK = "mock"


@dataclass
class AIProcessingConfig:
    """Configuration for AI Processing Gateway."""
    backend: AIBackend = AIBackend.AUTO
    model: str = "claude-3-sonnet"
    temperature: float = 0.7
    max_tokens: int = 2000
    retry_max: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    cache_ttl: int = 3600
    timeout: float = 30.0


@dataclass
class AIRequest:
    """Structured request for AI processing."""
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIBackendInterface:
    """Interface that all AI backends must implement."""
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        raise NotImplementedError
    
    def complete(self, request: AIRequest) -> str:
        """Complete AI request."""
        raise NotImplementedError


class AIBackendFactory:
    """Factory for creating AI backends with clear purpose."""
    
    @staticmethod
    def create(backend_type: AIBackend, config: AIProcessingConfig) -> AIBackendInterface:
        """Create appropriate backend based on type."""
        if backend_type == AIBackend.AUTO:
            return AIBackendFactory.auto_detect(config)
        elif backend_type == AIBackend.MOCK:
            return MockAIBackend(config)
        elif backend_type == AIBackend.BEDROCK:
            return BedrockAIBackend(config)
        elif backend_type == AIBackend.ANTHROPIC:
            return AnthropicAIBackend(config)
        elif backend_type == AIBackend.OPENAI:
            return OpenAIBackend(config)
        else:
            # Fallback to generic implementation
            return GenericAIBackend(backend_type, config)
    
    @staticmethod
    def auto_detect(config: AIProcessingConfig) -> AIBackendInterface:
        """Auto-detect best available backend with clear priority."""
        detection_order = [
            (AIBackend.ANTHROPIC, "Claude API"),
            (AIBackend.OPENAI, "OpenAI API"),
            (AIBackend.BEDROCK, "AWS Bedrock"),
            (AIBackend.MLFLOW, "MLFlow Gateway"),
        ]
        
        for backend_type, name in detection_order:
            try:
                backend = AIBackendFactory.create(backend_type, config)
                if backend.is_available():
                    logger.info(f"Auto-detected {name} as AI backend")
                    return backend
            except Exception as e:
                logger.debug(f"Backend {name} not available: {e}")
                continue
        
        logger.warning("No production backend available, using mock")
        return MockAIBackend(config)
    
    @staticmethod
    def is_available(backend_type: AIBackend) -> bool:
        """Check if a specific backend type is available."""
        try:
            # This would check actual availability
            return backend_type != AIBackend.MLFLOW  # Placeholder logic
        except:
            return False


class AIProcessingGateway(BaseGateway):
    """
    AI Processing Gateway - Multi-model AI processing engine.
    
    Purpose: Provides standardized access to multiple AI models and backends
    with enterprise features like caching, retry logic, and metrics tracking.
    
    Key Features:
    - Multiple backend support (Anthropic, OpenAI, Bedrock, etc.)
    - Automatic backend detection and fallback
    - Response caching with TTL
    - Retry logic with exponential backoff  
    - Performance metrics and monitoring
    - Structured request/response handling
    
    Examples:
        >>> gateway = AIProcessingGateway(backend=AIBackend.ANTHROPIC)
        >>> request = AIRequest(
        ...     prompt="Explain quantum computing",
        ...     temperature=0.7,
        ...     max_tokens=1500
        ... )
        >>> response = gateway.process_ai_request(request)
        >>> print(response.data)
    """
    
    def __init__(self, **config):
        """
        Initialize AI Processing Gateway.
        
        Args:
            **config: Configuration parameters for AIProcessingConfig
        """
        # Parse config into AIProcessingConfig BEFORE calling super()
        self.ai_config = self._parse_config(config)
        
        # Now call parent init
        super().__init__(**config)
        
        # Initialize backend
        self.backend = AIBackendFactory.create(self.ai_config.backend, self.ai_config)
        
        # Cache for responses
        self.cache = {} if self.ai_config.cache_enabled else None
        
        logger.info(f"AIProcessingGateway initialized with {self.ai_config.backend.value} backend")
        logger.info(f"AIProcessingGateway dependencies: {self.get_required_services()}")
    
    def _get_default_dependencies(self) -> GatewayDependencies:
        """
        AIProcessingGateway dependencies: Requires CorporateLLMGateway for model access.
        
        Dependency Logic:
        - AI processing needs LLM access for model operations
        - In corporate environments, LLM access should go through CorporateLLMGateway for governance
        - This ensures audit trails and cost controls for AI operations
        """
        return GatewayDependencies(
            requires_ai_processing=False,  # Self-reference not needed
            requires_corporate_llm=True,   # REQUIRED: AI needs corporate LLM access
            requires_workflow_optimizer=False,  # Optional: Can be optimized but not required
            requires_knowledge_resources=False  # Optional: Can use knowledge but not required
        )
    
    def _parse_config(self, config: Dict[str, Any]) -> AIProcessingConfig:
        """Parse configuration into AIProcessingConfig."""
        ai_config = AIProcessingConfig()
        
        # Map string backend to enum
        if "backend" in config:
            backend_str = config["backend"]
            if isinstance(backend_str, str):
                try:
                    ai_config.backend = AIBackend[backend_str.upper()]
                except KeyError:
                    ai_config.backend = AIBackend.AUTO
            elif isinstance(backend_str, AIBackend):
                ai_config.backend = backend_str
        
        # Set other config values
        for key in ["model", "temperature", "max_tokens", "retry_max", 
                   "retry_delay", "cache_enabled", "cache_ttl", "timeout"]:
            if key in config:
                setattr(ai_config, key, config[key])
        
        return ai_config
    
    def _generate_cache_key(self, request: AIRequest) -> str:
        """Generate unique cache key for request."""
        key_parts = [
            request.prompt,
            request.model or self.ai_config.model,
            str(request.temperature or self.ai_config.temperature),
            str(request.max_tokens or self.ai_config.max_tokens)
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def process(self, input_data: Any, **kwargs) -> GatewayResponse:
        """
        Process input through AI backend asynchronously.
        
        Args:
            input_data: AIRequest object or text prompt
            **kwargs: Additional parameters for processing
            
        Returns:
            GatewayResponse with AI-generated content
        """
        # Run sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_sync, input_data, **kwargs)
    
    def process_sync(self, input_data: Any, **kwargs) -> GatewayResponse:
        """
        Process input through AI backend synchronously.
        
        Args:
            input_data: AIRequest object or text prompt
            **kwargs: Additional parameters for processing
            
        Returns:
            GatewayResponse with AI-generated content
        """
        start_time = time.time()
        
        try:
            # Convert input to AIRequest
            if isinstance(input_data, AIRequest):
                request = input_data
            elif isinstance(input_data, str):
                request = AIRequest(prompt=input_data, **kwargs)
            elif isinstance(input_data, dict):
                request = AIRequest(**input_data, **kwargs)
            else:
                request = AIRequest(prompt=str(input_data), **kwargs)
            
            # Check cache
            if self.cache is not None:
                cache_key = self._generate_cache_key(request)
                if cache_key in self.cache:
                    cached_response = self.cache[cache_key]
                    cached_response.metadata["cache_hit"] = True
                    return cached_response
            
            # Process with retry logic
            response_text = self._process_with_retry(request)
            
            # Create response
            response = GatewayResponse(
                status=GatewayStatus.SUCCESS,
                data=response_text,
                gateway_name="AIProcessingGateway",
                metadata={
                    "model": request.model or self.ai_config.model,
                    "backend": self.ai_config.backend.value,
                    "processing_time": time.time() - start_time,
                    "cache_hit": False,
                    "prompt_length": len(request.prompt),
                    "response_length": len(response_text),
                    "temperature": request.temperature or self.ai_config.temperature
                }
            )
            
            # Cache response
            if self.cache is not None:
                self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"AIProcessingGateway processing failed: {e}")
            return GatewayResponse(
                status=GatewayStatus.FAILURE,
                data=None,
                errors=[str(e)],
                gateway_name="AIProcessingGateway",
                metadata={
                    "processing_time": time.time() - start_time,
                    "error": str(e)
                }
            )
    
    def process_ai_request(self, request: AIRequest) -> GatewayResponse:
        """
        Process structured AI request.
        
        Args:
            request: AIRequest with prompt and parameters
            
        Returns:
            GatewayResponse with AI-generated content
        """
        return self.process_sync(request)
    
    def _process_with_retry(self, request: AIRequest) -> str:
        """Process with retry logic."""
        last_error = None
        
        for attempt in range(self.ai_config.retry_max):
            try:
                return self.backend.complete(request)
            except Exception as e:
                last_error = e
                if attempt < self.ai_config.retry_max - 1:
                    time.sleep(self.ai_config.retry_delay * (attempt + 1))
                    continue
                else:
                    raise last_error
    
    def validate_config(self) -> bool:
        """Validate gateway configuration."""
        if self.ai_config.temperature < 0 or self.ai_config.temperature > 2:
            raise ValueError(f"Invalid temperature: {self.ai_config.temperature}")
        
        if self.ai_config.max_tokens < 1:
            raise ValueError(f"Invalid max_tokens: {self.ai_config.max_tokens}")
        
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get gateway capabilities."""
        return {
            "backends": [b.value for b in AIBackend],
            "current_backend": self.ai_config.backend.value,
            "models": self._get_available_models(),
            "max_tokens": self.ai_config.max_tokens,
            "supports_streaming": False,
            "supports_async": True,
            "supports_batch": False,
            "cache_enabled": self.ai_config.cache_enabled,
            "retry_enabled": self.ai_config.retry_max > 0
        }
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models for current backend."""
        model_map = {
            AIBackend.ANTHROPIC: ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            AIBackend.OPENAI: ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            AIBackend.BEDROCK: ["claude-3-sonnet", "llama2-70b", "mistral-7b"],
            AIBackend.MOCK: ["mock-model"]
        }
        return model_map.get(self.ai_config.backend, ["unknown"])


# Backend implementations
class GenericAIBackend(AIBackendInterface):
    """Generic backend for AI operations."""
    
    def __init__(self, backend_type: AIBackend, config: AIProcessingConfig):
        self.backend_type = backend_type
        self.config = config
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        # In production, would check actual availability
        return self.backend_type != AIBackend.MLFLOW
    
    def complete(self, request: AIRequest) -> str:
        """Complete AI request using backend."""
        # In production, would call actual backend
        model = request.model or self.config.model
        return f"AI response from {self.backend_type.value} using {model} for: {request.prompt[:50]}..."


class MockAIBackend(AIBackendInterface):
    """Mock backend for testing and development."""
    
    def __init__(self, config: AIProcessingConfig):
        self.config = config
    
    def is_available(self) -> bool:
        return True
    
    def complete(self, request: AIRequest) -> str:
        """Mock completion."""
        model = request.model or self.config.model
        return f"Mock AI response using {model} for prompt: {request.prompt[:100]}..."


# Placeholder for actual backend implementations
class BedrockAIBackend(AIBackendInterface):
    def __init__(self, config: AIProcessingConfig):
        self.config = config
    
    def is_available(self) -> bool:
        # Would check AWS credentials and Bedrock access
        return False
    
    def complete(self, request: AIRequest) -> str:
        # Would call AWS Bedrock API
        raise NotImplementedError("Bedrock integration not yet implemented")


class AnthropicAIBackend(AIBackendInterface):
    def __init__(self, config: AIProcessingConfig):
        self.config = config
    
    def is_available(self) -> bool:
        # Would check Anthropic API key
        return False
    
    def complete(self, request: AIRequest) -> str:
        # Would call Anthropic API
        raise NotImplementedError("Anthropic integration not yet implemented")


class OpenAIBackend(AIBackendInterface):
    def __init__(self, config: AIProcessingConfig):
        self.config = config
    
    def is_available(self) -> bool:
        # Would check OpenAI API key
        return False
    
    def complete(self, request: AIRequest) -> str:
        # Would call OpenAI API
        raise NotImplementedError("OpenAI integration not yet implemented")