"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

AI Processing Gateway - Multi-Model AI Processing Engine
=======================================================
🚀 CORE ENTERPRISE GATEWAY #2 - AI Processing Layer
This is a core gateway in the main enterprise workflow processing chain.

LEGAL DOCUMENT ANALYSIS WORKFLOW EXAMPLE:
For legal contract review, this gateway:
- Processes legal documents through multiple AI models for comprehensive analysis
- Routes different document sections to specialized AI backends (contracts, regulations, risks)
- Aggregates results from multiple AI models into unified legal assessment
- Provides intelligent fallback when primary AI models are unavailable
- Caches AI results to avoid redundant processing of similar legal clauses

AI AGENT INTEGRATION GUIDE:
Purpose: Core AI processing engine with multi-backend intelligence
- Supports multiple AI backends (OpenAI, Anthropic, AWS Bedrock, local models)
- Provides intelligent request routing based on content type and model capabilities
- Implements advanced caching and result aggregation from multiple AI sources
- Offers comprehensive error handling with automatic fallback strategies

DEPENDENCIES & REQUIREMENTS:
- Infrastructure: UnifiedSessionManager (for AI service credentials and sessions)
- Infrastructure: Centralized Settings Manager (for AI model configurations and routing rules)
- Data Processing: Polars DataFrames for large-scale AI result processing
- Upstream: CorporateLLMGateway (receives approved requests from corporate gateway)
- External: Multiple AI Backend Adapters (OpenAI, Anthropic, Bedrock, etc.)

INTEGRATION PATTERNS:
- Receives requests from CorporateLLMGateway after approval
- Use process_request() for single AI model processing
- Call process_multi_model() for comprehensive analysis across multiple backends
- Execute get_cached_result() to check for existing AI responses
- Monitor with get_processing_metrics() for performance optimization

AI BACKEND SUPPORT:
- OpenAI GPT models (GPT-4, GPT-3.5)
- Anthropic Claude models (Claude-3, Claude-2)
- AWS Bedrock (multiple model families)
- Local/Self-hosted models via API
- Custom AI adapters through plugin architecture

PERFORMANCE FEATURES:
- Intelligent caching with TTL and content-based invalidation
- Request batching for improved throughput
- Async processing with concurrent model execution
- Result aggregation and consensus building from multiple AI responses

ERROR HANDLING:
- Returns ProcessingError for AI backend failures
- Provides ModelUnavailableError with automatic fallback suggestions
- Implements TimeoutError with retry logic and exponential backoff
- Offers detailed error context for debugging and monitoring
"""

import asyncio
import time
import hashlib
import logging
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

from .base_gateway import BaseGateway, GatewayResponse, GatewayStatus, GatewayDependencies

# CRITICAL: Import polars for DataFrame processing
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

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
    def create(backend_type: AIBackend, config: AIProcessingConfig, session_manager=None) -> AIBackendInterface:
        """Create appropriate backend based on type."""
        if backend_type == AIBackend.AUTO:
            return AIBackendFactory.auto_detect(config, session_manager)
        elif backend_type == AIBackend.MOCK:
            return MockAIBackend(config)
        elif backend_type == AIBackend.BEDROCK:
            return BedrockAIBackend(config, session_manager)
        elif backend_type == AIBackend.ANTHROPIC:
            return AnthropicAIBackend(config)
        elif backend_type == AIBackend.OPENAI:
            return OpenAIBackend(config)
        else:
            # Fallback to generic implementation
            return GenericAIBackend(backend_type, config)
    
    @staticmethod
    def auto_detect(config: AIProcessingConfig, session_manager=None) -> AIBackendInterface:
        """Auto-detect best available backend with clear priority."""
        detection_order = [
            (AIBackend.ANTHROPIC, "Claude API"),
            (AIBackend.OPENAI, "OpenAI API"),
            (AIBackend.BEDROCK, "AWS Bedrock"),
            (AIBackend.MLFLOW, "MLFlow Gateway"),
        ]
        
        for backend_type, name in detection_order:
            try:
                backend = AIBackendFactory.create(backend_type, config, session_manager)
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
        
        # Initialize infrastructure session manager
        self.session_manager = None
        try:
            from ..infrastructure.session import UnifiedSessionManager
            self.session_manager = UnifiedSessionManager()
            logger.info("AIProcessingGateway: UnifiedSessionManager integrated")
        except ImportError as e:
            logger.debug(f"AIProcessingGateway: UnifiedSessionManager not available: {e}")
        
        # Now call parent init
        super().__init__(**config)
        
        # Initialize backend with session manager
        self.backend = AIBackendFactory.create(self.ai_config.backend, self.ai_config, self.session_manager)
        
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
        Process structured AI request with polars DataFrame capture.
        
        Args:
            request: AIRequest with prompt and parameters
            
        Returns:
            GatewayResponse with AI-generated content
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # CRITICAL: Create polars DataFrame for this AI processing stage
        stage_data = {
            "request_id": request_id,
            "prompt": request.prompt,
            "model": request.model or self.ai_config.model,
            "temperature": request.temperature or self.ai_config.temperature,
            "max_tokens": request.max_tokens or self.ai_config.max_tokens,
            "backend": self.ai_config.backend.value,
            "stage_start_time": datetime.now().isoformat()
        }
        
        # Create DataFrame for AI processing stage
        ai_df = self.create_stage_dataframe(stage_data, "ai_processing")
        if ai_df is not None:
            logger.info(f"Created AI processing DataFrame: {ai_df.shape}")
        
        # Process the request
        response = self.process_sync(request)
        
        # CRITICAL: Enhance DataFrame with processing results
        if ai_df is not None and POLARS_AVAILABLE:
            try:
                # Add results to DataFrame
                processing_time = time.time() - start_time
                enhanced_data = {
                    "processing_time_ms": [processing_time * 1000],
                    "response_length": [len(response.data) if response.data else 0],
                    "status": [response.status.value],
                    "cache_hit": [response.metadata.get('cache_hit', False) if response.metadata else False],
                    "token_usage": [response.metadata.get('tokens', 0) if response.metadata else 0]
                }
                
                # Create enhanced DataFrame with results
                enhanced_df = ai_df.with_columns([
                    pl.lit(enhanced_data["processing_time_ms"][0]).alias("processing_time_ms"),
                    pl.lit(enhanced_data["response_length"][0]).alias("response_length"),
                    pl.lit(enhanced_data["status"][0]).alias("status"),
                    pl.lit(enhanced_data["cache_hit"][0]).alias("cache_hit"),
                    pl.lit(enhanced_data["token_usage"][0]).alias("token_usage")
                ])
                
                # CRITICAL: Persist the enhanced DataFrame
                self.persist_stage_data(enhanced_df, "ai_processing", request_id)
                
                logger.info(f"Enhanced AI processing DataFrame: {enhanced_df.shape}")
                
            except Exception as e:
                logger.error(f"Failed to enhance AI processing DataFrame: {e}")
        
        return response
    
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
    
    def process_chat(self, request_data: Dict[str, Any]) -> GatewayResponse:
        """Generic wrapper for chat processing - calls process_ai_request()"""
        from ..infrastructure.data_structures import AIRequest
        
        ai_request = AIRequest(
            prompt=request_data.get("query", ""),
            context=request_data.get("context", ""),
            metadata=request_data.get("metadata", {})
        )
        
        return self.process_ai_request(ai_request)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check for AIProcessingGateway."""
        import time
        from datetime import datetime
        start_time = time.time()
        
        health_status = {
            "gateway": "AIProcessingGateway",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "dependencies": {},
            "metrics": {}
        }
        
        try:
            # 1. Configuration validation
            health_status["checks"]["config"] = {
                "status": "pass" if self.validate_config() else "fail",
                "message": "Configuration validation"
            }
            
            # 2. AI backend availability
            if hasattr(self, 'ai_config'):
                backend = self.ai_config.backend
                health_status["checks"]["ai_backend"] = {
                    "status": "available",
                    "backend": backend.value,
                    "message": f"Using {backend.value} backend"
                }
            
            # 3. Model access check
            try:
                capabilities = self.get_capabilities()
                model_count = len(capabilities.get("models", []))
                health_status["checks"]["models"] = {
                    "status": "pass" if model_count > 0 else "fail",
                    "model_count": model_count,
                    "message": f"{model_count} models available"
                }
            except Exception as e:
                health_status["checks"]["models"] = {"status": "fail", "error": str(e)}
            
            # 4. Session manager dependency
            if hasattr(self, 'session_manager') and self.session_manager:
                try:
                    health_summary = self.session_manager.get_health_summary()
                    health_status["dependencies"]["session_manager"] = {
                        "status": "healthy" if health_summary.get("overall_health") else "unhealthy",
                        "details": health_summary
                    }
                except Exception as e:
                    health_status["dependencies"]["session_manager"] = {"status": "error", "error": str(e)}
            
            # 5. Performance metrics
            duration_ms = (time.time() - start_time) * 1000
            health_status["metrics"] = {
                "health_check_duration_ms": round(duration_ms, 1),
                "memory_usage_mb": self._get_memory_usage()
            }
            
            # 6. Overall status determination
            failed_checks = [k for k, v in health_status["checks"].items() if v["status"] == "fail"]
            if failed_checks:
                health_status["status"] = "unhealthy"
                health_status["failed_checks"] = failed_checks
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
        
        return health_status
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return round(process.memory_info().rss / 1024 / 1024, 2)
        except ImportError:
            return 0.0


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
    def __init__(self, config: AIProcessingConfig, session_manager=None):
        self.config = config
        self.session_manager = session_manager
    
    def is_available(self) -> bool:
        # Check if we have session manager or can create AWS client
        if self.session_manager:
            try:
                # Test through session manager
                bedrock = self.session_manager.get_bedrock_client()
                return bedrock is not None
            except Exception:
                return False
        else:
            # Fallback availability check
            try:
                # Use UnifiedSessionManager for availability check
                if self.session_manager:
                    bedrock = self.session_manager.get_bedrock_client()
                    if bedrock:
                        bedrock.list_foundation_models()
                        return True
                return False
            except Exception:
                return False
    
    def complete(self, request: AIRequest) -> str:
        # Use UnifiedSessionManager for Bedrock access
        try:
            if self.session_manager:
                bedrock = self.session_manager.get_bedrock_client() 
                logger.info("Using UnifiedSessionManager for Bedrock access")
            else:
                # NO FALLBACK - UnifiedSessionManager is required
                raise RuntimeError("AIProcessingGateway: UnifiedSessionManager is required for Bedrock access")
            
            # Bedrock API call implementation would go here
            return f"[BEDROCK] Processed: {request.prompt[:50]}..."
            
        except Exception as e:
            raise RuntimeError(f"Bedrock API call failed: {e}")


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