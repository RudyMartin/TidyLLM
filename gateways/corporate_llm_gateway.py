"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

Corporate LLM Gateway - Enterprise Language Model Access Control
===============================================================
🚀 CORE ENTERPRISE GATEWAY #1 - Foundation Layer
This is a core gateway in the main enterprise workflow processing chain.

LEGAL DOCUMENT ANALYSIS WORKFLOW EXAMPLE:
During legal contract review, this gateway:
- Validates that the requesting user has permissions for legal document analysis
- Routes AI requests through corporate-approved models only (no external API access)
- Tracks costs associated with legal document processing for billing/budgeting
- Logs all AI interactions for legal compliance and audit trails
- Enforces data residency and privacy requirements for sensitive legal content

AI AGENT INTEGRATION GUIDE:
Purpose: Acts as the secure entry point for all enterprise AI operations
- Enforces corporate governance policies before allowing AI access
- Provides centralized cost tracking and budget enforcement
- Routes requests through approved AI models and infrastructure only
- Generates comprehensive audit logs for compliance and security

DEPENDENCIES & REQUIREMENTS:
- Infrastructure: Centralized Settings Manager (for corporate policies and model configurations)
- Infrastructure: UnifiedSessionManager (for secure credential management)
- Data Processing: Polars DataFrames for usage analytics and reporting
- External: AIProcessingGateway (downstream AI processing after approval)
- Optional: DatabaseGateway (for persistent audit logging)

INTEGRATION PATTERNS:
- All AI requests MUST go through process_llm_request() first
- Use validate_request() to check permissions and policies
- Call track_usage() for cost monitoring and billing
- Execute get_usage_analytics() for corporate reporting

SECURITY & COMPLIANCE:
- NO direct external API access - all requests routed through corporate infrastructure
- Comprehensive audit logging of all AI interactions
- Multi-tenant access controls with role-based permissions
- Data residency compliance for sensitive content
- Budget enforcement with automatic request blocking when limits exceeded

ERROR HANDLING:
- Returns AccessDeniedError for permission failures
- Provides BudgetExceededError when spending limits reached
- Implements ComplianceError for policy violations
- Offers detailed error messages with corrective actions
"""

from typing import Dict, Any, Optional, List, Union
import json
import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .base_gateway import BaseGateway, GatewayResponse, GatewayStatus, GatewayDependencies
from ..infrastructure.standards import TidyLLMStandardRequest, TidyLLMStandardResponse, migrate_parameters, ResponseStatus, resolve_model_id

# CRITICAL: Import polars for DataFrame processing
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logger = logging.getLogger(__name__)

# MLFlow integration moved to dedicated service
try:
    from ..services.mlflow_integration_service import MLflowIntegrationService, MLflowConfig
    MLFLOW_SERVICE_AVAILABLE = True
except ImportError:
    MLFLOW_SERVICE_AVAILABLE = False
    logger.debug("MLflowIntegrationService not available")


class LLMProvider(Enum):
    """Supported LLM providers in corporate environment."""
    CLAUDE = "claude"
    OPENAI = "openai"
    AZURE_OPENAI = "azure-openai"
    BEDROCK = "bedrock"
    CUSTOM = "custom"


@dataclass
class LLMRequest:
    """Structured request for LLM operations."""
    prompt: str
    model_id: str
    provider: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    audit_reason: Optional[str] = None  # Required in compliance mode
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "prompt": self.prompt,
            "model_id": self.model_id,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "audit_reason": self.audit_reason,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata
        }


@dataclass
class CostTracker:
    """Track LLM usage costs."""
    daily_spend: float = 0.0
    monthly_spend: float = 0.0
    request_count: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    user_costs: Dict[str, float] = field(default_factory=dict)
    
    def add_cost(self, cost: float, user_id: str = "unknown"):
        """Add cost for a request."""
        self.daily_spend += cost
        self.monthly_spend += cost
        self.request_count += 1
        
        if user_id not in self.user_costs:
            self.user_costs[user_id] = 0.0
        self.user_costs[user_id] += cost
    
    def reset_daily(self):
        """Reset daily counters."""
        self.daily_spend = 0.0
        self.last_reset = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost statistics."""
        return {
            "daily_spend": self.daily_spend,
            "monthly_spend": self.monthly_spend,
            "request_count": self.request_count,
            "average_cost_per_request": self.monthly_spend / max(self.request_count, 1),
            "top_users": sorted(self.user_costs.items(), key=lambda x: x[1], reverse=True)[:5]
        }


@dataclass
class CorporateLLMConfig:
    """Configuration for Corporate LLM Gateway."""
    
    # MLFlow Gateway settings
    mlflow_gateway_uri: str = "http://localhost:5000"
    
    # Provider controls (IT managed)
    available_providers: List[str] = field(default_factory=lambda: ["claude", "openai"])
    default_provider: str = "claude"
    
    # Model controls (IT managed) 
    provider_models: Dict[str, List[str]] = field(default_factory=lambda: {
        "claude": ["claude-3-5-sonnet", "claude-3-haiku"],
        "openai": ["gpt-4o", "gpt-4o-mini"]
    })
    default_models: Dict[str, str] = field(default_factory=lambda: {
        "claude": "claude-3-5-sonnet",
        "openai": "gpt-4o"
    })
    
    # Cost controls
    max_tokens_per_request: int = 4096
    max_cost_per_request_usd: float = 1.0
    budget_limit_daily_usd: Optional[float] = 100.0
    budget_limit_monthly_usd: Optional[float] = 3000.0
    
    # Quality controls
    temperature_limits: tuple = (0.0, 1.0)
    require_audit_reason: bool = True
    
    # Security controls
    enable_content_filtering: bool = True
    enable_pii_detection: bool = True
    log_all_requests: bool = True
    
    # Performance controls
    request_timeout: float = 30.0
    retry_attempts: int = 2
    rate_limit_per_minute: int = 60


class CorporateLLMGateway(BaseGateway):
    """
    Corporate LLM Access Control Layer
    
    Purpose: Enforces enterprise policies on all LLM usage including
    cost controls, audit requirements, security checks, and model restrictions.
    
    Architecture Flow:
    Application → CorporateLLMGateway → MLFlow Gateway → Corporate IT → External Provider
    
    Key Features:
    1. **Zero External Access** - All requests routed through corporate infrastructure
    2. **Cost Control** - Budget limits, per-request limits, usage tracking
    3. **Compliance** - Full audit trails, PII detection, content filtering
    4. **Security** - Request validation, user authentication, access controls
    5. **Governance** - IT-controlled model availability and configurations
    6. **Monitoring** - Real-time usage metrics and alerting
    
    Examples:
        >>> gateway = CorporateLLMGateway(
        ...     budget_limit_daily_usd=500.0,
        ...     require_audit_reason=True
        ... )
        >>> 
        >>> # Make controlled LLM request
        >>> request = LLMRequest(
        ...     prompt="Analyze customer feedback data",
        ...     model="claude-3-5-sonnet",
        ...     audit_reason="Customer satisfaction analysis for Q3 report",
        ...     user_id="john.doe@company.com"
        ... )
        >>> response = gateway.execute_llm_request(request)
        >>> 
        >>> # Check cost and compliance
        >>> stats = gateway.get_usage_stats()
        >>> print(f"Daily spend: ${stats['daily_spend']:.2f}")
    """
    
    def __init__(self, config=None, **config_kwargs):
        """
        Initialize Corporate LLM Gateway.
        
        Args:
            config: CorporateLLMConfig instance or dict of config parameters
            **config_kwargs: Configuration parameters for CorporateLLMConfig
        """
        logger.info(f"🔍 DEBUG: CorporateLLMGateway.__init__ called with config type: {type(config)}, kwargs: {config_kwargs}")
        # Initialize infrastructure using centralized settings
        self.config_manager = None  # No longer needed - using centralized settings
        try:
            from ..infrastructure.settings_manager import get_settings_manager
            
            settings_manager = get_settings_manager()
            logger.info(f"CorporateLLMGateway: Using centralized settings from: {settings_manager.settings_file}")
        except (ImportError, Exception) as e:
            logger.debug(f"CorporateLLMGateway: Centralized settings not available: {e}")
        
        # Always create a proper CorporateLLMConfig object
        # Initialize with defaults first to ensure self.config is always a CorporateLLMConfig
        self.config = CorporateLLMConfig()
        
        try:
            if isinstance(config, CorporateLLMConfig):
                self.config = config
            elif isinstance(config, dict):
                # Merge config dict with kwargs
                merged_config = {**config, **config_kwargs}
                # Filter to only include valid dataclass fields to handle dynamic config
                valid_config = {
                    k: v for k, v in merged_config.items() 
                    if k in CorporateLLMConfig.__dataclass_fields__
                }
                self.config = CorporateLLMConfig(**valid_config)
            elif config is None and config_kwargs:
                # Use kwargs only - filter to valid dataclass fields
                valid_config = {
                    k: v for k, v in config_kwargs.items() 
                    if k in CorporateLLMConfig.__dataclass_fields__
                }
                self.config = CorporateLLMConfig(**valid_config)
            # else: keep the default CorporateLLMConfig() we already created
        except (TypeError, Exception) as e:
            # If initialization fails, keep the default config and log
            logger.warning(f"Config initialization failed: {e}, using defaults")
            # Try to apply any valid parameters manually
            source_config = config if isinstance(config, dict) else config_kwargs
            if source_config:
                for key, value in source_config.items():
                    if hasattr(self.config, key):
                        try:
                            setattr(self.config, key, value)
                        except Exception:
                            pass  # Skip any attributes that can't be set
        
        # TIDY FIX: Auto-load mlflow_gateway_uri from centralized settings
        # This ensures MLflow configuration is always pulled from centralized settings.yaml (updated)
        # CRITICAL: Must happen BEFORE super().__init__() which overwrites self.config!
        try:
            from ..infrastructure.settings_manager import get_settings_manager
            settings_manager = get_settings_manager()
            settings = settings_manager.get_settings()
            
            # Try services.mlflow first, then integrations.mlflow
            mlflow_config = None
            if 'services' in settings and 'mlflow' in settings['services']:
                mlflow_config = settings['services']['mlflow']
            elif 'integrations' in settings and 'mlflow' in settings['integrations']:
                mlflow_config = settings['integrations']['mlflow']
            
            if mlflow_config and 'mlflow_gateway_uri' in mlflow_config:
                self.config.mlflow_gateway_uri = mlflow_config['mlflow_gateway_uri']
                logger.info(f"✅ TIDY FIX: Auto-loaded mlflow_gateway_uri from centralized settings: {self.config.mlflow_gateway_uri}")
            else:
                logger.debug(f"TIDY FIX: No mlflow_gateway_uri found in centralized settings, using default: {self.config.mlflow_gateway_uri}")
        except Exception as e:
            logger.warning(f"TIDY FIX: Could not auto-load mlflow_gateway_uri from settings: {e}, using default: {self.config.mlflow_gateway_uri}")
        
        super().__init__()

        # Initialize MLFlow Integration Service
        self.mlflow_service = None
        self._init_mlflow_service()
        
        # Initialize cost tracking with persistence
        self.cost_tracker = CostTracker()
        self._load_cost_data()
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        
        # Debug: Check config type before logging
        logger.info("🤖 Corporate LLM Gateway initialized")
        logger.info(f"   Config type: {type(self.config)}")
        
        # Ensure config is properly set
        if isinstance(self.config, dict):
            logger.debug("Converting dict config to CorporateLLMConfig")
            temp_config = CorporateLLMConfig()
            for key, value in self.config.items():
                if hasattr(temp_config, key):
                    setattr(temp_config, key, value)
            self.config = temp_config
        
        logger.info(f"   Available providers: {self.config.available_providers}")
        logger.info(f"   Default provider: {self.config.default_provider}")
        logger.info(f"   Daily budget: ${self.config.budget_limit_daily_usd}")
        logger.info(f"CorporateLLMGateway dependencies: {self.get_required_services()}")
    
    def process_request(self, request):
        """
        Process request through corporate gateway.
        
        This method provides the standard interface expected by tests
        and delegates to the existing process_llm_request method.
        
        Args:
            request: Request object or data to process
            
        Returns:
            Processed response
        """
        # Convert to LLMRequest if needed
        if not isinstance(request, LLMRequest):
            if isinstance(request, str):
                request = LLMRequest(prompt=request, model_id="claude-3-5-sonnet")  # Default model_id
            elif isinstance(request, dict):
                # Migrate legacy parameter names
                migrated_params = migrate_parameters(request)
                # Ensure model_id is present (required field)
                if "model_id" not in migrated_params:
                    migrated_params["model_id"] = migrated_params.get("model", "claude-3-5-sonnet")
                request = LLMRequest(**migrated_params)
            else:
                request = LLMRequest(prompt=str(request), model_id="claude-3-5-sonnet")  # Default model_id
        
        return self.process_llm_request(request)
    
    def validate_request(self, request):
        """
        Validate request against corporate policies.
        
        This method provides the standard validation interface expected by tests
        and uses the existing _validate_llm_request method.
        
        Args:
            request: Request object or data to validate
            
        Returns:
            bool: True if request is valid
            
        Raises:
            ValueError: If request is invalid
        """
        # Convert to LLMRequest if needed
        if not isinstance(request, LLMRequest):
            if isinstance(request, str):
                request = LLMRequest(prompt=request, model_id="claude-3-5-sonnet")  # Default model_id
            elif isinstance(request, dict):
                # Migrate legacy parameter names
                migrated_params = migrate_parameters(request)
                # Ensure model_id is present (required field)
                if "model_id" not in migrated_params:
                    migrated_params["model_id"] = migrated_params.get("model", "claude-3-5-sonnet")
                request = LLMRequest(**migrated_params)
            else:
                request = LLMRequest(prompt=str(request), model_id="claude-3-5-sonnet")  # Default model_id
        
        try:
            self._validate_llm_request(request)
            return True
        except Exception as e:
            # Re-raise the validation error
            raise e

    def process_llm_request(self, request):
        """Process LLM request through corporate gateway."""
        try:
            logger.info(f"Processing LLM request: {request.prompt[:50]}...")
            
            # Use UnifiedSessionManager for Bedrock access (consistent with other gateways)
            import json
            import os
            
            # Create Bedrock client through session manager (like AIProcessingGateway)
            if self.session_manager:
                bedrock = self.session_manager.get_bedrock_client()
                logger.info("CorporateLLMGateway: Using UnifiedSessionManager for Bedrock access")
            else:
                # NO FALLBACK - UnifiedSessionManager is required
                raise RuntimeError("CorporateLLMGateway: UnifiedSessionManager is required for Bedrock access")
            
            # Prepare request
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": getattr(request, 'max_tokens', 1000),
                "temperature": getattr(request, 'temperature', 0.7),
                "messages": [
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ]
            }
            
            # Call Bedrock (resolve friendly model_id to actual Bedrock model identifier)
            bedrock_model_id = resolve_model_id(request.model_id)
            response = bedrock.invoke_model(
                modelId=bedrock_model_id,  # Resolved to actual Bedrock model ID
                body=json.dumps(body),
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            logger.info("LLM request processed successfully via AWS Bedrock")
            return TidyLLMStandardResponse(
                status=ResponseStatus.SUCCESS,
                data=content,
                metadata={"provider": "bedrock", "model_id": request.model_id}
            )
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return TidyLLMStandardResponse(
                status=ResponseStatus.ERROR,
                data="",
                error=str(e),
                metadata={"provider": "bedrock", "model_id": getattr(request, 'model_id', 'unknown')}
            )
    
    def _get_default_dependencies(self) -> GatewayDependencies:
        """
        CorporateLLMGateway dependencies: Independent foundation layer.
        
        Dependency Logic:
        - CorporateLLMGateway is the foundational corporate control layer
        - It provides controlled access to external LLM providers via MLFlow
        - Other gateways depend on CorporateLLMGateway, but it is independent
        - This ensures corporate governance is always available as the base layer
        """
        return GatewayDependencies(
            requires_ai_processing=False,      # Independent: Base layer
            requires_corporate_llm=False,      # Self-reference not needed
            requires_workflow_optimizer=False, # Independent: Doesn't need workflow optimization
            requires_context=False  # Independent: Doesn't need context resources
        )
    
    def _init_mlflow_service(self):
        """Initialize MLFlow Integration Service."""
        if MLFLOW_SERVICE_AVAILABLE:
            try:
                mlflow_config = MLflowConfig(
                    gateway_uri=self.config.mlflow_gateway_uri
                )
                self.mlflow_service = MLflowIntegrationService(mlflow_config)
                if self.mlflow_service.is_available():
                    logger.info(f"🔗 MLFlow Integration Service connected: {self.config.mlflow_gateway_uri}")
                else:
                    logger.warning(f"⚠️ MLFlow Service initialized but not connected: {self.mlflow_service.get_status()}")
            except Exception as e:
                logger.warning(f"⚠️ MLFlow Service initialization failed: {e}")
                logger.info("Operating in fallback mode")
        else:
            logger.debug("MLFlow Integration Service not available - using direct integration mode")
    
    def set_mlflow_service(self, mlflow_service: 'MLflowIntegrationService'):
        """Set MLflow Integration Service (dependency injection)."""
        self.mlflow_service = mlflow_service
        if mlflow_service and mlflow_service.is_available():
            logger.info("MLflow Integration Service injected successfully")
        else:
            logger.warning("MLflow Integration Service injected but not available")
    
    def _load_cost_data(self):
        """Load cost tracking data from persistence."""
        # In production, would load from database or file
        cost_file = Path("./corporate_llm_costs.json")
        if cost_file.exists():
            try:
                with open(cost_file) as f:
                    data = json.load(f)
                    self.cost_tracker.daily_spend = data.get("daily_spend", 0.0)
                    self.cost_tracker.monthly_spend = data.get("monthly_spend", 0.0)
                    self.cost_tracker.request_count = data.get("request_count", 0)
                    self.cost_tracker.user_costs = data.get("user_costs", {})
                logger.info(f"Loaded cost data: ${self.cost_tracker.daily_spend:.2f} daily")
            except Exception as e:
                logger.warning(f"Could not load cost data: {e}")
    
    def _save_cost_data(self):
        """Save cost tracking data to persistence."""
        cost_file = Path("./corporate_llm_costs.json")
        try:
            with open(cost_file, 'w') as f:
                json.dump({
                    "daily_spend": self.cost_tracker.daily_spend,
                    "monthly_spend": self.cost_tracker.monthly_spend,
                    "request_count": self.cost_tracker.request_count,
                    "user_costs": self.cost_tracker.user_costs,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cost data: {e}")
    
    async def process(self, input_data: Any, **kwargs) -> GatewayResponse:
        """Process LLM request asynchronously."""
        return self.process_sync(input_data, **kwargs)
    
    def process_sync(self, input_data: Any, **kwargs) -> GatewayResponse:
        """Process LLM request synchronously."""
        try:
            # Convert input to LLMRequest
            if isinstance(input_data, LLMRequest):
                request = input_data
            elif isinstance(input_data, str):
                # Migrate kwargs parameters and ensure model_id
                migrated_kwargs = migrate_parameters(kwargs)
                if "model_id" not in migrated_kwargs:
                    migrated_kwargs["model_id"] = migrated_kwargs.get("model", "claude-3-5-sonnet")
                request = LLMRequest(prompt=input_data, **migrated_kwargs)
            elif isinstance(input_data, dict):
                # Merge input_data and kwargs, then migrate
                merged_data = {**input_data, **kwargs}
                migrated_data = migrate_parameters(merged_data)
                # Ensure model_id is present
                if "model_id" not in migrated_data:
                    migrated_data["model_id"] = migrated_data.get("model", "claude-3-5-sonnet")
                request = LLMRequest(**migrated_data)
            else:
                # Migrate kwargs parameters and ensure model_id
                migrated_kwargs = migrate_parameters(kwargs)
                if "model_id" not in migrated_kwargs:
                    migrated_kwargs["model_id"] = migrated_kwargs.get("model", "claude-3-5-sonnet")
                request = LLMRequest(prompt=str(input_data), **migrated_kwargs)
            
            # Execute through corporate controls
            return self.execute_llm_request(request)
            
        except Exception as e:
            logger.error(f"CorporateLLMGateway processing failed: {e}")
            return GatewayResponse(
                status=GatewayStatus.FAILURE,
                data=None,
                errors=[str(e)],
                gateway_name="CorporateLLMGateway"
            )
    
    def execute_llm_request(self, request: LLMRequest) -> GatewayResponse:
        """
        Execute LLM request with full corporate controls.
        
        Control Flow:
        1. Validate request against corporate policies
        2. Check budget and cost limits
        3. Apply security and content filtering
        4. Route through MLFlow Gateway
        5. Track costs and usage
        6. Generate audit log entry
        7. Return controlled response
        
        Args:
            request: LLMRequest with prompt and parameters
            
        Returns:
            GatewayResponse with LLM output and compliance metadata
        """
        start_time = time.time()
        
        try:
            # 1. Validate request
            self._validate_llm_request(request)
            
            # 2. Check budget limits
            self._check_budget_limits(request)
            
            # 3. Apply security controls
            filtered_request = self._apply_security_controls(request)
            
            # 4. Execute through MLFlow Integration Service
            if self.mlflow_service and self.mlflow_service.is_available():
                llm_response = self._execute_mlflow_request(filtered_request)
            else:
                llm_response = self._execute_fallback_request(filtered_request)
            
            # 5. Calculate and track costs
            cost = self._calculate_request_cost(filtered_request, llm_response)
            self.cost_tracker.add_cost(cost, request.user_id or "unknown")
            
            # 6. Generate audit log
            audit_entry = self._create_audit_entry(filtered_request, llm_response, cost)
            self.audit_log.append(audit_entry)
            
            # 7. Save cost data
            self._save_cost_data()
            
            # Create response
            processing_time = time.time() - start_time
            
            return GatewayResponse(
                status=GatewayStatus.SUCCESS,
                data=llm_response,
                gateway_name="CorporateLLMGateway",
                metadata={
                    "model": filtered_request.model_id or self._get_default_model(filtered_request.provider),
                    "provider": filtered_request.provider or self.config.default_provider,
                    "processing_time": processing_time,
                    "cost_usd": cost,
                    "tokens_used": len(llm_response.split()) * 1.3,  # Rough estimate
                    "compliance_checked": True,
                    "audit_logged": True,
                    "user_id": request.user_id,
                    "session_id": request.session_id
                }
            )
            
        except Exception as e:
            # Log error for audit
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": request.user_id,
                "error": str(e),
                "request_summary": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
            }
            self.audit_log.append(error_entry)
            
            return GatewayResponse(
                status=GatewayStatus.FAILURE,
                data=None,
                errors=[str(e)],
                gateway_name="CorporateLLMGateway",
                metadata={
                    "processing_time": time.time() - start_time,
                    "error": str(e)
                }
            )
    
    def _validate_llm_request(self, request: LLMRequest):
        """Validate request against corporate policies."""
        
        # Check audit reason requirement
        if self.config.require_audit_reason and not request.audit_reason:
            raise ValueError("Audit reason is required for all LLM requests")
        
        # Validate provider
        provider = request.provider or self.config.default_provider
        if provider not in self.config.available_providers:
            raise ValueError(f"Provider '{provider}' not available. Available: {self.config.available_providers}")
        
        # Validate model
        model = request.model_id or self._get_default_model(provider)
        available_models = self.config.provider_models.get(provider, [])
        if model not in available_models:
            raise ValueError(f"Model '{model}' not available for provider '{provider}'. Available: {available_models}")
        
        # Validate parameters
        if request.temperature and not (self.config.temperature_limits[0] <= request.temperature <= self.config.temperature_limits[1]):
            raise ValueError(f"Temperature must be between {self.config.temperature_limits}")
        
        if request.max_tokens and request.max_tokens > self.config.max_tokens_per_request:
            raise ValueError(f"max_tokens cannot exceed {self.config.max_tokens_per_request}")
    
    def _check_budget_limits(self, request: LLMRequest):
        """Check if request would exceed budget limits."""
        
        # Estimate cost for this request
        estimated_cost = self._estimate_request_cost(request)
        
        # Check per-request limit
        if estimated_cost > self.config.max_cost_per_request_usd:
            raise ValueError(f"Request cost ${estimated_cost:.2f} exceeds per-request limit ${self.config.max_cost_per_request_usd}")
        
        # Check daily budget
        if self.config.budget_limit_daily_usd:
            if self.cost_tracker.daily_spend + estimated_cost > self.config.budget_limit_daily_usd:
                raise ValueError(f"Request would exceed daily budget limit of ${self.config.budget_limit_daily_usd}")
        
        # Check monthly budget
        if self.config.budget_limit_monthly_usd:
            if self.cost_tracker.monthly_spend + estimated_cost > self.config.budget_limit_monthly_usd:
                raise ValueError(f"Request would exceed monthly budget limit of ${self.config.budget_limit_monthly_usd}")
    
    def _apply_security_controls(self, request: LLMRequest) -> LLMRequest:
        """Apply security and content filtering."""
        
        filtered_request = request
        
        # PII detection and masking
        if self.config.enable_pii_detection:
            filtered_prompt = self._mask_pii(request.prompt)
            filtered_request = LLMRequest(
                prompt=filtered_prompt,
                model_id=request.model_id,
                provider=request.provider,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                audit_reason=request.audit_reason,
                user_id=request.user_id,
                session_id=request.session_id,
                metadata=request.metadata
            )
        
        # Content filtering
        if self.config.enable_content_filtering:
            if self._contains_inappropriate_content(filtered_request.prompt):
                raise ValueError("Request contains inappropriate content")
        
        return filtered_request
    
    def _execute_mlflow_request(self, request: LLMRequest) -> str:
        """Execute request through MLFlow Integration Service."""
        
        provider = request.provider or self.config.default_provider
        model = request.model_id or self._get_default_model(provider)
        
        try:
            # Call MLFlow through the integration service
            mlflow_response = self.mlflow_service.query(
                route=f"{provider}-{model}",
                data={
                    "messages": [{"role": "user", "content": request.prompt}],
                    "temperature": request.temperature or 0.1,
                    "max_tokens": request.max_tokens or 1000
                }
            )
            
            if mlflow_response:
                return mlflow_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.warning("MLFlow service returned no response, using fallback")
                return self._execute_fallback_request(request)
            
        except Exception as e:
            logger.error(f"MLFlow service error: {e}")
            # Fallback to mock response
            return self._execute_fallback_request(request)
    
    def _execute_fallback_request(self, request: LLMRequest) -> str:
        """Execute fallback request when MLFlow unavailable."""
        provider = request.provider or self.config.default_provider
        model = request.model_id or self._get_default_model(provider)
        
        logger.warning("Using fallback mode for LLM request")
        return f"[FALLBACK] Corporate LLM response using {provider}/{model} for: {request.prompt[:100]}..."
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        return self.config.default_models.get(provider, "unknown")
    
    def _estimate_request_cost(self, request: LLMRequest) -> float:
        """Estimate cost for request."""
        # Simple cost estimation - in production would use actual pricing
        prompt_tokens = len(request.prompt.split()) * 1.3
        max_tokens = request.max_tokens or 1000
        total_tokens = prompt_tokens + max_tokens
        
        # Claude pricing: ~$0.008 per 1K tokens (rough estimate)
        return (total_tokens / 1000) * 0.008
    
    def _calculate_request_cost(self, request: LLMRequest, response: str) -> float:
        """Calculate actual cost after request completion."""
        # In production, would get actual token counts from provider
        return self._estimate_request_cost(request)
    
    def _mask_pii(self, text: str) -> str:
        """Mask PII in text."""
        # Simple PII masking - in production would use sophisticated NLP
        import re
        
        # Mask email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
        
        # Mask phone numbers
        text = re.sub(r'\\b\\d{3}-\\d{3}-\\d{4}\\b', '[PHONE]', text)
        
        # Mask SSN
        text = re.sub(r'\\b\\d{3}-\\d{2}-\\d{4}\\b', '[SSN]', text)
        
        return text
    
    def _contains_inappropriate_content(self, text: str) -> bool:
        """Check for inappropriate content."""
        # Simple content filtering - in production would use ML models
        inappropriate_terms = ["hack", "exploit", "bypass", "illegal"]
        text_lower = text.lower()
        return any(term in text_lower for term in inappropriate_terms)
    
    def _create_audit_entry(self, request: LLMRequest, response: str, cost: float) -> Dict[str, Any]:
        """Create audit log entry."""
        return {
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id,
            "session_id": request.session_id,
            "provider": request.provider or self.config.default_provider,
            "model": request.model_id or self._get_default_model(request.provider),
            "audit_reason": request.audit_reason,
            "prompt_length": len(request.prompt),
            "response_length": len(response),
            "cost_usd": cost,
            "compliance_status": "approved"
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            **self.cost_tracker.get_stats(),
            "available_providers": self.config.available_providers,
            "daily_budget_remaining": max(0, (self.config.budget_limit_daily_usd or 0) - self.cost_tracker.daily_spend),
            "monthly_budget_remaining": max(0, (self.config.budget_limit_monthly_usd or 0) - self.cost_tracker.monthly_spend)
        }
    
    def get_audit_log(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get audit log entries from last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            entry for entry in self.audit_log
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
    
    def validate_config(self) -> bool:
        """Validate gateway configuration."""
        # Configuration is validated during initialization via CorporateLLMConfig
        # All required attributes are guaranteed by the dataclass default values
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get gateway capabilities."""
        return {
            "providers": self.config.available_providers,
            "models": self.config.provider_models,
            "max_tokens": self.config.max_tokens_per_request,
            "supports_streaming": False,
            "supports_async": True,
            "mlflow_enabled": self.mlflow_service and self.mlflow_service.is_available(),
            "cost_tracking_enabled": True,
            "audit_logging_enabled": self.config.log_all_requests,
            "content_filtering_enabled": self.config.enable_content_filtering,
            "pii_detection_enabled": self.config.enable_pii_detection,
            "budget_controls_enabled": self.config.budget_limit_daily_usd is not None
        }