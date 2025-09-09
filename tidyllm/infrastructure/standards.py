"""
TidyLLM Standard Data Structures
===============================

This module defines the standard data structures and parameter names used throughout
TidyLLM to eliminate transformation overhead and ensure consistency.

These standards replace the multiple incompatible Config classes and response formats
found throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ResponseStatus(Enum):
    """Standard response status values."""
    SUCCESS = "success"
    ERROR = "error" 
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class ModelProvider(Enum):
    """Standard model provider identifiers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    BEDROCK = "bedrock"
    AZURE = "azure"
    LOCAL = "local"


@dataclass
class TidyLLMStandardConfig:
    """
    Unified configuration structure for all TidyLLM components.
    
    This replaces the multiple incompatible config classes:
    - ServiceConfig, CorporateLLMConfig, DatabaseConfig, etc.
    
    Standard parameter names (use these everywhere):
    - model_id (not model_name, model, llm_model)
    - user_id (not userId, username) 
    - session_id (not sessionId)
    - postgres_username (not db_user)
    - aws_access_key_id (not s3_access_key_id)
    """
    
    # User/Session identifiers (standardized names)
    user_id: str
    session_id: str
    
    # Model/AI parameters (standardized names)
    model_id: str  # Use everywhere instead of model_name/model/llm_model
    provider: ModelProvider = ModelProvider.ANTHROPIC
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Database parameters (standardized names)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_username: str = ""  # Not db_user
    postgres_password: str = ""  # Not db_password  
    postgres_database: str = ""  # Not db_name
    
    # AWS parameters (standardized names)
    aws_access_key_id: str = ""  # Not s3_access_key_id
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    aws_session_token: str = ""
    
    # System parameters
    environment: str = "development"
    debug_mode: bool = False
    log_level: str = "INFO"


@dataclass 
class TidyLLMStandardResponse:
    """
    Unified response structure for all TidyLLM operations.
    
    This replaces the multiple incompatible response formats:
    - GatewayResponse, LLMResponse, dict responses, etc.
    
    Standard response structure eliminates transformation overhead.
    """
    status: ResponseStatus
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    request_id: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Backward compatibility property."""
        return self.status == ResponseStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format without transformation overhead."""
        return {
            "status": self.status.value,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "request_id": self.request_id
        }


@dataclass
class TidyLLMStandardRequest:
    """
    Unified request structure for all TidyLLM operations.
    
    Standard request format eliminates parameter transformation chains.
    """
    # Core request parameters (standardized names)
    model_id: str  # Standard name for model identification
    user_id: str   # Standard name for user identification
    session_id: str  # Standard name for session identification
    
    # Request content
    messages: List[Dict[str, str]] = field(default_factory=list)
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    
    # Model parameters (using standard names)
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    
    # Request metadata
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Standard parameter migration mappings
PARAMETER_MIGRATIONS = {
    # Model identification - migrate all to model_id
    "model_name": "model_id",
    "model": "model_id", 
    "llm_model": "model_id",
    "modelId": "model_id",
    
    # User identification - migrate all to user_id
    "userId": "user_id",
    "username": "user_id",  # In most contexts
    
    # Session identification - migrate all to session_id
    "sessionId": "session_id",
    
    # Database parameters - standardize naming
    "db_user": "postgres_username",
    "db_password": "postgres_password", 
    "db_name": "postgres_database",
    
    # AWS parameters - standardize naming
    "s3_access_key_id": "aws_access_key_id",
}


def migrate_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate old parameter names to standard names.
    
    This function helps transition existing code to use standard parameter names
    without breaking existing functionality.
    """
    migrated = params.copy()
    
    for old_name, new_name in PARAMETER_MIGRATIONS.items():
        if old_name in migrated:
            migrated[new_name] = migrated.pop(old_name)
    
    return migrated


def create_standard_response(
    success: bool,
    data: Any = None, 
    error: Optional[str] = None,
    **kwargs
) -> TidyLLMStandardResponse:
    """
    Helper function to create standard responses.
    
    This provides an easy migration path from existing response creation patterns.
    """
    status = ResponseStatus.SUCCESS if success else ResponseStatus.ERROR
    return TidyLLMStandardResponse(
        status=status,
        data=data,
        error=error,
        **kwargs
    )


def resolve_model_id(model_id: str, settings_config: Dict[str, Any] = None, warn_unmapped: bool = True) -> str:
    """
    Resolve internal model_id to external API model identifier.
    
    This function handles the mapping between our friendly internal model names
    and the actual model identifiers required by external APIs like Bedrock.
    
    Args:
        model_id: Internal model identifier (e.g., "claude-3-sonnet")
        settings_config: Settings configuration dict (optional)
    
    Returns:
        External API model identifier (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
    
    Examples:
        >>> resolve_model_id("claude-3-sonnet")
        "anthropic.claude-3-sonnet-20240229-v1:0"
        
        >>> resolve_model_id("gpt-4")
        "gpt-4"
        
        >>> resolve_model_id("unknown-model")  # Falls back to original
        "unknown-model"
    """
    # Default model mappings (fallback if settings not available)
    default_mappings = {
        # Claude models
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0", 
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-v2": "anthropic.claude-v2:1",
        "claude-instant": "anthropic.claude-instant-v1",
        
        # Titan models
        "titan-text-express": "amazon.titan-text-express-v1",
        "titan-text-lite": "amazon.titan-text-lite-v1",
        "titan-embed-text": "amazon.titan-embed-text-v1",
        
        # Llama models
        "llama-2-13b": "meta.llama2-13b-chat-v1",
        "llama-2-70b": "meta.llama2-70b-chat-v1", 
        "llama-2-7b": "meta.llama2-7b-chat-v1",
        
        # OpenAI models (pass-through)
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-turbo-preview",
        "gpt-3.5": "gpt-3.5-turbo",
    }
    
    # Use settings config if provided
    if settings_config:
        model_mapping = settings_config.get("aws", {}).get("bedrock", {}).get("model_mapping", {})
        if model_mapping and model_id in model_mapping:
            return model_mapping[model_id]
    
    # Fall back to default mappings
    resolved_id = default_mappings.get(model_id, model_id)
    
    # Log warning for unmapped models (helps catch missing mappings)
    if warn_unmapped and resolved_id == model_id and model_id not in default_mappings:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Unmapped model_id '{model_id}' - using as-is. Consider adding to settings.yaml model_mapping.")
    
    return resolved_id


# Cache for model mappings to avoid repeated YAML loading
_model_mappings_cache = {}
_cache_timestamp = {}

def load_model_mappings_from_settings(settings_path: str = None, use_cache: bool = True) -> Dict[str, str]:
    """
    Load model mappings from settings.yaml file with caching.
    
    Args:
        settings_path: Path to settings.yaml (optional)
        use_cache: Whether to use cached mappings (default: True)
    
    Returns:
        Dictionary of model_id -> external_api_id mappings
    """
    import yaml
    import os
    from pathlib import Path
    from time import time
    
    if not settings_path:
        # Default settings path
        settings_path = Path(__file__).parent.parent / "admin" / "settings.yaml"
    
    settings_key = str(settings_path)
    
    # Check cache validity (5 minute cache timeout)
    if use_cache and settings_key in _model_mappings_cache:
        cache_age = time() - _cache_timestamp.get(settings_key, 0)
        if cache_age < 300:  # 5 minutes
            return _model_mappings_cache[settings_key]
    
    try:
        # Check if file was modified since cache
        if use_cache and os.path.exists(settings_path):
            file_mtime = os.path.getmtime(settings_path)
            cached_mtime = _cache_timestamp.get(settings_key + "_file", 0)
            if file_mtime <= cached_mtime and settings_key in _model_mappings_cache:
                return _model_mappings_cache[settings_key]
        
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
            mappings = settings.get("aws", {}).get("bedrock", {}).get("model_mapping", {})
            
            # Cache the results
            _model_mappings_cache[settings_key] = mappings
            _cache_timestamp[settings_key] = time()
            if os.path.exists(settings_path):
                _cache_timestamp[settings_key + "_file"] = os.path.getmtime(settings_path)
            
            return mappings
            
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load model mappings from {settings_path}: {e}")
        # Return empty dict if settings can't be loaded
        return {}