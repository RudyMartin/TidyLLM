#!/usr/bin/env python3
"""
Unified Configuration Management

Provides a single, environment-agnostic configuration system that works
consistently across development, staging, and production environments.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self):
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from multiple sources"""
        # Base configuration
        self._config.update(self._get_base_config())
        
        # Environment-specific overrides
        env = os.getenv('VECTORQA_ENV', 'local')
        env_config = self._get_environment_config(env)
        self._config.update(env_config)
        
        # Runtime overrides from environment variables
        self._apply_env_overrides()
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration that applies to all environments"""
        return {
            # AWS Configuration
            "bucket_name": "sagemaker-us-east-1-XXXXXXXXXXXX",
            "region_name": "us-east-1",
            "prefix": "dev",
            
            # Model Configuration
            "default_model": "amazon.titan-embed-text-v2:0",
            "embedding_models": {
                "amazon.titan-embed-text-v1": 768,
                "amazon.titan-embed-text-v2:0": 1024,
                "cohere.embed-english-v3": 1024,
                "anthropic.claude-v2": 1536
            },
            
            # Processing Configuration
            "num_training_samples": 1000,
            "nlist": 512,
            "nprobe": 16,
            
            # Folder Structure
            "json_folder": "json",
            "index_folder": "idx",
            "page_folder": "page",
            "pdf_folder": "pdf",
            "compiled_folder": "compiled_modules",
            
            # Security Configuration
            "security": {
                "aws_only": True,
                "allow_external_apis": False,
                "require_iam_roles": True,
                "audit_logging": True,
                "data_encryption": True
            }
        }
    
    def _get_environment_config(self, env: str) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        configs = {
            "local": {
                "debug_mode": True,
                "log_level": "INFO",
                "cache_dir": "~/vectorqa_cache"
            },
            "development": {
                "debug_mode": True,
                "log_level": "INFO",
                "cache_dir": "/tmp/vectorqa_cache"
            },
            "staging": {
                "debug_mode": False,
                "log_level": "WARNING",
                "cache_dir": "/tmp/vectorqa_cache"
            },
            "production": {
                "debug_mode": False,
                "log_level": "ERROR",
                "cache_dir": "/tmp/vectorqa_cache"
            }
        }
        return configs.get(env, configs["local"])
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            "VECTORQA_BUCKET": "bucket_name",
            "VECTORQA_REGION": "region_name",
            "VECTORQA_PREFIX": "prefix",
            "VECTORQA_LOG_LEVEL": "log_level",
            "VECTORQA_DEBUG": "debug_mode"
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle boolean conversion
                if config_key == "debug_mode":
                    value = value.lower() in ("true", "1", "yes")
                self._config[config_key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration"""
        self._config.update(updates)

# Global configuration instance
config = ConfigManager()

# Convenience function for backward compatibility
def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary"""
    return config.get_all()

# Export the CONFIG constant for backward compatibility
CONFIG = config.get_all()
