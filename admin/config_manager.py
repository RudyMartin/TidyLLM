"""
TidyLLM Configuration Manager - Backend Settings Management

Centralized configuration management for all TidyLLM ecosystem modules.
Handles settings for databases, gateways, compliance, research tools, etc.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModuleConfig:
    """Base configuration for TidyLLM modules"""
    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    
    
@dataclass
class DatabaseConfig:
    """Database configuration (PostgreSQL + MLFlow)"""
    # PostgreSQL settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "tidyllm_db"
    postgres_username: str = "tidyllm_user"
    postgres_password: str = ""  # Set via environment variable DB_PASSWORD
    postgres_ssl_mode: str = "prefer"
    
    # MLFlow backend
    mlflow_backend_store_uri: Optional[str] = None
    mlflow_artifact_root: Optional[str] = None
    mlflow_tracking_uri: str = "file://./mlruns"
    
    # Connection settings
    connection_pool_size: int = 10
    connection_timeout: int = 30


@dataclass
class GatewayConfig:
    """Gateway configuration (MLFlow + Enterprise)"""
    enabled: bool = True
    base_url: str = "http://localhost:5000"
    
    # Model routing
    model_routing_enabled: bool = True
    model_routing_strategy: str = "cost_optimized"  # cost_optimized, performance, balanced
    fallback_model: str = "gpt-3.5-turbo"
    
    # Quality control
    response_time_threshold: float = 5.0
    cost_threshold: float = 0.10
    quality_score_threshold: float = 7.0
    
    # Security
    rate_limiting_enabled: bool = True
    requests_per_minute: int = 60
    audit_logging: bool = True
    
    # Enterprise features
    enterprise_governance: bool = True
    compliance_monitoring: bool = True


@dataclass
class AWSConfig:
    """AWS configuration for integrations"""
    region: str = "us-east-1"
    profile: str = "default"
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: str = ""
    kms_key_id: str = ""


@dataclass
class ResearchConfig:
    """Research module configuration"""
    enabled: bool = True
    arxiv_enabled: bool = True
    arxiv_max_results: int = 100
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class ComplianceConfig:
    """Compliance monitoring configuration"""
    enabled: bool = True
    
    # Regulatory frameworks
    sox_compliance: bool = False
    gdpr_compliance: bool = False  
    hipaa_compliance: bool = False
    pci_compliance: bool = False
    
    # Audit settings
    audit_logging: bool = True
    audit_retention_days: int = 365
    full_transparency: bool = True
    
    # Model risk management
    model_risk_monitoring: bool = True
    evidence_validation: bool = True


@dataclass
class TidyLLMConfig:
    """Complete TidyLLM ecosystem configuration"""
    # Module configurations
    database: DatabaseConfig
    gateway: GatewayConfig
    aws: AWSConfig
    research: ResearchConfig
    compliance: ComplianceConfig
    
    # Global settings
    environment: str = "development"  # development, staging, production
    debug_mode: bool = False
    log_level: str = "INFO"
    config_version: str = "1.0"
    last_updated: Optional[str] = None


class ConfigManager:
    """Centralized configuration management for TidyLLM ecosystem"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[TidyLLMConfig] = None
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Look for config in multiple locations
        possible_paths = [
            os.environ.get("TIDYLLM_CONFIG_PATH"),
            "tidyllm_config.yaml",
            "config/tidyllm.yaml", 
            os.path.expanduser("~/.tidyllm/config.yaml"),
            "/etc/tidyllm/config.yaml"
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
                
        # Return first writable location
        return "tidyllm_config.yaml"
    
    def _load_config(self) -> None:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                self._config = self._dict_to_config(data)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.info("No config file found, using defaults")
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._config = self._get_default_config()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> TidyLLMConfig:
        """Convert dictionary to TidyLLMConfig"""
        return TidyLLMConfig(
            database=DatabaseConfig(**data.get("database", {})),
            gateway=GatewayConfig(**data.get("gateway", {})),
            aws=AWSConfig(**data.get("aws", {})),
            research=ResearchConfig(**data.get("research", {})),
            compliance=ComplianceConfig(**data.get("compliance", {})),
            environment=data.get("environment", "development"),
            debug_mode=data.get("debug_mode", False),
            log_level=data.get("log_level", "INFO"),
            config_version=data.get("config_version", "1.0"),
            last_updated=data.get("last_updated")
        )
    
    def _get_default_config(self) -> TidyLLMConfig:
        """Get default configuration"""
        return TidyLLMConfig(
            database=DatabaseConfig(),
            gateway=GatewayConfig(),
            aws=AWSConfig(),
            research=ResearchConfig(),
            compliance=ComplianceConfig(),
            last_updated=datetime.now().isoformat()
        )
    
    @property
    def config(self) -> TidyLLMConfig:
        """Get current configuration"""
        return self._config
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            # Update timestamp
            self._config.last_updated = datetime.now().isoformat()
            
            # Convert to dictionary
            config_dict = asdict(self._config)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path) or ".", exist_ok=True)
            
            # Save to YAML
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            # Apply updates to current config
            for key, value in updates.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                elif key in ["database", "gateway", "aws", "research", "compliance"]:
                    # Update nested configuration
                    nested_config = getattr(self._config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
            
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
    
    def get_module_config(self, module_name: str) -> Optional[Any]:
        """Get configuration for specific module"""
        return getattr(self._config, module_name, None)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return issues"""
        issues = []
        warnings = []
        
        # Validate database config
        db_config = self._config.database
        if not db_config.postgres_host:
            issues.append("PostgreSQL host not configured")
        if not db_config.postgres_username:
            issues.append("PostgreSQL username not configured")
            
        # Validate gateway config
        gw_config = self._config.gateway
        if gw_config.enabled and not gw_config.base_url:
            issues.append("Gateway enabled but no base URL configured")
            
        # Validate AWS config for production
        if self._config.environment == "production":
            aws_config = self._config.aws
            if not aws_config.access_key_id and not aws_config.profile:
                warnings.append("No AWS credentials configured for production")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }


# Utility functions
def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Load TidyLLM configuration"""
    return ConfigManager(config_path)


def save_config(config: TidyLLMConfig, config_path: Optional[str] = None) -> bool:
    """Save TidyLLM configuration"""
    manager = ConfigManager(config_path)
    manager._config = config
    return manager.save_config()


def get_default_config() -> TidyLLMConfig:
    """Get default TidyLLM configuration"""
    return TidyLLMConfig(
        database=DatabaseConfig(),
        gateway=GatewayConfig(), 
        aws=AWSConfig(),
        research=ResearchConfig(),
        compliance=ComplianceConfig()
    )