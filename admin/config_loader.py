"""
V2 TidyLLM Configuration Loader
===============================

Multi-environment configuration management with security best practices.
NO HARDCODED CREDENTIALS - Uses AWS Secrets Manager for production.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class V2ConfigLoader:
    """
    Clean Architecture configuration loader for V2 TidyLLM.
    Manages multi-environment settings with zero hardcoded credentials.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader."""
        self.config_path = config_path or self._get_default_config_path()
        self.environment = self._get_current_environment()
        self._config_cache: Optional[Dict[str, Any]] = None
        
        logger.info(f"V2 Config initialized for environment: {self.environment}")
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        current_dir = Path(__file__).parent
        return str(current_dir / "settings.yaml")
    
    def _get_current_environment(self) -> str:
        """Get current environment from environment variables."""
        env = os.getenv('ENVIRONMENT', 'development')
        valid_envs = ['development', 'staging', 'production']
        
        if env not in valid_envs:
            logger.warning(f"Invalid environment '{env}', defaulting to 'development'")
            return 'development'
        
        return env
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration for current environment."""
        if self._config_cache is not None:
            return self._config_cache
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
            
            # Get environment-specific config
            env_config = full_config['environments'][self.environment]
            
            # Merge with global settings
            global_config = {
                'features': full_config.get('features', {}),
                'security': full_config.get('security', {}),
                'cost_management': full_config.get('cost_management', {}),
                'integrations': full_config.get('integrations', {}),
                'data_retention': full_config.get('data_retention', {}),
                'backup': full_config.get('backup', {}),
            }
            
            # Merge configurations
            config = {**env_config, **global_config}
            
            # Add environment metadata
            config['_metadata'] = {
                'environment': self.environment,
                'config_path': self.config_path,
                'loaded_at': self._get_timestamp()
            }
            
            self._config_cache = config
            logger.info(f"Configuration loaded successfully for {self.environment}")
            
            return config
        
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Examples:
            config.get('aws.region')
            config.get('database.pool_size') 
            config.get('monitoring.thresholds.response_time_ms')
        """
        config = self.load_config()
        
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Configuration key '{key_path}' not found, using default: {default}")
            return default
    
    def get_aws_config(self) -> Dict[str, Any]:
        """Get AWS-specific configuration."""
        return self.get('aws', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get('database', {})
    
    def get_application_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.get('application', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.get('monitoring', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.get('security', {})
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return self.get(f'features.{feature_name}', False)
    
    def get_credentials_config(self) -> Dict[str, str]:
        """
        Get credential configuration.
        IMPORTANT: Returns SECRET NAMES, not actual credentials.
        Actual credentials must be retrieved from AWS Secrets Manager.
        """
        aws_config = self.get_aws_config()
        secrets_config = aws_config.get('secrets', {})
        
        return {
            'database_secret_name': secrets_config.get('database_secret'),
            'api_keys_secret_name': secrets_config.get('api_keys_secret'),
            'aws_creds_secret_name': secrets_config.get('aws_creds_secret'),
            'region': aws_config.get('region', 'us-east-1')
        }
    
    def validate_config(self) -> bool:
        """Validate configuration completeness."""
        config = self.load_config()
        
        required_keys = [
            'aws.region',
            'aws.s3.document_bucket',
            'database.host_key',
            'application.name',
            'monitoring.pulse_interval_seconds'
        ]
        
        missing_keys = []
        
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        
        # Validate environment-specific requirements
        if self.is_production():
            prod_required = [
                'aws.secrets.database_secret',
                'security.use_secrets_manager'
            ]
            
            for key in prod_required:
                if self.get(key) is None:
                    missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing production configuration keys: {missing_keys}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def reload_config(self) -> Dict[str, Any]:
        """Reload configuration from file (clears cache)."""
        self._config_cache = None
        return self.load_config()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.utcnow().isoformat()


# Global configuration instance
_global_config: Optional[V2ConfigLoader] = None


def get_config() -> V2ConfigLoader:
    """Get global configuration instance."""
    global _global_config
    
    if _global_config is None:
        _global_config = V2ConfigLoader()
    
    return _global_config


def load_v2_config() -> Dict[str, Any]:
    """Load V2 configuration (convenience function)."""
    return get_config().load_config()


def get_v2_setting(key_path: str, default: Any = None) -> Any:
    """Get V2 configuration setting (convenience function)."""
    return get_config().get(key_path, default)


# Example usage functions for your boss to test
def test_config_loading():
    """Test configuration loading functionality."""
    print("Testing V2 Configuration Loading...")
    print("=" * 40)
    
    config = get_config()
    
    # Test basic loading
    print(f"Environment: {config.environment}")
    print(f"App Name: {config.get('application.name')}")
    print(f"AWS Region: {config.get('aws.region')}")
    print(f"Document Bucket: {config.get('aws.s3.document_bucket')}")
    print(f"PDF Max Size: {config.get('application.pdf_processing.max_file_size_mb')} MB")
    
    # Test feature flags
    print(f"\nFeature Flags:")
    print(f"- PDF Processing: {config.is_feature_enabled('pdf_processing')}")
    print(f"- Health Monitoring: {config.is_feature_enabled('health_monitoring')}")
    print(f"- Cost Tracking: {config.is_feature_enabled('cost_tracking')}")
    
    # Test credentials config (SECRET NAMES ONLY)
    creds_config = config.get_credentials_config()
    print(f"\nCredential Configuration (Secret Names):")
    for key, value in creds_config.items():
        print(f"- {key}: {value}")
    
    # Validate configuration
    print(f"\nConfiguration Valid: {config.validate_config()}")
    
    print("\nTEST COMPLETE: V2 Configuration system working!")


if __name__ == "__main__":
    test_config_loading()