#!/usr/bin/env python3
"""
MLflow Configuration Management

This module provides environment-aware MLflow configuration that integrates
with the existing multi-environment system. It handles different MLflow
setups for local, staging, and production environments.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MLflowConfig:
    """Environment-aware MLflow configuration"""
    
    def __init__(self, env_name: str = None):
        self.env_name = env_name or self._detect_environment()
        self.config = self._get_environment_config()
        
    def _detect_environment(self) -> str:
        """Auto-detect environment based on system properties"""
        # Check for explicit environment variable
        if os.getenv('VECTORQA_ENV'):
            return os.getenv('VECTORQA_ENV')
        
        # Check for AWS environment
        if os.getenv('AWS_REGION') or os.getenv('AWS_ACCESS_KEY_ID'):
            return 'aws'
        
        # Check for production indicators
        if os.getenv('MLFLOW_TRACKING_URI') and 'postgresql' in os.getenv('MLFLOW_TRACKING_URI', ''):
            return 'production'
        
        # Default to local
        return 'local'
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific MLflow configuration"""
        
        # Base configuration
        base_config = {
            'enabled': True,
            'experiment_name': 'unified-llm-gateway',
            'enable_tracking': True,
            'artifact_root': 'file://./mlflow_artifacts',
            'tracking_uri': 'file://./mlflow_tracking',
            'gateway': {
                'local_url': 'http://localhost:11434',
                'remote_url': None,
                'default_gateway': 'local'
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'mlflow',
                'user': 'mlflow',
                'password': 'password'
            },
            'security': {
                'require_authentication': False,
                'encrypt_artifacts': False,
                'audit_logging': True
            },
            'performance': {
                'batch_size': 100,
                'cache_enabled': True,
                'async_logging': False
            }
        }
        
        # Environment-specific overrides
        if self.env_name == 'local':
            base_config.update({
                'experiment_name': 'unified-llm-gateway-local',
                'artifact_root': 'file://./mlflow_artifacts',
                'tracking_uri': 'file://./mlflow_tracking',
                'gateway': {
                    'local_url': 'http://localhost:11434',
                    'remote_url': None,
                    'default_gateway': 'local'
                },
                'security': {
                    'require_authentication': False,
                    'encrypt_artifacts': False,
                    'audit_logging': True
                },
                'performance': {
                    'batch_size': 50,
                    'cache_enabled': True,
                    'async_logging': False
                }
            })
            
        elif self.env_name == 'staging':
            base_config.update({
                'experiment_name': 'unified-llm-gateway-staging',
                'artifact_root': 's3://mlflow-artifacts-staging',
                'tracking_uri': 'postgresql://mlflow:password@staging-db:5432/mlflow',
                'gateway': {
                    'local_url': 'http://localhost:11434',
                    'remote_url': 'https://api.openai.com/v1',
                    'default_gateway': 'auto'
                },
                'database': {
                    'host': 'staging-db',
                    'port': 5432,
                    'name': 'mlflow',
                    'user': 'mlflow',
                    'password': os.getenv('MLFLOW_DB_PASSWORD', 'password')
                },
                'security': {
                    'require_authentication': True,
                    'encrypt_artifacts': True,
                    'audit_logging': True
                },
                'performance': {
                    'batch_size': 100,
                    'cache_enabled': True,
                    'async_logging': True
                }
            })
            
        elif self.env_name == 'production':
            base_config.update({
                'experiment_name': 'unified-llm-gateway-production',
                'artifact_root': 's3://mlflow-artifacts-production',
                'tracking_uri': 'postgresql://mlflow:password@prod-db:5432/mlflow',
                'gateway': {
                    'local_url': 'http://zllm-cluster:11434',
                    'remote_url': 'https://api.openai.com/v1',
                    'default_gateway': 'auto'
                },
                'database': {
                    'host': 'prod-db',
                    'port': 5432,
                    'name': 'mlflow',
                    'user': 'mlflow',
                    'password': os.getenv('MLFLOW_DB_PASSWORD')
                },
                'security': {
                    'require_authentication': True,
                    'encrypt_artifacts': True,
                    'audit_logging': True
                },
                'performance': {
                    'batch_size': 200,
                    'cache_enabled': True,
                    'async_logging': True
                }
            })
            
        elif self.env_name == 'aws':
            base_config.update({
                'experiment_name': 'unified-llm-gateway-aws',
                'artifact_root': f"s3://{os.getenv('S3_BUCKET', 'mlflow-artifacts')}/mlflow-artifacts",
                'tracking_uri': f"postgresql://{os.getenv('DB_USER', 'mlflow')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'mlflow')}",
                'gateway': {
                    'local_url': 'http://localhost:11434',
                    'remote_url': 'https://api.openai.com/v1',
                    'default_gateway': 'auto'
                },
                'database': {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': int(os.getenv('DB_PORT', '5432')),
                    'name': os.getenv('DB_NAME', 'mlflow'),
                    'user': os.getenv('DB_USER', 'mlflow'),
                    'password': os.getenv('DB_PASSWORD', 'password')
                },
                'security': {
                    'require_authentication': True,
                    'encrypt_artifacts': True,
                    'audit_logging': True
                },
                'performance': {
                    'batch_size': 150,
                    'cache_enabled': True,
                    'async_logging': True
                }
            })
        
        # Override with environment variables if present
        self._override_from_env(base_config)
        
        return base_config
    
    def _override_from_env(self, config: Dict[str, Any]):
        """Override configuration with environment variables"""
        env_overrides = {
            'MLFLOW_TRACKING_URI': ('tracking_uri',),
            'MLFLOW_ARTIFACT_ROOT': ('artifact_root',),
            'MLFLOW_EXPERIMENT_NAME': ('experiment_name',),
            'ZLLM_GATEWAY_URL': ('gateway', 'local_url'),
            'REMOTE_GATEWAY_URL': ('gateway', 'remote_url'),
            'MLFLOW_DB_HOST': ('database', 'host'),
            'MLFLOW_DB_PORT': ('database', 'port'),
            'MLFLOW_DB_NAME': ('database', 'name'),
            'MLFLOW_DB_USER': ('database', 'user'),
            'MLFLOW_DB_PASSWORD': ('database', 'password'),
        }
        
        for env_var, config_path in env_overrides.items():
            if os.getenv(env_var):
                value = os.getenv(env_var)
                
                # Navigate to the nested config location
                current = config
                for key in config_path[:-1]:
                    current = current[key]
                
                # Set the value
                current[config_path[-1]] = value
                logger.info(f"Override {'.'.join(config_path)} = {value} (from {env_var})")
    
    def get_tracking_uri(self) -> str:
        """Get MLflow tracking URI for current environment"""
        return self.config['tracking_uri']
    
    def get_artifact_root(self) -> str:
        """Get MLflow artifact root for current environment"""
        return self.config['artifact_root']
    
    def get_experiment_name(self) -> str:
        """Get MLflow experiment name for current environment"""
        return self.config['experiment_name']
    
    def get_gateway_config(self) -> Dict[str, Any]:
        """Get gateway configuration for current environment"""
        return self.config['gateway']
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for current environment"""
        return self.config['database']
    
    def is_enabled(self) -> bool:
        """Check if MLflow is enabled for current environment"""
        return self.config['enabled']
    
    def is_tracking_enabled(self) -> bool:
        """Check if MLflow tracking is enabled"""
        return self.config['enable_tracking']
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.config['security']
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config['performance']
    
    def setup_environment(self) -> bool:
        """Setup MLflow environment variables"""
        try:
            logger.info(f"🔧 Setting up MLflow environment: {self.env_name}")
            
            # Set MLflow environment variables
            os.environ['MLFLOW_TRACKING_URI'] = self.get_tracking_uri()
            os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = self.get_artifact_root()
            os.environ['MLFLOW_EXPERIMENT_NAME'] = self.get_experiment_name()
            
            # Set gateway environment variables
            gateway_config = self.get_gateway_config()
            os.environ['ZLLM_GATEWAY_URL'] = gateway_config['local_url']
            if gateway_config['remote_url']:
                os.environ['REMOTE_GATEWAY_URL'] = gateway_config['remote_url']
            
            # Set database environment variables
            db_config = self.get_database_config()
            os.environ['MLFLOW_DB_HOST'] = db_config['host']
            os.environ['MLFLOW_DB_PORT'] = str(db_config['port'])
            os.environ['MLFLOW_DB_NAME'] = db_config['name']
            os.environ['MLFLOW_DB_USER'] = db_config['user']
            os.environ['MLFLOW_DB_PASSWORD'] = db_config['password']
            
            logger.info("✅ MLflow environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ MLflow environment setup failed: {e}")
            return False
    
    def validate_config(self) -> bool:
        """Validate MLflow configuration"""
        try:
            logger.info("🔍 Validating MLflow configuration...")
            
            # Check required fields
            required_fields = ['tracking_uri', 'artifact_root', 'experiment_name']
            for field in required_fields:
                if not self.config.get(field):
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Check database configuration
            db_config = self.get_database_config()
            required_db_fields = ['host', 'port', 'name', 'user', 'password']
            for field in required_db_fields:
                if not db_config.get(field):
                    logger.error(f"Missing database field: {field}")
                    return False
            
            # Check gateway configuration
            gateway_config = self.get_gateway_config()
            if not gateway_config.get('local_url'):
                logger.error("Missing local gateway URL")
                return False
            
            logger.info("✅ MLflow configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ MLflow configuration validation failed: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging"""
        return {
            'environment': self.env_name,
            'enabled': self.is_enabled(),
            'tracking_enabled': self.is_tracking_enabled(),
            'tracking_uri': self.get_tracking_uri(),
            'artifact_root': self.get_artifact_root(),
            'experiment_name': self.get_experiment_name(),
            'gateway': self.get_gateway_config(),
            'database': {
                'host': self.get_database_config()['host'],
                'port': self.get_database_config()['port'],
                'name': self.get_database_config()['name'],
                'user': self.get_database_config()['user']
            },
            'security': self.get_security_config(),
            'performance': self.get_performance_config()
        }


# Global MLflow configuration instance
mlflow_config = MLflowConfig()


def get_mlflow_config(env_name: str = None) -> MLflowConfig:
    """Get MLflow configuration for specified environment"""
    if env_name:
        return MLflowConfig(env_name)
    return mlflow_config


def setup_mlflow_environment(env_name: str = None) -> bool:
    """Setup MLflow environment"""
    config = get_mlflow_config(env_name)
    return config.setup_environment()


def validate_mlflow_config(env_name: str = None) -> bool:
    """Validate MLflow configuration"""
    config = get_mlflow_config(env_name)
    return config.validate_config()


if __name__ == "__main__":
    # Test MLflow configuration
    config = MLflowConfig()
    print(f"🌍 Environment: {config.env_name}")
    print("📋 Configuration Summary:")
    for key, value in config.get_config_summary().items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    if config.validate_config():
        config.setup_environment()
    else:
        print("❌ Configuration validation failed")
