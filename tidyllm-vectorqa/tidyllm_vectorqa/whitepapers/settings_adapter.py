#!/usr/bin/env python3
"""
Settings Adapter for Working Demo Settings.yaml
===============================================

Adapts the comprehensive working demo settings.yaml to work with the 
existing Streamlit app and backend configuration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AdaptedPostgresConfig:
    """Adapted PostgreSQL configuration from working demo settings"""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str
    
    @classmethod
    def from_settings(cls, settings: Dict[str, Any]):
        """Create from settings.yaml postgres section"""
        postgres = settings.get('postgres', {})
        
        return cls(
            host=postgres.get('host', 'localhost'),
            port=postgres.get('port', 5432),
            database=postgres.get('db_name', postgres.get('database', 'vectorqa')),
            username=postgres.get('db_user', postgres.get('username', 'postgres')),
            password=postgres.get('db_password', postgres.get('password', '')),
            ssl_mode=postgres.get('ssl_mode', 'require')
        )


@dataclass
class AdaptedAWSConfig:
    """Adapted AWS configuration from working demo settings"""
    region: str
    kms_key_id: Optional[str]
    default_bucket: Optional[str]
    default_prefix: Optional[str]
    
    @classmethod
    def from_settings(cls, settings: Dict[str, Any]):
        """Create from settings.yaml aws section"""
        aws = settings.get('aws', {})
        s3_config = aws.get('s3', {})
        
        return cls(
            region=aws.get('region', 'us-east-1'),
            kms_key_id=aws.get('kms_key_id'),
            default_bucket=s3_config.get('default_bucket'),
            default_prefix=s3_config.get('default_prefix')
        )


class SettingsAdapter:
    """Adapter to bridge working demo settings.yaml with existing app"""
    
    def __init__(self, settings_path: str = None):
        """Initialize with settings file path"""
        if settings_path is None:
            # Look for settings.yaml in current and parent directories
            current_dir = Path(__file__).parent
            possible_paths = [
                current_dir / "settings.yaml",
                current_dir.parent / "settings.yaml",
                current_dir.parent.parent / "settings.yaml"
            ]
            
            for path in possible_paths:
                if path.exists():
                    settings_path = str(path)
                    break
        
        if not settings_path or not Path(settings_path).exists():
            raise FileNotFoundError("settings.yaml not found")
        
        self.settings_path = Path(settings_path)
        self.settings = self._load_settings()
        
        # Create adapted configs
        self.postgres = AdaptedPostgresConfig.from_settings(self.settings)
        self.aws = AdaptedAWSConfig.from_settings(self.settings)
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from YAML file"""
        try:
            with open(self.settings_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading settings: {e}")
            return {}
    
    def get_postgres_connection_params(self) -> Dict[str, Any]:
        """Get PostgreSQL connection parameters"""
        return {
            'host': self.postgres.host,
            'port': self.postgres.port,
            'database': self.postgres.database,
            'user': self.postgres.username,
            'password': self.postgres.password,
            'sslmode': self.postgres.ssl_mode
        }
    
    def get_s3_config(self) -> Dict[str, Any]:
        """Get S3 configuration"""
        return {
            'region': self.aws.region,
            'kms_key_id': self.aws.kms_key_id,
            'default_bucket': self.aws.default_bucket,
            'default_prefix': self.aws.default_prefix
        }
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration (placeholder for future use)"""
        return self.settings.get('embeddings', {})
    
    def get_aws_bedrock_config(self) -> Dict[str, Any]:
        """Get AWS Bedrock configuration"""
        return self.settings.get('aws', {}).get('bedrock', {})


# Global instance
_settings_adapter = None

def get_settings_adapter() -> SettingsAdapter:
    """Get global settings adapter instance"""
    global _settings_adapter
    
    if _settings_adapter is None:
        _settings_adapter = SettingsAdapter()
    
    return _settings_adapter


def create_compatible_backend_config():
    """Create a backend config object compatible with existing app"""
    adapter = get_settings_adapter()
    
    # Create a simple object that mimics the original backend_config interface
    class CompatibleConfig:
        def __init__(self):
            # Create postgres config with all required attribute aliases
            postgres_params = adapter.get_postgres_connection_params()
            postgres_params['username'] = postgres_params['user']  # Add username alias
            postgres_params['ssl_mode'] = postgres_params['sslmode']  # Add ssl_mode alias
            
            self.settings = type('Settings', (), {
                'postgres': type('Postgres', (), postgres_params)()
            })()
            self.connection_params = adapter.get_postgres_connection_params()
            self.s3_config = adapter.get_s3_config()
    
    return CompatibleConfig()


if __name__ == '__main__':
    # Test the adapter
    try:
        adapter = get_settings_adapter()
        print("Settings Adapter Test")
        print("=" * 30)
        print(f"PostgreSQL: {adapter.postgres.host}:{adapter.postgres.port}")
        print(f"Database: {adapter.postgres.database}")
        print(f"AWS Region: {adapter.aws.region}")
        print(f"S3 Bucket: {adapter.aws.default_bucket}")
        print(f"KMS Key: {adapter.aws.kms_key_id}")
        print("✅ Settings adapter working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")