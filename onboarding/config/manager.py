"""
TidyLLM Onboarding Configuration Manager
=======================================

Manages configuration for the TidyLLM onboarding system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages configuration for TidyLLM onboarding."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        self.config_file = self.config_dir / "settings.yaml"
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'aws': {
                'region': 'us-east-1',
                'profile': 'default'
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'tidyllm',
                'user': 'tidyllm_user'
            },
            'security': {
                'encryption': True,
                'audit_logging': True,
                'data_masking': True
            },
            'models': {
                'default_provider': 'bedrock',
                'default_model': 'anthropic.claude-3-sonnet-20240229-v1:0'
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration."""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['aws', 'database', 'security', 'models']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate AWS config
        if 'aws' in config:
            aws_config = config['aws']
            if 'region' not in aws_config:
                warnings.append("AWS region not specified, using default")
        
        # Validate database config
        if 'database' in config:
            db_config = config['database']
            if 'host' not in db_config:
                errors.append("Database host is required")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
