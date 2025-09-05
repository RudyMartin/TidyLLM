#!/usr/bin/env python3
"""
Configuration Validation Script

Validates the deployment configuration.
"""

import json
import sys
from pathlib import Path

def validate_config():
    """Validate deployment configuration"""
    config_file = Path(__file__).parent.parent / "deployment_config.json"
    
    try:
        with open(config_file) as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['target_environment', 'environment_variables', 'streamlit_config']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate environment variables
        required_env_vars = ['VECTORQA_ENV', 'LOG_LEVEL']
        for var in required_env_vars:
            if var not in config['environment_variables']:
                print(f"❌ Missing environment variable: {var}")
                return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_config()
    sys.exit(0 if success else 1)
