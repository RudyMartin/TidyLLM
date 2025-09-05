#!/usr/bin/env python3
"""
Settings Loader

Dynamically loads environment-specific settings.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

def load_environment_settings(env_name: str = None) -> Dict[str, Any]:
    """Load settings for the specified environment"""
    if not env_name:
        env_name = os.getenv('VECTORQA_ENV', 'local')
    
    # Add settings directory to path
    settings_dir = Path(__file__).parent
    if str(settings_dir) not in sys.path:
        sys.path.insert(0, str(settings_dir))
    
    # Import environment-specific settings
    try:
        settings_module = __import__(f"{env_name}_settings")
        return settings_module.get_settings()
    except ImportError as e:
        print(f"❌ Could not load settings for environment '{env_name}': {e}")
        # Fallback to local settings
        try:
            settings_module = __import__("local_settings")
            return settings_module.get_settings()
        except ImportError:
            print("❌ Could not load any settings")
            return {}

def setup_environment(env_name: str = None):
    """Setup environment with settings"""
    settings = load_environment_settings(env_name)
    
    if not settings:
        print("❌ No settings loaded")
        return False
    
    # Set environment variables
    for key, value in settings.get('env_vars', {}).items():
        os.environ[key] = str(value)
    
    # Load secrets
    try:
        settings_module = __import__(f"{settings['environment']}_settings")
        settings_module.load_secrets()
    except ImportError:
        print(f"⚠️  Could not load secrets for {settings['environment']}")
    
    print(f"✅ Environment setup complete: {settings['environment']}")
    return True

if __name__ == "__main__":
    # Test settings loader
    env = os.getenv('VECTORQA_ENV', 'local')
    settings = load_environment_settings(env)
    print(f"🌍 Loaded settings for: {env}")
    print(f"📋 Settings: {settings}")
