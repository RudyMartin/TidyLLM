#!/usr/bin/env python3
"""
Environment Setup Module

Simple setup file that can be included at the top of any app to automatically
configure the environment. Usage:

    from config.setup import setup_env
    setup_env()
"""

import sys
import os
from pathlib import Path

def setup_env(env_name: str = None):
    """
    Setup the environment for VectorQA Sage applications.
    
    Args:
        env_name: Optional environment name ('local' or 'aws')
    
    Returns:
        Environment configuration object
    """
    # Add config directory to path
    config_path = Path(__file__).parent
    if str(config_path) not in sys.path:
        sys.path.insert(0, str(config_path))
    
    # Import and setup environment
    from environments import setup_environment
    return setup_environment(env_name)

def get_env(env_name: str = None):
    """
    Get environment configuration without setup.
    
    Args:
        env_name: Optional environment name ('local' or 'aws')
    
    Returns:
        Environment configuration object
    """
    # Add config directory to path
    config_path = Path(__file__).parent
    if str(config_path) not in sys.path:
        sys.path.insert(0, str(config_path))
    
    # Import and get environment
    from environments import get_environment
    return get_environment(env_name)

# Auto-setup when imported
if __name__ != "__main__":
    # Only auto-setup if not running as main
    try:
        setup_env()
    except Exception as e:
        print(f"⚠️ Auto-setup failed: {e}")
        print("💡 You can manually setup with: from config.setup import setup_env; setup_env()")

if __name__ == "__main__":
    # Test setup
    print("🧪 Testing environment setup...")
    env = setup_env()
    print(f"✅ Environment: {env.env_name}")
    print(f"📁 Project root: {env.project_root}")

