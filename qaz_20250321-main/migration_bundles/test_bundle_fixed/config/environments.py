#!/usr/bin/env python3
"""
Environment Configuration System

Centralized environment setup for VectorQA Sage applications.
Handles environment detection, path setup, and configuration management.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

class Environment:
    """Environment configuration and setup"""
    
    def __init__(self, env_name: str = None):
        # Calculate project root (2 levels up from src/config to get to project root)
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent
        self.env_name = env_name or self._detect_environment()
        self.config = self._get_config()
    
    def _detect_environment(self) -> str:
        """Auto-detect environment based on system properties"""
        # Check for explicit environment variable
        if os.getenv('VECTORQA_ENV'):
            return os.getenv('VECTORQA_ENV')
        
        # Check for AWS environment
        if os.getenv('AWS_REGION') or os.getenv('AWS_ACCESS_KEY_ID'):
            return 'aws'
        
        # Default to local
        return 'local'
    
    def _get_config(self) -> Dict[str, Any]:
        """Get environment configuration"""
        base_config = {
            'python_paths': ['src', 'src/backend'],
            'env_vars': {
                'VECTORQA_ENV': self.env_name,
                'LOG_LEVEL': 'INFO',
                'CACHE_DIR': '~/vectorqa_cache'
            },
            'streamlit_config': {
                'server.port': 8501,
                'server.address': 'localhost',
                'browser.gatherUsageStats': False,
                'server.runOnSave': True
            },
            'backend_config': {
                'use_local_llm': True,
                'cache_enabled': True,
                'debug_mode': False
            }
        }
        
        # Environment-specific overrides
        if self.env_name == 'aws':
            base_config.update({
                'env_vars': {
                    'VECTORQA_ENV': 'aws',
                    'LOG_LEVEL': 'WARNING',
                    'CACHE_DIR': '/tmp/vectorqa_cache'
                },
                'streamlit_config': {
                    'server.port': 8501,
                    'server.address': '0.0.0.0',
                    'browser.gatherUsageStats': False,
                    'server.runOnSave': False,
                    'server.headless': True
                },
                'backend_config': {
                    'use_local_llm': False,
                    'cache_enabled': True,
                    'debug_mode': False
                }
            })
        
        return base_config
    
    def setup(self) -> bool:
        """Setup the environment"""
        try:
            print(f"🔧 Setting up environment: {self.env_name}")
            
            # Set environment variables
            self._set_env_vars()
            
            # Setup Python paths
            self._setup_python_paths()
            
            # Verify critical directories
            self._verify_directories()
            
            # Check dependencies
            self._check_dependencies()
            
            print("✅ Environment setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
    
    def _set_env_vars(self):
        """Set environment variables"""
        for key, value in self.config['env_vars'].items():
            os.environ[key] = str(value)
            print(f"✅ Set {key}={value}")
    
    def _setup_python_paths(self):
        """Setup Python paths"""
        for path_name in self.config['python_paths']:
            path = self.project_root / path_name
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
                print(f"✅ Added to Python path: {path}")
    
    def _verify_directories(self):
        """Verify critical directories exist"""
        critical_dirs = ['src', 'src/backend']
        
        for dir_name in critical_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(f"Critical directory not found: {dir_path}")
            print(f"✅ Verified directory: {dir_path}")
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        required_packages = [
            "streamlit",
            "pandas", 
            "numpy"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} available")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} not available")
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("💡 Install with: pip install -r requirements_demo.txt")
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit configuration"""
        return self.config['streamlit_config']
    
    def get_backend_config(self) -> Dict[str, Any]:
        """Get backend configuration"""
        return self.config['backend_config']
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information"""
        return {
            'name': self.env_name,
            'project_root': str(self.project_root),
            'python_paths': self.config['python_paths'],
            'env_vars': self.config['env_vars']
        }

def setup_environment(env_name: str = None) -> Environment:
    """Setup and return environment configuration"""
    env = Environment(env_name)
    env.setup()
    return env

def get_environment(env_name: str = None) -> Environment:
    """Get environment configuration without setup"""
    return Environment(env_name)

if __name__ == "__main__":
    # Test environment setup
    env = Environment()
    print(f"🌍 Environment: {env.env_name}")
    print(f"📁 Project root: {env.project_root}")
    print("=" * 60)
    env.setup()
