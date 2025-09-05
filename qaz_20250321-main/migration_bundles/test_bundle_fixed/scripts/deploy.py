#!/usr/bin/env python3
"""
VectorQA Sage Deployment Script

Automated deployment script for staging environment.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def main():
    """Main deployment function"""
    print("🚀 Starting VectorQA Sage deployment...")
    
    # Load deployment configuration
    config_file = Path(__file__).parent.parent / "deployment_config.json"
    with open(config_file) as f:
        config = json.load(f)
    
    print(f"🎯 Target environment: {config['target_environment']}")
    
    # Setup environment
    print("🔧 Setting up environment...")
    setup_script = Path(__file__).parent / "setup_environment.py"
    subprocess.run([sys.executable, str(setup_script)], check=True)
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_demo.txt"], check=True)
    
    # Start application
    print("🚀 Starting application...")
    app_script = Path(__file__).parent.parent / "src" / "main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_script)], check=True)

if __name__ == "__main__":
    main()
