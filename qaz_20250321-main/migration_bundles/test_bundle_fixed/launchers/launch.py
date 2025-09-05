#!/usr/bin/env python3
"""
VectorQA Sage Launcher

Main launcher script that detects environment and launches appropriately.
"""

import os
import sys
import subprocess
from pathlib import Path

def detect_environment():
    """Detect the current environment"""
    # Check for explicit environment variable
    if os.getenv('VECTORQA_ENV'):
        return os.getenv('VECTORQA_ENV')
    
    # Check for AWS environment
    if os.getenv('AWS_REGION') or os.getenv('AWS_ACCESS_KEY_ID'):
        return 'production'
    
    # Check for staging indicators
    if os.getenv('STAGING') or 'staging' in os.getenv('HOSTNAME', '').lower():
        return 'staging'
    
    # Check for development indicators
    if os.getenv('DEVELOPMENT') or 'dev' in os.getenv('HOSTNAME', '').lower():
        return 'development'
    
    # Default to local
    return 'local'

def main():
    """Main launcher function"""
    env = detect_environment()
    print(f"🌍 Detected environment: {env}")
    
    # Get launcher script path
    launcher_script = Path(__file__).parent / f"launch_{env}.py"
    
    if launcher_script.exists():
        print(f"🚀 Launching with {env} configuration...")
        subprocess.run([sys.executable, str(launcher_script)])
    else:
        print(f"❌ Launcher not found: {launcher_script}")
        print("Available launchers:")
        for launcher in Path(__file__).parent.glob("launch_*.py"):
            print(f"  - {launcher.name}")
        sys.exit(1)

if __name__ == "__main__":
    main()
