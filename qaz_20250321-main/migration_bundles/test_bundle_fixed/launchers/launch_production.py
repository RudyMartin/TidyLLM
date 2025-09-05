#!/usr/bin/env python3
"""
Production Environment Launcher

Launches VectorQA Sage in production environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch application in production environment"""
    print(f"🚀 Launching VectorQA Sage in {env} environment...")
    
    # Set environment variable
    os.environ['VECTORQA_ENV'] = 'production'
    
    # Get project root (2 levels up from launchers directory)
    project_root = Path(__file__).parent.parent
    
    # Add settings to Python path
    settings_dir = project_root / "settings"
    if str(settings_dir) not in sys.path:
        sys.path.insert(0, str(settings_dir))
    
    # Load environment settings
    try:
        from settings_loader import setup_environment
        setup_environment('production')
    except ImportError as e:
        print(f"⚠️  Could not load settings: {e}")
    
    # Launch Streamlit app
    app_script = project_root / "src" / "main.py"
    if app_script.exists():
        print(f"🎯 Starting Streamlit app: {app_script}")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_script)
        ])
    else:
        print(f"❌ App script not found: {app_script}")
        sys.exit(1)

if __name__ == "__main__":
    main()
