#!/usr/bin/env python3
"""
Local Environment Launcher

Launches the VectorQA application optimized for local environment.
"""

import os
import sys
from pathlib import Path

def setup_local_environment():
    """Setup local-specific environment"""
    # Set environment variable
    os.environ["VECTORQA_ENV"] = "local"
    
    # Setup Python path for local
    bundle_root = Path(__file__).parent.parent
    src_dir = bundle_root / "src"
    config_dir = bundle_root / "config"
    
    # Add to Python path
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(config_dir))
    
    print(f"🚀 Starting VectorQA in {environment} mode...")
    print(f"📁 Bundle root: {bundle_root}")
    print(f"🐍 Python path configured for {environment}")

def main():
    """Main launcher function"""
    setup_local_environment()
    
    try:
        # Import and run the main application
        from main import main as app_main
        app_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Trying alternative import...")
        try:
            import main
            main.main()
        except Exception as e2:
            print(f"❌ Failed to start application: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    main()
