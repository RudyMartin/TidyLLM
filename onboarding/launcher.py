#!/usr/bin/env python3
"""
TidyLLM Onboarding System Launcher
==================================

Single entry point for the TidyLLM onboarding system.
Handles environment setup and launches the Streamlit application.

Usage:
    python launcher.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Set up AWS environment variables for TidyLLM onboarding."""
    print("[SETUP] Configuring AWS environment...")
    
    # Standard AWS credentials for TidyLLM onboarding
    os.environ["AWS_ACCESS_KEY_ID"] = "REMOVED_AWS_KEY"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "REMOVED_AWS_SECRET"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    print("[OK] AWS environment configured")

def launch_streamlit():
    """Launch Streamlit onboarding app with auto-reload."""
    
    # Set working directory to onboarding folder
    onboarding_dir = Path(__file__).parent
    os.chdir(onboarding_dir)
    
    app_file = "app.py"
    
    # Check if app file exists
    if not Path(app_file).exists():
        print(f"[ERROR] App file not found: {app_file}")
        return False
    
    print(f"[LAUNCH] Starting TidyLLM Onboarding System")
    print(f"[FILE] {app_file}")
    print(f"[AUTO-RELOAD] Enabled - saves will refresh the app")
    print(f"[URL] http://localhost:8501")
    
    # Build Streamlit command with auto-reload
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port=8501",
        "--server.address=0.0.0.0", 
        "--server.headless=true",
        "--server.runOnSave=true",  # AUTO-RELOAD ON SAVE!
        "--browser.serverAddress=localhost",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.maxUploadSize=200"
    ]
    
    try:
        # Run Streamlit with current environment (includes AWS vars)
        subprocess.run(cmd, check=True, env=os.environ.copy())
        return True
    except KeyboardInterrupt:
        print("\n[STOP] Onboarding app stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Streamlit failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TidyLLM Onboarding System Launcher")
    parser.add_argument("--validate", action="store_true", help="Run validation tests only")
    args = parser.parse_args()
    
    print("TidyLLM Onboarding System Launcher")
    print("=" * 40)
    
    if args.validate:
        print("[VALIDATE] Running validation tests...")
        # TODO: Add validation logic
        return
    
    # Step 1: Setup environment
    setup_environment()
    
    # Step 2: Launch Streamlit
    success = launch_streamlit()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
