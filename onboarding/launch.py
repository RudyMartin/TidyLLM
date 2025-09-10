#!/usr/bin/env python3
"""
TidyLLM Onboarding Wizard Launcher
==================================

Simple script to launch the corporate onboarding wizard.

Usage:
    python -m tidyllm.onboarding.launch
    
Or:
    python tidyllm/onboarding/launch.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit onboarding wizard."""
    
    print("🚀 Starting TidyLLM Corporate Onboarding Wizard...")
    print("=" * 60)
    
    # Get the path to the streamlit app
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Error: Streamlit is not installed")
        print("Please install it with: pip install streamlit")
        sys.exit(1)
    
    print(f"📁 App location: {app_path}")
    print("🌐 Starting Streamlit server...")
    print("")
    print("💡 The wizard will open in your browser automatically.")
    print("   If it doesn't, navigate to: http://localhost:8501")
    print("")
    print("📋 This wizard will help you:")
    print("   • Detect your corporate environment")
    print("   • Configure AWS credentials safely")  
    print("   • Validate service connectivity")
    print("   • Generate deployment-ready configuration")
    print("")
    print("🔒 Security: This wizard does not store credentials permanently.")
    print("   All sensitive data is handled securely.")
    print("")
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Onboarding wizard stopped by user.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()