#!/usr/bin/env python3
"""
Advanced Demo Launcher

Quick launcher for the advanced multi-tab interface.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the advanced demo"""
    app_file = Path(__file__).parent / "advanced.py"
    
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        return
    
    print("🚀 Starting Advanced Demo...")
    print(f"📁 App: {app_file}")
    print("🌐 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    # Launch with streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_file),
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    main()
