#!/usr/bin/env python3
"""
MVR Demo Launcher

Quick launcher for the MVR Review Streamlit demo.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the MVR demo"""
    app_file = Path(__file__).parent / "streamlit_mvr_demo.py"
    
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        return
    
    print("🧹 Clearing any existing Streamlit processes...")
    subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
    
    print("🚀 Starting MVR Review Streamlit Demo...")
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

