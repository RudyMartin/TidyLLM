#!/usr/bin/env python3
"""
LLM Gateway Control Dashboard Launcher
Launches the gateway control dashboard
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    print("🚀 Starting LLM Gateway Control Dashboard...")
    print(f"📁 Working directory: {script_dir}")
    print("🌐 Opening browser at: http://localhost:8502")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        subprocess.run([
            "python3", "-m", "streamlit", "run", 
            "gateway_control_dashboard.py",
            "--server.port", "8502",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Gateway Control Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
