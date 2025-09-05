#!/usr/bin/env python3
"""
Live AI Ticker Launcher
Launches the live AI question ticker
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    print("📡 Starting Live AI Gateway Ticker...")
    print(f"📁 Working directory: {script_dir}")
    print("🌐 Opening browser at: http://localhost:8503")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        subprocess.run([
            "python3", "-m", "streamlit", "run", 
            "live_ticker.py",
            "--server.port", "8503",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Live Ticker stopped")
    except Exception as e:
        print(f"❌ Error starting ticker: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
