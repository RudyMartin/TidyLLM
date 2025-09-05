#!/usr/bin/env python3
"""
Start Settings Configurator
Simple script to start the Streamlit settings configurator
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    print("🚀 Starting Settings Configurator...")
    print(f"📁 Working directory: {script_dir}")
    print("🌐 Opening browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        subprocess.run([
            "python3", "-m", "streamlit", "run", 
            "settings_configurator.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Settings Configurator stopped")
    except Exception as e:
        print(f"❌ Error starting configurator: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
