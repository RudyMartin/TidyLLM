#!/usr/bin/env python3
"""
TidyLLM Demos Launcher

Quick launcher for the TidyLLM Demos Streamlit application.
Supports automatic restart and process cleanup.

Usage:
    python start_demos.py           # Launch demos
    python start_demos.py --install # Just install dependencies, don't launch
    python start_demos.py --restart # Force restart (kill existing processes)
"""

import subprocess
import sys
import argparse
import time
import signal
import os
import platform
from pathlib import Path

def kill_existing_processes():
    """Kill any existing Streamlit processes"""
    print("🧹 Clearing any existing Streamlit processes...")
    
    # Kill streamlit processes
    subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
    subprocess.run(["pkill", "-f", "launcher.py"], capture_output=True)
    
    # Wait a moment for processes to terminate
    time.sleep(2)
    
    # Double-check and force kill if needed
    subprocess.run(["pkill", "-9", "-f", "streamlit"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "launcher.py"], capture_output=True)
    
    print("✅ Process cleanup completed")

def check_mac_conda_environment():
    """Check if we're on Mac and suggest conda activation"""
    if platform.system() == "Darwin":  # macOS
        print("🍎 Detected macOS")
        
        # Check if conda is available
        conda_result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if conda_result.returncode == 0:
            print("✅ Conda is available")
            
            # Check current conda environment
            env_result = subprocess.run(["conda", "info", "--envs"], capture_output=True, text=True)
            if env_result.returncode == 0:
                # Look for py311 environment
                if "py311" in env_result.stdout:
                    print("🔍 Found py311 conda environment")
                    
                    # Check if we're currently in py311
                    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
                    if current_env != 'py311':
                        print("⚠️  You're not in the py311 conda environment")
                        print("💡 Please run: conda activate py311")
                        print("   Then run this script again")
                        return False
                    else:
                        print("✅ Currently in py311 conda environment")
                        return True
                else:
                    print("⚠️  py311 conda environment not found")
                    print("💡 Please create it with: conda create -n py311 python=3.11")
                    return False
        else:
            print("⚠️  Conda not found - continuing with system Python")
            return True
    
    return True

def install_requirements():
    """Install requirements"""
    req_file = "requirements.txt"
    
    if not Path(req_file).exists():
        print(f"❌ Requirements file not found: {req_file}")
        return False
    
    print("📦 Installing TidyLLM Demos requirements...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", req_file, "-q"
    ])
    
    if result.returncode == 0:
        print("✅ Requirements installed successfully")
        return True
    else:
        print("❌ Failed to install requirements")
        return False

def test_dependencies():
    """Test if all dependencies are available"""
    print("🔍 Testing TidyLLM Demos Dependencies...")
    
    test_script = """
try:
    import streamlit
    import yaml
    import pandas as pd
    import plotly.express as px
    print('✅ Core dependencies available')
    
    # Test shared modules
    import sys
    sys.path.append('shared')
    from utils import load_settings, save_settings
    print('✅ Shared utilities available')
    
    print('🎉 TidyLLM Demos - All Dependencies Resolved')
    print('🚀 System Ready: True')
    
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
"""
    
    result = subprocess.run([
        sys.executable, "-c", test_script
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print("⚠️ Warning: Some dependencies may be missing")
        print(result.stderr)
        return False

def launch_demos():
    """Launch the demos application"""
    app_file = Path(__file__).parent / "launcher.py"
    
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        return False
    
    print("🚀 Starting TidyLLM Demos...")
    print(f"📁 App: {app_file}")
    print("🌐 URL: http://localhost:8501")
    print("📋 Available Demos:")
    print("   • Settings Configurator")
    print("   • Live Ticker")
    print("   • Gateway Control Dashboard")
    print("   • MVR Demo")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    # Launch with streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "False",
            "--server.runOnSave", "True"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Demos stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching demos: {e}")
        return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='TidyLLM Demos Launcher')
    parser.add_argument('--install', action='store_true',
                       help='Only install requirements, do not launch demos')
    parser.add_argument('--restart', action='store_true',
                       help='Force restart (kill existing processes)')
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check Mac conda environment first
    if not check_mac_conda_environment():
        return
    
    # Force restart if requested
    if args.restart:
        kill_existing_processes()
    
    # Install requirements
    if not install_requirements():
        return
    
    # If only installing, exit here
    if args.install:
        print("🎯 Installation complete. Run without --install to launch demos.")
        return
    
    # Test dependencies
    if not test_dependencies():
        print("⚠️ Continuing anyway...")
    
    # Kill any existing processes before launch
    kill_existing_processes()
    
    # Launch the demos
    launch_demos()

if __name__ == "__main__":
    main()
