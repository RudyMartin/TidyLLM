#!/usr/bin/env python3
"""
Simple Demo Launcher

Quick launcher for the simple RAG demo with favorites prompt.

Need more help? See docs/STREAMLIT_SETUP_NOTES.md for streamlit setup troubleshooting.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the simple demo"""
    app_file = Path(__file__).parent / "simple_demo.py"
    
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        return
    
    print("🚀 Starting Simple Demo...")
    print(f"📁 App: {app_file}")
    print("🌐 URL: http://localhost:8555")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    # Check if we're in a conda environment
    conda_env = os.getenv('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"🐍 Using conda environment: {conda_env}")
        
        # If we're in base, try to use py311
        if conda_env == "base":
            print("🔄 Switching to py311 environment for streamlit...")
            try:
                # Use the full path to the py311 Python executable
                py311_python = "/Users/rudy/opt/anaconda3/envs/py311/bin/python"
                if os.path.exists(py311_python):
                    # Use echo to provide empty input for streamlit onboarding
                    cmd = f'echo "" | {py311_python} -m streamlit run {app_file} --server.port 8555 --server.address 0.0.0.0'
                    subprocess.run(cmd, shell=True)
                    return
                else:
                    print("❌ py311 Python not found at expected location")
                    return
            except Exception as e:
                print(f"❌ Error launching with py311: {e}")
                return
        else:
            # Use current conda environment
            # Get the conda environment's Python executable
            conda_prefix = os.getenv('CONDA_PREFIX')
            if conda_prefix:
                python_executable = os.path.join(conda_prefix, 'bin', 'python')
                if os.path.exists(python_executable):
                    subprocess.run([
                        python_executable, "-m", "streamlit", "run", str(app_file),
                        "--server.port", "8555",
                        "--server.address", "0.0.0.0"
                    ])
                else:
                    # Fallback to sys.executable
                    subprocess.run([
                        sys.executable, "-m", "streamlit", "run", str(app_file),
                        "--server.port", "8555",
                        "--server.address", "0.0.0.0"
                    ])
            else:
                # Fallback to sys.executable
                subprocess.run([
                    sys.executable, "-m", "streamlit", "run", str(app_file),
                    "--server.port", "8555",
                    "--server.address", "0.0.0.0"
                ])
    else:
        print("⚠️  No conda environment detected")
        print("💡 Consider running: conda activate py311")
        print("🔄 Attempting to activate py311 environment...")
        
        # Try to activate conda environment
        try:
            # Use conda run to execute in the py311 environment
            cmd = [
                "conda", "run", "-n", "py311", "python", "-m", "streamlit", "run", str(app_file),
                "--server.port", "8555",
                "--server.address", "0.0.0.0"
            ]
            subprocess.run(cmd)
            return
        except FileNotFoundError:
            print("❌ conda not found. Please activate py311 manually:")
            print("   conda activate py311")
            print("   python start_simple.py")
            return

if __name__ == "__main__":
    main()
