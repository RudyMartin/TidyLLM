#!/usr/bin/env python3
"""
Simple RAG Demo Launcher

A launcher that automatically sets up a virtual environment, installs dependencies,
and launches the simple RAG demo. Perfect for non-technical users.
"""

import sys
import os
import subprocess
import argparse
import venv
from pathlib import Path
import platform

# Requirements list (embedded in the launcher)
REQUIREMENTS = [
    "streamlit>=1.48.1",
    "PyPDF2>=3.0.0",  # Fallback PDF processor
    "pdfplumber>=0.11.0",  # Advanced PDF text and table extraction
    "pypdfium2>=4.30.0",  # Advanced PDF image extraction
    "pypdf>=6.0.0",  # Modern PDF library
    "requests>=2.31.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0"
]

def get_python_executable():
    """Get the Python executable path"""
    return sys.executable

def check_virtual_environment():
    """Check if we're running in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def create_virtual_environment(venv_path: Path):
    """Create a virtual environment"""
    try:
        print(f"📦 Creating virtual environment at: {venv_path}")
        venv.create(venv_path, with_pip=True)
        return True
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_venv_python(venv_path: Path):
    """Get the Python executable path in the virtual environment"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def get_venv_pip(venv_path: Path):
    """Get the pip executable path in the virtual environment"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"

def install_requirements(venv_path: Path):
    """Install requirements in the virtual environment"""
    try:
        pip_path = get_venv_pip(venv_path)
        
        print("📦 Installing requirements...")
        for requirement in REQUIREMENTS:
            print(f"   Installing: {requirement}")
            result = subprocess.run([
                str(pip_path), "install", requirement
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to install {requirement}: {result.stderr}")
                return False
        
        print("✅ All requirements installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_streamlit(venv_path: Path):
    """Check if streamlit is available in the virtual environment"""
    try:
        python_path = get_venv_python(venv_path)
        result = subprocess.run([
            str(python_path), "-c", "import streamlit; print('Streamlit available')"
        ], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def launch_demo(venv_path: Path, port: int = 8555, headless: bool = False):
    """Launch the simple demo"""
    try:
        python_path = get_venv_python(venv_path)
        demo_file = Path(__file__).parent / "simple_demo.py"
        
        if not demo_file.exists():
            print(f"❌ Demo file not found: {demo_file}")
            return False
        
        print("🚀 Launching Simple RAG Demo")
        print(f"📁 App file: {demo_file}")
        print(f"🌐 URL: http://localhost:{port}")
        print("📝 Description: Simple RAG demo with document upload and chat")
        print("💡 Features: Upload up to 5 documents, chat with AI, ZLLM gateway integration")
        
        # Build command
        cmd = [
            str(python_path), "-m", "streamlit", "run", str(demo_file),
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "False",
            "--server.runOnSave", "True"
        ]
        
        if headless:
            cmd.extend(["--server.headless", "true"])
        
        print(f"⚡ Command: {' '.join(cmd)}")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 60)
        
        # Launch the app
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        return False
    
    return True

def setup_environment():
    """Setup the environment for the demo"""
    try:
        # Get project root
        project_root = Path(__file__).parent.absolute()
        venv_path = project_root / "simple_demo_env"
        
        print("🔧 Setting up environment...")
        
        # Check if virtual environment exists
        if venv_path.exists():
            print(f"✅ Virtual environment found at: {venv_path}")
            
            # Check if streamlit is available
            if check_streamlit(venv_path):
                print("✅ Streamlit is available in virtual environment")
                return venv_path
            else:
                print("⚠️ Virtual environment exists but streamlit not found, reinstalling...")
                if not install_requirements(venv_path):
                    return None
        else:
            # Create virtual environment
            if not create_virtual_environment(venv_path):
                return None
            
            # Install requirements
            if not install_requirements(venv_path):
                return None
        
        return venv_path
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Simple RAG Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_simple_demo.py                    # Launch demo on port 8555
  python start_simple_demo.py --port 8560       # Launch demo on custom port
  python start_simple_demo.py --headless        # Launch in headless mode
  python start_simple_demo.py --reinstall       # Reinstall dependencies

Quick Start:
  1. python start_simple_demo.py                # One command to start everything
  2. Open browser to http://localhost:8555
  3. Upload documents and start chatting!
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8555,
        help='Port to run the demo on (default: 8555)'
    )
    
    parser.add_argument(
        '--headless', '-H',
        action='store_true',
        help='Run in headless mode (no browser)'
    )
    
    parser.add_argument(
        '--reinstall', '-r',
        action='store_true',
        help='Reinstall dependencies even if virtual environment exists'
    )
    
    args = parser.parse_args()
    
    print("🎯 Simple RAG Demo Launcher")
    print("=" * 60)
    print("🤖 Simple RAG Demo with Document Upload and Chat")
    print("📚 Upload up to 5 documents and chat with them using AI")
    print("🔧 Automatic virtual environment setup and dependency installation")
    print("=" * 60)
    
    # Setup environment
    venv_path = setup_environment()
    if not venv_path:
        print("❌ Failed to setup environment")
        return False
    
    # Reinstall if requested
    if args.reinstall:
        print("🔄 Reinstalling dependencies...")
        if not install_requirements(venv_path):
            print("❌ Failed to reinstall dependencies")
            return False
    
    # Launch demo
    return launch_demo(venv_path, args.port, args.headless)

if __name__ == "__main__":
    main()
