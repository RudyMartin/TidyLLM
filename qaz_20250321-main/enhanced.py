#!/usr/bin/env python3
"""
QA Enhanced Demo Launcher

Simple entry point for the QA Enhanced demo. This script provides a quick way
to launch the demo with minimal setup and dependencies.

Usage:
  python start_demo.py                    # Launch demo on default port 8502
  python start_demo.py --port 8505       # Launch demo on custom port
  python start_demo.py --headless        # Launch in headless mode
  python start_demo.py --install         # Install streamlit if missing
  python start_demo.py --advanced        # Launch advanced launcher instead
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def check_streamlit():
    """Check if streamlit is available"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install streamlit if not available"""
    try:
        print("📦 Installing streamlit...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Streamlit installed successfully")
            return True
        else:
            print(f"❌ Failed to install streamlit: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing streamlit: {e}")
        return False

def setup_environment():
    """Basic environment setup"""
    try:
        # Add src to Python path
        project_root = Path(__file__).parent.absolute()
        src_path = project_root / "src"
        
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            print(f"✅ Added to Python path: {src_path}")
        
        # Set basic environment variables
        os.environ['VECTORQA_ENV'] = 'local'
        os.environ['LOG_LEVEL'] = 'INFO'
        
        return True
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return False

def launch_demo(port: int = 8502, headless: bool = False):
    """Launch the QA Enhanced demo"""
    try:
        # Check if demo file exists
        project_root = Path(__file__).parent.absolute()
        demo_file = project_root / "src" / "demo.py"
        
        if not demo_file.exists():
            print(f"❌ Demo file not found: {demo_file}")
            return False
        
        print("🚀 Launching QA Enhanced Demo")
        print(f"📁 App file: {demo_file}")
        print(f"🌐 URL: http://localhost:{port}")
        print("📝 Description: Simple chatbox with document upload and comparison")
        print("💡 Features: Document upload, TOC extraction, Bibliography analysis, Chat interface")
        
        # Build command
        cmd = [
            'streamlit', 'run', str(demo_file),
            '--server.port', str(port),
            '--server.address', '0.0.0.0',
            '--browser.gatherUsageStats', 'False',
            '--server.runOnSave', 'True'
        ]
        
        if headless:
            cmd.extend(['--server.headless', 'true'])
        
        print(f"⚡ Command: {' '.join(cmd)}")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 60)
        
        # Launch the app
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found.")
        print("💡 Install with: pip install streamlit")
        return False
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        return False
    
    return True

def launch_advanced_launcher():
    """Launch the advanced launcher"""
    try:
        project_root = Path(__file__).parent.absolute()
        advanced_launcher = project_root / "start_advanced.py"
        
        if not advanced_launcher.exists():
            print("❌ Advanced launcher not found: start_advanced.py")
            return False
        
        print("🚀 Launching Advanced Launcher...")
        subprocess.run([sys.executable, str(advanced_launcher)])
        
    except Exception as e:
        print(f"❌ Error launching advanced launcher: {e}")
        return False
    
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="QA Enhanced Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_demo.py                    # Launch demo on default port 8502
  python start_demo.py --port 8505       # Launch demo on custom port
  python start_demo.py --headless        # Launch in headless mode
  python start_demo.py --install         # Install streamlit if missing
  python start_demo.py --advanced        # Launch advanced launcher instead

Quick Start:
  1. python start_demo.py --install      # Install dependencies
  2. python start_demo.py                # Launch demo
  3. Open browser to http://localhost:8502
  4. Upload PDF documents and start chatting!
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8502,
        help='Port to run the demo on (default: 8502)'
    )
    
    parser.add_argument(
        '--headless', '-H',
        action='store_true',
        help='Run in headless mode (no browser)'
    )
    
    parser.add_argument(
        '--install', '-i',
        action='store_true',
        help='Install streamlit if not available'
    )
    
    parser.add_argument(
        '--advanced', '-a',
        action='store_true',
        help='Launch advanced launcher instead (for all apps)'
    )
    
    args = parser.parse_args()
    
    print("🎯 QA Enhanced Demo Launcher")
    print("=" * 60)
    
    # Launch advanced launcher if requested
    if args.advanced:
        return launch_advanced_launcher()
    
    # Check if streamlit is available
    if not check_streamlit():
        if args.install:
            if not install_streamlit():
                return False
        else:
            print("❌ Streamlit not available")
            print("💡 Install with: pip install streamlit")
            print("💡 Or run with: python start_demo.py --install")
            return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Launch demo
    return launch_demo(args.port, args.headless)

if __name__ == "__main__":
    main()
