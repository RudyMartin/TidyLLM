#!/usr/bin/env python3
"""
VectorQA Sage Demo Launcher

This script handles environment initialization and launches different Streamlit apps
with proper configuration and error handling.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Import environment configuration
sys.path.insert(0, str(Path(__file__).parent / "src" / "config"))
from environments import setup_environment, get_environment

class DemoLauncher:
    """Handles environment setup and app launching"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.src_path = self.project_root / "src"
        self.backend_path = self.src_path / "backend"
        self.env = None
        
        # Available apps
        self.apps = {
            "advanced": {
                "file": "advanced.py",
                "name": "Advanced App (8-Tab Interface)",
                "description": "Complete VectorQA Sage interface with all tabs",
                "default_port": 8501
            },
            "demo": {
                "file": "demo.py", 
                "name": "QA Enhanced Demo",
                "description": "Simple chatbox with document upload and comparison",
                "default_port": 8502
            },
            "qa_demo": {
                "file": "qa_demo.py", 
                "name": "QA Document Processing Demo",
                "description": "MCP-based document processing interface",
                "default_port": 8503
            },
            "mcp_dashboard": {
                "file": "mcp_dashboard.py",
                "name": "MCP Dashboard",
                "description": "MCP hierarchical LLM monitoring dashboard",
                "default_port": 8504
            },
            "model_eval": {
                "file": "model_eval_dashboard.py",
                "name": "Model Evaluation Dashboard", 
                "description": "Model performance analytics dashboard",
                "default_port": 8505
            }
        }
    
    def setup_environment(self) -> bool:
        """Setup the Python environment for the application"""
        try:
            print("🔧 Setting up environment...")
            
            # Use centralized environment setup
            self.env = setup_environment()
            
            # Check if critical dependencies are available
            if not self._check_critical_dependencies():
                print("\n❌ Critical dependencies missing. Please install required packages.")
                print("💡 Run: pip install -r requirements_demo.txt")
                print("💡 Or install individually: pip install streamlit pandas numpy")
                return False
            
            print("✅ Environment setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
    
    def _check_critical_dependencies(self) -> bool:
        """Check if critical dependencies are available"""
        critical_packages = ["streamlit"]
        
        for package in critical_packages:
            try:
                __import__(package)
            except ImportError:
                print(f"❌ Critical package missing: {package}")
                return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        try:
            print("📦 Installing dependencies...")
            
            # Check if requirements file exists
            requirements_file = self.project_root / "requirements_demo.txt"
            if not requirements_file.exists():
                print("❌ requirements_demo.txt not found")
                return False
            
            # Install dependencies
            cmd = ["pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        # Dependencies are now checked by the environment setup
        pass
    
    def validate_app(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Validate that the requested app exists and is accessible"""
        if app_name not in self.apps:
            print(f"❌ Unknown app: {app_name}")
            print(f"Available apps: {', '.join(self.apps.keys())}")
            return None
        
        app_config = self.apps[app_name]
        app_file = self.src_path / app_config["file"]
        
        if not app_file.exists():
            print(f"❌ App file not found: {app_file}")
            return None
        
        return app_config
    
    def launch_app(self, app_name: str, port: Optional[int] = None, headless: bool = False):
        """Launch the specified Streamlit app"""
        app_config = self.validate_app(app_name)
        if not app_config:
            return False
        
        # Use provided port or default
        port = port or app_config["default_port"]
        app_file = self.src_path / app_config["file"]
        
        print(f"🚀 Launching {app_config['name']}")
        print(f"📁 App file: {app_file}")
        print(f"🌐 URL: http://localhost:{port}")
        print(f"📝 Description: {app_config['description']}")
        
        # Get environment-specific Streamlit configuration
        streamlit_config = self.env.get_streamlit_config() if self.env else {}
        
        # Build command with environment-specific settings
        cmd = [
            'streamlit', 'run', str(app_file),
            '--server.port', str(port),
            '--server.address', streamlit_config.get('server.address', '0.0.0.0'),
            '--browser.gatherUsageStats', str(streamlit_config.get('browser.gatherUsageStats', False)),
            '--server.runOnSave', str(streamlit_config.get('server.runOnSave', False))
        ]
        
        if headless or streamlit_config.get('server.headless', False):
            cmd.extend(['--server.headless', 'true'])
        
        try:
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
            print("💡 Or activate your conda environment: conda activate py311")
            return False
        except Exception as e:
            print(f"❌ Error launching app: {e}")
            print("💡 Try running: streamlit run src/demo.py --server.port 8502")
            return False
        
        return True
    
    def list_apps(self):
        """List all available apps"""
        print("📱 Available Applications:")
        print("=" * 60)
        
        for app_id, config in self.apps.items():
            print(f"\n🔹 {app_id}")
            print(f"   Name: {config['name']}")
            print(f"   File: {config['file']}")
            print(f"   Port: {config['default_port']}")
            print(f"   Description: {config['description']}")
        
        print("\n💡 Usage: python start_advanced.py <app_name> [--port PORT] [--headless] [--install-deps]")
    
    def run(self, app_name: Optional[str] = None, port: Optional[int] = None, headless: bool = False):
        """Main run method"""
        print("🎯 VectorQA Sage Demo Launcher")
        print("=" * 60)
        
        # Setup environment
        if not self.setup_environment():
            return False
        
        # If no app specified, show list
        if not app_name:
            self.list_apps()
            return True
        
        # Launch the specified app
        return self.launch_app(app_name, port, headless)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="VectorQA Sage Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_advanced.py                    # List all apps
  python start_advanced.py demo               # Launch QA Enhanced demo
  python start_advanced.py advanced           # Launch advanced app
  python start_advanced.py qa_demo --port 8505  # Launch QA demo on custom port
  python start_advanced.py mcp_dashboard --headless  # Launch MCP dashboard headless
  python start_advanced.py demo --install-deps  # Install dependencies and launch demo
        """
    )
    
    parser.add_argument(
        'app',
        nargs='?',
        help='App to launch (demo, advanced, qa_demo, mcp_dashboard, model_eval)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        help='Port to run the app on (default: app-specific)'
    )
    
    parser.add_argument(
        '--headless', '-H',
        action='store_true',
        help='Run in headless mode (no browser)'
    )
    
    parser.add_argument(
        '--install-deps', '-i',
        action='store_true',
        help='Install required dependencies before launching'
    )
    
    args = parser.parse_args()
    
    launcher = DemoLauncher()
    
    # Install dependencies if requested
    if args.install_deps:
        if not launcher.install_dependencies():
            return False
    
    launcher.run(args.app, args.port, args.headless)

if __name__ == "__main__":
    main()
