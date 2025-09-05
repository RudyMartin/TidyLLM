#!/usr/bin/env python3
"""
Universal Streamlit Launcher for TidyLLM Demos
==============================================

SOLVES THE STREAMLIT DEMO CHAOS:
- Standardizes ALL demo launchers to 'start_{demo_name}' pattern
- Enables run-on-save for ALL demos (no more manual refresh hassle!)
- Uses unified session management (no more scattered connections)
- Cross-platform process management
- Dynamic port assignment with fallbacks

STANDARDIZED NAMING PATTERN:
- start_whitepapers_demo.py
- start_heiros_demo.py  
- start_rag_demo.py
- start_vectorqa_demo.py
- start_gateway_demo.py

FEATURES THAT END THE HASSLE:
✅ --server.runOnSave=true (auto-reload on file changes)
✅ Unified session management (S3, PostgreSQL, Bedrock)
✅ Smart port discovery (no conflicts)
✅ Process cleanup (no orphaned processes)
✅ Health monitoring for all services
✅ Consistent logging and error handling
"""

import os
import sys
import subprocess
import platform
import signal
import time
import socket
import psutil
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Import our unified session manager
sys.path.append(str(Path(__file__).parent))
from start_unified_sessions import get_global_session_manager, ServiceType

class StreamlitDemoLauncher:
    """Universal launcher for all TidyLLM Streamlit demos"""
    
    DEMO_REGISTRY = {
        # Main demos (prioritized)
        "whitepapers": {
            "file": "tidyllm-whitepapers/streamlit_demo/app.py",
            "name": "TidyLLM Whitepapers Research",
            "description": "Mathematical decomposition research papers",
            "services": [ServiceType.S3, ServiceType.POSTGRESQL]
        },
        "heiros": {
            "file": "heiros_streamlit_demo.py", 
            "name": "TidyLLM-HeirOS Workflow Dashboard",
            "description": "Hierarchical workflow management",
            "services": [ServiceType.POSTGRESQL]
        },
        "vectorqa": {
            "file": "tidyllm-vectorqa/tidyllm_vectorqa/whitepapers/app.py",
            "name": "TidyLLM Vector QA System",
            "description": "Vector-based question answering",
            "services": [ServiceType.S3, ServiceType.POSTGRESQL]
        },
        "rag": {
            "file": "tidyllm-vectorqa/tidyllm_vectorqa/whitepapers/streamlit_rag_demo.py",
            "name": "TidyLLM RAG Demo",
            "description": "Retrieval-augmented generation",
            "services": [ServiceType.S3, ServiceType.POSTGRESQL, ServiceType.BEDROCK]
        },
        
        # Secondary demos
        "gateway": {
            "file": "tidyllm/demo-standalone/my_config/gateway_control_dashboard.py",
            "name": "TidyLLM Gateway Control",
            "description": "Gateway management dashboard",
            "services": [ServiceType.POSTGRESQL]
        },
        "ticker": {
            "file": "tidyllm/demo-standalone/my_config/live_ticker.py",
            "name": "TidyLLM Live Ticker",
            "description": "Live data ticker dashboard",
            "services": [ServiceType.POSTGRESQL]
        },
        "settings": {
            "file": "tidyllm-demos/demos/settings-config/settings_config.py",
            "name": "TidyLLM Settings Configuration",
            "description": "System settings configuration",
            "services": []
        }
    }
    
    def __init__(self, demo_name: str):
        if demo_name not in self.DEMO_REGISTRY:
            available = ", ".join(self.DEMO_REGISTRY.keys())
            raise ValueError(f"Unknown demo '{demo_name}'. Available: {available}")
        
        self.demo_name = demo_name
        self.demo_config = self.DEMO_REGISTRY[demo_name]
        self.app_file = self.demo_config["file"]
        self.system = platform.system().lower()
        self.process = None
        self.port = None
        self.host = "0.0.0.0"
        
        # Initialize unified session manager
        self.session_mgr = get_global_session_manager()
    
    def find_available_port(self, start_port: int = 8501) -> int:
        """Find an available port starting from start_port"""
        port = start_port
        while port < start_port + 100:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('localhost', port))
                    return port
            except OSError:
                port += 1
        raise RuntimeError(f"No available ports found starting from {start_port}")
    
    def kill_existing_streamlit_processes(self):
        """Kill existing Streamlit processes for this demo"""
        killed_count = 0
        
        print("🔍 Checking for existing Streamlit processes...")
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('streamlit' in arg.lower() for arg in cmdline):
                        # Check if it's our specific demo
                        demo_file = Path(self.app_file).name
                        if (any(demo_file in arg for arg in cmdline) or 
                            any(self.demo_name in arg.lower() for arg in cmdline)):
                            
                            print(f"🔪 Killing Streamlit process: PID {proc.info['pid']}")
                            
                            if self.system == "windows":
                                subprocess.run([
                                    'taskkill', '/PID', str(proc.info['pid']), '/F'
                                ], capture_output=True, check=False)
                            else:
                                try:
                                    os.kill(proc.info['pid'], signal.SIGTERM)
                                    time.sleep(1)
                                    if psutil.pid_exists(proc.info['pid']):
                                        os.kill(proc.info['pid'], signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            
                            killed_count += 1
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            print(f"⚠️  Warning checking processes: {e}")
        
        if killed_count > 0:
            print(f"✅ Killed {killed_count} existing process(es)")
            time.sleep(2)
        else:
            print("✅ No existing processes found")
    
    def check_demo_file_exists(self) -> bool:
        """Check if demo file exists"""
        file_path = Path(self.app_file)
        if not file_path.exists():
            # Try relative to current directory
            if not file_path.is_absolute():
                abs_path = Path.cwd() / file_path
                if abs_path.exists():
                    self.app_file = str(abs_path)
                    return True
            
            print(f"❌ Demo file not found: {self.app_file}")
            print(f"   Current directory: {Path.cwd()}")
            return False
        
        return True
    
    def check_service_health(self):
        """Check health of required services for this demo"""
        print(f"🏥 Checking health of required services...")
        
        required_services = self.demo_config.get("services", [])
        
        if not required_services:
            print("✅ No external services required")
            return True
        
        all_healthy = True
        health_summary = self.session_mgr.get_health_summary()
        
        for service in required_services:
            service_health = health_summary["services"][service.value]
            is_healthy = service_health["healthy"]
            
            status_icon = "✅" if is_healthy else "⚠️"
            status_text = "HEALTHY" if is_healthy else "DEGRADED"
            
            print(f"  {status_icon} {service.value}: {status_text}")
            
            if not is_healthy:
                error = service_health.get("error")
                if error:
                    print(f"    Error: {error}")
                all_healthy = False
        
        if not all_healthy:
            print("\n💡 Some services are degraded but demo will continue with fallback behavior")
        
        return True  # Always continue, even with degraded services
    
    def install_requirements_if_needed(self):
        """Install requirements if requirements file exists"""
        possible_req_files = [
            Path(self.app_file).parent / "requirements.txt",
            Path("requirements.txt"),
            Path(f"{self.demo_name}_requirements.txt")
        ]
        
        for req_file in possible_req_files:
            if req_file.exists():
                print(f"📦 Installing requirements from {req_file}")
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", "-r", str(req_file), "--upgrade"
                    ], capture_output=True, text=True, check=False)
                    
                    if result.returncode == 0:
                        print("✅ Requirements installed successfully")
                    else:
                        print(f"⚠️  Requirements installation had issues: {result.stderr[:200]}")
                    return
                    
                except Exception as e:
                    print(f"❌ Error installing requirements: {e}")
                    return
        
        print("📦 No requirements.txt found, continuing...")
    
    def launch_streamlit_with_unified_config(self):
        """Launch Streamlit with optimized settings for TidyLLM demos"""
        
        # Find available port
        self.port = self.find_available_port()
        print(f"🚀 Using port: {self.port}")
        
        # Build optimized Streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run", self.app_file,
            f"--server.port={self.port}",
            f"--server.address={self.host}",
            "--server.headless=true",  # Don't auto-open browser
            "--server.runOnSave=true",  # 🎯 AUTO-RELOAD ON SAVE!
            "--browser.serverAddress=localhost",
            "--server.enableCORS=false",  # Disable CORS for local dev
            "--server.enableXsrfProtection=false",  # Disable XSRF for local dev
            "--server.maxUploadSize=200",  # Allow larger file uploads
            "--server.enableStaticServing=true",  # Enable static file serving
        ]
        
        print(f"🎬 Starting {self.demo_config['name']}")
        print(f"📄 Description: {self.demo_config['description']}")
        print(f"🔄 Auto-reload: ENABLED (runOnSave=true)")
        print(f"🌐 URL: http://localhost:{self.port}")
        print(f"📱 Network URL: http://{self.get_local_ip()}:{self.port}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"✅ Streamlit started with PID: {self.process.pid}")
            return True
            
        except Exception as e:
            print(f"❌ Error starting Streamlit: {e}")
            return False
    
    def get_local_ip(self) -> str:
        """Get local IP address for external access"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except:
            return "localhost"
    
    def monitor_process(self, show_logs: bool = True):
        """Monitor the Streamlit process"""
        if not self.process:
            print("❌ No process to monitor")
            return
        
        print("\n" + "="*70)
        print(f"🖥️  MONITORING {self.demo_config['name'].upper()}")
        print("Press Ctrl+C to stop the demo")
        print("="*70)
        
        try:
            while True:
                if self.process.poll() is not None:
                    print(f"\n💀 Process terminated with exit code: {self.process.returncode}")
                    break
                
                if show_logs:
                    line = self.process.stdout.readline()
                    if line:
                        print(line.strip())
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Shutting down {self.demo_config['name']}...")
            self.cleanup()
    
    def cleanup(self):
        """Clean up processes on exit"""
        if self.process and self.process.poll() is None:
            print("🧹 Cleaning up processes...")
            
            try:
                if self.system == "windows":
                    subprocess.run([
                        'taskkill', '/PID', str(self.process.pid), '/T', '/F'
                    ], capture_output=True, check=False)
                else:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                
                print("✅ Process cleanup complete")
                
            except Exception as e:
                print(f"⚠️  Warning during cleanup: {e}")
    
    def full_launch_sequence(self, install_deps: bool = False, show_logs: bool = True):
        """Complete launch sequence for any TidyLLM demo"""
        
        print("🚀 TidyLLM Universal Streamlit Launcher")
        print("=" * 60)
        print(f"Demo: {self.demo_config['name']}")
        print(f"Description: {self.demo_config['description']}")
        print(f"File: {self.app_file}")
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Working Directory: {os.getcwd()}")
        print("=" * 60)
        
        # Step 1: Check demo file exists
        if not self.check_demo_file_exists():
            return False
        
        # Step 2: Kill existing processes
        self.kill_existing_streamlit_processes()
        
        # Step 3: Install dependencies if requested
        if install_deps:
            self.install_requirements_if_needed()
        
        # Step 4: Check service health
        if not self.check_service_health():
            print("⚠️  Service health check failed, but continuing...")
        
        # Step 5: Launch Streamlit
        if not self.launch_streamlit_with_unified_config():
            return False
        
        # Step 6: Show service summary
        health_summary = self.session_mgr.get_health_summary()
        print("\n" + "SERVICE STATUS".center(60, "="))
        print(f"Overall Health: {'✅ HEALTHY' if health_summary['overall_healthy'] else '⚠️  DEGRADED'}")
        print(f"Credential Source: {health_summary['credential_source']}")
        for service, status in health_summary['services'].items():
            if status['healthy']:
                latency = f" ({status['latency_ms']:.1f}ms)" if status['latency_ms'] else ""
                print(f"  ✅ {service.upper()}: READY{latency}")
            else:
                print(f"  ❌ {service.upper()}: FAILED")
        print("=" * 60)
        
        # Step 7: Monitor process
        self.monitor_process(show_logs)
        
        return True

def create_standardized_launcher(demo_name: str):
    """Create a standardized launcher script for a specific demo"""
    
    if demo_name not in StreamlitDemoLauncher.DEMO_REGISTRY:
        available = ", ".join(StreamlitDemoLauncher.DEMO_REGISTRY.keys())
        raise ValueError(f"Unknown demo '{demo_name}'. Available: {available}")
    
    launcher_name = f"start_{demo_name}_demo.py"
    launcher_path = Path(__file__).parent / launcher_name
    
    launcher_content = f'''#!/usr/bin/env python3
"""
Standardized Launcher for {StreamlitDemoLauncher.DEMO_REGISTRY[demo_name]["name"]}
Generated by TidyLLM Universal Streamlit Launcher System
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from start_streamlit_universal import StreamlitDemoLauncher
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="{StreamlitDemoLauncher.DEMO_REGISTRY[demo_name]['description']}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Install/update requirements before launching"
    )
    
    parser.add_argument(
        "--no-logs", 
        action="store_true",
        help="Don't show Streamlit logs in console"
    )
    
    args = parser.parse_args()
    
    launcher = StreamlitDemoLauncher("{demo_name}")
    
    # Register cleanup on exit
    import atexit
    atexit.register(launcher.cleanup)
    
    try:
        success = launcher.full_launch_sequence(
            install_deps=args.install_deps,
            show_logs=not args.no_logs
        )
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\\nGoodbye!")
        launcher.cleanup()
    except Exception as e:
        print(f"Unexpected error: {{e}}")
        launcher.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    # Create launcher file
    launcher_path.parent.mkdir(exist_ok=True)
    launcher_path.write_text(launcher_content)
    
    # Make executable on Unix systems
    if not sys.platform.startswith('win'):
        import stat
        launcher_path.chmod(launcher_path.stat().st_mode | stat.S_IEXEC)
    
    return launcher_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TidyLLM Universal Streamlit Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available demos:
{chr(10).join(f'  {name}: {config["description"]}' for name, config in StreamlitDemoLauncher.DEMO_REGISTRY.items())}

Examples:
  python start_streamlit_universal.py whitepapers
  python start_streamlit_universal.py heiros --install-deps
  python start_streamlit_universal.py rag --no-logs
        """
    )
    
    parser.add_argument(
        "demo",
        nargs='?',
        choices=list(StreamlitDemoLauncher.DEMO_REGISTRY.keys()),
        help="Demo to launch"
    )
    
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Install/update requirements before launching"
    )
    
    parser.add_argument(
        "--no-logs", 
        action="store_true",
        help="Don't show Streamlit logs in console"
    )
    
    parser.add_argument(
        "--create-launchers",
        action="store_true",
        help="Create standardized launcher scripts for all demos"
    )
    
    args = parser.parse_args()
    
    if args.create_launchers:
        print("Creating standardized launcher scripts...")
        for demo_name in StreamlitDemoLauncher.DEMO_REGISTRY.keys():
            launcher_path = create_standardized_launcher(demo_name)
            print(f"Created: {launcher_path}")
        print(f"Created {len(StreamlitDemoLauncher.DEMO_REGISTRY)} launcher scripts")
        return
    
    if not args.demo:
        parser.print_help()
        return
    
    launcher = StreamlitDemoLauncher(args.demo)
    
    # Register cleanup on exit
    import atexit
    atexit.register(launcher.cleanup)
    
    try:
        success = launcher.full_launch_sequence(
            install_deps=args.install_deps,
            show_logs=not args.no_logs
        )
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        launcher.cleanup()
    except Exception as e:
        print(f"Unexpected error: {e}")
        launcher.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()