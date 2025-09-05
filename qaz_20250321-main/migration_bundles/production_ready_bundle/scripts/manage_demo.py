#!/usr/bin/env python3
"""
QA Demo Manager

Custom script to manage the QA demo Streamlit process.
Provides easy start, stop, restart, and status checking functionality.
"""

import os
import sys
import time
import signal
import subprocess
import psutil
from pathlib import Path
import argparse

class QADemoManager:
    """Manages the QA demo Streamlit process"""
    
    def __init__(self, port=8501):
        self.port = port
        self.app_path = "src/qa_demo.py"
        self.process = None
        self.pid_file = f"/tmp/qa_demo_{port}.pid"
        
    def find_streamlit_process(self):
        """Find existing Streamlit process for this app"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                # Look for Python process running streamlit with qa_demo.py
                if (proc.info['name'] in ['Python', 'python', 'python3'] and 
                    any('streamlit' in cmd for cmd in cmdline) and
                    any('qa_demo.py' in cmd for cmd in cmdline)):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def is_port_in_use(self):
        """Check if the port is in use"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', self.port))
                return False
        except OSError:
            return True
    
    def kill_existing_process(self):
        """Kill any existing Streamlit process"""
        print(f"🔍 Looking for existing Streamlit processes...")
        
        # Kill by process name
        try:
            subprocess.run(['pkill', '-f', 'streamlit run'], 
                         capture_output=True, check=False)
            print("✅ Killed existing Streamlit processes")
        except Exception as e:
            print(f"⚠️ Could not kill processes: {e}")
        
        # Kill by port
        if self.is_port_in_use():
            try:
                subprocess.run(['lsof', '-ti', f':{self.port}'], 
                             capture_output=True, check=False)
                subprocess.run(['kill', '-9', f'$(lsof -ti:{self.port})'], 
                             shell=True, capture_output=True, check=False)
                print(f"✅ Freed port {self.port}")
            except Exception as e:
                print(f"⚠️ Could not free port: {e}")
        
        # Wait a moment for processes to terminate
        time.sleep(2)
    
    def start_demo(self, headless=True):
        """Start the QA demo"""
        print(f"🚀 Starting QA Demo on port {self.port}...")
        
        # Kill any existing processes
        self.kill_existing_process()
        
        # Check if app file exists
        if not Path(self.app_path).exists():
            print(f"❌ App file not found: {self.app_path}")
            return False
        
        # Build command
        cmd = [
            'streamlit', 'run', self.app_path,
            '--server.port', str(self.port),
            '--server.runOnSave', 'false'
        ]
        
        if headless:
            cmd.extend(['--server.headless', 'true'])
        
        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(self.process.pid))
            
            print(f"✅ QA Demo started with PID: {self.process.pid}")
            print(f"🌐 Access at: http://localhost:{self.port}")
            
            # Wait a moment and check if it started successfully
            time.sleep(3)
            if self.process.poll() is None:
                print("✅ Demo is running successfully!")
                return True
            else:
                print("❌ Demo failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start demo: {e}")
            return False
    
    def stop_demo(self):
        """Stop the QA demo"""
        print("🛑 Stopping QA Demo...")
        
        # Kill by PID file
        if Path(self.pid_file).exists():
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGTERM)
                print(f"✅ Sent SIGTERM to PID: {pid}")
            except Exception as e:
                print(f"⚠️ Could not kill by PID: {e}")
        
        # Kill by process name
        self.kill_existing_process()
        
        # Remove PID file
        if Path(self.pid_file).exists():
            Path(self.pid_file).unlink()
        
        print("✅ Demo stopped")
    
    def restart_demo(self, headless=True):
        """Restart the QA demo"""
        print("🔄 Restarting QA Demo...")
        self.stop_demo()
        time.sleep(2)
        return self.start_demo(headless)
    
    def status(self):
        """Check demo status"""
        print("📊 QA Demo Status:")
        print(f"   Port: {self.port}")
        print(f"   App: {self.app_path}")
        
        # Check if app file exists
        if Path(self.app_path).exists():
            print("   ✅ App file exists")
        else:
            print("   ❌ App file missing")
            return False
        
        # Check if port is in use
        if self.is_port_in_use():
            print(f"   ✅ Port {self.port} is in use")
        else:
            print(f"   ❌ Port {self.port} is free")
            return False
        
        # Check if process is running
        proc = self.find_streamlit_process()
        if proc:
            print(f"   ✅ Streamlit process running (PID: {proc.pid})")
            return True
        else:
            print("   ❌ No Streamlit process found")
            return False
    
    def test_imports(self):
        """Test if all imports work"""
        print("🧪 Testing imports...")
        
        try:
            sys.path.append('src')
            from qa_demo import main, validate_review_id
            print("   ✅ QA Demo imports work")
            
            from backend.mcp.orchestrators.qa_orchestrator import QAOrchestrator
            print("   ✅ QA Orchestrator imports work")
            
            from backend.mcp.orchestrators.dspy_coordinator import DSPyCoordinator
            print("   ✅ DSPy Coordinator imports work")
            
            return True
            
        except ImportError as e:
            print(f"   ❌ Import failed: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='QA Demo Manager')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status', 'test'],
                       help='Action to perform')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port to run on (default: 8501)')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run in headless mode (default: True)')
    
    args = parser.parse_args()
    
    manager = QADemoManager(args.port)
    
    if args.action == 'start':
        success = manager.start_demo(args.headless)
        sys.exit(0 if success else 1)
        
    elif args.action == 'stop':
        manager.stop_demo()
        
    elif args.action == 'restart':
        success = manager.restart_demo(args.headless)
        sys.exit(0 if success else 1)
        
    elif args.action == 'status':
        is_running = manager.status()
        sys.exit(0 if is_running else 1)
        
    elif args.action == 'test':
        success = manager.test_imports()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
