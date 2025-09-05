#!/usr/bin/env python3
"""
TidyLLM RAG TidyMart Demo Launcher
Smart launcher for TidyMart-enhanced RAG demonstration

LAUNCHER BEHAVIOR SPECIFICATION:
===============================
a) OS Detection & Process Management:
   - Detect Windows/Unix/Linux/Mac automatically
   - Kill existing Streamlit processes using appropriate OS commands:
     * Windows: taskkill /PID /T /F (process tree termination)
     * Unix/Linux/Mac: SIGTERM -> SIGKILL progression
   - Use psutil for cross-platform process discovery and management
   
b) Auto-Reload & No Browser Restrictions:
   - Enable --server.runOnSave=true for automatic file change detection
   - Use --server.headless=true to prevent forced browser opening
   - Disable CORS and XSRF for local development freedom
   - Allow external connections with --server.address=0.0.0.0
   
c) Dynamic Port Assignment:
   - NO hardcoded ports - scan from 8511 upward for available port
   - Handle port conflicts gracefully with automatic fallback
   - Display multiple access URLs: localhost, network IP, mobile access
   - Support up to 100 port attempts before failure

EXECUTION FLOW:
==============
1. Display system info (OS, Python version, working directory)
2. Kill any existing Streamlit processes (cross-platform)
3. Install/check dependencies if requested
4. Validate required packages are available
5. Find available port dynamically
6. Launch Streamlit with optimal settings for development
7. Monitor process and display logs in real-time
8. Handle graceful shutdown on Ctrl+C with proper cleanup
"""

import os
import sys
import subprocess
import platform
import signal
import time
import socket
import psutil
from pathlib import Path
from typing import Optional, List
import argparse

class RAGTidyMartLauncher:
    """Smart launcher for RAG TidyMart demo"""
    
    def __init__(self):
        self.app_file = "tidyllm/examples/rag_demo_with_tidymart.py"
        self.system = platform.system().lower()
        self.process = None
        self.port = None
        self.host = "0.0.0.0"
        
    def find_available_port(self, start_port: int = 8511) -> int:
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
        """Kill all existing Streamlit processes across platforms"""
        killed_count = 0
        
        print("Checking for existing Streamlit processes...")
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('streamlit' in arg.lower() for arg in cmdline):
                        if (any('rag' in arg.lower() for arg in cmdline) or 
                            any('tidymart' in arg.lower() for arg in cmdline) or
                            any('8511' in arg for arg in cmdline) or
                            any('rag_demo_with_tidymart.py' in arg for arg in cmdline)):
                            
                            print(f"Killing Streamlit process: PID {proc.info['pid']}")
                            
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
            print(f"Warning: Error checking processes: {e}")
        
        if killed_count > 0:
            print(f"SUCCESS: Killed {killed_count} existing Streamlit process(es)")
            time.sleep(2)
        else:
            print("SUCCESS: No existing Streamlit processes found")
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        required_packages = ['streamlit', 'polars', 'numpy']
        missing_packages = []
        
        print("Checking dependencies...")
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  OK: {package}")
            except ImportError:
                print(f"  MISSING: {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\nMissing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("SUCCESS: All dependencies available")
        return True
    
    def launch_streamlit(self):
        """Launch Streamlit with optimal settings"""
        
        if not os.path.exists(self.app_file):
            print(f"ERROR: Application file '{self.app_file}' not found!")
            return False
        
        # Find available port
        self.port = self.find_available_port()
        print(f"Using port: {self.port}")
        
        # Build Streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run", self.app_file,
            f"--server.port={self.port}",
            f"--server.address={self.host}",
            "--server.headless=true",
            "--server.runOnSave=true",
            "--browser.serverAddress=localhost",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
        ]
        
        print("Starting RAG TidyMart Demo...")
        print(f"URL: http://localhost:{self.port}")
        print(f"External access: http://{self.get_local_ip()}:{self.port}")
        
        try:
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"SUCCESS: Streamlit started with PID: {self.process.pid}")
            return True
            
        except Exception as e:
            print(f"ERROR: Error starting Streamlit: {e}")
            return False
    
    def get_local_ip(self) -> str:
        """Get local IP address for external access"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except:
            return "localhost"
    
    def monitor_process(self):
        """Monitor the Streamlit process"""
        if not self.process:
            print("ERROR: No process to monitor")
            return
        
        print("\n" + "="*60)
        print("RAG TIDYMART DEMO - RUNNING")
        print("Features: TidyMart Integration, Config Optimization, Performance Tracking")
        print("Press Ctrl+C to stop the application")
        print("="*60)
        
        try:
            while True:
                if self.process.poll() is not None:
                    print(f"\nProcess terminated with exit code: {self.process.returncode}")
                    break
                time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\nShutting down RAG TidyMart Demo...")
            self.cleanup()
    
    def cleanup(self):
        """Clean up processes on exit"""
        if self.process and self.process.poll() is None:
            print("Cleaning up processes...")
            
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
                
                print("SUCCESS: Process cleanup complete")
                
            except Exception as e:
                print(f"Warning during cleanup: {e}")
    
    def full_launch(self):
        """Complete launch sequence"""
        
        print("TidyLLM RAG TidyMart Demo Launcher")
        print("=" * 50)
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"App: RAG with TidyMart Integration")
        print("=" * 50)
        
        # Step 1: Kill existing processes
        self.kill_existing_streamlit_processes()
        
        # Step 2: Check dependencies
        if not self.check_dependencies():
            return False
        
        # Step 3: Launch Streamlit
        if not self.launch_streamlit():
            return False
        
        # Step 4: Show connection info
        print("\n" + "CONNECTION INFO".center(60, "="))
        print(f"Local URL:     http://localhost:{self.port}")
        print(f"Network URL:   http://{self.get_local_ip()}:{self.port}")
        print(f"Demo Type:     RAG with TidyMart Integration")
        print("=" * 60)
        
        # Step 5: Monitor process
        self.monitor_process()
        
        return True

def main():
    """Main entry point"""
    launcher = RAGTidyMartLauncher()
    
    import atexit
    atexit.register(launcher.cleanup)
    
    try:
        success = launcher.full_launch()
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