#!/usr/bin/env python3
"""
TidyLLM-HeirOS Smart Launcher
Cross-platform launcher with process management and auto-reload

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
   - NO hardcoded ports - scan from 8501 upward for available port
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

class HeirOSLauncher:
    """Smart launcher for HeirOS Streamlit dashboard"""
    
    def __init__(self, app_file: str = "heiros_streamlit_demo.py"):
        self.app_file = app_file
        self.system = platform.system().lower()
        self.process = None
        self.port = None
        self.host = "0.0.0.0"  # Allow external connections
        
    def find_available_port(self, start_port: int = 8501) -> int:
        """Find an available port starting from start_port"""
        port = start_port
        while port < start_port + 100:  # Check 100 ports max
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
                        # Check if it's running our specific app or any streamlit
                        if (any(self.app_file in arg for arg in cmdline) or 
                            any('heiros' in arg.lower() for arg in cmdline) or
                            any('streamlit run' in ' '.join(cmdline).lower() for arg in cmdline)):
                            
                            print(f"Killing Streamlit process: PID {proc.info['pid']}")
                            
                            if self.system == "windows":
                                # Windows: Use taskkill for forceful termination
                                subprocess.run([
                                    'taskkill', '/PID', str(proc.info['pid']), '/F'
                                ], capture_output=True, check=False)
                            else:
                                # Unix/Linux/Mac: Use SIGTERM then SIGKILL
                                try:
                                    os.kill(proc.info['pid'], signal.SIGTERM)
                                    time.sleep(1)  # Give process time to terminate gracefully
                                    # Check if still running
                                    if psutil.pid_exists(proc.info['pid']):
                                        os.kill(proc.info['pid'], signal.SIGKILL)
                                except ProcessLookupError:
                                    pass  # Process already terminated
                            
                            killed_count += 1
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            print(f"Warning: Error checking processes: {e}")
        
        if killed_count > 0:
            print(f"SUCCESS: Killed {killed_count} existing Streamlit process(es)")
            time.sleep(2)  # Wait for cleanup
        else:
            print("SUCCESS: No existing Streamlit processes found")
    
    def install_requirements(self):
        """Install required packages"""
        requirements_file = "heiros_requirements.txt"
        
        if not os.path.exists(requirements_file):
            print(f"Requirements file {requirements_file} not found, skipping...")
            return
        
        print("Installing/updating requirements...")
        try:
            # Use pip install with upgrade flag
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", requirements_file, "--upgrade"
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print("SUCCESS: Requirements installed successfully")
            else:
                print(f"Warning: Requirements installation had issues:\n{result.stderr}")
                
        except Exception as e:
            print(f"ERROR: Error installing requirements: {e}")
    
    def launch_streamlit(self, auto_reload: bool = True, open_browser: bool = False):
        """Launch Streamlit with optimal settings"""
        
        if not os.path.exists(self.app_file):
            print(f"ERROR: Application file '{self.app_file}' not found!")
            return False
        
        # Find available port
        self.port = self.find_available_port()
        print(f"Using port: {self.port}")
        
        # Build Streamlit command with optimized settings
        cmd = [
            sys.executable, "-m", "streamlit", "run", self.app_file,
            f"--server.port={self.port}",
            f"--server.address={self.host}",
            "--server.headless=true",  # Don't auto-open browser
            "--server.runOnSave=true" if auto_reload else "--server.runOnSave=false",
            "--browser.serverAddress=localhost",  # Browser connection address
            "--server.enableCORS=false",  # Disable CORS for local dev
            "--server.enableXsrfProtection=false",  # Disable XSRF for local dev
        ]
        
        if not open_browser:
            cmd.append("--server.headless=true")
        
        print("Starting HeirOS Streamlit Dashboard...")
        print(f"URL: http://localhost:{self.port}")
        print(f"Auto-reload: {'Enabled' if auto_reload else 'Disabled'}")
        print(f"External access: http://{self.get_local_ip()}:{self.port}")
        
        try:
            # Start process
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
            # Connect to a remote address to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except:
            return "localhost"
    
    def monitor_process(self, show_logs: bool = True):
        """Monitor the Streamlit process and show logs"""
        if not self.process:
            print("ERROR: No process to monitor")
            return
        
        print("\n" + "="*60)
        print("MONITORING STREAMLIT PROCESS")
        print("Press Ctrl+C to stop the application")
        print("="*60)
        
        try:
            while True:
                if self.process.poll() is not None:
                    print(f"\nProcess terminated with exit code: {self.process.returncode}")
                    break
                
                if show_logs:
                    # Read and display output
                    line = self.process.stdout.readline()
                    if line:
                        print(line.strip())
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\nShutting down HeirOS Dashboard...")
            self.cleanup()
    
    def cleanup(self):
        """Clean up processes on exit"""
        if self.process and self.process.poll() is None:
            print("Cleaning up processes...")
            
            try:
                if self.system == "windows":
                    # Windows: Kill process tree
                    subprocess.run([
                        'taskkill', '/PID', str(self.process.pid), '/T', '/F'
                    ], capture_output=True, check=False)
                else:
                    # Unix: Send SIGTERM first, then SIGKILL if needed
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                
                print("SUCCESS: Process cleanup complete")
                
            except Exception as e:
                print(f"Warning during cleanup: {e}")
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        required_packages = ['streamlit', 'pandas', 'psycopg2', 'plotly']
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
            print("Run with --install-deps to install them")
            return False
        
        print("SUCCESS: All dependencies available")
        return True
    
    def full_launch(self, auto_reload: bool = True, open_browser: bool = False, 
                   install_deps: bool = False, show_logs: bool = True):
        """Complete launch sequence"""
        
        print("TidyLLM-HeirOS Smart Launcher")
        print("=" * 50)
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Working Directory: {os.getcwd()}")
        print(f"App File: {self.app_file}")
        print("=" * 50)
        
        # Step 1: Kill existing processes
        self.kill_existing_streamlit_processes()
        
        # Step 2: Install dependencies if requested
        if install_deps:
            self.install_requirements()
        
        # Step 3: Check dependencies
        if not self.check_dependencies():
            print("\n💡 Try running with --install-deps flag")
            return False
        
        # Step 4: Launch Streamlit
        if not self.launch_streamlit(auto_reload, open_browser):
            return False
        
        # Step 5: Show connection info
        print("\n" + "CONNECTION INFO".center(60, "="))
        print(f"Local URL:     http://localhost:{self.port}")
        print(f"Network URL:   http://{self.get_local_ip()}:{self.port}")
        print(f"Mobile URL:    http://{self.get_local_ip()}:{self.port}")
        print("=" * 60)
        
        # Step 6: Monitor process
        self.monitor_process(show_logs)
        
        return True

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="TidyLLM-HeirOS Smart Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--app-file", 
        default="heiros_streamlit_demo.py",
        help="Streamlit app file to launch (default: heiros_streamlit_demo.py)"
    )
    
    parser.add_argument(
        "--no-auto-reload", 
        action="store_true",
        help="Disable auto-reload on file changes"
    )
    
    parser.add_argument(
        "--open-browser", 
        action="store_true",
        help="Open browser automatically"
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
        "--kill-only", 
        action="store_true",
        help="Only kill existing Streamlit processes and exit"
    )
    
    args = parser.parse_args()
    
    launcher = HeirOSLauncher(args.app_file)
    
    # Register cleanup on exit
    import atexit
    atexit.register(launcher.cleanup)
    
    try:
        if args.kill_only:
            launcher.kill_existing_streamlit_processes()
            print("✅ Process cleanup complete")
            return
        
        success = launcher.full_launch(
            auto_reload=not args.no_auto_reload,
            open_browser=args.open_browser,
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