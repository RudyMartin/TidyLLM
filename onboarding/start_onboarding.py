#!/usr/bin/env python3
"""
TidyLLM Onboarding System Starter
=================================

Cross-platform launcher with:
- OS detection and appropriate handling
- Kill old Streamlit processes
- Run-on-save enabled
- No browser restrictions
- Auto-reload for development

Usage:
    python start_onboarding.py
"""

import os
import sys
import subprocess
import platform
import signal
import time
import psutil
from pathlib import Path

def detect_os():
    """Detect operating system and return appropriate settings."""
    system = platform.system().lower()
    
    if system == "windows":
        return {
            "os": "windows",
            "shell": True,
            "kill_cmd": "taskkill /F /IM streamlit.exe",
            "port_check_cmd": "netstat -ano | findstr :8501"
        }
    elif system == "darwin":  # macOS
        return {
            "os": "macos", 
            "shell": False,
            "kill_cmd": "pkill -f streamlit",
            "port_check_cmd": "lsof -ti:8501"
        }
    elif system == "linux":
        return {
            "os": "linux",
            "shell": False, 
            "kill_cmd": "pkill -f streamlit",
            "port_check_cmd": "lsof -ti:8501"
        }
    else:
        return {
            "os": "unknown",
            "shell": False,
            "kill_cmd": "pkill -f streamlit",
            "port_check_cmd": "lsof -ti:8501"
        }

def kill_streamlit_processes():
    """Kill any existing Streamlit processes."""
    print("[CLEANUP] Checking for existing Streamlit processes...")
    
    try:
        # Method 1: Use psutil to find and kill processes
        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'streamlit' in proc.info['name'].lower():
                    print(f"[CLEANUP] Killing Streamlit process: PID {proc.info['pid']}")
                    proc.kill()
                    killed_count += 1
                elif proc.info['cmdline'] and any('streamlit' in str(cmd).lower() for cmd in proc.info['cmdline']):
                    print(f"[CLEANUP] Killing Streamlit process: PID {proc.info['pid']}")
                    proc.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if killed_count > 0:
            print(f"[CLEANUP] Killed {killed_count} existing Streamlit processes")
            time.sleep(2)  # Give processes time to die
        else:
            print("[CLEANUP] No existing Streamlit processes found")
            
    except Exception as e:
        print(f"[CLEANUP] Error killing processes: {e}")

def check_port_8501():
    """Check if port 8501 is in use."""
    print("[CHECK] Checking if port 8501 is available...")
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8501))
        sock.close()
        
        if result == 0:
            print("[WARNING] Port 8501 is already in use")
            return False
        else:
            print("[OK] Port 8501 is available")
            return True
    except Exception as e:
        print(f"[ERROR] Could not check port 8501: {e}")
        return True

def setup_environment():
    """Set up environment variables for TidyLLM onboarding."""
    print("[SETUP] Configuring environment...")
    
    # AWS credentials for TidyLLM onboarding
    os.environ["AWS_ACCESS_KEY_ID"] = "REMOVED_AWS_KEY"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "REMOVED_AWS_SECRET"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # Streamlit configuration
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    print("[OK] Environment configured")

def launch_streamlit():
    """Launch Streamlit with optimal settings for development."""
    
    # Get OS-specific settings
    os_info = detect_os()
    print(f"[INFO] Detected OS: {os_info['os']}")
    
    # Set working directory to onboarding folder
    onboarding_dir = Path(__file__).parent
    os.chdir(onboarding_dir)
    
    app_file = "app.py"
    
    # Check if app file exists
    if not Path(app_file).exists():
        print(f"[ERROR] App file not found: {app_file}")
        return False
    
    print(f"[LAUNCH] Starting TidyLLM Onboarding System")
    print(f"[FILE] {app_file}")
    print(f"[AUTO-RELOAD] Enabled - saves will refresh the app")
    print(f"[BROWSER] No restrictions - will open automatically")
    print(f"[URL] http://localhost:8501")
    
    # Build Streamlit command with optimal settings
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=false",  # Allow browser to open
        "--server.runOnSave=true",  # AUTO-RELOAD ON SAVE!
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.maxUploadSize=200",
        "--browser.serverAddress=localhost",
        "--browser.gatherUsageStats=false",
        "--logger.level=info"
    ]
    
    try:
        print(f"[COMMAND] {' '.join(cmd)}")
        print(f"[STARTING] Launching Streamlit...")
        
        # Run Streamlit with current environment
        process = subprocess.Popen(
            cmd, 
            env=os.environ.copy(),
            shell=os_info['shell']
        )
        
        print(f"[SUCCESS] Streamlit launched with PID: {process.pid}")
        print(f"[INFO] Press Ctrl+C to stop the server")
        
        # Wait for process to complete
        try:
            process.wait()
        except KeyboardInterrupt:
            print(f"\n[STOP] Stopping Streamlit server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print(f"[STOP] Server stopped")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Streamlit failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def main():
    """Main entry point."""
    print("TidyLLM Onboarding System Starter")
    print("=" * 50)
    
    # Step 1: Kill existing Streamlit processes
    kill_streamlit_processes()
    
    # Step 2: Check port availability
    if not check_port_8501():
        print("[WARNING] Port 8501 is in use. Attempting to kill processes...")
        kill_streamlit_processes()
        time.sleep(3)
    
    # Step 3: Setup environment
    setup_environment()
    
    # Step 4: Launch Streamlit
    success = launch_streamlit()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
