#!/usr/bin/env python3
"""
AI Dropzone Manager Dashboard Launcher
======================================

Quick launcher script for the Streamlit dashboard with proper environment setup.
"""

import subprocess
import sys
import os
from pathlib import Path

def setup_environment():
    """Set up environment for dashboard."""
    print("[SETUP] Setting up dashboard environment...")
    
    # Set Python path
    current_dir = Path(__file__).parent
    tidyllm_path = current_dir / "tidyllm" 
    
    if str(tidyllm_path) not in sys.path:
        sys.path.insert(0, str(tidyllm_path))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(current_dir)
    os.environ["STREAMLIT_THEME_BASE"] = "light"
    os.environ["STREAMLIT_THEME_PRIMARY_COLOR"] = "#3b82f6"
    
    print("[OK] Environment configured")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("[CHECK] Verifying dashboard dependencies...")
    
    required_packages = [
        "streamlit",
        "plotly", 
        "pandas",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n[INSTALL] Installing missing packages: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"  [INSTALLED] {package}")
            except subprocess.CalledProcessError:
                print(f"  [ERROR] Failed to install {package}")
                return False
    
    print("[OK] All dependencies satisfied")
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("[LAUNCH] Starting AI Dropzone Manager Dashboard...")
    
    dashboard_path = Path("tidyllm/web/ai_dropzone_dashboard.py")
    
    if not dashboard_path.exists():
        print(f"[ERROR] Dashboard not found at {dashboard_path}")
        return False
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"[CMD] {' '.join(cmd)}")
        print("[INFO] Dashboard will open at: http://localhost:8501")
        print("[INFO] Press Ctrl+C to stop the dashboard")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n[STOP] Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to launch dashboard: {e}")
        return False

def show_dashboard_info():
    """Show dashboard information and features."""
    print("\n" + "=" * 70)
    print("AI DROPZONE MANAGER DASHBOARD")  
    print("=" * 70)
    print()
    print("Features:")
    print("- Real-time drop zone monitoring")
    print("- Interactive file processing")
    print("- Bracket command execution")
    print("- System performance metrics")
    print("- Processing history and analytics")
    print("- Worker status monitoring")
    print("- Error tracking and alerts")
    print()
    print("Drop Zones Available:")
    print("- MVR Analysis - [Process MVR]")
    print("- Financial Analysis - [Financial Analysis]") 
    print("- Contract Review - [Contract Review]")
    print("- Compliance Check - [Compliance Check]")
    print("- Quality Check - [Quality Check]")
    print("- Data Extraction - [Data Extraction]")
    print()
    print("Usage:")
    print("1. Drop documents in tidyllm/drop_zones/{workflow}/ folders")
    print("2. Monitor processing status in real-time")
    print("3. Execute bracket commands manually")
    print("4. View system metrics and worker status")
    print()

def main():
    """Main launcher function."""
    print("=" * 70)
    print("[LAUNCH] AI DROPZONE MANAGER DASHBOARD LAUNCHER")
    print("=" * 70)
    
    # Show dashboard info
    show_dashboard_info()
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("[ERROR] Dependency check failed. Please install missing packages.")
        return False
    
    # Launch dashboard
    success = launch_dashboard()
    
    if success:
        print("[SUCCESS] Dashboard session completed")
    else:
        print("[ERROR] Dashboard launch failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)