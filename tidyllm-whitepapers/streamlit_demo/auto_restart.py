#!/usr/bin/env python3
"""
Auto-restart script for Streamlit demo
Automatically restarts Streamlit when app.py changes
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class StreamlitHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.demo_dir = Path(__file__).parent
        self.start_streamlit()
    
    def start_streamlit(self):
        """Start Streamlit process"""
        if self.process:
            self.process.terminate()
            self.process.wait()
        
        print("[+] Starting Streamlit...")
        self.process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.port", "8501"
        ], cwd=self.demo_dir)
        
        time.sleep(3)  # Wait for startup
        print("[+] Streamlit running at http://localhost:8501")
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Only restart on Python file changes
        if event.src_path.endswith(('.py',)):
            print(f"[~] File changed: {event.src_path}")
            print("[~] Restarting Streamlit...")
            self.start_streamlit()

if __name__ == "__main__":
    handler = StreamlitHandler()
    observer = Observer()
    observer.schedule(handler, str(Path(__file__).parent), recursive=False)
    observer.start()
    
    print("[*] Watching for changes... (Ctrl+C to stop)")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if handler.process:
            handler.process.terminate()
        print("\n[!] Auto-restart stopped")
    
    observer.join()