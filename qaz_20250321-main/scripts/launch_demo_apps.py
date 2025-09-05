#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Apps Launcher
Launches the 4 main Streamlit apps for VectorQA Sage demo
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def launch_streamlit_app(app_path, app_name, port):
    """Launch a Streamlit app on a specific port"""
    print(f"🚀 Launching {app_name} on port {port}...")
    
    # Change to src directory
    src_dir = Path(__file__).parent.parent / "src"
    
    # Launch the app
    cmd = f"cd {src_dir} && streamlit run {app_path} --server.port {port} --server.headless true"
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the app to start
        time.sleep(3)
        
        if process.poll() is None:
            print(f"✅ {app_name} is running on http://localhost:{port}")
            return process
        else:
            print(f"❌ Failed to start {app_name}")
            return None
            
    except Exception as e:
        print(f"❌ Error launching {app_name}: {e}")
        return None

def main():
    """Launch all demo apps"""
    print("🎬 VectorQA Sage Demo Apps Launcher")
    print("="*50)
    print("Launching 4 main Streamlit apps for demo...")
    print()
    
    # Define the apps to launch
    apps = [
        {
            "path": "main.py",
            "name": "Main VectorQA Sage App",
            "port": 8501,
            "description": "Complete QA pipeline with all features"
        },
        {
            "path": "enhanced_qa_demo.py", 
            "name": "Enhanced QA Demo",
            "port": 8502,
            "description": "Smart document processing with live context"
        },
        {
            "path": "rag_query_demo.py",
            "name": "RAG Query Demo", 
            "port": 8503,
            "description": "Intelligent document search and reasoning"
        },
        {
            "path": "mcp_dashboard.py",
            "name": "MCP Dashboard",
            "port": 8504,
            "description": "System architecture and performance monitoring"
        }
    ]
    
    processes = []
    
    # Launch each app
    for app in apps:
        process = launch_streamlit_app(app["path"], app["name"], app["port"])
        if process:
            processes.append(process)
        print(f"   📋 {app['description']}")
        print()
    
    # Show summary
    print("="*50)
    print("🎉 Demo Apps Launched Successfully!")
    print()
    print("📱 Available Apps:")
    print("1. Main App:        http://localhost:8501")
    print("2. Enhanced QA:     http://localhost:8502") 
    print("3. RAG Query:       http://localhost:8503")
    print("4. MCP Dashboard:   http://localhost:8504")
    print()
    print("💡 Demo Flow:")
    print("   Start with MCP Dashboard (port 8504) for system overview")
    print("   Then Enhanced QA (port 8502) for document processing")
    print("   Then RAG Query (port 8503) for intelligent search")
    print("   Finally Main App (port 8501) for complete pipeline")
    print()
    print("⏹️  To stop all apps: Press Ctrl+C")
    print()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all demo apps...")
        for process in processes:
            if process:
                process.terminate()
        print("✅ All apps stopped")

if __name__ == "__main__":
    main()
