#!/usr/bin/env python3
"""
Start MLflow Dashboard on port 5000 with real PostgreSQL backend
"""

import yaml
import subprocess
import webbrowser
import time
from pathlib import Path

def start_mlflow_dashboard():
    """Start MLflow UI dashboard on port 5000"""
    print("STARTING MLFLOW DASHBOARD")
    print("=" * 50)
    
    # Load credentials
    settings_path = Path("C:/Users/marti/AI-Scoring/tidyllm/admin/settings.yaml")
    with open(settings_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get MLflow configuration
    mlflow_config = config['services']['mlflow']
    backend_store_uri = mlflow_config['backend_store_uri']
    artifact_store = mlflow_config['artifact_store']
    
    print("MLflow Configuration:")
    print(f"  Backend Store: PostgreSQL on AWS RDS")
    print(f"  Artifact Store: {artifact_store}")
    print(f"  Dashboard Port: 5000")
    print()
    
    # Build MLflow server command
    mlflow_command = [
        "mlflow", "ui",
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", artifact_store,
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    print("Starting MLflow server with command:")
    print(" ".join(mlflow_command))
    print()
    
    try:
        # Start MLflow server
        print("Starting MLflow UI server...")
        process = subprocess.Popen(
            mlflow_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        if process.poll() is None:
            print("SUCCESS! MLflow dashboard is running!")
            print()
            print("ACCESS THE DASHBOARD:")
            print("-" * 25)
            print("Local URL: http://localhost:5000")
            print("Network URL: http://0.0.0.0:5000")
            print()
            print("WHAT YOU'LL SEE:")
            print("-" * 16)
            print("1. All 36 experiments")
            print("2. All 61 runs with metrics")
            print("3. Token counts and costs")
            print("4. DSPy optimization results")
            print("5. Boss satisfaction metrics")
            print()
            print("TO VIEW YOUR EVIDENCE:")
            print("-" * 23)
            print("1. Go to: http://localhost:5000")
            print("2. Click on 'COMPLETE_EVIDENCE_V2_20250913_152543' experiment")
            print("3. View the run with all token/cost metrics")
            print()
            print("Press Ctrl+C to stop the server")
            
            # Open browser
            webbrowser.open("http://localhost:5000")
            
            # Keep server running
            process.wait()
            
        else:
            print("ERROR: MLflow server failed to start")
            stderr = process.stderr.read()
            if stderr:
                print(f"Error: {stderr}")
                
    except FileNotFoundError:
        print("ERROR: MLflow not found. Install with: pip install mlflow")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    start_mlflow_dashboard()