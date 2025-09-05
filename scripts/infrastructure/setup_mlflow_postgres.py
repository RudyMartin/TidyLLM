#!/usr/bin/env python3
"""
MLflow PostgreSQL Backend Setup Script

This script helps set up MLflow with PostgreSQL backend for TidyLLM Gateway.
It handles database creation, schema initialization, and configuration validation.
"""

import os
import sys
import subprocess
from typing import Optional

def create_postgres_database(
    host: str = "localhost",
    port: int = 5432,
    db_name: str = "mlflowdb", 
    username: str = "mlflowuser",
    password: str = "mlflowpass",
    admin_user: str = "postgres"
) -> bool:
    """
    Create PostgreSQL database and user for MLflow.
    
    Returns:
        True if successful, False otherwise
    """
    
    print(f"Creating PostgreSQL database '{db_name}' for MLflow...")
    
    # SQL commands to create database and user
    sql_commands = f"""
-- Create database
CREATE DATABASE {db_name};

-- Create user
CREATE USER {username} WITH PASSWORD '{password}';

-- Configure user settings
ALTER ROLE {username} SET client_encoding TO 'utf8';
ALTER ROLE {username} SET default_transaction_isolation TO 'read committed';
ALTER ROLE {username} SET timezone TO 'UTC';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {username};
GRANT USAGE ON SCHEMA public TO {username};
GRANT CREATE ON SCHEMA public TO {username};
ALTER ROLE {username} CREATEDB;

-- Exit
\\q
"""
    
    try:
        # Write SQL to temp file
        with open("setup_mlflow_db.sql", "w") as f:
            f.write(sql_commands)
        
        # Execute as postgres admin user
        cmd = f"psql -h {host} -p {port} -U {admin_user} -f setup_mlflow_db.sql"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Database and user created successfully")
            return True
        else:
            print(f"❌ Database creation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False
        
    finally:
        # Clean up temp file
        if os.path.exists("setup_mlflow_db.sql"):
            os.remove("setup_mlflow_db.sql")


def initialize_mlflow_schema(
    postgres_uri: str,
    artifact_root: Optional[str] = None
) -> bool:
    """
    Initialize MLflow database schema.
    
    Args:
        postgres_uri: PostgreSQL connection URI
        artifact_root: Artifact storage root (optional)
    
    Returns:
        True if successful, False otherwise
    """
    
    print("Initializing MLflow database schema...")
    
    try:
        # Set environment variables for MLflow
        os.environ["MLFLOW_BACKEND_STORE_URI"] = postgres_uri
        
        if artifact_root:
            os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = artifact_root
        
        # Run MLflow db upgrade
        cmd = f"mlflow db upgrade {postgres_uri}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MLflow schema initialized successfully")
            return True
        else:
            print(f"❌ Schema initialization failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing schema: {e}")
        return False


def start_mlflow_server(
    postgres_uri: str,
    host: str = "0.0.0.0",
    port: int = 5000,
    artifact_root: Optional[str] = None,
    background: bool = False
) -> bool:
    """
    Start MLflow server with PostgreSQL backend.
    
    Args:
        postgres_uri: PostgreSQL connection URI
        host: Server host
        port: Server port  
        artifact_root: Artifact storage root
        background: Run in background
        
    Returns:
        True if server starts, False otherwise
    """
    
    print(f"Starting MLflow server on {host}:{port}...")
    
    try:
        # Build server command
        cmd_parts = [
            "mlflow", "server",
            f"--backend-store-uri", postgres_uri,
            f"--host", host,
            f"--port", str(port)
        ]
        
        if artifact_root:
            cmd_parts.extend(["--default-artifact-root", artifact_root])
        
        cmd = " ".join(cmd_parts)
        
        if background:
            cmd += " &"
            
        print(f"Command: {cmd}")
        
        if background:
            subprocess.Popen(cmd, shell=True)
            print("✅ MLflow server started in background")
            return True
        else:
            print("Starting MLflow server (press Ctrl+C to stop)...")
            subprocess.run(cmd, shell=True)
            return True
            
    except Exception as e:
        print(f"❌ Error starting MLflow server: {e}")
        return False


def validate_setup(postgres_uri: str, server_url: str = "http://localhost:5000") -> bool:
    """
    Validate MLflow PostgreSQL setup.
    
    Args:
        postgres_uri: PostgreSQL connection URI
        server_url: MLflow server URL
        
    Returns:
        True if setup is valid, False otherwise
    """
    
    print("Validating MLflow PostgreSQL setup...")
    
    try:
        import mlflow
        import requests
        
        # Test database connection
        mlflow.set_tracking_uri(postgres_uri)
        
        # Try to create a test experiment
        experiment_name = "test_setup_validation"
        
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Database connection successful - created experiment {experiment_id}")
            
            # Clean up test experiment
            mlflow.delete_experiment(experiment_id)
            
        except Exception as e:
            if "already exists" in str(e):
                print("✅ Database connection successful - experiment exists")
            else:
                raise e
        
        # Test server endpoint
        try:
            response = requests.get(f"{server_url}/api/2.0/mlflow/experiments/list")
            if response.status_code == 200:
                print("✅ MLflow server accessible")
                return True
            else:
                print(f"⚠️ Server returned status {response.status_code}")
                return False
                
        except requests.ConnectionError:
            print("⚠️ MLflow server not accessible - may not be running")
            return False
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    print("=== MLflow PostgreSQL Setup for TidyLLM Gateway ===")
    
    # Default configuration
    config = {
        "postgres_host": "localhost",
        "postgres_port": 5432,
        "postgres_db": "mlflowdb", 
        "postgres_user": "mlflowuser",
        "postgres_password": "mlflowpass",
        "mlflow_host": "0.0.0.0",
        "mlflow_port": 5000,
        "artifact_root": None  # Will default to local filesystem
    }
    
    # Build PostgreSQL URI
    postgres_uri = f"postgresql://{config['postgres_user']}:{config['postgres_password']}@{config['postgres_host']}:{config['postgres_port']}/{config['postgres_db']}"
    
    print(f"PostgreSQL URI: {postgres_uri.replace(config['postgres_password'], '***')}")
    
    # Steps
    print("\nStep 1: Create database and user")
    if create_postgres_database(
        host=config["postgres_host"],
        port=config["postgres_port"],
        db_name=config["postgres_db"],
        username=config["postgres_user"],
        password=config["postgres_password"]
    ):
        print("\nStep 2: Initialize MLflow schema")
        if initialize_mlflow_schema(postgres_uri, config["artifact_root"]):
            
            print("\nStep 3: Validate setup")
            if validate_setup(postgres_uri):
                
                print("\nStep 4: Start MLflow server")
                print("To start the server manually, run:")
                print(f"mlflow server --backend-store-uri {postgres_uri} --host {config['mlflow_host']} --port {config['mlflow_port']}")
                
                # Optionally start server
                start_choice = input("\nStart MLflow server now? (y/n): ").lower()
                if start_choice == 'y':
                    start_mlflow_server(
                        postgres_uri=postgres_uri,
                        host=config["mlflow_host"],
                        port=config["mlflow_port"],
                        artifact_root=config["artifact_root"]
                    )
                
                print("\n✅ MLflow PostgreSQL setup complete!")
                print("\nTo use with TidyLLM Gateway:")
                print(f"gateway = create_gateway(postgres_uri='{postgres_uri}')")
            
            else:
                print("❌ Setup validation failed")
        else:
            print("❌ Schema initialization failed")
    else:
        print("❌ Database creation failed")