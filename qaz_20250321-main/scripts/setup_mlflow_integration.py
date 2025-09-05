#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLflow Integration Setup Script

This script sets up the MLflow integration with PostgreSQL database
and configures the Unified LLM Gateway for production use.
"""

import os
import sys
import logging
import subprocess
import psycopg2
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backend.llm.unified_llm_gateway import UnifiedLLMGateway

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLflowIntegrationSetup:
    """Setup MLflow integration with PostgreSQL and Unified LLM Gateway"""
    
    def __init__(self, config_path: str = "config/mlflow_integration.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'database': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'name': os.getenv('DB_NAME', 'mlflow'),
                'user': os.getenv('DB_USER', 'mlflow'),
                'password': os.getenv('DB_PASSWORD', 'password')
            },
            'mlflow': {
                'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
                'artifact_root': os.getenv('MLFLOW_ARTIFACT_ROOT', 's3://mlflow-artifacts'),
                'experiment_name': 'unified-llm-gateway'
            },
            'gateway': {
                'local_url': os.getenv('ZLLM_GATEWAY_URL', 'http://localhost:11434'),
                'remote_url': os.getenv('REMOTE_GATEWAY_URL'),
                'default_gateway': 'local'
            }
        }
    
    def setup_database(self) -> bool:
        """Set up PostgreSQL database and schema"""
        logger.info("Setting up PostgreSQL database...")
        
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host=self.config['database']['host'],
                port=self.config['database']['port'],
                database=self.config['database']['name'],
                user=self.config['database']['user'],
                password=self.config['database']['password']
            )
            
            # Check if MLflow tables exist
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'experiments'
            """)
            
            if cursor.fetchone()[0] == 0:
                logger.info("Creating MLflow database schema...")
                
                # Read and execute schema file
                schema_file = Path(__file__).parent.parent / "database" / "mlflow_integration_schema.sql"
                
                if not schema_file.exists():
                    logger.error(f"Schema file not found: {schema_file}")
                    return False
                
                with open(schema_file, 'r') as f:
                    schema_sql = f.read()
                
                # Execute schema
                cursor.execute(schema_sql)
                conn.commit()
                logger.info("✅ Database schema created successfully")
            else:
                logger.info("✅ Database schema already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            return False
    
    def setup_mlflow_server(self) -> bool:
        """Set up MLflow server with PostgreSQL backend"""
        logger.info("Setting up MLflow server...")
        
        try:
            # Check if MLflow is installed
            result = subprocess.run(['mlflow', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("MLflow not installed. Please install with: pip install mlflow")
                return False
            
            # Create MLflow server startup script
            startup_script = self._create_mlflow_startup_script()
            
            logger.info("✅ MLflow server setup complete")
            logger.info(f"Start MLflow server with: {startup_script}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow server: {e}")
            return False
    
    def _create_mlflow_startup_script(self) -> str:
        """Create MLflow server startup script"""
        script_content = f"""#!/bin/bash
# MLflow Server Startup Script

export MLFLOW_TRACKING_URI="postgresql://{self.config['database']['user']}:{self.config['database']['password']}@{self.config['database']['host']}:{self.config['database']['port']}/{self.config['database']['name']}"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="{self.config['mlflow']['artifact_root']}"

echo "Starting MLflow server..."
echo "Database: $MLFLOW_TRACKING_URI"
echo "Artifacts: $MLFLOW_DEFAULT_ARTIFACT_ROOT"

mlflow server \\
  --host 0.0.0.0 \\
  --port 5000 \\
  --backend-store-uri "$MLFLOW_TRACKING_URI" \\
  --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT"
"""
        
        script_path = Path(__file__).parent.parent / "scripts" / "start_mlflow_server.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return str(script_path)
    
    def test_unified_gateway(self) -> bool:
        """Test the Unified LLM Gateway with MLflow integration"""
        logger.info("Testing Unified LLM Gateway...")
        
        try:
            # Initialize gateway
            gateway = UnifiedLLMGateway(
                experiment_name=self.config['mlflow']['experiment_name'],
                tracking_uri=self.config['mlflow']['tracking_uri'],
                enable_tracking=True,
                local_gateway_url=self.config['gateway']['local_url'],
                remote_gateway_url=self.config['gateway']['remote_url'],
                default_gateway=self.config['gateway']['default_gateway']
            )
            
            # Test LLM call
            test_prompt = "Hello, this is a test of the Unified LLM Gateway with MLflow integration."
            
            response = gateway.call_llm(
                agent_name="setup_test",
                task_type="test",
                prompt=test_prompt,
                model_preference="llama2"
            )
            
            logger.info(f"✅ Gateway test successful: {response.content[:100]}...")
            logger.info(f"   Tokens: {response.input_tokens + response.output_tokens}")
            logger.info(f"   Cost: ${response.cost:.6f}")
            logger.info(f"   Response time: {response.response_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Gateway test failed: {e}")
            return False
    
    def create_docker_compose(self) -> bool:
        """Create Docker Compose file for easy deployment"""
        logger.info("Creating Docker Compose configuration...")
        
        docker_compose = f"""version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: {self.config['database']['name']}
      POSTGRES_USER: {self.config['database']['user']}
      POSTGRES_PASSWORD: {self.config['database']['password']}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "{self.config['database']['port']}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U {self.config['database']['user']}"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    environment:
      MLFLOW_TRACKING_URI: postgresql://{self.config['database']['user']}:{self.config['database']['password']}@postgres:5432/{self.config['database']['name']}
      MLFLOW_DEFAULT_ARTIFACT_ROOT: {self.config['mlflow']['artifact_root']}
    ports:
      - "5000:5000"
    depends_on:
      postgres:
        condition: service_healthy
    command: mlflow server --host 0.0.0.0 --port 5000

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - mlflow

volumes:
  postgres_data:
"""
        
        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(docker_compose)
        
        logger.info(f"✅ Docker Compose file created: {compose_path}")
        return True
    
    def create_nginx_config(self) -> bool:
        """Create Nginx configuration for load balancing"""
        logger.info("Creating Nginx configuration...")
        
        nginx_config = """upstream mlflow_backend {
    server mlflow:5000;
}

server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://mlflow_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
"""
        
        nginx_path = Path(__file__).parent.parent / "nginx.conf"
        with open(nginx_path, 'w') as f:
            f.write(nginx_config)
        
        logger.info(f"✅ Nginx configuration created: {nginx_path}")
        return True
    
    def setup_monitoring(self) -> bool:
        """Set up monitoring and health checks"""
        logger.info("Setting up monitoring...")
        
        # Create health check script
        health_check = """#!/usr/bin/env python3
import requests
import psycopg2
import os

def check_mlflow_health():
    try:
        response = requests.get("http://localhost:5000/health")
        return response.status_code == 200
    except:
        return False

def check_database_health():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'mlflow'),
            user=os.getenv('DB_USER', 'mlflow'),
            password=os.getenv('DB_PASSWORD', 'password')
        )
        conn.close()
        return True
    except:
        return False

if __name__ == "__main__":
    mlflow_ok = check_mlflow_health()
    db_ok = check_database_health()
    
    print(f"MLflow: {'✅' if mlflow_ok else '❌'}")
    print(f"Database: {'✅' if db_ok else '❌'}")
    
    exit(0 if mlflow_ok and db_ok else 1)
"""
        
        health_check_path = Path(__file__).parent.parent / "scripts" / "health_check.py"
        with open(health_check_path, 'w') as f:
            f.write(health_check)
        
        os.chmod(health_check_path, 0o755)
        
        logger.info(f"✅ Health check script created: {health_check_path}")
        return True
    
    def run_setup(self) -> bool:
        """Run complete setup process"""
        logger.info("🚀 Starting MLflow Integration Setup")
        
        steps = [
            ("Database Setup", self.setup_database),
            ("MLflow Server Setup", self.setup_mlflow_server),
            ("Docker Compose", self.create_docker_compose),
            ("Nginx Configuration", self.create_nginx_config),
            ("Monitoring Setup", self.setup_monitoring),
            ("Gateway Test", self.test_unified_gateway)
        ]
        
        success_count = 0
        total_steps = len(steps)
        
        for step_name, step_func in steps:
            logger.info(f"\n📋 {step_name}...")
            try:
                if step_func():
                    logger.info(f"✅ {step_name} completed successfully")
                    success_count += 1
                else:
                    logger.error(f"❌ {step_name} failed")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with error: {e}")
        
        logger.info(f"\n🎯 Setup Summary: {success_count}/{total_steps} steps completed")
        
        if success_count == total_steps:
            logger.info("🎉 MLflow Integration Setup completed successfully!")
            self._print_next_steps()
            return True
        else:
            logger.warning("⚠️ Some setup steps failed. Please check the logs above.")
            return False
    
    def _print_next_steps(self):
        """Print next steps for the user"""
        logger.info("\n📋 Next Steps:")
        logger.info("1. Start MLflow server: ./scripts/start_mlflow_server.sh")
        logger.info("2. Or use Docker: docker-compose up -d")
        logger.info("3. Access MLflow UI: http://localhost:5000")
        logger.info("4. Test health: python scripts/health_check.py")
        logger.info("5. Run your application with Unified LLM Gateway")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup MLflow Integration")
    parser.add_argument("--config", default="config/mlflow_integration.yaml", 
                       help="Configuration file path")
    parser.add_argument("--skip-db", action="store_true", 
                       help="Skip database setup")
    parser.add_argument("--skip-test", action="store_true", 
                       help="Skip gateway test")
    
    args = parser.parse_args()
    
    setup = MLflowIntegrationSetup(args.config)
    
    if setup.run_setup():
        logger.info("✅ Setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
