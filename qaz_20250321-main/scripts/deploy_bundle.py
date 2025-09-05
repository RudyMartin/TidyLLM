#!/usr/bin/env python3
"""
Migration Bundle Deployer

Deploys a VectorQA Sage migration bundle to any environment.
Handles environment setup, dependency installation, and application startup.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any

class BundleDeployer:
    """Deploys migration bundles"""
    
    def __init__(self, bundle_path: str):
        self.bundle_path = Path(bundle_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        config_file = self.bundle_path / "deployment_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file) as f:
            return json.load(f)
    
    def deploy(self, validate_only: bool = False) -> bool:
        """Deploy the bundle"""
        try:
            print(f"🚀 Deploying VectorQA Sage bundle")
            print(f"📦 Bundle: {self.bundle_path}")
            print(f"🎯 Environment: {self.config['target_environment']}")
            print("=" * 60)
            
            # Step 1: Validate configuration
            if not self._validate_config():
                return False
            
            if validate_only:
                print("✅ Configuration validation passed")
                return True
            
            # Step 2: Setup environment
            if not self._setup_environment():
                return False
            
            # Step 3: Install dependencies
            if not self._install_dependencies():
                return False
            
            # Step 4: Start application
            if not self._start_application():
                return False
            
            # Step 5: Verify deployment
            if not self._verify_deployment():
                return False
            
            print("✅ Deployment completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate deployment configuration"""
        print("🔍 Validating configuration...")
        
        try:
            # Run validation script
            validation_script = self.bundle_path / "scripts" / "validate_config.py"
            if validation_script.exists():
                result = subprocess.run([sys.executable, str(validation_script)], 
                                      capture_output=True, text=True, cwd=self.bundle_path)
                if result.returncode != 0:
                    print(f"❌ Configuration validation failed: {result.stderr}")
                    return False
            
            print("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def _setup_environment(self) -> bool:
        """Setup the deployment environment"""
        print("🔧 Setting up environment...")
        
        try:
            # Set environment variables
            for key, value in self.config['environment_variables'].items():
                os.environ[key] = str(value)
                print(f"✅ Set {key}={value}")
            
            # Run environment setup script
            setup_script = self.bundle_path / "scripts" / "setup_environment.py"
            if setup_script.exists():
                result = subprocess.run([sys.executable, str(setup_script)], 
                                      capture_output=True, text=True, cwd=self.bundle_path)
                if result.returncode != 0:
                    print(f"❌ Environment setup failed: {result.stderr}")
                    return False
            
            print("✅ Environment setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
    
    def _install_dependencies(self) -> bool:
        """Install Python dependencies"""
        print("📦 Installing dependencies...")
        
        try:
            # Install requirements
            requirements_file = self.bundle_path / "requirements_demo.txt"
            if requirements_file.exists():
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], capture_output=True, text=True, cwd=self.bundle_path)
                
                if result.returncode != 0:
                    print(f"❌ Dependency installation failed: {result.stderr}")
                    return False
                
                print("✅ Dependencies installed")
            else:
                print("⚠️ No requirements file found, skipping dependency installation")
            
            return True
            
        except Exception as e:
            print(f"❌ Dependency installation failed: {e}")
            return False
    
    def _start_application(self) -> bool:
        """Start the Streamlit application"""
        print("🚀 Starting application...")
        
        try:
            # Get Streamlit configuration
            streamlit_config = self.config['streamlit_config']
            
            # Build Streamlit command
            app_script = self.bundle_path / "src" / "main.py"
            if not app_script.exists():
                print(f"❌ Application script not found: {app_script}")
                return False
            
            cmd = [
                sys.executable, "-m", "streamlit", "run", str(app_script),
                "--server.port", str(streamlit_config['server.port']),
                "--server.address", streamlit_config['server.address'],
                "--browser.gatherUsageStats", str(streamlit_config['browser.gatherUsageStats']),
                "--server.runOnSave", str(streamlit_config['server.runOnSave'])
            ]
            
            if streamlit_config.get('server.headless', False):
                cmd.extend(["--server.headless", "true"])
            
            print(f"⚡ Starting with command: {' '.join(cmd)}")
            
            # Start application in background
            process = subprocess.Popen(cmd, cwd=self.bundle_path)
            
            # Wait a moment for startup
            import time
            time.sleep(5)
            
            # Check if process is still running
            if process.poll() is None:
                print("✅ Application started successfully")
                return True
            else:
                print("❌ Application failed to start")
                return False
            
        except Exception as e:
            print(f"❌ Application startup failed: {e}")
            return False
    
    def _verify_deployment(self) -> bool:
        """Verify the deployment is working"""
        print("🔍 Verifying deployment...")
        
        try:
            # Run health check
            health_script = self.bundle_path / "scripts" / "health_check.py"
            if health_script.exists():
                result = subprocess.run([sys.executable, str(health_script)], 
                                      capture_output=True, text=True, cwd=self.bundle_path)
                if result.returncode != 0:
                    print(f"❌ Health check failed: {result.stderr}")
                    return False
            
            print("✅ Deployment verification passed")
            return True
            
        except Exception as e:
            print(f"❌ Deployment verification failed: {e}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy VectorQA Sage migration bundle")
    parser.add_argument("bundle_path", help="Path to the migration bundle")
    parser.add_argument("--validate-only", "-v", action="store_true",
                       help="Only validate configuration, don't deploy")
    
    args = parser.parse_args()
    
    try:
        deployer = BundleDeployer(args.bundle_path)
        success = deployer.deploy(validate_only=args.validate_only)
        
        if success:
            print("\n🎉 Deployment successful!")
            if not args.validate_only:
                print("🌐 Application should be available at: http://localhost:8501")
        else:
            print("\n❌ Deployment failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

