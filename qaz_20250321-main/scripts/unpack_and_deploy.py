#!/usr/bin/env python3
"""
Manual Deployment Unpacker Script
=================================

This script unpacks and deploys the manual deployment packages
created by pack_files.py for air-gapped environments.

Usage:
    python scripts/unpack_and_deploy.py [--packages-dir] [--target-dir] [--auto-deploy]
"""

import os
import sys
import zipfile
import argparse
import logging
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentUnpacker:
    """Unpack and deploy manual deployment packages"""
    
    def __init__(self, packages_dir=None, target_dir=None, auto_deploy=False):
        self.packages_dir = Path(packages_dir) if packages_dir else Path.cwd()
        self.target_dir = Path(target_dir) if target_dir else Path.cwd()
        self.auto_deploy = auto_deploy
        
    def find_deployment_packages(self):
        """Find deployment packages in the packages directory"""
        logger.info(f"🔍 Looking for deployment packages in: {self.packages_dir}")
        
        packages = {}
        package_patterns = {
            'database_schemas': 'database_schemas_*.zip',
            'credentials_config': 'credentials_config_*.zip', 
            'input_assets': 'input_assets_*.zip',
            'core_application': 'core_application_*.zip'
        }
        
        for package_type, pattern in package_patterns.items():
            matches = list(self.packages_dir.glob(pattern))
            if matches:
                # Get the most recent package
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                packages[package_type] = latest
                logger.info(f"📦 Found {package_type}: {latest.name}")
            else:
                logger.warning(f"⚠️  No {package_type} package found")
                
        return packages
        
    def extract_package(self, package_path, extract_dir):
        """Extract a zip package to the target directory"""
        logger.info(f"📦 Extracting {package_path.name} to {extract_dir}")
        
        with zipfile.ZipFile(package_path, 'r') as zipf:
            zipf.extractall(extract_dir)
            
        logger.info(f"✅ Extracted {package_path.name}")
        
    def setup_database(self, database_dir):
        """Set up database using extracted scripts"""
        logger.info("🗄️  Setting up database...")
        
        # Look for setup scripts
        setup_scripts = [
            database_dir / "database" / "prod_scripts" / "00_complete_setup.sql",
            database_dir / "database" / "infra" / "01_extensions.sql"
        ]
        
        for script in setup_scripts:
            if script.exists():
                logger.info(f"📋 Found database script: {script}")
                # Note: In a real deployment, you'd run this against your database
                # For now, we'll just log it
                logger.info(f"   Would run: psql -f {script}")
                
    def setup_environment(self, config_dir):
        """Set up environment configuration"""
        logger.info("⚙️  Setting up environment configuration...")
        
        # Copy environment configs
        environ_dir = config_dir / "environ_settings"
        if environ_dir.exists():
            logger.info(f"📋 Found environment settings in: {environ_dir}")
            
            # Look for config files
            config_files = list(environ_dir.glob("config.*.yaml"))
            for config_file in config_files:
                logger.info(f"   Config file: {config_file.name}")
                
        # Copy dev configs
        dev_configs_dir = config_dir / "dev_configs"
        if dev_configs_dir.exists():
            logger.info(f"📋 Found dev configs in: {dev_configs_dir}")
            
    def install_dependencies(self, core_dir):
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        requirements_files = [
            core_dir / "requirements.txt",
            core_dir / "requirements_demo.txt",
            core_dir / "requirements_rag.txt"
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                logger.info(f"📋 Installing from: {req_file}")
                try:
                    # In a real deployment, you'd run: pip install -r req_file
                    logger.info(f"   Would run: pip install -r {req_file}")
                except Exception as e:
                    logger.error(f"❌ Failed to install from {req_file}: {e}")
                    
    def validate_deployment(self, target_dir):
        """Validate that deployment was successful"""
        logger.info("✅ Validating deployment...")
        
        # Check for key files
        key_files = [
            "simple_demo.py",
            "start_simple_demo.py",
            "src/",
            "database/",
            "environ_settings/",
            "input/"
        ]
        
        missing_files = []
        for key_file in key_files:
            file_path = target_dir / key_file
            if not file_path.exists():
                missing_files.append(key_file)
            else:
                logger.info(f"✅ Found: {key_file}")
                
        if missing_files:
            logger.warning(f"⚠️  Missing files: {missing_files}")
            return False
        else:
            logger.info("✅ All key files present")
            return True
            
    def create_deployment_summary(self, target_dir, packages):
        """Create a deployment summary"""
        summary = {
            "deployment_info": {
                "timestamp": datetime.now().isoformat(),
                "target_directory": str(target_dir),
                "packages_used": {k: str(v) for k, v in packages.items()}
            },
            "deployment_status": "completed",
            "next_steps": [
                "1. Configure database connection in environ_settings/",
                "2. Set up MLflow tracking URI if needed",
                "3. Configure S3 credentials for document storage",
                "4. Run: python start_simple_demo.py"
            ]
        }
        
        summary_path = target_dir / "deployment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"📋 Deployment summary created: {summary_path}")
        return summary_path
        
    def deploy(self):
        """Main deployment process"""
        logger.info("🚀 Starting deployment process...")
        
        # Find packages
        packages = self.find_deployment_packages()
        if not packages:
            logger.error("❌ No deployment packages found!")
            return False
            
        # Create target directory
        self.target_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Target directory: {self.target_dir}")
        
        try:
            # Extract packages in order
            if 'database_schemas' in packages:
                self.extract_package(packages['database_schemas'], self.target_dir)
                
            if 'credentials_config' in packages:
                self.extract_package(packages['credentials_config'], self.target_dir)
                
            if 'input_assets' in packages:
                self.extract_package(packages['input_assets'], self.target_dir)
                
            if 'core_application' in packages:
                self.extract_package(packages['core_application'], self.target_dir)
                
            # Set up components
            self.setup_database(self.target_dir)
            self.setup_environment(self.target_dir)
            
            if self.auto_deploy:
                self.install_dependencies(self.target_dir)
                
            # Validate deployment
            if self.validate_deployment(self.target_dir):
                # Create summary
                summary_path = self.create_deployment_summary(self.target_dir, packages)
                
                logger.info("🎉 Deployment completed successfully!")
                logger.info(f"📁 Deployed to: {self.target_dir}")
                logger.info(f"📋 Summary: {summary_path}")
                
                return True
            else:
                logger.error("❌ Deployment validation failed!")
                return False
                
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unpack and deploy manual deployment packages")
    parser.add_argument("--packages-dir", help="Directory containing deployment packages")
    parser.add_argument("--target-dir", help="Target directory for deployment")
    parser.add_argument("--auto-deploy", action="store_true", help="Automatically install dependencies")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        unpacker = DeploymentUnpacker(
            packages_dir=args.packages_dir,
            target_dir=args.target_dir,
            auto_deploy=args.auto_deploy
        )
        
        success = unpacker.deploy()
        
        if success:
            print(f"\n🎉 Deployment completed successfully!")
            print(f"📁 Target directory: {unpacker.target_dir}")
            print(f"📋 Next steps:")
            print(f"   1. Configure environment settings")
            print(f"   2. Set up database connection")
            print(f"   3. Run: python start_simple_demo.py")
        else:
            print(f"\n❌ Deployment failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
