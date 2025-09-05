#!/usr/bin/env python3
"""
Migration Bundle Creator

Creates a self-contained deployment bundle for VectorQA Sage that includes:
- Application code
- Environment configuration
- Dependencies
- Deployment scripts
- Health checks
- Configuration validation

This eliminates the traditional "thousands of pages" migration nightmare.
"""

import os
import sys
import shutil
import json
import yaml
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class MigrationBundleCreator:
    """Creates production-ready migration bundles"""
    
    def __init__(self, bundle_name: str = None):
        self.project_root = Path(__file__).parent.parent.absolute()
        self.bundle_name = bundle_name or f"vectorqa_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.bundle_path = self.project_root / "migration_bundles" / self.bundle_name
        
        # Define what to include in the bundle
        self.include_patterns = [
            "src/**/*.py",
            "src/**/*.txt",
            "src/**/*.yaml",
            "src/**/*.yml",
            "src/**/*.json",
            "requirements*.txt",
            "README.md",
            "scripts/create_migration_bundle.py",
            "scripts/deploy_aws_only.sh",
            "scripts/quick_*.sh",
            "config/*.yaml",
            "config/*.yml",
            "database/infra/*.sql",
            "docs/architecture/*.md",
            "docs/getting-started/*.md"
        ]
        
        self.exclude_patterns = [
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.DS_Store",
            "**/node_modules/**",
            "**/.git/**",
            "**/logs/**",
            "**/mlruns/**",
            "**/llm_cache/**",
            "**/test_outputs/**",
            "**/output/**",
            "**/llm_metrics/**",
            "**/llm_enhanced_output/**",
            "**/mlflow_export/**",
            "**/mlflow_real_doc_export/**",
            "**/mlflow_real_document_output/**",
            "**/llm_utilization_reports/**"
        ]
    
    def create_bundle(self, target_env: str = "aws") -> bool:
        """Create the complete migration bundle"""
        try:
            print(f"🚀 Creating migration bundle: {self.bundle_name}")
            print(f"🎯 Target environment: {target_env}")
            print("=" * 60)
            
            # Create bundle directory
            self.bundle_path.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Copy application files
            self._copy_application_files()
            
            # Step 2: Create environment configuration
            self._create_environment_config(target_env)
            
            # Step 3: Create deployment scripts
            self._create_deployment_scripts(target_env)
            
            # Step 4: Create health checks
            self._create_health_checks()
            
            # Step 5: Create environment-specific settings
            self._create_environment_settings(target_env)
            
            # Step 6: Create environment launchers
            self._create_environment_launchers()
            
            # Step 7: Create service account migration script
            self._create_service_account_migration_script()
            
            # Step 8: Create configuration validation
            self._create_config_validation()
            
            # Step 9: Create bundle manifest
            self._create_bundle_manifest(target_env)
            
            # Step 10: Create deployment guide
            self._create_deployment_guide(target_env)
            
            # Step 11: Package everything
            self._package_bundle()
            
            print(f"✅ Migration bundle created successfully!")
            print(f"📦 Bundle location: {self.bundle_path}")
            print(f"📋 Deployment guide: {self.bundle_path}/DEPLOYMENT_GUIDE.md")
            
            return True
            
        except Exception as e:
            print(f"❌ Bundle creation failed: {e}")
            return False
    
    def _copy_application_files(self):
        """Copy application files to bundle"""
        print("📁 Copying application files...")
        
        # Copy src directory
        src_src = self.project_root / "src"
        src_dest = self.bundle_path / "src"
        if src_src.exists():
            shutil.copytree(src_src, src_dest, dirs_exist_ok=True)
            print(f"✅ Copied src directory")
        
        # Copy requirements files
        for req_file in ["requirements.txt", "requirements_demo.txt"]:
            req_path = self.project_root / req_file
            if req_path.exists():
                shutil.copy2(req_path, self.bundle_path)
                print(f"✅ Copied {req_file}")
        
        # Copy README
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            shutil.copy2(readme_path, self.bundle_path)
            print(f"✅ Copied README.md")
        
        # Copy config directory from src/config
        config_src = self.project_root / "src" / "config"
        config_dest = self.bundle_path / "config"
        if config_src.exists():
            shutil.copytree(config_src, config_dest, dirs_exist_ok=True)
            print(f"✅ Copied config directory from src/config")
        else:
            print(f"⚠️ Config directory not found at {config_src}")
        
        # Copy database schema
        db_src = self.project_root / "database"
        db_dest = self.bundle_path / "database"
        if db_src.exists():
            shutil.copytree(db_src, db_dest, dirs_exist_ok=True)
            print(f"✅ Copied database directory")
        
        # Fix import paths for deployment
        self._fix_import_paths()
    
    def _fix_import_paths(self):
        """Fix import paths for deployment environment"""
        print("🔧 Fixing import paths for deployment...")
        
        # Files that need import fixes
        import_fixes = [
            {
                "file": "src/backend/core/config.py",
                "patterns": [
                    (r"try:\s*try:\s*from \.\.config\.credential_manager import credential_manager\s*except ImportError:\s*from config\.credential_manager import credential_manager\s*except ImportError:\s*# Fallback for deployment environment\s*from config\.credential_manager import credential_manager",
                     "try:\n    from ..config.credential_manager import credential_manager\nexcept ImportError:\n    # Fallback for deployment environment\n    from config.credential_manager import credential_manager"),
                    (r"from \.\.config\.credential_manager import credential_manager",
                     "try:\n    from ..config.credential_manager import credential_manager\nexcept ImportError:\n    # Fallback for deployment environment\n    from config.credential_manager import credential_manager")
                ]
            },
            {
                "file": "src/backend/core/report_export.py", 
                "patterns": [
                    (r"from core\.config import CONFIG",
                     "try:\n    from core.config import CONFIG\nexcept ImportError:\n    from backend.core.config import CONFIG")
                ]
            },
            {
                "file": "src/t_dashboard.py",
                "patterns": [
                    (r"from backend\.core\.normalize_labels import normalize_label",
                     "try:\n    from backend.core.normalize_labels import normalize_label\nexcept ImportError:\n    from core.normalize_labels import normalize_label"),
                    (r"from backend\.core\.report_export import export_dashboard_to_pdf",
                     "try:\n    from backend.core.report_export import export_dashboard_to_pdf\nexcept ImportError:\n    from core.report_export import export_dashboard_to_pdf")
                ]
            }
        ]
        
        for fix in import_fixes:
            file_path = self.bundle_path / fix["file"]
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    original_content = content
                    for pattern, replacement in fix["patterns"]:
                        import re
                        content = re.sub(pattern, replacement, content)
                    
                    if content != original_content:
                        with open(file_path, 'w') as f:
                            f.write(content)
                        print(f"✅ Fixed imports in {fix['file']}")
                    else:
                        print(f"ℹ️  No changes needed in {fix['file']}")
                        
                except Exception as e:
                    print(f"⚠️  Error fixing imports in {fix['file']}: {e}")
            else:
                print(f"⚠️  File not found: {fix['file']}")
    
    def _create_environment_config(self, target_env: str):
        """Create environment-specific configuration"""
        print("⚙️ Creating environment configuration...")
        
        # Environment-specific configurations
        env_configs = {
            "local": {
                "LOG_LEVEL": "INFO",
                "CACHE_DIR": "~/vectorqa_cache",
                "server.address": "localhost",
                "server.runOnSave": True,
                "server.headless": False,
                "use_local_llm": True,
                "debug_mode": True
            },
            "development": {
                "LOG_LEVEL": "INFO", 
                "CACHE_DIR": "/tmp/vectorqa_cache",
                "server.address": "0.0.0.0",
                "server.runOnSave": True,
                "server.headless": True,
                "use_local_llm": False,
                "debug_mode": True
            },
            "staging": {
                "LOG_LEVEL": "WARNING",
                "CACHE_DIR": "/tmp/vectorqa_cache", 
                "server.address": "0.0.0.0",
                "server.runOnSave": False,
                "server.headless": True,
                "use_local_llm": False,
                "debug_mode": False
            },
            "production": {
                "LOG_LEVEL": "ERROR",
                "CACHE_DIR": "/tmp/vectorqa_cache",
                "server.address": "0.0.0.0", 
                "server.runOnSave": False,
                "server.headless": True,
                "use_local_llm": False,
                "debug_mode": False
            }
        }
        
        # Get config for target environment (default to staging if not found)
        env_config = env_configs.get(target_env, env_configs["staging"])
        
        env_config = {
            "target_environment": target_env,
            "deployment_timestamp": datetime.now().isoformat(),
            "bundle_version": "1.0.0",
            "environment_variables": {
                "VECTORQA_ENV": target_env,
                "LOG_LEVEL": env_config["LOG_LEVEL"],
                "CACHE_DIR": env_config["CACHE_DIR"]
            },
            "streamlit_config": {
                "server.port": 8501,
                "server.address": env_config["server.address"],
                "browser.gatherUsageStats": False,
                "server.runOnSave": env_config["server.runOnSave"],
                "server.headless": env_config["server.headless"]
            },
            "backend_config": {
                "use_local_llm": env_config["use_local_llm"],
                "cache_enabled": True,
                "debug_mode": env_config["debug_mode"]
            }
        }
        
        # Save environment config
        env_file = self.bundle_path / "deployment_config.json"
        with open(env_file, 'w') as f:
            json.dump(env_config, f, indent=2)
        
        print(f"✅ Created deployment configuration")
    
    def _create_deployment_scripts(self, target_env: str):
        """Create deployment scripts"""
        print("📜 Creating deployment scripts...")
        
        scripts_dir = self.bundle_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Create main deployment script
        deploy_script = scripts_dir / "deploy.py"
        with open(deploy_script, 'w') as f:
            f.write(self._get_deploy_script_content(target_env))
        
        # Create environment setup script
        setup_script = scripts_dir / "setup_environment.py"
        with open(setup_script, 'w') as f:
            f.write(self._get_setup_script_content())
        
        # Create health check script
        health_script = scripts_dir / "health_check.py"
        with open(health_script, 'w') as f:
            f.write(self._get_health_check_content())
        
        # Make scripts executable
        for script in [deploy_script, setup_script, health_script]:
            script.chmod(0o755)
        
        print(f"✅ Created deployment scripts")
    
    def _get_deploy_script_content(self, target_env: str) -> str:
        """Get deployment script content"""
        return f'''#!/usr/bin/env python3
"""
VectorQA Sage Deployment Script

Automated deployment script for {target_env} environment.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def main():
    """Main deployment function"""
    print("🚀 Starting VectorQA Sage deployment...")
    
    # Load deployment configuration
    config_file = Path(__file__).parent.parent / "deployment_config.json"
    with open(config_file) as f:
        config = json.load(f)
    
    print(f"🎯 Target environment: {{config['target_environment']}}")
    
    # Setup environment
    print("🔧 Setting up environment...")
    setup_script = Path(__file__).parent / "setup_environment.py"
    subprocess.run([sys.executable, str(setup_script)], check=True)
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_demo.txt"], check=True)
    
    # Start application
    print("🚀 Starting application...")
    app_script = Path(__file__).parent.parent / "src" / "main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_script)], check=True)

if __name__ == "__main__":
    main()
'''
    
    def _get_setup_script_content(self) -> str:
        """Get environment setup script content"""
        return '''#!/usr/bin/env python3
"""
Environment Setup Script

Sets up the environment for VectorQA Sage deployment.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup the deployment environment"""
    # Add src to Python path
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Add backend to Python path
    backend_path = src_path / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    
    # Set environment variables
    os.environ['VECTORQA_ENV'] = 'aws'
    os.environ['LOG_LEVEL'] = 'INFO'
    
    print("✅ Environment setup complete")

if __name__ == "__main__":
    setup_environment()
'''
    
    def _get_health_check_content(self) -> str:
        """Get health check script content"""
        return '''#!/usr/bin/env python3
"""
Health Check Script

Checks the health of the VectorQA Sage deployment.
"""

import requests
import sys
import time

def check_health():
    """Check application health"""
    try:
        # Check if Streamlit is running
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        if response.status_code == 200:
            print("✅ Application is healthy")
            return True
        else:
            print(f"❌ Application health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    success = check_health()
    sys.exit(0 if success else 1)
'''
    
    def _create_health_checks(self):
        """Create health check configuration"""
        print("🏥 Creating health checks...")
        
        health_config = {
            "health_checks": [
                {
                    "name": "streamlit_app",
                    "url": "http://localhost:8501/_stcore/health",
                    "timeout": 5,
                    "interval": 30
                },
                {
                    "name": "backend_services",
                    "script": "scripts/health_check.py",
                    "timeout": 10,
                    "interval": 60
                }
            ],
            "alerts": {
                "email": [],
                "slack": [],
                "pagerduty": []
            }
        }
        
        health_file = self.bundle_path / "health_config.json"
        with open(health_file, 'w') as f:
            json.dump(health_config, f, indent=2)
        
        print(f"✅ Created health check configuration")
    
    def _create_config_validation(self):
        """Create configuration validation"""
        print("🔍 Creating configuration validation...")
        
        validation_script = self.bundle_path / "scripts" / "validate_config.py"
        with open(validation_script, 'w') as f:
            f.write(self._get_validation_script_content())
        
        validation_script.chmod(0o755)
        print(f"✅ Created configuration validation")
    
    def _get_validation_script_content(self) -> str:
        """Get configuration validation script content"""
        return '''#!/usr/bin/env python3
"""
Configuration Validation Script

Validates the deployment configuration.
"""

import json
import sys
from pathlib import Path

def validate_config():
    """Validate deployment configuration"""
    config_file = Path(__file__).parent.parent / "deployment_config.json"
    
    try:
        with open(config_file) as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['target_environment', 'environment_variables', 'streamlit_config']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate environment variables
        required_env_vars = ['VECTORQA_ENV', 'LOG_LEVEL']
        for var in required_env_vars:
            if var not in config['environment_variables']:
                print(f"❌ Missing environment variable: {var}")
                return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_config()
    sys.exit(0 if success else 1)
'''
    
    def _create_bundle_manifest(self, target_env: str):
        """Create bundle manifest"""
        print("📋 Creating bundle manifest...")
        
        manifest = {
            "bundle_name": self.bundle_name,
            "target_environment": target_env,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "contents": {
                "application_files": [
                    "src/",
                    "requirements*.txt",
                    "README.md"
                ],
                "configuration": [
                    "deployment_config.json",
                    "health_config.json"
                ],
                "scripts": [
                    "scripts/deploy.py",
                    "scripts/setup_environment.py",
                    "scripts/health_check.py",
                    "scripts/validate_config.py",
                    "scripts/migrate_to_service_accounts.py"
                ],
                "settings": [
                    "settings/local_settings.py",
                    "settings/development_settings.py",
                    "settings/staging_settings.py",
                    "settings/production_settings.py",
                    "settings/settings_loader.py"
                ],
                "launchers": [
                    "launchers/launch.py",
                    "launchers/launch_local.py",
                    "launchers/launch_development.py",
                    "launchers/launch_staging.py",
                    "launchers/launch_production.py"
                ],
                "documentation": [
                    "DEPLOYMENT_GUIDE.md"
                ]
            },
            "dependencies": {
                "python_version": "3.8+",
                "packages": [
                    "streamlit>=1.48.1",
                    "pandas",
                    "numpy",
                    "plotly"
                ]
            }
        }
        
        manifest_file = self.bundle_path / "bundle_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Created bundle manifest")
    
    def _create_deployment_guide(self, target_env: str):
        """Create deployment guide"""
        print("📖 Creating deployment guide...")
        
        guide_content = f"""# VectorQA Sage Deployment Guide

## 🚀 Quick Start

This bundle contains everything needed to deploy VectorQA Sage to {target_env}.

### Prerequisites

- Python 3.8+
- pip
- Internet connection (for package installation)

### Deployment Steps

1. **Extract the bundle:**
   ```bash
   unzip {self.bundle_name}.zip
   cd {self.bundle_name}
   ```

2. **Validate configuration:**
   ```bash
   python scripts/validate_config.py
   ```

3. **Setup environment:**
   ```bash
   python scripts/setup_environment.py
   ```

4. **Launch application:**
   
   **Option A: Use environment-specific launcher (Recommended)**
   ```bash
   # Auto-detect environment
   python launchers/launch.py
   
   # Or specify environment
   python launchers/launch_production.py
   python launchers/launch_staging.py
   python launchers/launch_development.py
   python launchers/launch_local.py
   ```
   
   **Option B: Use deployment script**
   ```bash
   python scripts/deploy.py
   ```

### Health Checks

Monitor the deployment:
```bash
python scripts/health_check.py
```

### Configuration

The bundle includes environment-specific settings files in the `settings/` directory:

- `local_settings.py` - Local development configuration
- `development_settings.py` - Development environment configuration  
- `staging_settings.py` - Staging environment configuration
- `production_settings.py` - Production environment configuration
- `settings_loader.py` - Dynamic settings loader

To use environment-specific settings:
```python
from settings.settings_loader import load_environment_settings
settings = load_environment_settings('production')
```

You can also edit `deployment_config.json` to customize:
- Environment variables
- Streamlit settings
- Backend configuration

### Account Migration (Individual → Service Accounts)

🚨 **CRITICAL**: Production deployments require service account migration!

#### Pre-Migration Checklist:
- [ ] Create service accounts for each environment
- [ ] Generate service account credentials
- [ ] Update access permissions
- [ ] Test service account access
- [ ] Plan rollback strategy

#### AWS Service Account Setup:
```bash
# Create IAM role for service account
aws iam create-role --role-name vectorqa-service-role --assume-role-policy-document file://trust-policy.json

# Attach required policies
aws iam attach-role-policy --role-name vectorqa-service-role --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

# Create service account user
aws iam create-user --user-name vectorqa-service-user

# Attach role to user
aws iam attach-user-policy --user-name vectorqa-service-user --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess
```

#### GCP Service Account Setup:
```bash
# Create service account
gcloud iam service-accounts create vectorqa-service --display-name="VectorQA Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:vectorqa-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Generate service account key
gcloud iam service-accounts keys create service-account-key.json \
    --iam-account=vectorqa-service@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

#### Migration Steps:
1. **Backup current credentials**
2. **Create service accounts** (see above)
3. **Update credential files** with service account keys
4. **Test with service accounts** in staging
5. **Deploy to production** with service accounts
6. **Monitor and validate** service account access
7. **Remove individual user access** (after validation)

#### Security Considerations:
- Service accounts should have minimal required permissions
- Rotate service account keys regularly
- Monitor service account usage
- Use cloud provider secret management (AWS Secrets Manager, GCP Secret Manager)

### Troubleshooting

1. **Port conflicts:** Change port in `deployment_config.json`
2. **Dependencies:** Run `pip install -r requirements_demo.txt`
3. **Permissions:** Ensure scripts are executable

### Support

For issues, check:
- Application logs
- Health check results
- Configuration validation

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        guide_file = self.bundle_path / "DEPLOYMENT_GUIDE.md"
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        print(f"✅ Created deployment guide")
    
    def _package_bundle(self):
        """Package the bundle into a zip file"""
        print("📦 Packaging bundle...")
        
        zip_path = self.bundle_path.parent / f"{self.bundle_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.bundle_path):
                for file in files:
                    file_path = Path(root) / file
                    arc_name = file_path.relative_to(self.bundle_path)
                    zipf.write(file_path, arc_name)
        
        print(f"✅ Bundle packaged: {zip_path}")
        print(f"📊 Bundle size: {zip_path.stat().st_size / (1024*1024):.1f} MB")

    def _create_environment_settings(self, target_env: str):
        """Create environment-specific settings files"""
        print("⚙️ Creating environment-specific settings...")
        
        settings_dir = self.bundle_path / "settings"
        settings_dir.mkdir(exist_ok=True)
        
        # Create environment-specific settings files
        for env in ["local", "development", "staging", "production"]:
            settings_file = settings_dir / f"{env}_settings.py"
            with open(settings_file, 'w') as f:
                f.write(self._get_environment_settings_content(env))
            print(f"✅ Created {env}_settings.py")
        
        # Create main settings loader
        loader_file = settings_dir / "settings_loader.py"
        with open(loader_file, 'w') as f:
            f.write(self._get_settings_loader_content())
        print(f"✅ Created settings_loader.py")
    
    def _get_environment_settings_content(self, env: str) -> str:
        """Get environment-specific settings content"""
        env_configs = {
            "local": {
                "description": "Local Development Environment",
                "log_level": "INFO",
                "cache_dir": "~/vectorqa_cache",
                "server_address": "localhost",
                "server_port": 8501,
                "run_on_save": True,
                "headless": False,
                "use_local_llm": True,
                "debug_mode": True,
                "database_url": "sqlite:///./local.db",
                "secrets_file": ".env.local"
            },
            "development": {
                "description": "Development Environment",
                "log_level": "INFO",
                "cache_dir": "/tmp/vectorqa_cache",
                "server_address": "0.0.0.0",
                "server_port": 8501,
                "run_on_save": True,
                "headless": True,
                "use_local_llm": False,
                "debug_mode": True,
                "database_url": "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}",
                "secrets_file": ".env.development"
            },
            "staging": {
                "description": "Staging Environment",
                "log_level": "WARNING",
                "cache_dir": "/tmp/vectorqa_cache",
                "server_address": "0.0.0.0",
                "server_port": 8501,
                "run_on_save": False,
                "headless": True,
                "use_local_llm": False,
                "debug_mode": False,
                "database_url": "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}",
                "secrets_file": ".env.staging"
            },
            "production": {
                "description": "Production Environment",
                "log_level": "ERROR",
                "cache_dir": "/tmp/vectorqa_cache",
                "server_address": "0.0.0.0",
                "server_port": 8501,
                "run_on_save": False,
                "headless": True,
                "use_local_llm": False,
                "debug_mode": False,
                "database_url": "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}",
                "secrets_file": ".env.production"
            }
        }
        
        config = env_configs[env]
        
        return f'''#!/usr/bin/env python3
"""
{config['description']} Settings

Environment-specific configuration for VectorQA Sage.
This file contains settings for the {env} environment.
"""

import os
from pathlib import Path

# Environment name
ENVIRONMENT = "{env}"

# Logging configuration
LOG_LEVEL = "{config['log_level']}"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Cache configuration
CACHE_DIR = "{config['cache_dir']}"
CACHE_ENABLED = True

# Server configuration
SERVER_ADDRESS = "{config['server_address']}"
SERVER_PORT = {config['server_port']}
RUN_ON_SAVE = {config['run_on_save']}
HEADLESS = {config['headless']}

# Backend configuration
USE_LOCAL_LLM = {config['use_local_llm']}
DEBUG_MODE = {config['debug_mode']}

# Database configuration
DATABASE_URL = "{config['database_url']}"

# Secrets file
SECRETS_FILE = "{config['secrets_file']}"

# Environment variables
ENV_VARS = {{
    "VECTORQA_ENV": ENVIRONMENT,
    "LOG_LEVEL": LOG_LEVEL,
    "CACHE_DIR": CACHE_DIR,
    "DATABASE_URL": DATABASE_URL,
    "DEBUG_MODE": str(DEBUG_MODE).lower()
}}

# Streamlit configuration
STREAMLIT_CONFIG = {{
    "server.port": SERVER_PORT,
    "server.address": SERVER_ADDRESS,
    "browser.gatherUsageStats": False,
    "server.runOnSave": RUN_ON_SAVE,
    "server.headless": HEADLESS
}}

# Backend configuration
BACKEND_CONFIG = {{
    "use_local_llm": USE_LOCAL_LLM,
    "cache_enabled": CACHE_ENABLED,
    "debug_mode": DEBUG_MODE
}}

def get_settings():
    """Get all settings for this environment"""
    return {{
        "environment": ENVIRONMENT,
        "env_vars": ENV_VARS,
        "streamlit_config": STREAMLIT_CONFIG,
        "backend_config": BACKEND_CONFIG,
        "database_url": DATABASE_URL,
        "secrets_file": SECRETS_FILE
    }}

def load_secrets():
    """Load secrets from environment file"""
    secrets_path = Path(SECRETS_FILE)
    if secrets_path.exists():
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        print(f"✅ Loaded secrets from {{SECRETS_FILE}}")
    else:
        print(f"⚠️  Secrets file not found: {{SECRETS_FILE}}")

if __name__ == "__main__":
    # Test settings
    settings = get_settings()
    print(f"🌍 Environment: {{settings['environment']}}")
    print(f"📁 Cache dir: {{settings['env_vars']['CACHE_DIR']}}")
    print(f"🔧 Debug mode: {{settings['backend_config']['debug_mode']}}")
'''
    
    def _get_settings_loader_content(self) -> str:
        """Get settings loader content"""
        return '''#!/usr/bin/env python3
"""
Settings Loader

Dynamically loads environment-specific settings.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

def load_environment_settings(env_name: str = None) -> Dict[str, Any]:
    """Load settings for the specified environment"""
    if not env_name:
        env_name = os.getenv('VECTORQA_ENV', 'local')
    
    # Add settings directory to path
    settings_dir = Path(__file__).parent
    if str(settings_dir) not in sys.path:
        sys.path.insert(0, str(settings_dir))
    
    # Import environment-specific settings
    try:
        settings_module = __import__(f"{env_name}_settings")
        return settings_module.get_settings()
    except ImportError as e:
        print(f"❌ Could not load settings for environment '{env_name}': {e}")
        # Fallback to local settings
        try:
            settings_module = __import__("local_settings")
            return settings_module.get_settings()
        except ImportError:
            print("❌ Could not load any settings")
            return {}

def setup_environment(env_name: str = None):
    """Setup environment with settings"""
    settings = load_environment_settings(env_name)
    
    if not settings:
        print("❌ No settings loaded")
        return False
    
    # Set environment variables
    for key, value in settings.get('env_vars', {}).items():
        os.environ[key] = str(value)
    
    # Load secrets
    try:
        settings_module = __import__(f"{settings['environment']}_settings")
        settings_module.load_secrets()
    except ImportError:
        print(f"⚠️  Could not load secrets for {settings['environment']}")
    
    print(f"✅ Environment setup complete: {settings['environment']}")
    return True

if __name__ == "__main__":
    # Test settings loader
    env = os.getenv('VECTORQA_ENV', 'local')
    settings = load_environment_settings(env)
    print(f"🌍 Loaded settings for: {env}")
    print(f"📋 Settings: {settings}")
'''

    def _create_environment_launchers(self):
        """Create environment-specific launcher scripts"""
        print("🚀 Creating environment launcher scripts...")
        
        launchers_dir = self.bundle_path / "launchers"
        launchers_dir.mkdir(exist_ok=True)
        
        # Create launcher scripts for each environment
        for env in ["local", "development", "staging", "production"]:
            launcher_file = launchers_dir / f"launch_{env}.py"
            with open(launcher_file, 'w') as f:
                f.write(self._get_launcher_script_content(env))
            launcher_file.chmod(0o755)
            print(f"✅ Created launch_{env}.py")
        
        # Create main launcher
        main_launcher = launchers_dir / "launch.py"
        with open(main_launcher, 'w') as f:
            f.write(self._get_main_launcher_content())
        main_launcher.chmod(0o755)
        print(f"✅ Created main launcher")
    
    def _get_launcher_script_content(self, env: str) -> str:
        """Get environment-specific launcher script content"""
        return f'''#!/usr/bin/env python3
"""
{env.title()} Environment Launcher

Launches VectorQA Sage in {env} environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch application in {env} environment"""
    print(f"🚀 Launching VectorQA Sage in {{env}} environment...")
    
    # Set environment variable
    os.environ['VECTORQA_ENV'] = '{env}'
    
    # Get project root (2 levels up from launchers directory)
    project_root = Path(__file__).parent.parent
    
    # Add settings to Python path
    settings_dir = project_root / "settings"
    if str(settings_dir) not in sys.path:
        sys.path.insert(0, str(settings_dir))
    
    # Load environment settings
    try:
        from settings_loader import setup_environment
        setup_environment('{env}')
    except ImportError as e:
        print(f"⚠️  Could not load settings: {{e}}")
    
    # Launch Streamlit app
    app_script = project_root / "src" / "main.py"
    if app_script.exists():
        print(f"🎯 Starting Streamlit app: {{app_script}}")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_script)
        ])
    else:
        print(f"❌ App script not found: {{app_script}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _get_main_launcher_content(self) -> str:
        """Get main launcher script content"""
        return '''#!/usr/bin/env python3
"""
VectorQA Sage Launcher

Main launcher script that detects environment and launches appropriately.
"""

import os
import sys
import subprocess
from pathlib import Path

def detect_environment():
    """Detect the current environment"""
    # Check for explicit environment variable
    if os.getenv('VECTORQA_ENV'):
        return os.getenv('VECTORQA_ENV')
    
    # Check for AWS environment
    if os.getenv('AWS_REGION') or os.getenv('AWS_ACCESS_KEY_ID'):
        return 'production'
    
    # Check for staging indicators
    if os.getenv('STAGING') or 'staging' in os.getenv('HOSTNAME', '').lower():
        return 'staging'
    
    # Check for development indicators
    if os.getenv('DEVELOPMENT') or 'dev' in os.getenv('HOSTNAME', '').lower():
        return 'development'
    
    # Default to local
    return 'local'

def main():
    """Main launcher function"""
    env = detect_environment()
    print(f"🌍 Detected environment: {env}")
    
    # Get launcher script path
    launcher_script = Path(__file__).parent / f"launch_{env}.py"
    
    if launcher_script.exists():
        print(f"🚀 Launching with {env} configuration...")
        subprocess.run([sys.executable, str(launcher_script)])
    else:
        print(f"❌ Launcher not found: {launcher_script}")
        print("Available launchers:")
        for launcher in Path(__file__).parent.glob("launch_*.py"):
            print(f"  - {launcher.name}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    def _create_service_account_migration_script(self):
        """Create service account migration script"""
        print("🔐 Creating service account migration script...")
        
        scripts_dir = self.bundle_path / "scripts"
        migration_script = scripts_dir / "migrate_to_service_accounts.py"
        
        with open(migration_script, 'w') as f:
            f.write(self._get_service_account_migration_content())
        
        migration_script.chmod(0o755)
        print(f"✅ Created service account migration script")
    
    def _get_service_account_migration_content(self) -> str:
        """Get service account migration script content"""
        return '''#!/usr/bin/env python3
"""
Service Account Migration Script

Migrates from individual user accounts to service accounts for production deployment.
This script provides step-by-step guidance for secure account migration.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class ServiceAccountMigrator:
    """Handles migration from individual to service accounts"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_file = self.project_root / "deployment_config.json"
        self.credentials_file = self.project_root / "src" / "backend" / "config" / "credentials.env"
        
    def run_migration_checklist(self):
        """Run the complete migration checklist"""
        print("🔐 SERVICE ACCOUNT MIGRATION CHECKLIST")
        print("=" * 50)
        
        # Pre-migration checks
        self._check_current_credentials()
        self._check_environment()
        self._show_migration_steps()
        self._show_security_considerations()
        
    def _check_current_credentials(self):
        """Check current credential setup"""
        print("\n📋 CURRENT CREDENTIALS CHECK:")
        
        if self.credentials_file.exists():
            print("✅ Credentials file found")
            self._analyze_credentials()
        else:
            print("⚠️  No credentials file found")
            
    def _analyze_credentials(self):
        """Analyze current credentials for migration needs"""
        try:
            with open(self.credentials_file) as f:
                content = f.read()
                
            print("\n🔍 CREDENTIAL ANALYSIS:")
            
            # Check for individual user credentials
            if "your_aws_access_key_here" in content or "your_openai_api_key_here" in content:
                print("⚠️  Found placeholder credentials - needs real values")
            else:
                print("✅ Found configured credentials")
                
            # Check for service account indicators
            if "service" in content.lower() or "role" in content.lower():
                print("✅ Appears to be service account credentials")
            else:
                print("⚠️  Appears to be individual user credentials")
                
        except Exception as e:
            print(f"❌ Error analyzing credentials: {e}")
    
    def _check_environment(self):
        """Check current environment"""
        print("\n🌍 ENVIRONMENT CHECK:")
        
        env = os.getenv('VECTORQA_ENV', 'unknown')
        print(f"Current environment: {env}")
        
        if env == 'production':
            print("🚨 PRODUCTION ENVIRONMENT - Service accounts REQUIRED!")
        elif env == 'staging':
            print("⚠️  STAGING ENVIRONMENT - Consider service accounts")
        else:
            print("ℹ️  DEVELOPMENT ENVIRONMENT - Individual accounts OK")
    
    def _show_migration_steps(self):
        """Show migration steps"""
        print("\n📋 MIGRATION STEPS:")
        print("1. Backup current credentials")
        print("2. Create service accounts (see commands below)")
        print("3. Generate service account credentials")
        print("4. Update credential files")
        print("5. Test in staging environment")
        print("6. Deploy to production")
        print("7. Monitor and validate")
        print("8. Remove individual access")
        
    def _show_security_considerations(self):
        """Show security considerations"""
        print("\n🔒 SECURITY CONSIDERATIONS:")
        print("• Use minimal required permissions")
        print("• Rotate service account keys regularly")
        print("• Monitor service account usage")
        print("• Use cloud provider secret management")
        print("• Implement proper access controls")
        
    def create_aws_service_account(self):
        """Create AWS service account"""
        print("\n☁️  AWS SERVICE ACCOUNT CREATION:")
        print("Run these commands:")
        print("\n# Create IAM role")
        print("aws iam create-role --role-name vectorqa-service-role \\")
        print("  --assume-role-policy-document file://trust-policy.json")
        print("\n# Attach policies")
        print("aws iam attach-role-policy --role-name vectorqa-service-role \\")
        print("  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess")
        print("\n# Create service user")
        print("aws iam create-user --user-name vectorqa-service-user")
        print("\n# Generate access keys")
        print("aws iam create-access-key --user-name vectorqa-service-user")
        
    def create_gcp_service_account(self):
        """Create GCP service account"""
        print("\n☁️  GCP SERVICE ACCOUNT CREATION:")
        print("Run these commands:")
        print("\n# Create service account")
        print("gcloud iam service-accounts create vectorqa-service \\")
        print("  --display-name='VectorQA Service Account'")
        print("\n# Grant roles")
        print("gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \\")
        print("  --member='serviceAccount:vectorqa-service@YOUR_PROJECT_ID.iam.gserviceaccount.com' \\")
        print("  --role='roles/aiplatform.user'")
        print("\n# Generate key")
        print("gcloud iam service-accounts keys create service-account-key.json \\")
        print("  --iam-account=vectorqa-service@YOUR_PROJECT_ID.iam.gserviceaccount.com")

def main():
    """Main migration function"""
    migrator = ServiceAccountMigrator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "checklist":
            migrator.run_migration_checklist()
        elif command == "aws":
            migrator.create_aws_service_account()
        elif command == "gcp":
            migrator.create_gcp_service_account()
        else:
            print("Usage: python migrate_to_service_accounts.py [checklist|aws|gcp]")
    else:
        migrator.run_migration_checklist()

if __name__ == "__main__":
    main()
'''

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create VectorQA Sage migration bundle")
    parser.add_argument("--env", "-e", default="staging", choices=["local", "development", "staging", "production"],
                       help="Target environment (default: staging)")
    parser.add_argument("--name", "-n", help="Bundle name (default: auto-generated)")
    
    args = parser.parse_args()
    
    creator = MigrationBundleCreator(args.name)
    success = creator.create_bundle(args.env)
    
    if success:
        print("\n🎉 Migration bundle created successfully!")
        print("📦 Ready for deployment!")
    else:
        print("\n❌ Bundle creation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

