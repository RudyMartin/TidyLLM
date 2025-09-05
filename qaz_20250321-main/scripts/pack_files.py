#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Deployment Packer Script
===============================

This script creates deployment packages for air-gapped or restricted environments
where git or external access is not possible.

Creates separate zip files for:
1. Database schemas and scripts
2. Credentials and configuration
3. Input assets (filtered for deployment)
4. Core application code

Usage:
    python scripts/pack_files.py [--clean] [--filter-input] [--include-demos]
"""

import os
import sys
import shutil
import zipfile
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentPacker:
    """Manual deployment packer for air-gapped environments"""
    
    def __init__(self, clean_site=True, filter_input=True, include_demos=False):
        self.clean_site = clean_site
        self.filter_input = filter_input
        self.include_demos = include_demos
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "deployment_packages"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define what to include/exclude
        self.setup_inclusion_rules()
        
    def setup_inclusion_rules(self):
        """Define what files to include/exclude for each package"""
        
        # Core application files (always included)
        self.core_includes = [
            "src/",
            "simple_demo.py",
            "start_simple_demo.py", 
            "start_demo.py",
            "start_advanced.py",
            "requirements.txt",
            "requirements_demo.txt",
            "requirements_rag.txt",
            "README.md",
            "README_SIMPLE_DEMO.md",
            "IMPORTANT_START_HERE.md"
        ]
        
        # Files to exclude from core package
        self.core_excludes = [
            ".git/",
            ".venv/",
            "__pycache__/",
            "*.pyc",
            ".DS_Store",
            "*.log",
            "logs/",
            "test_outputs/",
            "rag_output/",
            "output/",
            "input/",
            "environ_settings/",
            "dev_configs/",
            "database/",
            "deployment_packages/",
            "migration_bundles/",
            "_archive/",
            "_documentation_backup/",
            "_old_documentation_backup/",
            ".claude/",
            "optional/",
            "notebooks/",
            ".ipynb_checkpoints/",
            "tests/",
            "data/"
        ]
        
        # Database files
        self.database_includes = [
            "database/"
        ]
        
        # Credential and config files
        self.credentials_includes = [
            "environ_settings/",
            "dev_configs/"
        ]
        
        # Input assets (filtered for deployment)
        self.input_includes = [
            "input/"
        ]
        
        # Input files to exclude (keep only essential demo files)
        self.input_excludes = [
            "*.DS_Store",
            "input/omnibus/",  # Large collection of demo files
            "input/omnibus/all/",
            "input/omnibus/reviews/", 
            "input/omnibus/wfc/",
            "input/omnibus/Data-Detective-Stats-Game/",
            "input/omnibus/bayesian-CNN-LSTM-Q-learning-main/",
            "input/omnibus/SSRI Network Tutorial Materials/"
        ]
        
        # Demo files to include if requested
        self.demo_includes = [
            "input/omnibus/Readme Rag Demo.pdf",
            "input/omnibus/Robot Presentation.pdf",
            "input/omnibus/Smart Fruit Ripeness System.pdf",
            "input/omnibus/helper_functions.txt"
        ]
        
    def clean_site_directory(self):
        """Clean up temporary and build artifacts"""
        logger.info("🧹 Cleaning site directory...")
        
        cleanup_patterns = [
            "**/__pycache__",
            "**/*.pyc", 
            "**/*.log",
            "**/.DS_Store",
            "**/test_outputs",
            "**/rag_output",
            "**/output",
            "**/logs"
        ]
        
        for pattern in cleanup_patterns:
            for path in self.project_root.glob(pattern):
                if path.is_file():
                    path.unlink()
                    logger.debug(f"Deleted file: {path}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    logger.debug(f"Deleted directory: {path}")
                    
        logger.info("✅ Site cleaning complete")
        
    def create_output_directory(self):
        """Create output directory for deployment packages"""
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Output directory: {self.output_dir}")
        
    def should_include_file(self, file_path, includes, excludes):
        """Check if file should be included based on patterns"""
        file_path_str = str(file_path)
        
        # Check excludes first
        for exclude_pattern in excludes:
            if exclude_pattern.endswith('/'):
                # Directory pattern
                if exclude_pattern.rstrip('/') in file_path_str:
                    return False
            else:
                # File pattern
                if file_path.name.endswith(exclude_pattern.lstrip('*')):
                    return False
                    
        # Check includes
        for include_pattern in includes:
            if include_pattern.endswith('/'):
                # Directory pattern
                if include_pattern.rstrip('/') in file_path_str:
                    return True
            else:
                # File pattern
                if file_path.name == include_pattern or file_path.name.endswith(include_pattern.lstrip('*')):
                    return True
                    
        return False
        
    def create_zip_package(self, package_name, includes, excludes, description):
        """Create a zip package with specified files"""
        logger.info(f"📦 Creating {package_name} package...")
        
        zip_path = self.output_dir / f"{package_name}_{self.timestamp}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            files_added = 0
            total_size = 0
            
            for include_pattern in includes:
                if include_pattern.endswith('/'):
                    # Directory pattern
                    dir_path = self.project_root / include_pattern.rstrip('/')
                    if dir_path.exists():
                        for file_path in dir_path.rglob('*'):
                            if file_path.is_file() and self.should_include_file(file_path, [include_pattern], excludes):
                                arcname = file_path.relative_to(self.project_root)
                                zipf.write(file_path, arcname)
                                files_added += 1
                                total_size += file_path.stat().st_size
                                logger.debug(f"Added: {arcname}")
                else:
                    # File pattern
                    file_path = self.project_root / include_pattern
                    if file_path.exists() and file_path.is_file():
                        zipf.write(file_path, include_pattern)
                        files_added += 1
                        total_size += file_path.stat().st_size
                        logger.debug(f"Added: {include_pattern}")
                        
        logger.info(f"✅ {package_name} package created: {zip_path}")
        logger.info(f"   Files: {files_added}, Size: {total_size / 1024 / 1024:.1f} MB")
        
        return zip_path
        
    def create_database_package(self):
        """Create database schema and scripts package"""
        return self.create_zip_package(
            "database_schemas",
            self.database_includes,
            [],
            "Database schemas, scripts, and migration files"
        )
        
    def create_credentials_package(self):
        """Create credentials and configuration package"""
        return self.create_zip_package(
            "credentials_config",
            self.credentials_includes,
            [],
            "Environment settings, configurations, and credentials"
        )
        
    def create_input_package(self):
        """Create filtered input assets package"""
        excludes = self.input_excludes.copy()
        
        if not self.include_demos:
            # Add demo files to excludes
            excludes.extend([
                "input/omnibus/*.pdf",
                "input/omnibus/*.txt"
            ])
            
        includes = self.input_includes.copy()
        
        if self.include_demos:
            includes.extend(self.demo_includes)
            
        return self.create_zip_package(
            "input_assets",
            includes,
            excludes,
            "Input assets and demo files (filtered)"
        )
        
    def create_core_package(self):
        """Create core application package"""
        return self.create_zip_package(
            "core_application",
            self.core_includes,
            self.core_excludes,
            "Core application code and dependencies"
        )
        
    def create_deployment_manifest(self, packages):
        """Create deployment manifest with package information"""
        manifest = {
            "deployment_info": {
                "timestamp": self.timestamp,
                "packer_version": "1.0.0",
                "total_packages": len(packages)
            },
            "packages": {k: str(v) for k, v in packages.items()},
            "deployment_instructions": {
                "1_database": "Extract database_schemas.zip and run setup scripts",
                "2_credentials": "Extract credentials_config.zip and configure environment",
                "3_input": "Extract input_assets.zip to input/ directory",
                "4_core": "Extract core_application.zip to project root",
                "5_setup": "Run: pip install -r requirements.txt",
                "6_start": "Run: python start_simple_demo.py"
            },
            "package_contents": {
                "database_schemas": "SQL scripts, database schemas, migration files",
                "credentials_config": "Environment configs, credentials, settings",
                "input_assets": "Demo documents, test files, sample data",
                "core_application": "Source code, requirements, documentation"
            }
        }
        
        manifest_path = self.output_dir / f"deployment_manifest_{self.timestamp}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        logger.info(f"📋 Deployment manifest created: {manifest_path}")
        return manifest_path
        
    def create_readme(self, packages):
        """Create README for deployment packages"""
        readme_content = f"""# Manual Deployment Packages

Generated on: {self.timestamp}

## 📦 Package Contents

### 1. Database Schemas (`database_schemas_{self.timestamp}.zip`)
- SQL scripts for database setup
- Schema definitions and migrations
- Database configuration files

### 2. Credentials & Config (`credentials_config_{self.timestamp}.zip`)
- Environment configuration files
- Development and production settings
- Credential templates and examples

### 3. Input Assets (`input_assets_{self.timestamp}.zip`)
- Demo documents and test files
- Sample data for testing
- Filtered for deployment (excludes large demo collections)

### 4. Core Application (`core_application_{self.timestamp}.zip`)
- Source code and application files
- Requirements and dependencies
- Documentation and README files

## 🚀 Deployment Instructions

1. **Extract all packages** to a clean directory
2. **Set up database** using scripts from database package
3. **Configure environment** using files from credentials package
4. **Install dependencies**: `pip install -r requirements.txt`
5. **Start application**: `python start_simple_demo.py`

## 🔧 Configuration

- Copy `environ_settings/config.local.yaml` to your environment
- Update database connection settings
- Configure MLflow tracking URI if needed
- Set up S3 credentials for document storage

## 📋 Package Details

"""
        
        for package_name, package_path in packages.items():
            size_mb = package_path.stat().st_size / 1024 / 1024
            readme_content += f"- **{package_name}**: {size_mb:.1f} MB\n"
            
        readme_path = self.output_dir / f"DEPLOYMENT_README_{self.timestamp}.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        logger.info(f"📖 Deployment README created: {readme_path}")
        return readme_path
        
    def pack_all(self):
        """Create all deployment packages"""
        logger.info("🚀 Starting manual deployment packaging...")
        
        # Clean site if requested
        if self.clean_site:
            self.clean_site_directory()
            
        # Create output directory
        self.create_output_directory()
        
        # Create packages
        packages = {}
        
        try:
            packages['database_schemas'] = self.create_database_package()
            packages['credentials_config'] = self.create_credentials_package()
            packages['input_assets'] = self.create_input_package()
            packages['core_application'] = self.create_core_package()
            
            # Create manifest and README
            manifest_path = self.create_deployment_manifest(packages)
            readme_path = self.create_readme(packages)
            
            # Summary
            total_size = sum(p.stat().st_size for p in packages.values())
            logger.info("🎉 Deployment packaging complete!")
            logger.info(f"📊 Total packages: {len(packages)}")
            logger.info(f"📊 Total size: {total_size / 1024 / 1024:.1f} MB")
            logger.info(f"📁 Output directory: {self.output_dir}")
            
            return packages, manifest_path, readme_path
            
        except Exception as e:
            logger.error(f"❌ Error during packaging: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Create manual deployment packages")
    parser.add_argument("--clean", action="store_true", help="Clean site before packaging")
    parser.add_argument("--filter-input", action="store_true", default=True, help="Filter input assets (default: True)")
    parser.add_argument("--include-demos", action="store_true", help="Include demo files in input package")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        packer = DeploymentPacker(
            clean_site=args.clean,
            filter_input=args.filter_input,
            include_demos=args.include_demos
        )
        
        packages, manifest, readme = packer.pack_all()
        
        print(f"\n🎉 Deployment packages created successfully!")
        print(f"📁 Location: {packer.output_dir}")
        print(f"📋 Manifest: {manifest.name}")
        print(f"📖 README: {readme.name}")
        
    except Exception as e:
        logger.error(f"❌ Packaging failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
