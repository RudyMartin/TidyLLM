#!/usr/bin/env python3
"""
QA-Scoring Environment Setup Script
Installs TidyLLM ecosystem with proper dependency management

Usage:
    python qa-scoring-setup.py [--minimal] [--full] [--dev]

Options:
    --minimal   Install core dependencies only
    --full      Install all optional dependencies
    --dev       Install development tools
    --help      Show this help message
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

class QAScoringSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.success_count = 0
        self.total_steps = 0

    def run_command(self, command, description, cwd=None, capture_output=False):
        """Run a command with error handling and progress tracking"""
        print(f"\n[{self.success_count + 1}/{self.total_steps}] {description}")
        print(f"Running: {' '.join(command) if isinstance(command, list) else command}")

        try:
            if capture_output:
                result = subprocess.run(
                    command,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout.strip()
            else:
                subprocess.run(command, cwd=cwd, check=True)

            self.success_count += 1
            print(f"✅ Success: {description}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {description}")
            print(f"Error: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error in {description}: {e}")
            return False

    def check_python_version(self):
        """Ensure Python 3.8+ is being used"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True

    def install_local_packages(self):
        """Install local TLM and tidyllm-sentence packages in development mode"""
        local_packages = [
            ("tlm", "Teaching Library Module - NumPy substitute"),
            ("tidyllm-sentence", "TidyLLM Sentence Processing"),
            ("tidyllm", "TidyLLM Core Framework")
        ]

        for package_dir, description in local_packages:
            package_path = self.base_dir / package_dir
            if package_path.exists():
                success = self.run_command(
                    [sys.executable, "-m", "pip", "install", "-e", "."],
                    f"Installing {description} (development mode)",
                    cwd=package_path
                )
                if not success:
                    print(f"⚠️  Warning: Failed to install {package_dir}")
            else:
                print(f"⚠️  Warning: {package_dir} directory not found at {package_path}")

    def install_core_requirements(self):
        """Install core requirements from requirements.txt"""
        core_packages = [
            "requests>=2.25.0",
            "pyyaml>=5.4.0",
            "boto3>=1.20.0",
            "botocore>=1.23.0",
            "psycopg2-binary>=2.8.6",
            "sqlalchemy>=1.4.0",
            "polars>=0.18.0",
            "mlflow>=2.0.0",
            "dspy-ai>=2.4.0"
        ]

        for package in core_packages:
            success = self.run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing core package: {package.split('>=')[0]}"
            )
            if not success:
                print(f"⚠️  Failed to install {package}")

    def install_web_packages(self):
        """Install web and visualization packages"""
        web_packages = [
            "streamlit>=1.28.0",
            "flask>=2.0.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "plotly>=5.0.0"
        ]

        for package in web_packages:
            self.run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing web package: {package.split('>=')[0]}"
            )

    def install_document_packages(self):
        """Install document processing packages"""
        doc_packages = [
            "PyMuPDF>=1.20.0",
            "sentence-transformers>=2.2.0",
            "langchain>=0.1.0",
            "reportlab>=3.6.0",
            "openai>=1.0.0"
        ]

        for package in doc_packages:
            self.run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing document package: {package.split('>=')[0]}"
            )

    def install_system_packages(self):
        """Install system and performance packages"""
        system_packages = [
            "psutil>=5.8.0",
            "schedule>=1.1.0"
        ]

        for package in system_packages:
            self.run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing system package: {package.split('>=')[0]}"
            )

    def install_dev_packages(self):
        """Install development tools"""
        dev_packages = [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0"
        ]

        for package in dev_packages:
            self.run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing dev package: {package.split('>=')[0]}"
            )

    def verify_installation(self):
        """Verify that key packages are installed correctly"""
        verification_tests = [
            ("import tlm; print(f'TLM v{tlm.__version__}')", "TLM"),
            ("import polars as pl; print(f'Polars v{pl.__version__}')", "Polars"),
            ("import requests; print(f'Requests v{requests.__version__}')", "Requests"),
            ("import boto3; print(f'Boto3 v{boto3.__version__}')", "Boto3"),
            ("import mlflow; print(f'MLflow v{mlflow.__version__}')", "MLflow")
        ]

        print("\n" + "="*60)
        print("VERIFYING INSTALLATION")
        print("="*60)

        for test_code, package_name in verification_tests:
            try:
                result = self.run_command(
                    [sys.executable, "-c", test_code],
                    f"Verifying {package_name}",
                    capture_output=True
                )
                if result:
                    print(f"   → {result}")
            except:
                print(f"⚠️  {package_name} verification failed")

    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "data", "logs", "temp", "cache",
            "admin/backups", "mlruns"
        ]

        for dir_path in directories:
            full_path = self.base_dir / "tidyllm" / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

        print("✅ Created necessary directories")

    def display_summary(self):
        """Display installation summary"""
        print("\n" + "="*60)
        print("QA-SCORING SETUP COMPLETE")
        print("="*60)
        print(f"✅ Successful steps: {self.success_count}/{self.total_steps}")

        if self.success_count == self.total_steps:
            print("🎉 All installations completed successfully!")
        else:
            print(f"⚠️  {self.total_steps - self.success_count} steps had warnings/errors")

        print("\nNext steps:")
        print("1. cd tidyllm")
        print("2. python -c \"import tidyllm; print('TidyLLM ready!')\"")
        print("3. Check admin/settings.yaml for configuration")
        print("\nEnvironment: qa-scoring")
        print("Root path: C:\\Users\\marti\\qa-scoring\\tidyllm")

def main():
    parser = argparse.ArgumentParser(description="QA-Scoring Environment Setup")
    parser.add_argument("--minimal", action="store_true", help="Install core dependencies only")
    parser.add_argument("--full", action="store_true", help="Install all dependencies")
    parser.add_argument("--dev", action="store_true", help="Include development tools")

    args = parser.parse_args()

    setup = QAScoringSetup()

    # Determine installation scope
    if args.minimal:
        install_scope = "minimal"
        setup.total_steps = 12  # Local packages + core + verification
    elif args.full:
        install_scope = "full"
        setup.total_steps = 25  # All packages
    else:
        install_scope = "standard"
        setup.total_steps = 18  # Local + core + web + verification

    print("="*60)
    print("QA-SCORING ENVIRONMENT SETUP")
    print("="*60)
    print(f"Installation scope: {install_scope}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {setup.base_dir}")
    print(f"Total steps: {setup.total_steps}")
    print("="*60)

    # Check Python version
    if not setup.check_python_version():
        sys.exit(1)

    # Setup directories
    setup.setup_directories()

    # Install packages based on scope
    print("\n🔧 Installing local packages...")
    setup.install_local_packages()

    print("\n📦 Installing core requirements...")
    setup.install_core_requirements()

    if not args.minimal:
        print("\n🌐 Installing web packages...")
        setup.install_web_packages()

        if args.full:
            print("\n📄 Installing document packages...")
            setup.install_document_packages()

            print("\n⚡ Installing system packages...")
            setup.install_system_packages()

    if args.dev or args.full:
        print("\n🛠️  Installing development tools...")
        setup.install_dev_packages()

    # Verification
    setup.verify_installation()

    # Summary
    setup.display_summary()

if __name__ == "__main__":
    main()