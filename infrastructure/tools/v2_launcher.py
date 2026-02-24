#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM V2 Startup Script
========================

Quick startup script for your boss to get PDF processing running immediately.

Usage:
    python start_v2.py                    # Start PDF processing app
    python start_v2.py --health           # Start health monitoring
    python start_v2.py --setup            # Run setup wizard
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Fix Windows Unicode issues
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Ensure we're in the V2 directory
V2_ROOT = Path(__file__).parent
os.chdir(V2_ROOT)

def setup_environment():
    """Set up environment variables for V2."""
    
    env_vars = {
        'ENVIRONMENT': 'development',
        'AWS_REGION': 'us-east-1',
        'AWS_DEFAULT_REGION': 'us-east-1',
        'DATA_DIRECTORY': './data',
        'DOCUMENT_BUCKET': 'tidyllm-v2-documents-dev',
        'PYTHONPATH': str(V2_ROOT),
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("CHECK Environment configured for TidyLLM V2")


def check_dependencies():
    """Check if required dependencies are installed."""
    
    required_packages = [
        'streamlit',
        'polars', 
        'boto3',
        'pydantic',
        'PyPDF2',
        'PyMuPDF',
        'python-docx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"X Missing required packages: {', '.join(missing_packages)}")
        print("PACKAGE Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', *missing_packages
            ])
            print("CHECK All dependencies installed")
        except subprocess.CalledProcessError:
            print("X Failed to install dependencies")
            print("Please run: pip install -r requirements.txt")
            return False
    
    else:
        print("CHECK All dependencies satisfied")
    
    return True


def start_pdf_app():
    """Start the PDF processing application."""
    
    print("ROCKET Starting TidyLLM V2 PDF Processing Application...")
    print("DOCUMENT Your boss can now upload PDFs and get AI analysis!")
    print("")
    print("WEB Application will be available at: http://localhost:8502")
    print("TOOLS Use Ctrl+C to stop the application")
    print("")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'src/presentation/pdf_app.py',
            '--server.port=8502',
            '--server.address=0.0.0.0',
            '--server.headless=true',
            '--server.enableCORS=false',
            '--server.enableXsrfProtection=false'
        ])
    except KeyboardInterrupt:
        print("\nWAVE TidyLLM V2 stopped")


def start_health_monitoring():
    """Start the health monitoring dashboard."""
    
    print("HEART Starting TidyLLM V2 Health Monitoring...")
    print("CHART System health dashboard for monitoring")
    print("")
    print("WEB Health dashboard will be available at: http://localhost:8503")
    print("")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'src/presentation/app.py',
            '--server.port=8503',
            '--server.address=0.0.0.0'
        ])
    except KeyboardInterrupt:
        print("\nWAVE Health monitoring stopped")


def run_setup_wizard():
    """Run interactive setup wizard."""
    
    print("WIZARD TidyLLM V2 Setup Wizard")
    print("=" * 40)
    print("")
    
    # AWS Configuration
    print("LOCK AWS Configuration")
    print("For production deployment, you'll need:")
    print("  1. AWS Account with Bedrock access")
    print("  2. S3 bucket for document storage")  
    print("  3. AWS Secrets Manager for credentials")
    print("")
    
    use_aws = input("Do you have AWS configured? (y/n): ").lower().strip()
    
    if use_aws == 'y':
        print("CHECK AWS mode enabled")
        
        # Get AWS settings
        region = input("AWS Region (default: us-east-1): ").strip() or 'us-east-1'
        bucket = input("S3 Bucket name (default: tidyllm-v2-documents): ").strip() or 'tidyllm-v2-documents'
        
        # Update environment
        os.environ['AWS_REGION'] = region
        os.environ['DOCUMENT_BUCKET'] = bucket
        
        print(f"LOCATION Region: {region}")
        print(f"BUCKET Bucket: {bucket}")
        
    else:
        print("WARNING Development mode - using local storage")
        print("NOTE Note: AI analysis will require AWS Bedrock")
    
    print("")
    print("TARGET Setup complete! Starting PDF processing app...")
    print("")
    
    start_pdf_app()


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='TidyLLM V2 Startup Script')
    parser.add_argument('--health', action='store_true', help='Start health monitoring dashboard')
    parser.add_argument('--setup', action='store_true', help='Run setup wizard')
    parser.add_argument('--check', action='store_true', help='Check dependencies only')
    
    args = parser.parse_args()
    
    print("ROCKET TidyLLM V2 - Clean Architecture PDF AI Analysis")
    print("=" * 50)
    print("CHECK Zero hardcoded credentials")
    print("LOCK Enterprise security with AWS Secrets Manager") 
    print("DOCUMENT PDF upload -> AI analysis -> Business insights")
    print("")
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.check:
        print("CHECK Dependency check complete")
        return
    
    if args.setup:
        run_setup_wizard()
    elif args.health:
        start_health_monitoring()
    else:
        start_pdf_app()


if __name__ == '__main__':
    main()