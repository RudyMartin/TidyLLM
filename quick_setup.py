#!/usr/bin/env python3
"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

TidyLLM Quick Setup - Simple Onboarding Experience
=================================================

Easy-to-follow setup process that uses existing admin infrastructure.
Runs a limited set of essential tests to verify everything works.

Usage:
    python quick_setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print(">> TidyLLM Quick Setup v1.0.4")
    print("=" * 60)
    print("Simple onboarding with essential tests")
    print()

def print_step(step_num, title):
    """Print step header."""
    print(f"\n[STEP {step_num}]: {title}")
    print("-" * 40)

def run_command(command, description, required=True):
    """Run a command and handle errors."""
    print(f"[RUNNING] {description}...")
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[SUCCESS] {description}")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"[FAILED] {description}")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            if required:
                print(f"   This step is required for TidyLLM to work properly.")
                return False
            return True
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        if required:
            return False
        return True

def main():
    """Main setup process."""
    print_banner()
    
    # Check if we're in the right directory
    if not Path("tidyllm/admin").exists():
        print("[ERROR] Please run this from the TidyLLM root directory")
        print("   Current directory should contain 'tidyllm/admin' folder")
        sys.exit(1)
    
    print("[INFO] Using existing admin infrastructure for setup...")
    print(f"[INFO] Working directory: {Path.cwd()}")
    
    # Step 1: Set up AWS credentials using existing admin script
    print_step(1, "AWS Credentials Setup")
    print("Using existing admin script: tidyllm/admin/set_aws_env.bat")
    
    if os.name == 'nt':  # Windows
        success = run_command("tidyllm\\admin\\set_aws_env.bat", "Setting up AWS credentials")
    else:  # Unix/Linux
        success = run_command("tidyllm/admin/set_aws_env.sh", "Setting up AWS credentials")
    
    if not success:
        print("\n[ERROR] AWS setup failed. Please check your credentials in tidyllm/admin/settings.yaml")
        return False
    
    # Step 2: Test configuration using existing admin test
    print_step(2, "Configuration Test")
    print("Using existing admin test: tidyllm/admin/test_config.py")
    
    success = run_command(
        [sys.executable, "tidyllm/admin/test_config.py"],
        "Testing TidyLLM configuration"
    )
    
    if not success:
        print("\n[WARNING] Configuration test had issues, but continuing...")
    
    # Step 3: Quick connectivity test
    print_step(3, "Essential Connectivity Tests")
    
    # Test S3 connectivity
    test_s3_cmd = [
        sys.executable, "-c",
        "from tidyllm.infrastructure.session.unified import UnifiedSessionManager; sm = UnifiedSessionManager(); client = sm.get_s3_client(); print('S3 buckets:', len(client.list_buckets()['Buckets']))"
    ]
    success = run_command(test_s3_cmd, "Testing S3 connectivity", required=False)
    
    # Test basic TidyLLM import
    test_import_cmd = [
        sys.executable, "-c",
        "import tidyllm; print('TidyLLM import successful')"
    ]
    success = run_command(test_import_cmd, "Testing TidyLLM import", required=False)
    
    # Step 4: Summary
    print_step(4, "Setup Complete")
    print("[SUCCESS] TidyLLM setup completed!")
    print()
    print("[NEXT STEPS]:")
    print("   1. Try: python -c \"import tidyllm; print(tidyllm.chat('Hello!'))\"")
    print("   2. Check: tidyllm/admin/settings.yaml for configuration")
    print("   3. Run tests: python tidyllm/admin/test_config.py")
    print("   4. View docs: docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md")
    print()
    print("[ADMIN TOOLS]:")
    print("   * tidyllm/admin/set_aws_env.bat - Reset AWS credentials")
    print("   * tidyllm/admin/test_config.py - Full configuration test")
    print("   * tidyllm/admin/run_diagnostics_real.py - Detailed diagnostics")
    print()
    print("[ARCHITECTURE]: 4-Gateway 2-Service Design")
    print("   * CorporateLLM -> AIProcessing -> WorkflowOptimizer -> Database")
    print("   * + UnifiedSessionManager + DomainRAG")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[WARNING] Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Setup failed with error: {e}")
        sys.exit(1)