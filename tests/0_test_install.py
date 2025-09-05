#!/usr/bin/env python3
"""
TidyLLM Strategic Test Suite #0: Installation & Dependencies
===========================================================

Verifies that all required packages are installed and accessible.
This should be the FIRST test to run before any other tests.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_python_version():
    """Check Python version meets minimum requirements."""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]
    required_major, required_minor = 3, 8
    
    if major >= required_major and minor >= required_minor:
        print(f"[OK] Python {major}.{minor} meets minimum requirement ({required_major}.{required_minor}+)")
        return True
    else:
        print(f"[FAIL] Python {major}.{minor} does not meet minimum requirement ({required_major}.{required_minor}+)")
        return False


def check_core_packages():
    """Check that core Python packages are installed."""
    print("\nChecking core packages...")
    core_packages = [
        'yaml',
        'json',
        'pathlib',
        'unittest',
        'datetime',
        'typing',
        'dataclasses',
        'concurrent.futures',
        'threading',
        'multiprocessing'
    ]
    
    all_good = True
    for package in core_packages:
        try:
            __import__(package)
            print(f"[OK] {package} available")
        except ImportError:
            print(f"[FAIL] {package} not available")
            all_good = False
    
    return all_good


def check_required_packages():
    """Check that required third-party packages are installed."""
    print("\nChecking required packages...")
    required_packages = {
        'yaml': 'pyyaml',
        'boto3': 'boto3',
        'psutil': 'psutil',
        'requests': 'requests'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"[OK] {package_name} installed")
        except ImportError:
            print(f"[WARN] {package_name} not installed (pip install {package_name})")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n[WARN] Missing packages can be installed with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_optional_packages():
    """Check optional packages that enhance functionality."""
    print("\nChecking optional packages...")
    optional_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'polars': 'polars',
        'streamlit': 'streamlit',
        'mlflow': 'mlflow',
        'dspy': 'dspy-ai',
        'openai': 'openai',
        'anthropic': 'anthropic'
    }
    
    available_count = 0
    
    for import_name, package_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"[OK] {package_name} installed")
            available_count += 1
        except ImportError:
            print(f"[INFO] {package_name} not installed (optional)")
    
    print(f"\nOptional packages available: {available_count}/{len(optional_packages)}")
    return True  # Optional packages don't fail the test


def check_tidyllm_package():
    """Check if TidyLLM package can be imported."""
    print("\nChecking TidyLLM package...")
    
    try:
        import tidyllm
        print("[OK] TidyLLM package can be imported")
        
        # Check for key components
        components = ['gateways', 'knowledge_systems']
        for component in components:
            if hasattr(tidyllm, component) or Path(project_root / 'tidyllm' / component).exists():
                print(f"[OK] TidyLLM component '{component}' found")
            else:
                print(f"[WARN] TidyLLM component '{component}' not found")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Cannot import TidyLLM: {e}")
        print("\nTrying to locate TidyLLM components...")
        
        # Check if we can find the tidyllm directory
        tidyllm_dir = project_root / 'tidyllm'
        if tidyllm_dir.exists():
            print(f"[OK] Found tidyllm directory at {tidyllm_dir}")
            
            # Check for __init__.py
            init_file = tidyllm_dir / '__init__.py'
            if init_file.exists():
                print("[OK] __init__.py exists")
            else:
                print("[WARN] __init__.py missing - creating it...")
                init_file.write_text("""# TidyLLM Package
from . import gateways
from . import knowledge_systems

__version__ = '1.0.0'
""")
                print("[OK] Created __init__.py")
        else:
            print(f"[FAIL] TidyLLM directory not found at {tidyllm_dir}")
        
        return False


def check_environment_setup():
    """Check environment variables and configuration."""
    print("\nChecking environment setup...")
    import os
    
    # Check for AWS credentials
    aws_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'AWS_PROFILE']
    aws_configured = False
    
    for var in aws_vars:
        if os.environ.get(var):
            print(f"[OK] {var} is set")
            aws_configured = True
        else:
            print(f"[INFO] {var} not set")
    
    if not aws_configured:
        print("[INFO] AWS credentials not configured (optional for S3 tests)")
    
    # Check for database configuration
    db_vars = ['DATABASE_URL', 'POSTGRES_HOST', 'POSTGRES_USER']
    db_configured = False
    
    for var in db_vars:
        if os.environ.get(var):
            print(f"[OK] {var} is set")
            db_configured = True
        else:
            print(f"[INFO] {var} not set")
    
    if not db_configured:
        print("[INFO] Database not configured (optional for database tests)")
    
    return True  # Environment setup is optional


def check_test_infrastructure():
    """Check that test infrastructure is in place."""
    print("\nChecking test infrastructure...")
    
    tests_dir = project_root / 'tests'
    if tests_dir.exists():
        print(f"[OK] Tests directory exists: {tests_dir}")
        
        # Count test files
        test_files = list(tests_dir.glob('[0-9]_test_*.py'))
        print(f"[OK] Found {len(test_files)} strategic test files")
        
        # Check for test runner
        runner = tests_dir / 'run_all_tests.py'
        if runner.exists():
            print("[OK] Test runner exists")
        else:
            print("[WARN] Test runner missing")
        
        return True
    else:
        print(f"[FAIL] Tests directory not found: {tests_dir}")
        return False


def suggest_installation_fixes(missing_core, missing_required):
    """Suggest how to fix installation issues."""
    print("\n" + "="*60)
    print("INSTALLATION FIXES NEEDED")
    print("="*60)
    
    if missing_core or missing_required:
        print("\n1. Install missing packages:")
        print("   pip install pyyaml boto3 psutil requests")
        
        print("\n2. For full functionality, consider:")
        print("   pip install pandas numpy polars streamlit mlflow")
        
        print("\n3. For AI capabilities:")
        print("   pip install dspy-ai openai anthropic")
    
    print("\n4. Ensure TidyLLM is in Python path:")
    print(f"   export PYTHONPATH={project_root}:$PYTHONPATH")
    
    print("\n5. For AWS/S3 testing:")
    print("   aws configure  # Set up AWS credentials")


def run_install_tests():
    """Run all installation tests."""
    print("="*60)
    print("TIDYLLM INSTALLATION & DEPENDENCY TESTS")
    print("="*60)
    
    tests = [
        ("Python Version", check_python_version),
        ("Core Packages", check_core_packages),
        ("Required Packages", check_required_packages),
        ("Optional Packages", check_optional_packages),
        ("TidyLLM Package", check_tidyllm_package),
        ("Environment Setup", check_environment_setup),
        ("Test Infrastructure", check_test_infrastructure)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n### {test_name} ###")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("INSTALLATION TEST SUMMARY")
    print("="*60)
    
    critical_pass = all([
        results.get("Python Version", False),
        results.get("Core Packages", False),
        results.get("TidyLLM Package", False),
        results.get("Test Infrastructure", False)
    ])
    
    required_pass = results.get("Required Packages", False)
    
    for test_name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        if test_name == "Optional Packages" or test_name == "Environment Setup":
            status = "[OK]" if passed else "[INFO]"
        print(f"{status} {test_name}")
    
    if critical_pass:
        if required_pass:
            print("\n[OK] System is ready for testing!")
            print("All critical components are installed and accessible.")
        else:
            print("\n[WARN] System has core components but missing some required packages.")
            print("Basic tests will work but some features may be limited.")
            suggest_installation_fixes(False, True)
    else:
        print("\n[FAIL] Critical components are missing!")
        print("System is not ready for testing.")
        suggest_installation_fixes(not results.get("Core Packages", False), 
                                  not results.get("Required Packages", False))
        return False
    
    return critical_pass


if __name__ == "__main__":
    success = run_install_tests()
    sys.exit(0 if success else 1)