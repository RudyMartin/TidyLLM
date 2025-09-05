#!/usr/bin/env python3
"""
Deployment Validation Runner

Runs comprehensive validation tests on the deployment bundle.
"""

import sys
import os
from pathlib import Path

def run_validation():
    """Run all validation tests"""
    print("🔍 Running deployment validation...")
    
    # Add bundle to path
    bundle_root = Path(__file__).parent.parent
    sys.path.insert(0, str(bundle_root / "src"))
    
    success = True
    
    # Run environment tests
    try:
        from test_environments import EnvironmentTester
        tester = EnvironmentTester()
        env_success = tester.test_all_environments()
        success = success and env_success
    except Exception as e:
        print(f"❌ Environment tests failed: {e}")
        success = False
    
    # Run import tests
    try:
        from test_imports import TestImportStructure
        import_tester = TestImportStructure()
        import_tester.setup_method()
        
        # Run key tests
        import_tester.test_config_imports()
        import_tester.test_core_module_imports()
        print("✅ Import tests passed")
    except Exception as e:
        print(f"❌ Import tests failed: {e}")
        success = False
    
    return success

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
