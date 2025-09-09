#!/usr/bin/env python3
"""
TidyLLM Critical Path Test Suite
===============================
Reduced to 3 critical test suites that MUST work:
1. AWS Connectivity (PRE-FLIGHT)
2. Core Package Structure  
3. Infrastructure Components

HARD REQUIREMENTS:
- AWS connectivity MUST pass first
- No fallbacks, no mocks
- All tests must pass for migration readiness
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_preflight_check():
    """Run AWS connectivity pre-flight check."""
    print("\n" + "="*70)
    print("STEP 1: PRE-FLIGHT AWS CONNECTIVITY CHECK")
    print("="*70)
    
    preflight_test = Path("tests/00_preflight/test_aws_connectivity.py")
    
    if not preflight_test.exists():
        print("[FAIL] Pre-flight test not found!")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(preflight_test)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("[OK] PRE-FLIGHT PASSED - AWS connectivity verified")
            return True
        else:
            print("[X] PRE-FLIGHT FAILED - AWS connectivity broken")
            return False
            
    except subprocess.TimeoutExpired:
        print("[X] PRE-FLIGHT TIMEOUT - AWS connectivity check took too long")
        return False
    except Exception as e:
        print(f"[X] PRE-FLIGHT ERROR - {e}")
        return False

def run_critical_suite(suite_path: Path, suite_name: str):
    """Run a critical test suite."""
    print(f"\n{'='*70}")
    print(f"Running: {suite_name}")
    print(f"{'='*70}")
    
    test_files = sorted(suite_path.glob("test_*.py"))
    
    if not test_files:
        print(f"  [SKIP] No test files in {suite_path}")
        return 0, 0
    
    suite_passed = 0
    suite_failed = 0
    
    for test_file in test_files:
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                suite_passed += 1
                print(f"  [PASS] {test_file.name}")
            else:
                suite_failed += 1
                print(f"  [FAIL] {test_file.name}")
                if result.stdout:
                    # Show last few lines of output
                    lines = result.stdout.strip().split('\n')[-3:]
                    for line in lines:
                        print(f"      {line}")
                if result.stderr:
                    print(f"    ERROR: {result.stderr[:200]}")
                    
        except subprocess.TimeoutExpired:
            suite_failed += 1
            print(f"  [TIMEOUT] {test_file.name}")
        except Exception as e:
            suite_failed += 1
            print(f"  [ERROR] {test_file.name}: {e}")
    
    return suite_passed, suite_failed

def main():
    """Run critical path test suite."""
    print("\n" + "="*70)
    print("TIDYLLM CRITICAL PATH VERIFICATION")
    print("3 Core Test Suites - AWS Required")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: PRE-FLIGHT AWS connectivity
    if not run_preflight_check():
        print("\n" + "="*70)
        print("[X] CRITICAL FAILURE: AWS NOT CONNECTED")
        print("Cannot proceed with system tests until AWS connectivity is established.")
        print("Fix AWS credentials and network connectivity first.")
        print("="*70)
        return 1
    
    # Step 2: Critical test suites (only 3)
    critical_suites = [
        ("01_core", "Core Package Structure"),
        ("02_infrastructure", "Infrastructure Components"),
        ("03_gateways", "Gateway System")
    ]
    
    total_passed = 0
    total_failed = 0
    failed_suites = []
    
    tests_dir = Path("tests")
    
    for suite_dir, suite_name in critical_suites:
        suite_path = tests_dir / suite_dir
        passed, failed = run_critical_suite(suite_path, suite_name)
        
        total_passed += passed
        total_failed += failed
        
        if failed > 0:
            failed_suites.append(suite_name)
    
    # Final Summary
    print("\n" + "="*70)
    print("CRITICAL PATH TEST RESULTS")
    print("="*70)
    
    print(f"Total Tests Run: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if failed_suites:
        print("\nFailed Critical Suites:")
        for suite in failed_suites:
            print(f"  [X] {suite}")
    
    print("\n" + "="*70)
    
    if total_failed == 0:
        print("[SUCCESS] ALL CRITICAL TESTS PASSED!")
        print("[OK] AWS connectivity verified")
        print("[OK] Core system functional")
        print("System is ready for migration.")
        return 0
    else:
        print(f"[X] FAILURE: {total_failed} critical tests failed")
        print("System is NOT ready for migration.")
        print("Fix critical failures before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())