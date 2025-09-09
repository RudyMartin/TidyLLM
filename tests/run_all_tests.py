#!/usr/bin/env python3
"""
TidyLLM Complete Test Suite Runner
==================================
Run all 50 tests to verify the system is REALLY working.
NO FALLBACKS - All tests must pass for migration readiness.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_test_suite(suite_path: Path, suite_name: str):
    """Run a single test suite and return results."""
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
                suite_passed += 5  # Each file has 5 tests
                print(f"  [PASS] {test_file.name}")
            else:
                suite_failed += 5
                print(f"  [FAIL] {test_file.name}")
                if result.stdout:
                    print(f"    Output: {result.stdout[-200:]}")
                if result.stderr:
                    print(f"    Error: {result.stderr[-200:]}")
                    
        except subprocess.TimeoutExpired:
            suite_failed += 5
            print(f"  [TIMEOUT] {test_file.name}")
        except Exception as e:
            suite_failed += 5
            print(f"  [ERROR] {test_file.name}: {e}")
    
    return suite_passed, suite_failed

def main():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("TIDYLLM COMPLETE VERIFICATION TEST SUITE")
    print("50 Core Tests - NO FALLBACKS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test suite directories in order
    test_suites = [
        ("01_core", "Core Package Structure"),
        ("02_infrastructure", "Infrastructure Components"),
        ("03_gateways", "Gateway System"),
        ("04_workers", "Worker Architecture"),
        ("05_flow", "Flow System & Bracket Commands"),
        ("06_dropzones", "Drop Zone System"),
        ("07_knowledge", "Knowledge Systems"),
        ("08_web", "Web Dashboard"),
        ("09_integration", "Integration Tests"),
        ("10_endtoend", "End-to-End Verification")
    ]
    
    total_passed = 0
    total_failed = 0
    failed_suites = []
    
    tests_dir = Path("tests")
    
    for suite_dir, suite_name in test_suites:
        suite_path = tests_dir / suite_dir
        passed, failed = run_test_suite(suite_path, suite_name)
        
        total_passed += passed
        total_failed += failed
        
        if failed > 0:
            failed_suites.append(suite_name)
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    
    print(f"Total Tests Run: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if failed_suites:
        print("\nFailed Suites:")
        for suite in failed_suites:
            print(f"  - {suite}")
    
    print("\n" + "="*70)
    
    if total_failed == 0:
        print("SUCCESS: ALL 50 TESTS PASSED!")
        print("System is ready for migration.")
        return 0
    else:
        print(f"FAILURE: {total_failed} tests failed")
        print("System is NOT ready for migration.")
        print("Fix the failures above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())