#!/usr/bin/env python3
"""
Test 01: Package Structure Validation
=====================================
Verify core TidyLLM package structure is correct with NO fallbacks.
Tests the restructuring work we just completed.
"""

import os
import sys
import importlib
from pathlib import Path

# NO FALLBACKS - Direct imports only
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_01_package_exists():
    """Test that tidyllm package exists at correct location."""
    print("[TEST 01] Checking tidyllm package exists...")
    
    tidyllm_path = Path("tidyllm")
    assert tidyllm_path.exists(), f"FAIL: tidyllm directory not found at {tidyllm_path.absolute()}"
    assert tidyllm_path.is_dir(), "FAIL: tidyllm is not a directory"
    
    init_file = tidyllm_path / "__init__.py"
    assert init_file.exists(), f"FAIL: __init__.py not found at {init_file}"
    
    print("  [PASS] Package structure exists")
    return True

def test_02_no_nested_tidyllm():
    """Test that nested tidyllm/tidyllm has been removed."""
    print("[TEST 02] Checking no nested tidyllm/tidyllm...")
    
    nested_path = Path("tidyllm/tidyllm")
    assert not nested_path.exists(), f"FAIL: Nested tidyllm/tidyllm still exists at {nested_path}"
    
    print("  [PASS] No nested structure found")
    return True

def test_03_core_files_present():
    """Test that core files are in correct location."""
    print("[TEST 03] Checking core files...")
    
    core_files = {
        "tidyllm/__init__.py": "Package init",
        "tidyllm/api.py": "API module", 
        "tidyllm/cli.py": "CLI module"
    }
    
    for file_path, description in core_files.items():
        path = Path(file_path)
        assert path.exists(), f"FAIL: {description} missing at {file_path}"
        assert path.stat().st_size > 0, f"FAIL: {description} is empty"
    
    print("  [PASS] All core files present")
    return True

def test_04_no_test_files_in_root():
    """Test that test/demo files have been moved from root."""
    print("[TEST 04] Checking no test files in package root...")
    
    root_path = Path("tidyllm")
    bad_patterns = ["*test*.py", "*demo*.py", "*example*.py", "simple_*.py"]
    
    for pattern in bad_patterns:
        matches = list(root_path.glob(pattern))
        # Exclude __init__.py and legitimate modules
        matches = [m for m in matches if m.name != "__init__.py"]
        assert len(matches) == 0, f"FAIL: Found {pattern} files in root: {matches}"
    
    print("  [PASS] No test/demo files in root")
    return True

def test_05_examples_directory_exists():
    """Test that examples directory was created and populated."""
    print("[TEST 05] Checking examples directory...")
    
    examples_path = Path("tidyllm/examples")
    assert examples_path.exists(), f"FAIL: examples directory not found at {examples_path}"
    
    example_files = list(examples_path.glob("*.py"))
    assert len(example_files) > 0, "FAIL: No files in examples directory"
    
    print(f"  [PASS] Examples directory has {len(example_files)} files")
    return True

def run_all_tests():
    """Run all package structure tests."""
    print("\n" + "="*60)
    print("CORE PACKAGE STRUCTURE TESTS")
    print("="*60)
    
    tests = [
        test_01_package_exists,
        test_02_no_nested_tidyllm,
        test_03_core_files_present,
        test_04_no_test_files_in_root,
        test_05_examples_directory_exists
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"  {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] Unexpected: {e}")
            failed += 1
    
    print("\n" + "-"*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)