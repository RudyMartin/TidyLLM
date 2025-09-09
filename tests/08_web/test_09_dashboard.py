#!/usr/bin/env python3
"""
Test 09: Web Dashboard
======================
Test web dashboard components and structure.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_41_dashboard_structure():
    """Test web dashboard directory structure."""
    print("[TEST 41] Testing dashboard structure...")
    
    web_path = Path("tidyllm/web")
    assert web_path.exists(), f"FAIL: Web directory not found at {web_path}"
    
    required_files = [
        "ai_dropzone_dashboard.py",
        "config.json",
        "requirements.txt",
        "README.md"
    ]
    
    for file in required_files:
        file_path = web_path / file
        assert file_path.exists(), f"FAIL: Required file {file} not found"
    
    print("  [PASS] Dashboard structure correct")
    return True

def test_42_dashboard_components():
    """Test dashboard components directory."""
    print("[TEST 42] Testing dashboard components...")
    
    components_path = Path("tidyllm/web/components")
    assert components_path.exists(), f"FAIL: Components directory not found"
    
    # Check for key component
    monitor_path = components_path / "dropzone_monitor.py"
    assert monitor_path.exists(), "FAIL: dropzone_monitor.py not found"
    
    print("  [PASS] Dashboard components exist")
    return True

def test_43_dashboard_config():
    """Test dashboard configuration file."""
    print("[TEST 43] Testing dashboard config...")
    
    config_path = Path("tidyllm/web/config.json")
    assert config_path.exists(), f"FAIL: Dashboard config not found"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Check required config sections
    required_sections = ["dashboard", "monitoring", "drop_zones"]
    for section in required_sections:
        assert section in config, f"FAIL: Config missing section {section}"
    
    print("  [PASS] Dashboard config valid")
    return True

def test_44_dashboard_pages():
    """Test dashboard pages directory."""
    print("[TEST 44] Testing dashboard pages...")
    
    pages_path = Path("tidyllm/web/pages")
    assert pages_path.exists(), f"FAIL: Pages directory not found"
    
    # Check for system monitor page
    monitor_page = pages_path / "system_monitor.py"
    assert monitor_page.exists(), "FAIL: system_monitor.py not found"
    
    print("  [PASS] Dashboard pages exist")
    return True

def test_45_dashboard_utils():
    """Test dashboard utilities."""
    print("[TEST 45] Testing dashboard utilities...")
    
    utils_path = Path("tidyllm/web/utils")
    assert utils_path.exists(), f"FAIL: Utils directory not found"
    
    # Check for dashboard helpers
    helpers_path = utils_path / "dashboard_helpers.py"
    assert helpers_path.exists(), "FAIL: dashboard_helpers.py not found"
    
    print("  [PASS] Dashboard utilities exist")
    return True

def run_all_tests():
    """Run all web dashboard tests."""
    print("\n" + "="*60)
    print("WEB DASHBOARD TESTS")
    print("="*60)
    
    tests = [
        test_41_dashboard_structure,
        test_42_dashboard_components,
        test_43_dashboard_config,
        test_44_dashboard_pages,
        test_45_dashboard_utils
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1
    
    print("\n" + "-"*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)