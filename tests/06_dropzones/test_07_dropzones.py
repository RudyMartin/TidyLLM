#!/usr/bin/env python3
"""
Test 07: Drop Zone System
=========================
Test drop zone directories and trigger mechanisms.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_31_dropzone_directories():
    """Test drop zone directories exist."""
    print("[TEST 31] Testing drop zone directories...")
    
    base_path = Path("tidyllm/drop_zones")
    assert base_path.exists(), f"FAIL: Drop zones directory not found at {base_path}"
    
    required_zones = [
        "mvr_analysis",
        "financial_analysis", 
        "contract_review",
        "compliance_check",
        "quality_check",
        "data_extraction",
        "processing",
        "completed",
        "failed"
    ]
    
    for zone in required_zones:
        zone_path = base_path / zone
        assert zone_path.exists(), f"FAIL: Drop zone {zone} not found"
    
    print(f"  [PASS] All {len(required_zones)} drop zones exist")
    return True

def test_32_dropzone_readme():
    """Test drop zone documentation exists."""
    print("[TEST 32] Testing drop zone documentation...")
    
    readme_path = Path("tidyllm/drop_zones/README.md")
    assert readme_path.exists(), f"FAIL: Drop zone README not found at {readme_path}"
    assert readme_path.stat().st_size > 100, "FAIL: Drop zone README is too small"
    
    print("  [PASS] Drop zone documentation exists")
    return True

def test_33_sample_documents():
    """Test sample documents in drop zones."""
    print("[TEST 33] Testing sample documents...")
    
    sample_files = {
        "tidyllm/drop_zones/mvr_analysis/sample_mvr_document.txt": "MVR sample",
        "tidyllm/drop_zones/financial_analysis/sample_financial_report.txt": "Financial sample",
        "tidyllm/drop_zones/contract_review/sample_contract.txt": "Contract sample"
    }
    
    found = 0
    for file_path, description in sample_files.items():
        path = Path(file_path)
        if path.exists():
            found += 1
            assert path.stat().st_size > 100, f"FAIL: {description} is too small"
    
    assert found > 0, "FAIL: No sample documents found"
    print(f"  [PASS] Found {found} sample documents")
    return True

def test_34_processing_folders():
    """Test processing state folders exist."""
    print("[TEST 34] Testing processing state folders...")
    
    base_path = Path("tidyllm/drop_zones")
    state_folders = ["processing", "completed", "failed"]
    
    for folder in state_folders:
        folder_path = base_path / folder
        assert folder_path.exists(), f"FAIL: State folder {folder} not found"
        assert folder_path.is_dir(), f"FAIL: {folder} is not a directory"
    
    print("  [PASS] All processing state folders exist")
    return True

def test_35_dropzone_structure():
    """Test overall drop zone structure is correct."""
    print("[TEST 35] Testing drop zone structure...")
    
    base_path = Path("tidyllm/drop_zones")
    
    # Count total directories
    all_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    assert len(all_dirs) >= 9, f"FAIL: Expected at least 9 drop zone directories, found {len(all_dirs)}"
    
    # Check no test files in root drop zone
    py_files = list(base_path.glob("*.py"))
    assert len(py_files) == 0, f"FAIL: Found Python files in drop zone root: {py_files}"
    
    print("  [PASS] Drop zone structure correct")
    return True

def run_all_tests():
    """Run all drop zone tests."""
    print("\n" + "="*60)
    print("DROP ZONE SYSTEM TESTS")
    print("="*60)
    
    tests = [
        test_31_dropzone_directories,
        test_32_dropzone_readme,
        test_33_sample_documents,
        test_34_processing_folders,
        test_35_dropzone_structure
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