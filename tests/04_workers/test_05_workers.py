#!/usr/bin/env python3
"""
Test 05: Worker Architecture
============================
Test worker classes and AI managers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_21_base_worker():
    """Test BaseWorker import and structure."""
    print("[TEST 21] Testing BaseWorker...")
    
    from tidyllm.infrastructure.workers import BaseWorker
    
    assert BaseWorker is not None, "FAIL: BaseWorker import failed"
    
    # Check expected methods
    expected_methods = ['process', 'validate_input']
    for method in expected_methods:
        assert hasattr(BaseWorker, method), f"FAIL: BaseWorker missing method {method}"
    
    print("  [PASS] BaseWorker available")
    return True

def test_22_extraction_worker():
    """Test ExtractionWorker import."""
    print("[TEST 22] Testing ExtractionWorker...")
    
    from tidyllm.infrastructure.workers import ExtractionWorker
    
    assert ExtractionWorker is not None, "FAIL: ExtractionWorker import failed"
    print("  [PASS] ExtractionWorker available")
    return True

def test_23_embedding_worker():
    """Test EmbeddingWorker import."""
    print("[TEST 23] Testing EmbeddingWorker...")
    
    from tidyllm.infrastructure.workers import EmbeddingWorker
    
    assert EmbeddingWorker is not None, "FAIL: EmbeddingWorker import failed"
    print("  [PASS] EmbeddingWorker available")
    return True

def test_24_ai_dropzone_manager():
    """Test AIDropzoneManager import."""
    print("[TEST 24] Testing AIDropzoneManager...")
    
    from tidyllm.infrastructure.workers import AIDropzoneManager
    
    assert AIDropzoneManager is not None, "FAIL: AIDropzoneManager import failed"
    
    # Check critical methods
    assert hasattr(AIDropzoneManager, 'initialize'), "FAIL: Missing initialize method"
    assert hasattr(AIDropzoneManager, 'process_document'), "FAIL: Missing process_document method"
    
    print("  [PASS] AIDropzoneManager available")
    return True

def test_25_flow_integration_manager():
    """Test FlowIntegrationManager import."""
    print("[TEST 25] Testing FlowIntegrationManager...")
    
    from tidyllm.infrastructure.workers import FlowIntegrationManager
    
    assert FlowIntegrationManager is not None, "FAIL: FlowIntegrationManager import failed"
    
    # Check bracket command methods
    assert hasattr(FlowIntegrationManager, 'validate_bracket_command'), "FAIL: Missing validate method"
    assert hasattr(FlowIntegrationManager, 'get_flow_mapping'), "FAIL: Missing mapping method"
    
    print("  [PASS] FlowIntegrationManager available")
    return True

def run_all_tests():
    """Run all worker tests."""
    print("\n" + "="*60)
    print("WORKER ARCHITECTURE TESTS")
    print("="*60)
    
    tests = [
        test_21_base_worker,
        test_22_extraction_worker,
        test_23_embedding_worker,
        test_24_ai_dropzone_manager,
        test_25_flow_integration_manager
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