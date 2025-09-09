#!/usr/bin/env python3
"""
Test 04: Gateway System
=======================
Test all gateways can be imported and initialized.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_16_corporate_llm_gateway():
    """Test CorporateLLMGateway import and structure."""
    print("[TEST 16] Testing CorporateLLMGateway...")
    
    from tidyllm.gateways import CorporateLLMGateway
    
    assert CorporateLLMGateway is not None, "FAIL: CorporateLLMGateway import failed"
    
    # Check expected methods exist
    expected_methods = ['process_request', 'validate_request']
    for method in expected_methods:
        assert hasattr(CorporateLLMGateway, method), f"FAIL: Missing method {method}"
    
    print("  [PASS] CorporateLLMGateway available")
    return True

def test_17_database_gateway():
    """Test DatabaseGateway import and structure."""
    print("[TEST 17] Testing DatabaseGateway...")
    
    from tidyllm.gateways import DatabaseGateway
    
    assert DatabaseGateway is not None, "FAIL: DatabaseGateway import failed"
    
    # Check it has expected attributes
    assert hasattr(DatabaseGateway, '__init__'), "FAIL: DatabaseGateway not properly defined"
    
    print("  [PASS] DatabaseGateway available")
    return True

def test_18_file_storage_gateway():
    """Test FileStorageGateway import and structure."""
    print("[TEST 18] Testing FileStorageGateway...")
    
    from tidyllm.gateways import FileStorageGateway
    
    assert FileStorageGateway is not None, "FAIL: FileStorageGateway import failed"
    
    print("  [PASS] FileStorageGateway available")
    return True

def test_19_ai_processing_gateway():
    """Test AIProcessingGateway import and structure."""
    print("[TEST 19] Testing AIProcessingGateway...")
    
    from tidyllm.gateways import AIProcessingGateway
    
    assert AIProcessingGateway is not None, "FAIL: AIProcessingGateway import failed"
    
    print("  [PASS] AIProcessingGateway available")
    return True

def test_20_gateway_registry():
    """Test gateway registry system."""
    print("[TEST 20] Testing gateway registry...")
    
    from tidyllm.gateways.gateway_registry import GatewayRegistry
    
    assert GatewayRegistry is not None, "FAIL: GatewayRegistry import failed"
    
    # Test singleton pattern
    registry1 = GatewayRegistry()
    registry2 = GatewayRegistry()
    assert registry1 is registry2, "FAIL: GatewayRegistry not a singleton"
    
    print("  [PASS] Gateway registry works")
    return True

def run_all_tests():
    """Run all gateway tests."""
    print("\n" + "="*60)
    print("GATEWAY SYSTEM TESTS")
    print("="*60)
    
    tests = [
        test_16_corporate_llm_gateway,
        test_17_database_gateway,
        test_18_file_storage_gateway,
        test_19_ai_processing_gateway,
        test_20_gateway_registry
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