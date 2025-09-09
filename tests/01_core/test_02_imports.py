#!/usr/bin/env python3
"""
Test 02: Import System Validation
=================================
Verify all imports work correctly with NO fallbacks or try/except.
"""

import sys
from pathlib import Path

# NO FALLBACKS - Direct imports only
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_06_core_imports():
    """Test core package imports work."""
    print("[TEST 06] Testing core imports...")
    
    # These MUST work - no fallbacks
    import tidyllm
    from tidyllm import __version__
    from tidyllm.api import TidyLLMAPI
    from tidyllm.cli import main
    
    assert tidyllm is not None, "FAIL: tidyllm import failed"
    assert hasattr(tidyllm, '__version__'), "FAIL: No __version__ attribute"
    
    print("  [PASS] Core imports work")
    return True

def test_07_infrastructure_imports():
    """Test infrastructure imports work."""
    print("[TEST 07] Testing infrastructure imports...")
    
    # These MUST work - no fallbacks
    from tidyllm.infrastructure import ConfigManager
    from tidyllm.infrastructure.session import UnifiedSessionManager
    from tidyllm.infrastructure.workers import BaseWorker
    
    assert ConfigManager is not None, "FAIL: ConfigManager import failed"
    assert UnifiedSessionManager is not None, "FAIL: UnifiedSessionManager import failed"
    assert BaseWorker is not None, "FAIL: BaseWorker import failed"
    
    print("  [PASS] Infrastructure imports work")
    return True

def test_08_gateway_imports():
    """Test gateway imports work."""
    print("[TEST 08] Testing gateway imports...")
    
    # These MUST work - no fallbacks
    from tidyllm.gateways import CorporateLLMGateway
    from tidyllm.gateways import DatabaseGateway
    from tidyllm.gateways import FileStorageGateway
    from tidyllm.gateways import AIProcessingGateway
    
    assert CorporateLLMGateway is not None, "FAIL: CorporateLLMGateway import failed"
    assert DatabaseGateway is not None, "FAIL: DatabaseGateway import failed"
    
    print("  [PASS] Gateway imports work")
    return True

def test_09_knowledge_system_imports():
    """Test knowledge system imports work."""
    print("[TEST 09] Testing knowledge system imports...")
    
    # These MUST work - no fallbacks
    from tidyllm.knowledge_systems.core import DomainRAG
    from tidyllm.knowledge_systems.facades import DocumentProcessor
    
    assert DomainRAG is not None, "FAIL: DomainRAG import failed"
    assert DocumentProcessor is not None, "FAIL: DocumentProcessor import failed"
    
    print("  [PASS] Knowledge system imports work")
    return True

def test_10_no_nested_imports():
    """Test that nested tidyllm.tidyllm imports fail."""
    print("[TEST 10] Testing nested imports fail correctly...")
    
    try:
        from tidyllm.tidyllm.gateways import CorporateLLMGateway
        assert False, "FAIL: Nested import should not work but it did!"
    except ImportError:
        print("  [PASS] Nested imports correctly fail")
        return True

def run_all_tests():
    """Run all import tests."""
    print("\n" + "="*60)
    print("IMPORT SYSTEM TESTS")
    print("="*60)
    
    tests = [
        test_06_core_imports,
        test_07_infrastructure_imports,
        test_08_gateway_imports,
        test_09_knowledge_system_imports,
        test_10_no_nested_imports
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
        except ImportError as e:
            print(f"  [IMPORT ERROR] {e}")
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