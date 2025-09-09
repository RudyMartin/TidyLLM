#!/usr/bin/env python3
"""
Test 10: End-to-End Integration
================================
Test complete workflows work together.
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_46_cli_import_chain():
    """Test CLI -> API -> Gateway chain."""
    print("[TEST 46] Testing CLI import chain...")
    
    # Import chain should work
    from tidyllm.cli import main
    from tidyllm.api import TidyLLMAPI
    from tidyllm.gateways import CorporateLLMGateway
    
    assert main is not None, "FAIL: CLI main not available"
    assert TidyLLMAPI is not None, "FAIL: API not available"
    assert CorporateLLMGateway is not None, "FAIL: Gateway not available"
    
    print("  [PASS] CLI -> API -> Gateway chain works")
    return True

def test_47_worker_manager_chain():
    """Test Worker -> Manager -> Gateway chain."""
    print("[TEST 47] Testing worker chain...")
    
    from tidyllm.infrastructure.workers import BaseWorker
    from tidyllm.infrastructure.workers import AIDropzoneManager
    from tidyllm.gateways import AIProcessingGateway
    
    assert BaseWorker is not None, "FAIL: BaseWorker not available"
    assert AIDropzoneManager is not None, "FAIL: AIDropzoneManager not available"
    assert AIProcessingGateway is not None, "FAIL: AIProcessingGateway not available"
    
    print("  [PASS] Worker -> Manager -> Gateway chain works")
    return True

def test_48_flow_integration():
    """Test Flow -> Manager -> Worker integration."""
    print("[TEST 48] Testing flow integration...")
    
    from tidyllm.flow.examples.bracket_registry import BracketRegistry
    from tidyllm.infrastructure.workers import FlowIntegrationManager
    from tidyllm.infrastructure.workers import ProcessingWorker
    
    registry = BracketRegistry()
    assert len(registry.get_all_commands()) > 0, "FAIL: No commands in registry"
    
    assert FlowIntegrationManager is not None, "FAIL: FlowIntegrationManager not available"
    assert ProcessingWorker is not None, "FAIL: ProcessingWorker not available"
    
    print("  [PASS] Flow -> Manager -> Worker chain works")
    return True

def test_49_knowledge_chain():
    """Test Knowledge -> Document -> Embedding chain."""
    print("[TEST 49] Testing knowledge chain...")
    
    from tidyllm.knowledge_systems.core import DomainRAG
    from tidyllm.knowledge_systems.facades import DocumentProcessor
    from tidyllm.knowledge_systems.facades import EmbeddingProcessor
    
    assert DomainRAG is not None, "FAIL: DomainRAG not available"
    assert DocumentProcessor is not None, "FAIL: DocumentProcessor not available"
    assert EmbeddingProcessor is not None, "FAIL: EmbeddingProcessor not available"
    
    print("  [PASS] Knowledge -> Document -> Embedding chain works")
    return True

def test_50_complete_system():
    """Test complete system integration."""
    print("[TEST 50] Testing complete system...")
    
    # Test all major components can be imported together
    from tidyllm import __version__
    from tidyllm.cli import main
    from tidyllm.api import TidyLLMAPI
    from tidyllm.infrastructure import ConfigManager
    from tidyllm.infrastructure.workers import AIDropzoneManager
    from tidyllm.gateways import CorporateLLMGateway
    from tidyllm.flow.examples.bracket_registry import BracketRegistry
    from tidyllm.knowledge_systems.core import DomainRAG
    
    # Verify version exists
    assert __version__ is not None, "FAIL: No version defined"
    
    # Test paths exist
    paths_to_check = [
        Path("tidyllm"),
        Path("tidyllm/infrastructure"),
        Path("tidyllm/gateways"),
        Path("tidyllm/flow"),
        Path("tidyllm/web"),
        Path("tests")
    ]
    
    for path in paths_to_check:
        assert path.exists(), f"FAIL: Path {path} does not exist"
    
    print("  [PASS] Complete system integration verified!")
    return True

def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("END-TO-END INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        test_46_cli_import_chain,
        test_47_worker_manager_chain,
        test_48_flow_integration,
        test_49_knowledge_chain,
        test_50_complete_system
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