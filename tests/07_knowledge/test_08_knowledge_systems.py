#!/usr/bin/env python3
"""
Test 08: Knowledge Systems
==========================
Test RAG and knowledge management components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_36_domain_rag_import():
    """Test DomainRAG can be imported."""
    print("[TEST 36] Testing DomainRAG import...")
    
    from tidyllm.knowledge_systems.core.domain_rag import DomainRAG
    
    assert DomainRAG is not None, "FAIL: DomainRAG import failed"
    print("  [PASS] DomainRAG imports")
    return True

def test_37_knowledge_manager():
    """Test KnowledgeManager import."""
    print("[TEST 37] Testing KnowledgeManager...")
    
    from tidyllm.knowledge_systems.core.knowledge_manager import KnowledgeManager
    
    assert KnowledgeManager is not None, "FAIL: KnowledgeManager import failed"
    assert hasattr(KnowledgeManager, '__init__'), "FAIL: KnowledgeManager not a proper class"
    
    print("  [PASS] KnowledgeManager available")
    return True

def test_38_document_processor():
    """Test DocumentProcessor facade."""
    print("[TEST 38] Testing DocumentProcessor...")
    
    from tidyllm.knowledge_systems.facades.document_processor import DocumentProcessor
    
    assert DocumentProcessor is not None, "FAIL: DocumentProcessor import failed"
    
    # Check expected methods
    expected_methods = ['process_document', 'extract_text']
    for method in expected_methods:
        assert hasattr(DocumentProcessor, method), f"FAIL: Missing method {method}"
    
    print("  [PASS] DocumentProcessor available")
    return True

def test_39_embedding_processor():
    """Test EmbeddingProcessor facade."""
    print("[TEST 39] Testing EmbeddingProcessor...")
    
    from tidyllm.knowledge_systems.facades.embedding_processor import EmbeddingProcessor
    
    assert EmbeddingProcessor is not None, "FAIL: EmbeddingProcessor import failed"
    print("  [PASS] EmbeddingProcessor available")
    return True

def test_40_vector_storage():
    """Test VectorStorage facade."""
    print("[TEST 40] Testing VectorStorage...")
    
    from tidyllm.knowledge_systems.facades.vector_storage import VectorStorage
    
    assert VectorStorage is not None, "FAIL: VectorStorage import failed"
    
    # Check expected methods
    assert hasattr(VectorStorage, 'store_vectors'), "FAIL: Missing store_vectors method"
    assert hasattr(VectorStorage, 'search'), "FAIL: Missing search method"
    
    print("  [PASS] VectorStorage available")
    return True

def run_all_tests():
    """Run all knowledge system tests."""
    print("\n" + "="*60)
    print("KNOWLEDGE SYSTEM TESTS")
    print("="*60)
    
    tests = [
        test_36_domain_rag_import,
        test_37_knowledge_manager,
        test_38_document_processor,
        test_39_embedding_processor,
        test_40_vector_storage
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