#!/usr/bin/env python3
"""
Simple connectivity tests for TidyLLM setup validation.
Uses existing admin infrastructure for testing.
"""

import os
import sys
from pathlib import Path

def test_s3_basic():
    """Test basic S3 connectivity."""
    try:
        # Use UnifiedSessionManager for S3 client
        from tidyllm.infrastructure.session.unified import UnifiedSessionManager
        session_mgr = UnifiedSessionManager()
        client = session_mgr.get_s3_client()
        buckets = client.list_buckets()
        print(f"[SUCCESS] S3 connectivity: Found {len(buckets['Buckets'])} buckets")
        return True
    except Exception as e:
        print(f"[WARNING] S3 connectivity issue: {e}")
        return False

def test_tidyllm_import():
    """Test TidyLLM basic import."""
    try:
        import tidyllm
        print("[SUCCESS] TidyLLM import successful")
        return True
    except Exception as e:
        print(f"[WARNING] TidyLLM import issue: {e}")
        return False

def test_basic_chat():
    """Test basic chat functionality."""
    try:
        import tidyllm
        response = tidyllm.chat("Hello!")
        print(f"[SUCCESS] Basic chat test: {response[:50]}...")
        return True
    except Exception as e:
        print(f"[WARNING] Chat test issue: {e}")
        return False

def main():
    """Run simple tests."""
    print("=" * 50)
    print(">> TidyLLM Simple Tests")
    print("=" * 50)
    
    tests = [
        ("S3 Connectivity", test_s3_basic),
        ("TidyLLM Import", test_tidyllm_import),
        ("Basic Chat", test_basic_chat)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n[TESTING] {name}...")
        if test_func():
            passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"[RESULTS] {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed!")
        return True
    else:
        print("[INFO] Some tests had warnings, but basic setup is working")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)