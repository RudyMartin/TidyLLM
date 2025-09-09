#!/usr/bin/env python3
"""
Test 03: Configuration Management
=================================
Test configuration system works without AWS or external dependencies.
"""

import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_11_config_manager_import():
    """Test ConfigManager can be imported."""
    print("[TEST 11] Testing ConfigManager import...")
    
    from tidyllm.infrastructure.config import ConfigManager
    
    assert ConfigManager is not None, "FAIL: ConfigManager import failed"
    print("  [PASS] ConfigManager imports")
    return True

def test_12_config_initialization():
    """Test ConfigManager can be initialized."""
    print("[TEST 12] Testing ConfigManager initialization...")
    
    from tidyllm.infrastructure.config import ConfigManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        
        # Create test config
        test_config = {
            "test_key": "test_value",
            "nested": {
                "key": "value"
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Initialize manager
        manager = ConfigManager(str(config_path))
        
        assert manager is not None, "FAIL: ConfigManager initialization failed"
        assert manager.get("test_key") == "test_value", "FAIL: Config value retrieval failed"
        assert manager.get("nested.key") == "value", "FAIL: Nested config retrieval failed"
    
    print("  [PASS] ConfigManager works")
    return True

def test_13_session_manager_import():
    """Test UnifiedSessionManager can be imported."""
    print("[TEST 13] Testing UnifiedSessionManager import...")
    
    from tidyllm.infrastructure.session import UnifiedSessionManager
    
    assert UnifiedSessionManager is not None, "FAIL: UnifiedSessionManager import failed"
    print("  [PASS] UnifiedSessionManager imports")
    return True

def test_14_worker_base_class():
    """Test BaseWorker can be imported and instantiated."""
    print("[TEST 14] Testing BaseWorker class...")
    
    from tidyllm.infrastructure.workers import BaseWorker
    
    assert BaseWorker is not None, "FAIL: BaseWorker import failed"
    
    # Check it's a class we can inspect
    assert hasattr(BaseWorker, '__init__'), "FAIL: BaseWorker is not a proper class"
    
    print("  [PASS] BaseWorker class available")
    return True

def test_15_api_infrastructure():
    """Test API infrastructure components."""
    print("[TEST 15] Testing API infrastructure...")
    
    from tidyllm.infrastructure.api import GatewayController
    
    assert GatewayController is not None, "FAIL: GatewayController import failed"
    
    print("  [PASS] API infrastructure available")
    return True

def run_all_tests():
    """Run all infrastructure tests."""
    print("\n" + "="*60)
    print("INFRASTRUCTURE TESTS")
    print("="*60)
    
    tests = [
        test_11_config_manager_import,
        test_12_config_initialization,
        test_13_session_manager_import,
        test_14_worker_base_class,
        test_15_api_infrastructure
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