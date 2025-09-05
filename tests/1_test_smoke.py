#!/usr/bin/env python3
"""
TidyLLM Strategic Test Suite: Smoke Tests
=========================================

Critical path verification - ensures core TidyLLM functionality works.
These tests must pass for system to be considered operational.
Replaces numbered import/config tests with focused critical checks.
"""

import unittest
import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCriticalImports(unittest.TestCase):
    """Test that critical TidyLLM components can be imported."""
    
    def test_tidyllm_package_import(self):
        """Test main TidyLLM package imports correctly."""
        try:
            import tidyllm
            self.assertTrue(hasattr(tidyllm, '__version__'))
            print(f"TidyLLM package imported successfully")
        except ImportError as e:
            self.fail(f"Critical import failed: {e}")
    
    def test_gateway_imports(self):
        """Test gateway module imports."""
        try:
            from tidyllm import gateways
            from tidyllm.gateways import init_gateways
            print("Gateway imports successful")
        except ImportError as e:
            self.fail(f"Gateway import failed: {e}")
    
    def test_knowledge_server_import(self):
        """Test knowledge server import (non-critical)."""
        try:
            from tidyllm.knowledge_resource_server import KnowledgeMCPServer
            print("Knowledge server import successful")
        except ImportError as e:
            print(f"Knowledge server import failed (non-critical): {e}")


class TestSystemInitialization(unittest.TestCase):
    """Test that TidyLLM system initializes correctly."""
    
    def test_gateway_registry_init(self):
        """Test gateway registry initializes without errors."""
        try:
            import tidyllm
            registry = tidyllm.init_gateways()
            self.assertIsNotNone(registry)
            
            # Should have at least some services available
            available = registry.get_available_services()
            self.assertIsInstance(available, list)
            print(f"Gateway registry initialized with {len(available)} services")
            
        except Exception as e:
            self.fail(f"Gateway registry initialization failed: {e}")
    
    def test_availability_flags(self):
        """Test TidyLLM availability flags are correctly set."""
        try:
            import tidyllm
            
            # Check availability flags exist
            self.assertTrue(hasattr(tidyllm, 'GATEWAYS_AVAILABLE'))
            self.assertTrue(hasattr(tidyllm, 'KNOWLEDGE_SERVER_AVAILABLE'))
            self.assertTrue(hasattr(tidyllm, 'KNOWLEDGE_SYSTEMS_AVAILABLE'))
            
            print(f"Availability flags - Gateways: {tidyllm.GATEWAYS_AVAILABLE}, "
                  f"Knowledge Server: {tidyllm.KNOWLEDGE_SERVER_AVAILABLE}, "
                  f"Knowledge Systems: {tidyllm.KNOWLEDGE_SYSTEMS_AVAILABLE}")
            
        except Exception as e:
            self.fail(f"Availability flags check failed: {e}")


class TestBasicFunctionality(unittest.TestCase):
    """Test basic TidyLLM functionality works."""
    
    def setUp(self):
        """Initialize system for functionality tests."""
        import tidyllm
        self.registry = tidyllm.init_gateways()
    
    def test_service_discovery(self):
        """Test service discovery works."""
        services = self.registry.get_available_services()
        print(f"Available services: {services}")
        
        # Should be able to enumerate services
        self.assertIsInstance(services, list)
        
        # Test getting service info
        for service_name in services:
            service = self.registry.get(service_name)
            # Service may be None if not configured, which is ok
            print(f"Service '{service_name}': {'Available' if service else 'Not configured'}")
    
    def test_health_check(self):
        """Test system health check functionality."""
        health = self.registry.health_check()
        
        self.assertIsInstance(health, dict)
        self.assertIn('overall_healthy', health)
        self.assertIn('services', health)
        self.assertIn('total_services', health)
        
        print(f"System health: {health['healthy_services']}/{health['total_services']} services healthy")
        print(f"Overall healthy: {health['overall_healthy']}")


class TestPerformanceCritical(unittest.TestCase):
    """Test performance-critical operations."""
    
    def test_import_performance(self):
        """Test that imports complete quickly."""
        start_time = time.time()
        
        # Re-import to test cold start
        import importlib
        if 'tidyllm' in sys.modules:
            importlib.reload(sys.modules['tidyllm'])
        else:
            import tidyllm
        
        end_time = time.time()
        import_time = end_time - start_time
        
        print(f"TidyLLM import time: {import_time:.3f} seconds")
        self.assertLess(import_time, 5.0, "Import time too slow")
    
    def test_initialization_performance(self):
        """Test that system initialization completes quickly."""
        import tidyllm
        
        start_time = time.time()
        registry = tidyllm.init_gateways()
        end_time = time.time()
        
        init_time = end_time - start_time
        print(f"System initialization time: {init_time:.3f} seconds")
        self.assertLess(init_time, 10.0, "Initialization too slow")


class TestConfigurationPaths(unittest.TestCase):
    """Test that configuration files and paths exist."""
    
    def test_package_structure(self):
        """Test basic package structure exists."""
        tidyllm_dir = project_root / "tidyllm"
        self.assertTrue(tidyllm_dir.exists(), "TidyLLM directory not found")
        self.assertTrue((tidyllm_dir / "__init__.py").exists(), "TidyLLM __init__.py not found")
        
        # Key directories should exist
        key_dirs = ["gateways", "knowledge_resource_server"]
        for dir_name in key_dirs:
            dir_path = tidyllm_dir / dir_name
            if dir_path.exists():
                print(f"[OK] Found {dir_name} directory")
            else:
                print(f"[WARN]  Missing {dir_name} directory (may be ok)")
    
    def test_admin_settings(self):
        """Test admin settings accessibility (non-critical)."""
        admin_dir = project_root / "tidyllm" / "admin"
        settings_file = admin_dir / "settings.yaml"
        
        if admin_dir.exists():
            print("[OK] Admin directory found")
            if settings_file.exists():
                print("[OK] Settings file found")
            else:
                print("[WARN]  Settings file not found (may use defaults)")
        else:
            print("[WARN]  Admin directory not found (may use defaults)")


def run_smoke_tests():
    """Run all smoke tests with detailed output."""
    print("="*60)
    print("TIDYLLM SMOKE TESTS - CRITICAL PATH VERIFICATION")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes in priority order
    test_classes = [
        TestCriticalImports,       # Must pass
        TestSystemInitialization, # Must pass
        TestBasicFunctionality,    # Should pass
        TestPerformanceCritical,   # Should pass
        TestConfigurationPaths     # Nice to have
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Detailed summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    # Show failures if any
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Determine overall status
    critical_failures = 0
    for test, _ in result.failures + result.errors:
        test_name = str(test)
        if 'TestCriticalImports' in test_name or 'TestSystemInitialization' in test_name:
            critical_failures += 1
    
    if critical_failures > 0:
        print(f"\n[CRITICAL] CRITICAL FAILURE: {critical_failures} critical tests failed")
        print("System is NOT operational")
        return False
    else:
        print("\n[OK] SMOKE TESTS PASSED")
        print("System is operational")
        return True


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)