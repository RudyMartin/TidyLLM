#!/usr/bin/env python3
"""
TidyLLM Strategic Test Suite #4: Cross-System Integration
========================================================

Tests integration between different TidyLLM components.
Replaces multiple integration/cross-service test files.
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import tidyllm
    TIDYLLM_AVAILABLE = True
except ImportError:
    TIDYLLM_AVAILABLE = False


class TestGatewayIntegration(unittest.TestCase):
    """Test integration between different gateways."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize gateway registry for integration tests."""
        if TIDYLLM_AVAILABLE:
            cls.registry = tidyllm.init_gateways()
        else:
            cls.registry = None
    
    def test_gateway_to_gateway_communication(self):
        """Test communication between gateways."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        available_services = self.registry.get_available_services()
        print(f"Available services for integration: {available_services}")
        
        # Test that gateways can discover each other
        if len(available_services) >= 2:
            service1 = self.registry.get(available_services[0])
            service2 = self.registry.get(available_services[1])
            
            self.assertIsNotNone(service1)
            self.assertIsNotNone(service2)
            print(f"[OK] Inter-gateway discovery working")
        else:
            print("[WARN] Not enough services for inter-gateway testing")
    
    def test_knowledge_server_gateway_integration(self):
        """Test knowledge server integration with other gateways."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        knowledge_server = self.registry.get('knowledge_resources')
        ai_gateway = self.registry.get('ai_processing')
        
        if knowledge_server and ai_gateway:
            print("[OK] Knowledge server and AI gateway both available")
            
            # Test that AI gateway can potentially use knowledge server
            self.assertTrue(hasattr(knowledge_server, 'handle_mcp_tool_call'))
            print("[OK] Knowledge server exposes MCP interface")
            
        elif knowledge_server:
            print("[OK] Knowledge server available (AI gateway not configured)")
        elif ai_gateway:
            print("[OK] AI gateway available (Knowledge server not configured)")
        else:
            print("[WARN] Neither knowledge server nor AI gateway configured")


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration system integration."""
    
    def test_settings_accessibility(self):
        """Test that configuration is accessible across components."""
        # Check for settings files in expected locations
        settings_locations = [
            project_root / "admin" / "settings.yaml",
            project_root / "tidyllm" / "admin" / "settings.yaml"
        ]
        
        settings_found = []
        for location in settings_locations:
            if location.exists():
                settings_found.append(str(location))
        
        if settings_found:
            print(f"[OK] Settings files found: {settings_found}")
        else:
            print("[WARN] No settings files found (using defaults)")
    
    def test_environment_integration(self):
        """Test environment variable integration."""
        # Check for common environment variables
        env_vars = ['AWS_PROFILE', 'AWS_REGION', 'PYTHONPATH']
        
        configured_vars = {}
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                configured_vars[var] = value
        
        if configured_vars:
            print(f"[OK] Environment integration: {list(configured_vars.keys())}")
        else:
            print("[WARN] No relevant environment variables configured")


class TestDataFlowIntegration(unittest.TestCase):
    """Test data flow between components."""
    
    def test_registry_health_integration(self):
        """Test that health checks work across all components."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        registry = tidyllm.init_gateways()
        health_report = registry.health_check()
        
        self.assertIsInstance(health_report, dict)
        self.assertIn('services', health_report)
        self.assertIn('overall_healthy', health_report)
        
        services = health_report['services']
        healthy_count = sum(1 for s in services.values() if s.get('healthy', False))
        total_count = len(services)
        
        print(f"[OK] Health check integration: {healthy_count}/{total_count} services healthy")
        
        if total_count > 0:
            health_percentage = (healthy_count / total_count) * 100
            print(f"System health: {health_percentage:.1f}%")
    
    def test_service_discovery_integration(self):
        """Test service discovery across the system."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        registry = tidyllm.init_gateways()
        
        # Test service enumeration
        all_services = registry.list_services()
        available_services = registry.get_available_services()
        
        print(f"[OK] Service discovery: {len(all_services)} total, {len(available_services)} available")
        
        # Test service info retrieval
        info_count = 0
        for service_name in available_services:
            info = registry.get_service_info(service_name)
            if info:
                info_count += 1
        
        print(f"[OK] Service info retrieval: {info_count}/{len(available_services)} services")


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling across integrated components."""
    
    def test_graceful_service_failure(self):
        """Test system behavior when services fail."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        registry = tidyllm.init_gateways()
        
        # Test getting non-existent service
        fake_service = registry.get('nonexistent_service')
        self.assertIsNone(fake_service)
        print("[OK] Graceful handling of non-existent service")
        
        # Test service info for non-existent service
        fake_info = registry.get_service_info('nonexistent_service')
        self.assertIsNone(fake_info)
        print("[OK] Graceful handling of non-existent service info")
    
    def test_partial_system_operation(self):
        """Test that system works with partially configured services."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        registry = tidyllm.init_gateways()
        available_services = registry.get_available_services()
        
        # System should work even if not all services are available
        self.assertIsInstance(available_services, list)
        
        if len(available_services) > 0:
            print(f"[OK] Partial system operation: {len(available_services)} services running")
        else:
            print("[WARN] No services available (all require configuration)")


def run_integration_tests():
    """Run all integration tests."""
    print("="*60)
    print("TIDYLLM INTEGRATION TESTS")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGatewayIntegration,
        TestConfigurationIntegration,
        TestDataFlowIntegration,
        TestErrorHandlingIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"Overall status: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)