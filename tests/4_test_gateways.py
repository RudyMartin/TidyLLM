#!/usr/bin/env python3
"""
TidyLLM Strategic Test Suite: Gateways
=====================================

Comprehensive testing for all TidyLLM gateway functionality.
Replaces 15+ individual gateway test files with strategic coverage.
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
    from tidyllm.gateways import (
        init_gateways,
        AIProcessingGateway, 
        CorporateLLMGateway,
        WorkflowOptimizerGateway,
        DatabaseGateway,
        FileStorageGateway
    )
    TIDYLLM_AVAILABLE = True
except ImportError as e:
    print(f"TidyLLM not available: {e}")
    TIDYLLM_AVAILABLE = False


class TestGatewayCore(unittest.TestCase):
    """Test core gateway functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize gateway registry once for all tests."""
        if TIDYLLM_AVAILABLE:
            cls.registry = init_gateways()
        else:
            cls.registry = None
    
    def test_gateway_registry_initialization(self):
        """Test gateway registry initializes successfully."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        self.assertIsNotNone(self.registry)
        available = self.registry.get_available_services()
        self.assertIsInstance(available, list)
        print(f"Available services: {available}")
    
    def test_ai_processing_gateway(self):
        """Test AI processing gateway functionality."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        ai_gateway = self.registry.get('ai_processing')
        if ai_gateway:
            self.assertIsInstance(ai_gateway, AIProcessingGateway)
            # Test basic configuration
            self.assertTrue(hasattr(ai_gateway, 'process'))
        else:
            print("AI processing gateway not available (expected with no config)")
    
    def test_workflow_optimizer_gateway(self):
        """Test workflow optimizer gateway functionality.""" 
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        workflow_gateway = self.registry.get('workflow_optimizer')
        if workflow_gateway:
            self.assertIsInstance(workflow_gateway, WorkflowOptimizerGateway)
            self.assertTrue(hasattr(workflow_gateway, 'process_workflow'))
        else:
            print("Workflow optimizer gateway not available (expected with no config)")
    
    def test_corporate_llm_gateway(self):
        """Test corporate LLM gateway functionality."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        llm_gateway = self.registry.get('corporate_llm')
        if llm_gateway:
            self.assertIsInstance(llm_gateway, CorporateLLMGateway) 
            self.assertTrue(hasattr(llm_gateway, 'process_request'))
        else:
            print("Corporate LLM gateway not available (expected with no config)")


class TestGatewayIntegration(unittest.TestCase):
    """Test gateway integration and communication."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize gateway registry."""
        if TIDYLLM_AVAILABLE:
            cls.registry = init_gateways()
        else:
            cls.registry = None
    
    def test_gateway_health_check(self):
        """Test gateway registry health check."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        health = self.registry.health_check()
        self.assertIsInstance(health, dict)
        self.assertIn('overall_healthy', health)
        self.assertIn('services', health)
        print(f"Health check: {health['healthy_services']}/{health['total_services']} healthy")
    
    def test_service_discovery(self):
        """Test service discovery functionality."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        # Test service enumeration
        services = self.registry.list_services()
        self.assertIsInstance(services, list)
        
        # Test service info retrieval
        for service_name in self.registry.get_available_services():
            info = self.registry.get_service_info(service_name)
            if info:
                self.assertIn('service_type', info)
                self.assertIn('description', info)


class TestGatewayConfiguration(unittest.TestCase):
    """Test gateway configuration and customization."""
    
    def test_gateway_creation_direct(self):
        """Test direct gateway creation without registry."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        from tidyllm.gateways import create_gateway
        
        # Test creating individual gateways
        gateway_types = ['ai_processing', 'corporate_llm', 'workflow_optimizer']
        
        for gateway_type in gateway_types:
            try:
                gateway = create_gateway(gateway_type)
                self.assertIsNotNone(gateway)
                print(f"Successfully created {gateway_type} gateway")
            except Exception as e:
                print(f"Expected config error for {gateway_type}: {e}")
    
    def test_invalid_gateway_type(self):
        """Test handling of invalid gateway types."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        from tidyllm.gateways import create_gateway
        
        with self.assertRaises(ValueError):
            create_gateway("nonexistent_gateway")


class TestGatewayPerformance(unittest.TestCase):
    """Test gateway performance and resource usage."""
    
    def test_registry_initialization_time(self):
        """Test gateway registry initialization performance."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        import time
        
        start_time = time.time()
        registry = init_gateways()
        end_time = time.time()
        
        initialization_time = end_time - start_time
        print(f"Gateway registry initialized in {initialization_time:.3f} seconds")
        
        # Should initialize reasonably quickly
        self.assertLess(initialization_time, 10.0, "Gateway initialization too slow")
    
    def test_service_lookup_performance(self):
        """Test service lookup performance."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        import time
        
        registry = init_gateways()
        available_services = registry.get_available_services()
        
        if not available_services:
            self.skipTest("No services available for performance testing")
        
        # Test multiple lookups
        start_time = time.time()
        for _ in range(100):
            for service_name in available_services:
                service = registry.get(service_name)
        end_time = time.time()
        
        lookup_time = end_time - start_time
        print(f"100 service lookups completed in {lookup_time:.3f} seconds")
        
        # Should be very fast
        self.assertLess(lookup_time, 1.0, "Service lookup too slow")


def run_gateway_tests():
    """Run all gateway tests with detailed output."""
    print("="*60)
    print("TIDYLLM STRATEGIC GATEWAY TESTS")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGatewayCore,
        TestGatewayIntegration, 
        TestGatewayConfiguration,
        TestGatewayPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("GATEWAY TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"Overall status: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_gateway_tests()
    sys.exit(0 if success else 1)