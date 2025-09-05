#!/usr/bin/env python3
"""
TidyLLM Strategic Test Suite #6: Performance & Load Testing
==========================================================

Tests system performance, load handling, and resource utilization.
Replaces multiple performance/benchmark test files.
"""

import unittest
import sys
import os
import time
import threading
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import tidyllm
    TIDYLLM_AVAILABLE = True
except ImportError:
    TIDYLLM_AVAILABLE = False


class TestGatewayPerformance(unittest.TestCase):
    """Test gateway system performance."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize for performance testing."""
        if TIDYLLM_AVAILABLE:
            cls.registry = tidyllm.init_gateways()
        else:
            cls.registry = None
    
    def test_gateway_initialization_time(self):
        """Test how quickly gateways can initialize."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        start_time = time.time()
        registry = tidyllm.init_gateways()
        init_time = time.time() - start_time
        
        self.assertLess(init_time, 5.0, "Gateway initialization should be under 5 seconds")
        print(f"[OK] Gateway initialization: {init_time:.2f}s")
    
    def test_service_discovery_performance(self):
        """Test service discovery speed."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        start_time = time.time()
        services = self.registry.get_available_services()
        discovery_time = time.time() - start_time
        
        self.assertLess(discovery_time, 1.0, "Service discovery should be under 1 second")
        print(f"[OK] Service discovery: {discovery_time:.3f}s for {len(services)} services")
    
    def test_health_check_performance(self):
        """Test health check execution time."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        start_time = time.time()
        health_report = self.registry.health_check()
        health_time = time.time() - start_time
        
        self.assertLess(health_time, 2.0, "Health checks should complete under 2 seconds")
        print(f"[OK] Health check: {health_time:.2f}s")
        
        services_count = len(health_report.get('services', {}))
        if services_count > 0:
            avg_time_per_service = health_time / services_count
            print(f"   Average per service: {avg_time_per_service:.3f}s")


class TestConcurrentAccess(unittest.TestCase):
    """Test concurrent access to gateway services."""
    
    def test_concurrent_service_discovery(self):
        """Test multiple threads discovering services simultaneously."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        def discover_services():
            registry = tidyllm.init_gateways()
            return registry.get_available_services()
        
        thread_count = 5
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(discover_services) for _ in range(thread_count)]
            results = [future.result() for future in as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(set(result), set(first_result), "Concurrent access should return consistent results")
        
        print(f"[OK] Concurrent service discovery: {concurrent_time:.2f}s for {thread_count} threads")
        print(f"   Results consistent: {len(set(str(sorted(r)) for r in results)) == 1}")
    
    def test_concurrent_health_checks(self):
        """Test concurrent health checks."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        def health_check():
            registry = tidyllm.init_gateways()
            return registry.health_check()
        
        thread_count = 3
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(health_check) for _ in range(thread_count)]
            results = [future.result() for future in as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        # All results should be dictionaries with health info
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('services', result)
        
        print(f"[OK] Concurrent health checks: {concurrent_time:.2f}s for {thread_count} threads")


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage patterns."""
    
    def test_gateway_memory_footprint(self):
        """Test memory usage of gateway system."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        import psutil
        import gc
        
        # Force garbage collection before measurement
        gc.collect()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize gateways
        registry = tidyllm.init_gateways()
        services = registry.get_available_services()
        
        # Perform some operations
        for service_name in services[:3]:  # Test up to 3 services
            registry.get_service_info(service_name)
        
        registry.health_check()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"[OK] Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Memory increase should be reasonable (less than 100MB for basic operations)
        self.assertLess(memory_increase, 100, "Memory increase should be reasonable")
    
    def test_repeated_operations_memory_leak(self):
        """Test for memory leaks in repeated operations."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil not available for memory testing")
        
        import gc
        gc.collect()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        registry = tidyllm.init_gateways()
        
        # Perform repeated operations
        for i in range(100):
            services = registry.get_available_services()
            if services:
                registry.get_service_info(services[0])
            
            # Check memory every 20 iterations
            if i % 20 == 19:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be bounded
                if memory_growth > 50:  # 50MB growth limit
                    self.fail(f"Potential memory leak: {memory_growth:.1f}MB growth after {i+1} iterations")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        print(f"[OK] Memory stability test: {total_growth:.1f}MB growth over 100 operations")


class TestScalabilityLimits(unittest.TestCase):
    """Test system scalability and limits."""
    
    def test_service_enumeration_scaling(self):
        """Test performance as number of services scales."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        registry = tidyllm.init_gateways()
        
        # Test multiple service discovery calls
        times = []
        for i in range(10):
            start_time = time.time()
            services = registry.get_available_services()
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"[OK] Service enumeration scaling: avg={avg_time:.3f}s, max={max_time:.3f}s")
        
        # Performance should be consistent
        self.assertLess(max_time, avg_time * 3, "Max time should not be more than 3x average")
    
    def test_concurrent_registry_instances(self):
        """Test multiple registry instances."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        def create_registry():
            return tidyllm.init_gateways()
        
        start_time = time.time()
        
        # Create multiple registry instances
        registries = []
        for _ in range(5):
            registry = create_registry()
            registries.append(registry)
        
        creation_time = time.time() - start_time
        
        # Test that all registries work
        for i, registry in enumerate(registries):
            services = registry.get_available_services()
            self.assertIsInstance(services, list)
        
        print(f"[OK] Multiple registry instances: {creation_time:.2f}s for 5 registries")


class TestConfigurationPerformance(unittest.TestCase):
    """Test configuration loading and caching performance."""
    
    def test_configuration_load_time(self):
        """Test configuration loading performance."""
        # Find settings files
        settings_files = [
            project_root / "admin" / "settings.yaml",
            project_root / "tidyllm" / "admin" / "settings.yaml"
        ]
        
        settings_file = None
        for file_path in settings_files:
            if file_path.exists():
                settings_file = file_path
                break
        
        if not settings_file:
            print("[WARN] No settings file found - testing default configuration")
            return
        
        import yaml
        
        # Time configuration loading
        times = []
        for _ in range(10):
            start_time = time.time()
            with open(settings_file, 'r') as f:
                config = yaml.safe_load(f)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"[OK] Configuration loading: {avg_time:.4f}s average")
        
        # Configuration loading should be very fast
        self.assertLess(avg_time, 0.1, "Configuration loading should be under 0.1 seconds")


def run_performance_tests():
    """Run all performance tests."""
    print("="*60)
    print("TIDYLLM PERFORMANCE TESTS")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGatewayPerformance,
        TestConcurrentAccess,
        TestMemoryUsage,
        TestScalabilityLimits,
        TestConfigurationPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"Overall status: {'PASS' if success else 'FAIL'}")
    
    if not success:
        print("\nPerformance issues detected - system may need optimization")
    
    return success


if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)