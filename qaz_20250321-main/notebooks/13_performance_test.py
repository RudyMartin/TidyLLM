#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Performance Testing Script

Tests the performance impact of logging on the SME Context Integration system.
Compares processing speed with different logging configurations to measure latency impact.

Key Features:
- Tests 3 scenarios with logging disabled
- Tests 3 scenarios with full logging enabled
- Measures precise timing for each operation
- Provides detailed performance metrics
- Generates recommendations for production use

Usage:
    python3 notebooks/13_performance_test.py
"""

import os
import sys
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime

# Add src to path for backend imports
sys.path.insert(0, '../src')

try:
    from src.backend.mcp.coordinators.sme_context_coordinator import SMEContextCoordinator
    from src.backend.mcp.config.debug_config import DebugConfig
    SME_AVAILABLE = True
except ImportError:
    print("⚠️ SME Context Coordinator not available - will use simulated testing")
    SME_AVAILABLE = False
    # Define placeholder classes for type hints
    class SMEContextCoordinator:
        pass
    class DebugConfig:
        @staticmethod
        def performance_test():
            return None
        @staticmethod  
        def development():
            return None


class PerformanceTester:
    """Performance testing for SME Context Integration with different logging configurations."""
    
    def __init__(self):
        self.test_scenarios = [
            {
                "name": "Simple Risk Analysis",
                "risk_event": "Model Performance Degradation",
                "risk_category": "Model Risk",
                "description": "Basic risk analysis with single SME context"
            },
            {
                "name": "Complex Multi-Category Analysis", 
                "risk_event": "Portfolio Risk Concentration",
                "risk_category": "Credit Risk",
                "description": "Complex analysis requiring multiple data sources"
            },
            {
                "name": "High-Frequency Operation",
                "risk_event": "Real-time Fraud Detection",
                "risk_category": "Operational Risk", 
                "description": "Rapid-fire analysis simulating high-frequency usage"
            }
        ]
        
        self.results = {
            "no_logging": {},
            "full_logging": {}
        }
    
    def run_single_test(self, coordinator: SMEContextCoordinator, scenario: Dict[str, Any], 
                       iterations: int = 10) -> Dict[str, Any]:
        """Run a single test scenario multiple times and collect timing data."""
        
        durations = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            # Perform the SME analysis
            if SME_AVAILABLE:
                result = coordinator.analyze_with_sme_context(
                    scenario["risk_event"],
                    scenario["risk_category"]
                )
            else:
                # Simulate analysis work
                time.sleep(0.001)  # Simulate 1ms of work
                result = {"status": "simulated", "risk_score": 5}
            
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            durations.append(duration)
        
        return {
            "scenario": scenario["name"],
            "iterations": iterations,
            "durations_ms": durations,
            "avg_duration_ms": statistics.mean(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "median_duration_ms": statistics.median(durations),
            "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "total_time_ms": sum(durations)
        }
    
    def run_batch_test(self, coordinator: SMEContextCoordinator, scenario: Dict[str, Any],
                      batch_size: int = 100) -> Dict[str, Any]:
        """Run a batch of operations to test sustained performance."""
        
        start_time = time.perf_counter()
        
        for i in range(batch_size):
            if SME_AVAILABLE:
                result = coordinator.analyze_with_sme_context(
                    f"{scenario['risk_event']} #{i+1}",
                    scenario["risk_category"]
                )
            else:
                # Simulate batch processing
                time.sleep(0.0001)  # Simulate 0.1ms per operation
        
        end_time = time.perf_counter()
        total_duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return {
            "scenario": scenario["name"],
            "batch_size": batch_size,
            "total_duration_ms": total_duration,
            "avg_per_operation_ms": total_duration / batch_size,
            "operations_per_second": batch_size / (total_duration / 1000)
        }
    
    def test_no_logging_configuration(self):
        """Test performance with logging disabled."""
        print("\n🚫 TESTING: No Logging Configuration")
        print("="*60)
        
        if SME_AVAILABLE:
            # Create coordinator with no logging
            config = DebugConfig.performance_test()
            coordinator = SMEContextCoordinator(debug_config=config)
            print(f"✅ Using real SME coordinator with config: {config}")
        else:
            coordinator = None
            print("⚠️ Using simulated coordinator")
        
        results = {}
        
        for scenario in self.test_scenarios:
            print(f"\n📋 Testing: {scenario['name']}")
            
            # Single operation test
            single_result = self.run_single_test(coordinator, scenario, iterations=50)
            print(f"   Average: {single_result['avg_duration_ms']:.3f}ms")
            print(f"   Range: {single_result['min_duration_ms']:.3f}ms - {single_result['max_duration_ms']:.3f}ms")
            
            # Batch operation test
            batch_result = self.run_batch_test(coordinator, scenario, batch_size=100)
            print(f"   Batch avg: {batch_result['avg_per_operation_ms']:.3f}ms per operation")
            print(f"   Throughput: {batch_result['operations_per_second']:.1f} ops/sec")
            
            results[scenario['name']] = {
                "single": single_result,
                "batch": batch_result
            }
        
        self.results["no_logging"] = results
        return results
    
    def test_full_logging_configuration(self):
        """Test performance with full logging enabled."""
        print("\n📝 TESTING: Full Logging Configuration")
        print("="*60)
        
        if SME_AVAILABLE:
            # Create coordinator with full logging
            config = DebugConfig.development()  # debug_full=True
            coordinator = SMEContextCoordinator(debug_config=config)
            print(f"✅ Using real SME coordinator with config: {config}")
        else:
            coordinator = None
            print("⚠️ Using simulated coordinator")
        
        results = {}
        
        for scenario in self.test_scenarios:
            print(f"\n📋 Testing: {scenario['name']}")
            
            # Single operation test
            single_result = self.run_single_test(coordinator, scenario, iterations=50)
            print(f"   Average: {single_result['avg_duration_ms']:.3f}ms")
            print(f"   Range: {single_result['min_duration_ms']:.3f}ms - {single_result['max_duration_ms']:.3f}ms")
            
            # Batch operation test
            batch_result = self.run_batch_test(coordinator, scenario, batch_size=100)
            print(f"   Batch avg: {batch_result['avg_per_operation_ms']:.3f}ms per operation")
            print(f"   Throughput: {batch_result['operations_per_second']:.1f} ops/sec")
            
            results[scenario['name']] = {
                "single": single_result,
                "batch": batch_result
            }
        
        self.results["full_logging"] = results
        return results
    
    def compare_results(self):
        """Compare performance between no logging and full logging."""
        print("\n📊 PERFORMANCE COMPARISON")
        print("="*80)
        
        if not self.results["no_logging"] or not self.results["full_logging"]:
            print("⚠️ Missing test results - cannot perform comparison")
            return
        
        print(f"{'Scenario':<30} {'No Logging (ms)':<15} {'Full Logging (ms)':<16} {'Overhead':<12} {'Impact'}")
        print("-" * 80)
        
        total_overhead = []
        
        for scenario_name in self.test_scenarios:
            scenario_display_name = scenario_name["name"]
            
            if scenario_display_name in self.results["no_logging"] and scenario_display_name in self.results["full_logging"]:
                no_log_avg = self.results["no_logging"][scenario_display_name]["single"]["avg_duration_ms"]
                full_log_avg = self.results["full_logging"][scenario_display_name]["single"]["avg_duration_ms"]
                
                overhead = full_log_avg - no_log_avg
                overhead_percent = (overhead / no_log_avg) * 100 if no_log_avg > 0 else 0
                
                total_overhead.append(overhead_percent)
                
                impact_level = "Low" if overhead_percent < 10 else "Medium" if overhead_percent < 25 else "High"
                
                print(f"{scenario_display_name:<30} {no_log_avg:<15.3f} {full_log_avg:<16.3f} {overhead:<12.3f} {impact_level} ({overhead_percent:.1f}%)")
        
        if total_overhead:
            avg_overhead = statistics.mean(total_overhead)
            print("-" * 80)
            print(f"{'AVERAGE OVERHEAD:':<30} {'':<15} {'':<16} {'':<12} {avg_overhead:.1f}%")
        
        # Throughput comparison
        print(f"\n📈 THROUGHPUT COMPARISON")
        print(f"{'Scenario':<30} {'No Logging (ops/s)':<18} {'Full Logging (ops/s)':<19} {'Reduction'}")
        print("-" * 80)
        
        for scenario_name in self.test_scenarios:
            scenario_display_name = scenario_name["name"]
            
            if scenario_display_name in self.results["no_logging"] and scenario_display_name in self.results["full_logging"]:
                no_log_ops = self.results["no_logging"][scenario_display_name]["batch"]["operations_per_second"]
                full_log_ops = self.results["full_logging"][scenario_display_name]["batch"]["operations_per_second"]
                
                reduction = ((no_log_ops - full_log_ops) / no_log_ops) * 100 if no_log_ops > 0 else 0
                
                print(f"{scenario_display_name:<30} {no_log_ops:<18.1f} {full_log_ops:<19.1f} {reduction:.1f}%")
    
    def generate_recommendations(self):
        """Generate performance recommendations based on test results."""
        print(f"\n💡 PERFORMANCE RECOMMENDATIONS")
        print("="*80)
        
        if not self.results["no_logging"] or not self.results["full_logging"]:
            print("⚠️ Insufficient data for recommendations")
            return
        
        # Calculate average overhead
        overheads = []
        for scenario_name in self.test_scenarios:
            scenario_display_name = scenario_name["name"]
            if scenario_display_name in self.results["no_logging"] and scenario_display_name in self.results["full_logging"]:
                no_log_avg = self.results["no_logging"][scenario_display_name]["single"]["avg_duration_ms"]
                full_log_avg = self.results["full_logging"][scenario_display_name]["single"]["avg_duration_ms"]
                overhead_percent = ((full_log_avg - no_log_avg) / no_log_avg) * 100 if no_log_avg > 0 else 0
                overheads.append(overhead_percent)
        
        if overheads:
            avg_overhead = statistics.mean(overheads)
            
            print(f"📊 Average Logging Overhead: {avg_overhead:.1f}%")
            print()
            
            if avg_overhead < 5:
                print("🟢 LOW IMPACT: Logging overhead is minimal")
                print("   ✅ Safe to use full logging in production")
                print("   ✅ No performance optimizations needed")
            elif avg_overhead < 15:
                print("🟡 MEDIUM IMPACT: Logging has moderate overhead")
                print("   ⚠️ Consider selective logging in high-traffic scenarios")
                print("   💡 Use production config (DebugConfig.production())")
                print("   💡 Disable detailed context logging")
            else:
                print("🔴 HIGH IMPACT: Logging significantly affects performance")
                print("   ⚠️ Use minimal logging in production")
                print("   💡 Enable only critical error logging")
                print("   💡 Consider asynchronous logging")
                print("   💡 Use compact JSON format (no pretty printing)")
        
        print(f"\n🔧 CONFIGURATION RECOMMENDATIONS:")
        print(f"   Production: Use DebugConfig.production()")
        print(f"   Development: Use DebugConfig.development()")
        print(f"   Performance Testing: Use DebugConfig.performance_test()")
        print(f"   Custom: Set specific log types based on needs")
        
        print(f"\n⚡ OPTIMIZATION TIPS:")
        print(f"   • Disable JSON pretty printing in production")
        print(f"   • Use selective logging (only critical events)")
        print(f"   • Consider log rotation for long-running systems")
        print(f"   • Monitor disk I/O impact in high-volume scenarios")
    
    def export_results(self, filename: str = "performance_test_results.json"):
        """Export test results to JSON file."""
        import json
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "test_configuration": {
                "scenarios_tested": len(self.test_scenarios),
                "iterations_per_test": 50,
                "batch_size": 100
            },
            "results": self.results,
            "scenarios": self.test_scenarios
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n📄 Results exported to: {filename}")
        return filename
    
    def run_complete_performance_test(self):
        """Run the complete performance test suite."""
        print("🚀 SME Context Integration - Performance Testing")
        print("="*80)
        print("Testing logging impact on system performance")
        print("="*80)
        
        # Test 1: No logging
        self.test_no_logging_configuration()
        
        # Test 2: Full logging  
        self.test_full_logging_configuration()
        
        # Compare results
        self.compare_results()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Export results
        self.export_results()
        
        print(f"\n🎉 PERFORMANCE TESTING COMPLETE!")
        print("="*80)


def main():
    """Main function to run performance testing."""
    tester = PerformanceTester()
    tester.run_complete_performance_test()


if __name__ == "__main__":
    main()
