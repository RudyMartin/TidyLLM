#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏭 Production Performance Test

Tests the optimized production configuration to validate that logging overhead
has been reduced to acceptable levels for client deployment.

Key Features:
- Tests production vs development configurations
- Measures actual overhead with selective logging
- Validates client performance requirements
- Provides deployment recommendations

Usage:
    python3 notebooks/15_production_performance_test.py
"""

import os
import sys
import time
import json
import tempfile
import shutil
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
    class SMEContextCoordinator:
        def __init__(self, debug_config=None):
            pass
        def analyze_with_sme_context(self, event, category):
            time.sleep(0.001)  # Simulate work
            return {"status": "simulated", "risk_score": 5}
    
    class DebugConfig:
        @staticmethod
        def production():
            return None
        @staticmethod
        def production_minimal():
            return None
        @staticmethod
        def development():
            return None
        @staticmethod
        def performance_test():
            return None


class ProductionPerformanceTester:
    """Test production-optimized logging configuration."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="prod_perf_test_")
        self.test_scenarios = [
            {
                "name": "Simple Risk Analysis",
                "risk_event": "Model Performance Degradation",
                "risk_category": "Model Risk"
            },
            {
                "name": "Complex Multi-Category Analysis", 
                "risk_event": "Portfolio Risk Concentration",
                "risk_category": "Credit Risk"
            },
            {
                "name": "High-Frequency Operation",
                "risk_event": "Real-time Fraud Detection",
                "risk_category": "Operational Risk"
            }
        ]
        
        print(f"📁 Test directory: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test directory")
    
    def run_scenario_test(self, coordinator: SMEContextCoordinator, scenario: Dict[str, Any], 
                         iterations: int = 100) -> Dict[str, Any]:
        """Run a single scenario test."""
        durations = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            # Perform the analysis
            result = coordinator.analyze_with_sme_context(
                scenario["risk_event"],
                scenario["risk_category"]
            )
            
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            durations.append(duration)
        
        return {
            "scenario": scenario["name"],
            "iterations": iterations,
            "avg_duration_ms": statistics.mean(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "median_duration_ms": statistics.median(durations),
            "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "p95_duration_ms": sorted(durations)[int(0.95 * len(durations))],
            "p99_duration_ms": sorted(durations)[int(0.99 * len(durations))]
        }
    
    def test_configuration(self, config_name: str, config: DebugConfig) -> Dict[str, Any]:
        """Test a specific configuration."""
        print(f"\n📊 TESTING: {config_name}")
        print("-" * 60)
        
        # Create coordinator with the specified config
        if SME_AVAILABLE:
            coordinator = SMEContextCoordinator(debug_config=config)
            print(f"✅ Using real SME coordinator")
        else:
            coordinator = SMEContextCoordinator()
            print(f"⚠️ Using simulated coordinator")
        
        results = {}
        
        for scenario in self.test_scenarios:
            result = self.run_scenario_test(coordinator, scenario, iterations=50)
            results[scenario["name"]] = result
            print(f"{scenario['name']:<30} {result['avg_duration_ms']:>8.3f}ms "
                  f"(P95: {result['p95_duration_ms']:.3f}ms, P99: {result['p99_duration_ms']:.3f}ms)")
        
        return results
    
    def compare_configurations(self, results: Dict[str, Dict[str, Any]]):
        """Compare performance across configurations."""
        print(f"\n📊 CONFIGURATION COMPARISON")
        print("="*90)
        
        configs = list(results.keys())
        scenarios = list(results[configs[0]].keys())
        
        # Header
        header = f"{'Scenario':<30}"
        for config in configs:
            header += f" {config:<15}"
        header += " Best Config"
        print(header)
        print("-" * 90)
        
        # Data rows
        for scenario in scenarios:
            row = f"{scenario:<30}"
            scenario_results = {}
            
            for config in configs:
                avg_duration = results[config][scenario]["avg_duration_ms"]
                scenario_results[config] = avg_duration
                row += f" {avg_duration:<15.3f}"
            
            # Find best (lowest) performance
            best_config = min(scenario_results, key=scenario_results.get)
            row += f" {best_config}"
            print(row)
        
        # Calculate overall averages
        print("-" * 90)
        row = f"{'AVERAGE':<30}"
        overall_averages = {}
        
        for config in configs:
            avg_across_scenarios = statistics.mean([
                results[config][scenario]["avg_duration_ms"] 
                for scenario in scenarios
            ])
            overall_averages[config] = avg_across_scenarios
            row += f" {avg_across_scenarios:<15.3f}"
        
        best_overall = min(overall_averages, key=overall_averages.get)
        row += f" {best_overall}"
        print(row)
        
        return overall_averages
    
    def analyze_production_readiness(self, results: Dict[str, Dict[str, Any]]):
        """Analyze if the production configuration meets performance requirements."""
        print(f"\n🎯 PRODUCTION READINESS ANALYSIS")
        print("="*80)
        
        # Define performance thresholds (example client requirements)
        thresholds = {
            "avg_latency_ms": 10.0,    # Average operation should be < 10ms
            "p95_latency_ms": 25.0,    # 95th percentile should be < 25ms
            "p99_latency_ms": 50.0,    # 99th percentile should be < 50ms
        }
        
        print(f"📋 Performance Requirements:")
        for metric, threshold in thresholds.items():
            print(f"   {metric}: < {threshold}ms")
        print()
        
        # Check each configuration against thresholds
        for config_name, config_results in results.items():
            print(f"📊 {config_name} Analysis:")
            
            meets_requirements = True
            
            for scenario_name, scenario_results in config_results.items():
                avg_latency = scenario_results["avg_duration_ms"]
                p95_latency = scenario_results["p95_duration_ms"]
                p99_latency = scenario_results["p99_duration_ms"]
                
                # Check thresholds
                avg_ok = avg_latency < thresholds["avg_latency_ms"]
                p95_ok = p95_latency < thresholds["p95_latency_ms"]
                p99_ok = p99_latency < thresholds["p99_latency_ms"]
                
                scenario_ok = avg_ok and p95_ok and p99_ok
                meets_requirements = meets_requirements and scenario_ok
                
                status = "✅" if scenario_ok else "❌"
                print(f"   {status} {scenario_name}")
                print(f"      Avg: {avg_latency:.3f}ms {'✅' if avg_ok else '❌'}")
                print(f"      P95: {p95_latency:.3f}ms {'✅' if p95_ok else '❌'}")
                print(f"      P99: {p99_latency:.3f}ms {'✅' if p99_ok else '❌'}")
            
            overall_status = "✅ READY" if meets_requirements else "❌ NOT READY"
            print(f"   Overall: {overall_status}")
            print()
    
    def generate_deployment_recommendations(self, results: Dict[str, Dict[str, Any]], 
                                          averages: Dict[str, float]):
        """Generate deployment recommendations based on test results."""
        print(f"\n💡 DEPLOYMENT RECOMMENDATIONS")
        print("="*80)
        
        # Find best performing configuration
        best_config = min(averages, key=averages.get)
        best_avg = averages[best_config]
        
        print(f"🏆 RECOMMENDED CONFIGURATION: {best_config}")
        print(f"   Average Latency: {best_avg:.3f}ms")
        print()
        
        # Configuration-specific recommendations
        if "Production Minimal" in best_config:
            print(f"📋 PRODUCTION MINIMAL DEPLOYMENT:")
            print(f"   ✅ Use DebugConfig.production_minimal()")
            print(f"   ✅ Minimal logging overhead")
            print(f"   ⚠️ Limited debugging capabilities")
            print(f"   💡 Best for high-performance requirements")
            
        elif "Production" in best_config:
            print(f"📋 PRODUCTION STANDARD DEPLOYMENT:")
            print(f"   ✅ Use DebugConfig.production()")
            print(f"   ✅ Balanced logging and performance")
            print(f"   ✅ Good debugging capabilities")
            print(f"   💡 Recommended for most production environments")
            
        else:
            print(f"📋 DEVELOPMENT/TESTING DEPLOYMENT:")
            print(f"   ⚠️ Not recommended for production")
            print(f"   ✅ Full debugging capabilities")
            print(f"   💡 Use only in development environments")
        
        print(f"\n🔧 IMPLEMENTATION GUIDE:")
        print(f"   1. Set environment variable: DEBUG_FULL=false")
        print(f"   2. Use production config: config = DebugConfig.production()")
        print(f"   3. Monitor log files for critical errors only")
        print(f"   4. Implement log rotation for long-running systems")
        print(f"   5. Set up alerting for critical error patterns")
        
        print(f"\n📈 EXPECTED PRODUCTION PERFORMANCE:")
        if best_avg < 5:
            print(f"   🟢 EXCELLENT: < 5ms average latency")
        elif best_avg < 10:
            print(f"   🟡 GOOD: < 10ms average latency")
        elif best_avg < 25:
            print(f"   🟠 ACCEPTABLE: < 25ms average latency")
        else:
            print(f"   🔴 POOR: > 25ms average latency - further optimization needed")
    
    def run_production_performance_test(self):
        """Run the complete production performance test."""
        print("🏭 Production Performance Testing - Optimized Configurations")
        print("="*80)
        print("Testing client-ready configurations for deployment validation")
        print("="*80)
        
        # Test configurations
        configurations = {
            "No Logging": DebugConfig.performance_test(),
            "Production Minimal": DebugConfig.production_minimal(),
            "Production Standard": DebugConfig.production(),
            "Development": DebugConfig.development()
        }
        
        results = {}
        
        # Run tests for each configuration
        for config_name, config in configurations.items():
            results[config_name] = self.test_configuration(config_name, config)
        
        # Compare configurations
        averages = self.compare_configurations(results)
        
        # Analyze production readiness
        self.analyze_production_readiness(results)
        
        # Generate recommendations
        self.generate_deployment_recommendations(results, averages)
        
        # Export results
        self.export_results(results)
        
        print(f"\n🎉 PRODUCTION PERFORMANCE TESTING COMPLETE!")
        print("="*80)
        print("✅ Client deployment configurations validated")
        print("✅ Performance requirements analyzed")
        print("✅ Deployment recommendations generated")
        
        return results
    
    def export_results(self, results: Dict[str, Any]):
        """Export test results."""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "production_performance_test",
            "results": results,
            "summary": {
                config_name: {
                    "avg_latency_ms": statistics.mean([
                        scenario_results["avg_duration_ms"] 
                        for scenario_results in config_results.values()
                    ])
                }
                for config_name, config_results in results.items()
            }
        }
        
        filename = "production_performance_results.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n📄 Results exported to: {filename}")


def main():
    """Main function."""
    tester = ProductionPerformanceTester()
    
    try:
        results = tester.run_production_performance_test()
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
