#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Realistic Performance Testing Script

Tests the actual performance impact of logging with real file I/O operations.
This provides accurate measurements of logging overhead by simulating real logging behavior.

Key Features:
- Real file I/O operations for logging
- JSON serialization overhead measurement
- File system write latency testing
- Realistic data structures and content
- Precise timing measurements

Usage:
    python3 notebooks/14_realistic_performance_test.py
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


class RealisticPerformanceTester:
    """Realistic performance testing that measures actual logging overhead."""
    
    def __init__(self):
        self.temp_dir = None
        self.setup_test_environment()
        
        # Realistic test data that would be logged
        self.test_data_samples = [
            {
                "timestamp": datetime.now().isoformat(),
                "operation": "sme_analysis",
                "error_type": "ValueError", 
                "error_message": "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. e.g, `dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`",
                "context": {
                    "risk_event": "Model Performance Degradation",
                    "risk_category": "Model Risk",
                    "sme_contexts_count": 1,
                    "historical_data_count": 2
                },
                "dspy_signatures_available": {
                    "sme_analysis": True,
                    "mvr_pattern": True,
                    "sme_validation": True
                }
            },
            {
                "timestamp": datetime.now().isoformat(),
                "operation": "sme_analysis",
                "duration_ms": 1.62,
                "success": True,
                "details": {
                    "risk_event": "Credit Portfolio Concentration",
                    "risk_category": "Credit Risk",
                    "used_dspy": False,
                    "used_synthetic": True
                }
            },
            {
                "timestamp": datetime.now().isoformat(),
                "risk_event": "VaR Model Breach",
                "risk_category": "Market Risk",
                "analysis_status": "success (synthetic)",
                "risk_score": 8,
                "confidence_level": "85%",
                "sme_contexts_used": 1,
                "historical_records_analyzed": 12,
                "used_fallback": True
            }
        ]
    
    def setup_test_environment(self):
        """Setup temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="perf_test_")
        print(f"📁 Test directory: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test directory")
    
    def simulate_core_operation(self, operation_complexity: str = "simple"):
        """Simulate the core operation without logging."""
        if operation_complexity == "simple":
            # Simple calculation
            result = sum(range(100))
        elif operation_complexity == "medium":
            # Medium complexity - some data processing
            data = list(range(1000))
            result = sum(x * x for x in data if x % 2 == 0)
        else:  # complex
            # Complex operation - nested processing
            result = 0
            for i in range(100):
                for j in range(50):
                    result += (i * j) % 7
        
        return result
    
    def simulate_logging_operation(self, data: Dict[str, Any], pretty_print: bool = False):
        """Simulate actual logging operation with file I/O."""
        # JSON serialization
        json_indent = 2 if pretty_print else None
        json_str = json.dumps(data, indent=json_indent)
        
        # File write operation
        log_file = os.path.join(self.temp_dir, "test.log")
        with open(log_file, 'a') as f:
            f.write(f"2025-08-22 04:15:00,000 - SMEContext.test - INFO - {json_str}\n")
            f.flush()  # Ensure data is written to disk
    
    def test_scenario_no_logging(self, scenario_name: str, iterations: int = 100):
        """Test scenario without any logging."""
        durations = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            # Core operation only
            if "Simple" in scenario_name:
                result = self.simulate_core_operation("simple")
            elif "Complex" in scenario_name:
                result = self.simulate_core_operation("complex")
            else:
                result = self.simulate_core_operation("medium")
            
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            durations.append(duration)
        
        return {
            "scenario": scenario_name,
            "iterations": iterations,
            "durations_ms": durations,
            "avg_duration_ms": statistics.mean(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "median_duration_ms": statistics.median(durations),
            "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0
        }
    
    def test_scenario_with_logging(self, scenario_name: str, iterations: int = 100, 
                                  pretty_print: bool = False):
        """Test scenario with realistic logging."""
        durations = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            # Core operation
            if "Simple" in scenario_name:
                result = self.simulate_core_operation("simple")
                log_data = self.test_data_samples[0].copy()
            elif "Complex" in scenario_name:
                result = self.simulate_core_operation("complex")
                log_data = self.test_data_samples[2].copy()
            else:
                result = self.simulate_core_operation("medium")
                log_data = self.test_data_samples[1].copy()
            
            # Logging operations (3-5 logs per operation to simulate real usage)
            num_logs = 3 if "Simple" in scenario_name else 5
            for _ in range(num_logs):
                log_data["iteration"] = i
                log_data["timestamp"] = datetime.now().isoformat()
                self.simulate_logging_operation(log_data, pretty_print)
            
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            durations.append(duration)
        
        return {
            "scenario": scenario_name,
            "iterations": iterations,
            "durations_ms": durations,
            "avg_duration_ms": statistics.mean(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "median_duration_ms": statistics.median(durations),
            "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "pretty_print": pretty_print
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive performance testing."""
        print("🎯 Realistic Performance Testing - Logging Impact Analysis")
        print("="*80)
        print("Testing actual file I/O and JSON serialization overhead")
        print("="*80)
        
        scenarios = [
            "Simple Risk Analysis",
            "Complex Multi-Category Analysis", 
            "High-Frequency Operation"
        ]
        
        results = {
            "no_logging": {},
            "compact_logging": {},
            "pretty_logging": {}
        }
        
        # Test 1: No Logging
        print("\n🚫 SCENARIO 1: No Logging (Baseline)")
        print("-" * 50)
        for scenario in scenarios:
            result = self.test_scenario_no_logging(scenario, iterations=100)
            results["no_logging"][scenario] = result
            print(f"{scenario:<30} {result['avg_duration_ms']:>8.3f}ms ± {result['std_dev_ms']:>6.3f}ms")
        
        # Test 2: Compact Logging
        print("\n📝 SCENARIO 2: Compact Logging (Production Style)")
        print("-" * 50)
        for scenario in scenarios:
            result = self.test_scenario_with_logging(scenario, iterations=100, pretty_print=False)
            results["compact_logging"][scenario] = result
            print(f"{scenario:<30} {result['avg_duration_ms']:>8.3f}ms ± {result['std_dev_ms']:>6.3f}ms")
        
        # Test 3: Pretty Logging
        print("\n📝 SCENARIO 3: Pretty Logging (Development Style)")
        print("-" * 50)
        for scenario in scenarios:
            result = self.test_scenario_with_logging(scenario, iterations=100, pretty_print=True)
            results["pretty_logging"][scenario] = result
            print(f"{scenario:<30} {result['avg_duration_ms']:>8.3f}ms ± {result['std_dev_ms']:>6.3f}ms")
        
        # Analysis
        self.analyze_results(results, scenarios)
        
        return results
    
    def analyze_results(self, results: Dict[str, Any], scenarios: List[str]):
        """Analyze and compare the results."""
        print(f"\n📊 PERFORMANCE IMPACT ANALYSIS")
        print("="*80)
        
        print(f"{'Scenario':<30} {'No Logging':<12} {'Compact Log':<12} {'Pretty Log':<12} {'Compact OH':<12} {'Pretty OH'}")
        print("-" * 90)
        
        compact_overheads = []
        pretty_overheads = []
        
        for scenario in scenarios:
            no_log = results["no_logging"][scenario]["avg_duration_ms"]
            compact_log = results["compact_logging"][scenario]["avg_duration_ms"]
            pretty_log = results["pretty_logging"][scenario]["avg_duration_ms"]
            
            compact_overhead = ((compact_log - no_log) / no_log) * 100
            pretty_overhead = ((pretty_log - no_log) / no_log) * 100
            
            compact_overheads.append(compact_overhead)
            pretty_overheads.append(pretty_overhead)
            
            print(f"{scenario:<30} {no_log:<12.3f} {compact_log:<12.3f} {pretty_log:<12.3f} {compact_overhead:<12.1f}% {pretty_overhead:<10.1f}%")
        
        # Summary
        avg_compact_overhead = statistics.mean(compact_overheads)
        avg_pretty_overhead = statistics.mean(pretty_overheads)
        
        print("-" * 90)
        print(f"{'AVERAGE OVERHEAD:':<30} {'':<12} {'':<12} {'':<12} {avg_compact_overhead:<12.1f}% {avg_pretty_overhead:<10.1f}%")
        
        # File size analysis
        self.analyze_log_file_sizes()
        
        # Recommendations
        self.generate_detailed_recommendations(avg_compact_overhead, avg_pretty_overhead)
    
    def analyze_log_file_sizes(self):
        """Analyze log file sizes generated during testing."""
        print(f"\n📁 LOG FILE SIZE ANALYSIS")
        print("-" * 50)
        
        log_file = os.path.join(self.temp_dir, "test.log")
        if os.path.exists(log_file):
            file_size = os.path.getsize(log_file)
            with open(log_file, 'r') as f:
                line_count = sum(1 for _ in f)
            
            print(f"Total log file size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"Total log entries: {line_count:,}")
            print(f"Average entry size: {file_size/line_count:.1f} bytes per entry")
            
            # Estimate daily/monthly sizes for production
            entries_per_hour = 1000  # Estimated
            daily_entries = entries_per_hour * 24
            monthly_entries = daily_entries * 30
            
            daily_size_mb = (daily_entries * file_size / line_count) / (1024 * 1024)
            monthly_size_gb = (monthly_entries * file_size / line_count) / (1024 * 1024 * 1024)
            
            print(f"\n📈 Production Estimates (1000 entries/hour):")
            print(f"Daily log size: {daily_size_mb:.1f} MB")
            print(f"Monthly log size: {monthly_size_gb:.1f} GB")
    
    def generate_detailed_recommendations(self, compact_overhead: float, pretty_overhead: float):
        """Generate detailed recommendations based on test results."""
        print(f"\n💡 DETAILED RECOMMENDATIONS")
        print("="*80)
        
        print(f"📊 Logging Overhead Summary:")
        print(f"   Compact Logging: {compact_overhead:.1f}% performance overhead")
        print(f"   Pretty Logging:  {pretty_overhead:.1f}% performance overhead")
        print()
        
        # Performance impact assessment
        if compact_overhead < 5:
            impact_level = "MINIMAL"
            color = "🟢"
        elif compact_overhead < 15:
            impact_level = "MODERATE" 
            color = "🟡"
        else:
            impact_level = "SIGNIFICANT"
            color = "🔴"
        
        print(f"{color} PERFORMANCE IMPACT: {impact_level}")
        print()
        
        # Environment-specific recommendations
        print(f"🏭 PRODUCTION RECOMMENDATIONS:")
        if compact_overhead < 10:
            print(f"   ✅ Safe to use compact logging in production")
            print(f"   ✅ Enable critical error logging (dspy_errors, fallback_usage)")
            print(f"   ⚠️ Disable performance metrics logging for high-traffic scenarios")
        else:
            print(f"   ⚠️ Use minimal logging in production")
            print(f"   ✅ Enable only critical error types")
            print(f"   🔴 Disable detailed context logging")
        
        print(f"\n🔬 DEVELOPMENT RECOMMENDATIONS:")
        if pretty_overhead < 25:
            print(f"   ✅ Safe to use pretty logging in development")
            print(f"   ✅ Enable all logging types for debugging")
        else:
            print(f"   ⚠️ Use compact logging even in development")
            print(f"   💡 Enable pretty logging only when actively debugging")
        
        print(f"\n⚡ OPTIMIZATION STRATEGIES:")
        print(f"   1. Use DebugConfig.production() for production environments")
        print(f"   2. Disable JSON pretty printing (saves {pretty_overhead - compact_overhead:.1f}% overhead)")
        print(f"   3. Enable selective logging based on criticality")
        print(f"   4. Consider asynchronous logging for high-volume scenarios")
        print(f"   5. Implement log rotation to manage disk usage")
        print(f"   6. Monitor disk I/O impact in production")
        
        print(f"\n🎛️ CONFIGURATION GUIDE:")
        print(f"   • debug_full=False + selective logging: Best for production")
        print(f"   • debug_full=True: Best for development/debugging")
        print(f"   • Custom config: Fine-tune based on specific needs")
    
    def export_results(self, results: Dict[str, Any], filename: str = "realistic_performance_results.json"):
        """Export detailed results to JSON."""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "realistic_performance_test",
            "test_environment": {
                "temp_directory": self.temp_dir,
                "iterations_per_test": 100,
                "logging_operations_per_test": "3-5 per iteration"
            },
            "results": results,
            "summary": {
                "compact_logging_overhead": statistics.mean([
                    ((results["compact_logging"][scenario]["avg_duration_ms"] - 
                      results["no_logging"][scenario]["avg_duration_ms"]) / 
                     results["no_logging"][scenario]["avg_duration_ms"]) * 100
                    for scenario in results["no_logging"].keys()
                ]),
                "pretty_logging_overhead": statistics.mean([
                    ((results["pretty_logging"][scenario]["avg_duration_ms"] - 
                      results["no_logging"][scenario]["avg_duration_ms"]) / 
                     results["no_logging"][scenario]["avg_duration_ms"]) * 100
                    for scenario in results["no_logging"].keys()
                ])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n📄 Detailed results exported to: {filename}")
        return filename


def main():
    """Main function to run realistic performance testing."""
    tester = RealisticPerformanceTester()
    
    try:
        results = tester.run_comprehensive_test()
        tester.export_results(results)
        
        print(f"\n🎉 REALISTIC PERFORMANCE TESTING COMPLETE!")
        print("="*80)
        print("✅ Actual logging overhead measured with real file I/O")
        print("✅ JSON serialization impact quantified")
        print("✅ Production recommendations generated")
        print("✅ Configuration guidelines provided")
        
    finally:
        tester.cleanup_test_environment()


if __name__ == "__main__":
    main()
