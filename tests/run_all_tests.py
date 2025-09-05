#!/usr/bin/env python3
"""
TidyLLM Strategic Test Runner
============================

Runs all strategic test suites with intelligent prioritization.
Replaces the nightmare of 50+ individual test files.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSuiteRunner:
    """Manages execution of strategic test suites."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = {}
        
        # Define strategic test suites in priority order
        self.test_suites = [
            {
                'name': 'install',
                'file': '0_test_install.py',
                'description': 'Installation & Dependencies',
                'critical': True,
                'timeout': 30
            },
            {
                'name': 'smoke',
                'file': '1_test_smoke.py',
                'description': 'Critical Path Verification',
                'critical': True,
                'timeout': 30
            },
            {
                'name': 's3_aws',
                'file': '2_test_s3_aws.py',
                'description': 'S3 & AWS Connectivity Tests', 
                'critical': False,
                'timeout': 60
            },
            {
                'name': 'config',
                'file': '3_test_config.py',
                'description': 'Configuration Management Tests',
                'critical': False,
                'timeout': 60
            },
            {
                'name': 'gateways',
                'file': '4_test_gateways.py', 
                'description': 'Gateway System Tests',
                'critical': True,
                'timeout': 60
            },
            {
                'name': 'knowledge_server',
                'file': '5_test_knowledge_server.py',
                'description': 'Knowledge MCP Server Tests',
                'critical': False,
                'timeout': 60
            },
            {
                'name': 'integrations',
                'file': '6_test_integrations.py',
                'description': 'Cross-System Integration Tests',
                'critical': False,
                'timeout': 90
            },
            {
                'name': 'performance',
                'file': '7_test_performance.py',
                'description': 'Performance & Load Tests',
                'critical': False,
                'timeout': 120
            },
            {
                'name': 'security',
                'file': '8_test_security.py',
                'description': 'Security & Authentication Tests',
                'critical': True,
                'timeout': 60
            }
        ]
    
    def run_single_suite(self, suite):
        """Run a single test suite."""
        print(f"\n{'='*60}")
        print(f"RUNNING: {suite['description']}")
        print(f"File: {suite['file']}")
        print(f"Critical: {'Yes' if suite['critical'] else 'No'}")
        print(f"{'='*60}")
        
        test_file = self.test_dir / suite['file']
        
        if not test_file.exists():
            print(f"[FAIL] Test file not found: {test_file}")
            return {
                'success': False,
                'runtime': 0,
                'error': 'File not found'
            }
        
        start_time = time.time()
        
        try:
            # Run the test suite
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=suite['timeout']
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Print output for visibility
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'runtime': runtime,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            runtime = end_time - start_time
            print(f"[FAIL] Test suite timed out after {suite['timeout']} seconds")
            
            return {
                'success': False,
                'runtime': runtime,
                'error': 'Timeout'
            }
            
        except Exception as e:
            end_time = time.time()
            runtime = end_time - start_time
            print(f"[FAIL] Unexpected error: {e}")
            
            return {
                'success': False,
                'runtime': runtime,
                'error': str(e)
            }
    
    def run_all_suites(self, fail_fast=False):
        """Run all test suites."""
        print("[START] TIDYLLM STRATEGIC TEST EXECUTION")
        print(f"Running {len(self.test_suites)} strategic test suites...")
        print(f"Fail fast: {'Enabled' if fail_fast else 'Disabled'}")
        
        overall_start = time.time()
        
        for suite in self.test_suites:
            result = self.run_single_suite(suite)
            self.results[suite['name']] = {
                **suite,
                **result
            }
            
            # Fail fast logic
            if fail_fast and not result['success'] and suite['critical']:
                print(f"\n[CRITICAL] CRITICAL TEST FAILED: {suite['name']}")
                print("Stopping execution due to fail-fast mode")
                break
        
        overall_end = time.time()
        total_runtime = overall_end - overall_start
        
        self.print_summary(total_runtime)
        return self.get_overall_success()
    
    def print_summary(self, total_runtime):
        """Print detailed test execution summary."""
        print(f"\n{'='*80}")
        print("TIDYLLM STRATEGIC TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total execution time: {total_runtime:.1f} seconds")
        print()
        
        # Suite-by-suite results
        print("SUITE RESULTS:")
        print("-" * 80)
        print(f"{'Suite':<20} {'Status':<10} {'Runtime':<10} {'Critical':<10} {'Details'}")
        print("-" * 80)
        
        critical_passed = 0
        critical_total = 0
        total_passed = 0
        total_suites = 0
        
        for name, result in self.results.items():
            status = "[OK] PASS" if result['success'] else "[FAIL] FAIL"
            runtime = f"{result['runtime']:.1f}s"
            critical = "Yes" if result['critical'] else "No"
            
            # Details
            if result['success']:
                details = "All tests passed"
            elif 'error' in result:
                details = result['error']
            else:
                details = f"Exit code: {result.get('returncode', 'unknown')}"
            
            print(f"{name:<20} {status:<10} {runtime:<10} {critical:<10} {details}")
            
            # Count results
            total_suites += 1
            if result['success']:
                total_passed += 1
            
            if result['critical']:
                critical_total += 1
                if result['success']:
                    critical_passed += 1
        
        print("-" * 80)
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        print(f"Total suites: {total_suites}")
        print(f"Passed: {total_passed}/{total_suites}")
        print(f"Critical passed: {critical_passed}/{critical_total}")
        print(f"Success rate: {(total_passed/total_suites)*100:.1f}%")
        
        # Final status
        overall_success = self.get_overall_success()
        if overall_success:
            print(f"\n[SUCCESS] OVERALL STATUS: [OK] SUCCESS")
            print("TidyLLM system is operational!")
        else:
            print(f"\n[CRITICAL] OVERALL STATUS: [FAIL] FAILURE") 
            if critical_passed < critical_total:
                print("Critical system components failed - system may not be operational")
            else:
                print("Non-critical components failed - system should still be functional")
    
    def get_overall_success(self):
        """Determine overall test success."""
        # Success if all critical tests pass
        critical_failures = 0
        for result in self.results.values():
            if result['critical'] and not result['success']:
                critical_failures += 1
        
        return critical_failures == 0
    
    def run_specific_suite(self, suite_name):
        """Run a specific test suite by name."""
        suite = next((s for s in self.test_suites if s['name'] == suite_name), None)
        
        if not suite:
            print(f"[FAIL] Unknown test suite: {suite_name}")
            print(f"Available suites: {[s['name'] for s in self.test_suites]}")
            return False
        
        result = self.run_single_suite(suite)
        self.results[suite_name] = {**suite, **result}
        
        if result['success']:
            print(f"\n[OK] {suite['description']} - PASSED")
        else:
            print(f"\n[FAIL] {suite['description']} - FAILED")
        
        return result['success']


def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TidyLLM Strategic Test Runner")
    parser.add_argument('--suite', help='Run specific test suite')
    parser.add_argument('--fail-fast', action='store_true', 
                       help='Stop on first critical failure')
    parser.add_argument('--list', action='store_true',
                       help='List available test suites')
    
    args = parser.parse_args()
    
    runner = TestSuiteRunner()
    
    if args.list:
        print("Available test suites:")
        for suite in runner.test_suites:
            critical = "Critical" if suite['critical'] else "Optional"
            print(f"  {suite['name']:<20} - {suite['description']} ({critical})")
        return True
    
    if args.suite:
        success = runner.run_specific_suite(args.suite)
    else:
        success = runner.run_all_suites(fail_fast=args.fail_fast)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)