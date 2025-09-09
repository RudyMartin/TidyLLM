#!/usr/bin/env python3
"""
Universal Pre-Flight Test Framework for TidyLLM Corporate Deployments
===================================================================

Based on the proven 00_preflight test architecture but designed to be:
- Universal across different corporate environments
- Configurable for different service combinations
- SSO and temporary credential aware
- Proxy and network restriction compatible
- Comprehensive PostgreSQL + MLflow testing

This replaces environment-specific pre-flight tests with a single
universal validator that adapts to the corporate environment.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Import our enhanced validator
try:
    from enhanced_session_validator import (
        EnhancedSessionValidator, 
        validate_corporate_aws_stack,
        print_validation_report
    )
except ImportError:
    print("[ERROR] Enhanced session validator not found!")
    print("Ensure enhanced_session_validator.py is in the same directory")
    sys.exit(1)

class UniversalPreflightTest:
    """
    Universal pre-flight test framework for TidyLLM corporate deployments.
    
    This class provides a standardized testing framework that can adapt
    to different corporate environments and service configurations.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self.load_configuration(config_file)
        self.test_results = []
        self.overall_success = True
        self.start_time = None
        
    def load_configuration(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load test configuration from file or use defaults.
        
        Configuration can specify:
        - Which services to test
        - Service-specific parameters
        - Corporate environment settings
        - Test thresholds and timeouts
        """
        default_config = {
            'aws': {
                'enabled': True,
                'region': 'us-east-1',
                'test_bedrock_models': False,
                's3_bucket': None,  # Will test general S3 access if None
            },
            'postgresql': {
                'enabled': False,  # Only test if explicitly configured
                'host': None,
                'port': 5432,
                'database': 'tidyllm',
                'username': None,
                'password': None,
                'ssl_mode': 'require'
            },
            'mlflow': {
                'enabled': False,  # Only test if explicitly configured
                'tracking_uri': None
            },
            'corporate': {
                'require_proxy': False,
                'require_sso': False,
                'require_encryption': True,
                'max_latency_ms': 5000,
                'timeout_seconds': 30
            },
            'tests': {
                'fail_on_warnings': False,
                'detailed_output': True,
                'save_report': True,
                'report_file': 'preflight_report.json'
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                # Merge with defaults (deep merge)
                def deep_merge(default, loaded):
                    if isinstance(default, dict) and isinstance(loaded, dict):
                        for key, value in loaded.items():
                            if key in default:
                                default[key] = deep_merge(default[key], value)
                            else:
                                default[key] = value
                    else:
                        return loaded
                    return default
                
                return deep_merge(default_config, loaded_config)
            except Exception as e:
                print(f"[WARNING] Could not load config file {config_file}: {e}")
                print("[INFO] Using default configuration")
        
        # Try to auto-detect configuration from environment
        self.auto_detect_config(default_config)
        
        return default_config
    
    def auto_detect_config(self, config: Dict[str, Any]):
        """
        Auto-detect configuration from environment variables and settings files.
        """
        # Auto-detect AWS region
        aws_region = os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')
        if aws_region:
            config['aws']['region'] = aws_region
        
        # Auto-detect S3 bucket from environment
        s3_bucket = os.environ.get('TIDYLLM_S3_BUCKET')
        if s3_bucket:
            config['aws']['s3_bucket'] = s3_bucket
        
        # Auto-detect PostgreSQL configuration
        pg_host = os.environ.get('TIDYLLM_DB_HOST') or os.environ.get('POSTGRES_HOST')
        pg_user = os.environ.get('TIDYLLM_DB_USER') or os.environ.get('POSTGRES_USER')
        pg_password = os.environ.get('TIDYLLM_DB_PASSWORD') or os.environ.get('POSTGRES_PASSWORD')
        pg_database = os.environ.get('TIDYLLM_DB_NAME') or os.environ.get('POSTGRES_DB')
        
        if pg_host and pg_user and pg_password:
            config['postgresql']['enabled'] = True
            config['postgresql']['host'] = pg_host
            config['postgresql']['username'] = pg_user
            config['postgresql']['password'] = pg_password
            if pg_database:
                config['postgresql']['database'] = pg_database
        
        # Auto-detect MLflow configuration
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
        if mlflow_uri:
            config['mlflow']['enabled'] = True
            config['mlflow']['tracking_uri'] = mlflow_uri
        
        # Auto-detect corporate proxy
        if any(os.environ.get(var) for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']):
            config['corporate']['require_proxy'] = True
    
    def print_test_header(self):
        """Print the test header with configuration summary."""
        print("\n" + "="*70)
        print("TIDYLLM UNIVERSAL PRE-FLIGHT TEST")
        print("="*70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"AWS Region: {self.config['aws']['region']}")
        
        # Show which services will be tested
        enabled_services = []
        if self.config['aws']['enabled']:
            enabled_services.append('AWS (S3, Bedrock)')
        if self.config['postgresql']['enabled']:
            enabled_services.append('PostgreSQL')
        if self.config['mlflow']['enabled']:
            enabled_services.append('MLflow')
        
        print(f"Testing Services: {', '.join(enabled_services)}")
        print("-" * 70)
    
    def run_environment_detection(self) -> Dict[str, Any]:
        """
        Test 1: Corporate Environment Detection
        """
        print("[TEST 01] Detecting corporate environment...")
        
        validator = EnhancedSessionValidator(region=self.config['aws']['region'])
        env_info = validator.detect_corporate_environment()
        
        # Validate corporate requirements
        issues = []
        warnings = []
        
        if self.config['corporate']['require_proxy'] and not env_info['proxy_detected']:
            issues.append("Corporate proxy required but not detected")
        
        if self.config['corporate']['require_sso'] and not env_info['sso_configured']:
            issues.append("SSO configuration required but not found")
        
        if env_info['proxy_detected'] and not env_info.get('ca_bundle_path'):
            warnings.append("Proxy detected but no custom CA bundle - may cause SSL issues")
        
        success = len(issues) == 0
        
        result = {
            'test_name': 'Environment Detection',
            'success': success,
            'environment_info': env_info,
            'issues': issues,
            'warnings': warnings,
            'message': f"Corporate environment detected - Proxy: {env_info['proxy_detected']}, SSO: {env_info['sso_configured']}"
        }
        
        print(f"  {'[PASS]' if success else '[FAIL]'} {result['message']}")
        for issue in issues:
            print(f"    [ERROR] {issue}")
        for warning in warnings:
            print(f"    [WARN] {warning}")
        
        self.test_results.append(result)
        if not success:
            self.overall_success = False
        
        return result
    
    def run_credential_discovery(self) -> Dict[str, Any]:
        """
        Test 2: AWS Credential Discovery and Validation
        """
        print("[TEST 02] Discovering and validating AWS credentials...")
        
        validator = EnhancedSessionValidator(region=self.config['aws']['region'])
        validator.detect_corporate_environment()  # Ensure environment is detected
        
        credentials = validator.discover_credentials()
        
        if not credentials:
            result = {
                'test_name': 'Credential Discovery',
                'success': False,
                'credentials_found': 0,
                'message': "No AWS credentials found",
                'recommendations': [
                    'Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables',
                    'Configure AWS CLI with "aws configure"',
                    'Set up AWS SSO with "aws sso login"'
                ]
            }
            print(f"  [FAIL] {result['message']}")
            self.test_results.append(result)
            self.overall_success = False
            return result
        
        # Try to create session with best available credentials
        session_success = False
        session_error = None
        active_credential = None
        
        for cred in credentials:
            try:
                validator.create_session_with_credentials(cred)
                session_success = True
                active_credential = cred
                break
            except Exception as e:
                session_error = str(e)
                continue
        
        result = {
            'test_name': 'Credential Discovery',
            'success': session_success,
            'credentials_found': len(credentials),
            'credential_sources': [cred.source for cred in credentials],
            'active_credential': {
                'source': active_credential.source,
                'account_id': active_credential.account_id,
                'region': active_credential.region,
                'expires': active_credential.expiration.isoformat() if active_credential and active_credential.expiration else None
            } if active_credential else None,
            'message': f"Found {len(credentials)} credential sources, using {active_credential.source}" if session_success else f"Credential validation failed: {session_error}"
        }
        
        print(f"  {'[PASS]' if session_success else '[FAIL]'} {result['message']}")
        
        # Check for expiring credentials
        if active_credential and active_credential.expiration:
            time_until_expiry = active_credential.expiration - datetime.now()
            if time_until_expiry.total_seconds() < 3600:  # Less than 1 hour
                print(f"    [WARN] Credentials expire in {time_until_expiry}")
        
        self.test_results.append(result)
        if not session_success:
            self.overall_success = False
        
        return result
    
    def run_aws_service_validation(self) -> Dict[str, Any]:
        """
        Test 3: AWS Service Connectivity (S3, Bedrock, STS)
        """
        print("[TEST 03] Testing AWS service connectivity...")
        
        if not self.config['aws']['enabled']:
            result = {
                'test_name': 'AWS Services',
                'success': True,
                'message': 'AWS testing disabled in configuration',
                'skipped': True
            }
            self.test_results.append(result)
            return result
        
        # Run AWS validation
        aws_config = self.config['aws']
        postgres_config = None
        mlflow_uri = None
        
        # We'll test AWS services only in this test
        validation_result = validate_corporate_aws_stack(
            region=aws_config['region'],
            s3_bucket=aws_config.get('s3_bucket'),
            test_bedrock_models=aws_config.get('test_bedrock_models', False)
        )
        
        # Extract AWS-specific results
        aws_services = ['aws', 's3', 'bedrock']
        aws_results = {
            service: validation_result['service_validations'].get(service)
            for service in aws_services
            if service in validation_result['service_validations']
        }
        
        # Check latency requirements
        latency_issues = []
        max_latency = self.config['corporate']['max_latency_ms']
        
        for service, service_result in aws_results.items():
            if service_result and service_result.get('latency_ms', 0) > max_latency:
                latency_issues.append(f"{service.upper()}: {service_result['latency_ms']:.1f}ms > {max_latency}ms")
        
        success = all(
            result and result['success'] 
            for result in aws_results.values()
        ) and len(latency_issues) == 0
        
        result = {
            'test_name': 'AWS Services',
            'success': success,
            'service_results': aws_results,
            'latency_issues': latency_issues,
            'total_latency_ms': sum(
                r.get('latency_ms', 0) for r in aws_results.values() if r
            ),
            'message': f"AWS services: {len([r for r in aws_results.values() if r and r['success']])}/{len(aws_results)} successful"
        }
        
        print(f"  {'[PASS]' if success else '[FAIL]'} {result['message']}")
        
        # Print individual service results
        for service, service_result in aws_results.items():
            if service_result:
                status = "[PASS]" if service_result['success'] else "[FAIL]"
                latency = f" ({service_result.get('latency_ms', 0):.1f}ms)" if service_result.get('latency_ms') else ""
                print(f"    {status} {service.upper()}: {service_result['message']}{latency}")
        
        for issue in latency_issues:
            print(f"    [WARN] High latency: {issue}")
        
        self.test_results.append(result)
        if not success:
            self.overall_success = False
        
        return result
    
    def run_database_validation(self) -> Dict[str, Any]:
        """
        Test 4: PostgreSQL Database Connectivity
        """
        print("[TEST 04] Testing PostgreSQL database connectivity...")
        
        if not self.config['postgresql']['enabled']:
            result = {
                'test_name': 'PostgreSQL Database',
                'success': True,
                'message': 'PostgreSQL testing disabled in configuration',
                'skipped': True
            }
            self.test_results.append(result)
            return result
        
        # Get the last successful AWS validator from previous tests
        validator = EnhancedSessionValidator(region=self.config['aws']['region'])
        
        pg_config = self.config['postgresql']
        pg_result = validator.validate_postgresql_connectivity(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            username=pg_config['username'],
            password=pg_config['password'],
            ssl_mode=pg_config.get('ssl_mode', 'require')
        )
        
        # Check if this looks like a TidyLLM database
        tidyllm_ready = False
        if pg_result.success and pg_result.details:
            table_count = pg_result.details.get('table_count', 0)
            mlflow_tables = pg_result.details.get('mlflow_tables', 0)
            tidyllm_ready = table_count > 0 and mlflow_tables > 0
        
        result = {
            'test_name': 'PostgreSQL Database',
            'success': pg_result.success,
            'message': pg_result.message,
            'details': pg_result.details,
            'tidyllm_ready': tidyllm_ready,
            'warnings': pg_result.warnings,
            'latency_ms': pg_result.latency_ms
        }
        
        print(f"  {'[PASS]' if pg_result.success else '[FAIL]'} {result['message']}")
        
        if pg_result.success:
            if tidyllm_ready:
                print(f"    [INFO] Database appears ready for TidyLLM (MLflow tables present)")
            else:
                print(f"    [WARN] Database may need TidyLLM initialization")
            
            if pg_result.details:
                print(f"    [INFO] Tables: {pg_result.details.get('table_count', 0)}, "
                      f"Version: {pg_result.details.get('version', 'unknown')}")
        
        if pg_result.warnings:
            for warning in pg_result.warnings:
                print(f"    [WARN] {warning}")
        
        self.test_results.append(result)
        if not pg_result.success:
            self.overall_success = False
        
        return result
    
    def run_mlflow_validation(self) -> Dict[str, Any]:
        """
        Test 5: MLflow Tracking Server Connectivity
        """
        print("[TEST 05] Testing MLflow tracking server connectivity...")
        
        if not self.config['mlflow']['enabled']:
            result = {
                'test_name': 'MLflow Tracking',
                'success': True,
                'message': 'MLflow testing disabled in configuration',
                'skipped': True
            }
            self.test_results.append(result)
            return result
        
        validator = EnhancedSessionValidator(region=self.config['aws']['region'])
        
        mlflow_uri = self.config['mlflow']['tracking_uri']
        mlflow_result = validator.validate_mlflow_connectivity(mlflow_uri)
        
        result = {
            'test_name': 'MLflow Tracking',
            'success': mlflow_result.success,
            'message': mlflow_result.message,
            'details': mlflow_result.details,
            'warnings': mlflow_result.warnings,
            'latency_ms': mlflow_result.latency_ms
        }
        
        print(f"  {'[PASS]' if mlflow_result.success else '[FAIL]'} {result['message']}")
        
        if mlflow_result.success and mlflow_result.details:
            exp_count = mlflow_result.details.get('experiment_count', 0)
            print(f"    [INFO] MLflow server ready with {exp_count} experiments")
        
        if mlflow_result.warnings:
            for warning in mlflow_result.warnings:
                print(f"    [WARN] {warning}")
        
        self.test_results.append(result)
        if not mlflow_result.success:
            self.overall_success = False
        
        return result
    
    def run_integration_validation(self) -> Dict[str, Any]:
        """
        Test 6: End-to-End Integration Validation
        """
        print("[TEST 06] Testing end-to-end integration...")
        
        # This test checks that all enabled services can work together
        enabled_services = [
            test for test in self.test_results 
            if test.get('success', False) and not test.get('skipped', False)
        ]
        
        integration_issues = []
        
        # Check if we have the minimum required services
        aws_available = any(test['test_name'] == 'AWS Services' and test['success'] for test in self.test_results)
        if not aws_available:
            integration_issues.append("AWS services not available - cannot proceed with TidyLLM")
        
        # Check for optimal service combinations
        db_available = any(test['test_name'] == 'PostgreSQL Database' and test['success'] for test in self.test_results)
        mlflow_available = any(test['test_name'] == 'MLflow Tracking' and test['success'] for test in self.test_results)
        
        if db_available and not mlflow_available:
            integration_issues.append("Database available but MLflow not configured - may limit functionality")
        
        # Check credential expiration vs. expected runtime
        cred_test = next((test for test in self.test_results if test['test_name'] == 'Credential Discovery'), None)
        if cred_test and cred_test.get('active_credential', {}).get('expires'):
            # This would be more sophisticated in a real deployment
            integration_issues.append("Temporary credentials detected - ensure refresh mechanism for long-running deployments")
        
        success = len(integration_issues) == 0 and aws_available
        
        result = {
            'test_name': 'Integration Validation',
            'success': success,
            'enabled_services': len(enabled_services),
            'integration_issues': integration_issues,
            'message': f"Integration check: {len(enabled_services)} services ready" if success else f"{len(integration_issues)} integration issues found"
        }
        
        print(f"  {'[PASS]' if success else '[FAIL]'} {result['message']}")
        
        for issue in integration_issues:
            print(f"    [WARN] {issue}")
        
        self.test_results.append(result)
        if not success:
            self.overall_success = False
        
        return result
    
    def generate_final_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive test report.
        """
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        # Count test results
        total_tests = len(self.test_results)
        passed_tests = len([test for test in self.test_results if test.get('success', False)])
        failed_tests = len([test for test in self.test_results if not test.get('success', True) and not test.get('skipped', False)])
        skipped_tests = len([test for test in self.test_results if test.get('skipped', False)])
        
        # Collect all warnings and recommendations
        all_warnings = []
        all_recommendations = []
        
        for test in self.test_results:
            if test.get('warnings'):
                all_warnings.extend(test['warnings'])
            if test.get('recommendations'):
                all_recommendations.extend(test['recommendations'])
            if test.get('integration_issues'):
                all_warnings.extend(test['integration_issues'])
        
        # Generate deployment readiness assessment
        deployment_ready = self.overall_success
        deployment_blockers = []
        deployment_warnings = []
        
        if not any(test['test_name'] == 'AWS Services' and test['success'] for test in self.test_results):
            deployment_blockers.append("AWS services not accessible")
        
        if not any(test['test_name'] == 'Credential Discovery' and test['success'] for test in self.test_results):
            deployment_blockers.append("No valid AWS credentials")
        
        # Check for warnings that should be addressed
        if any('expire' in warning.lower() for warning in all_warnings):
            deployment_warnings.append("Temporary credentials may expire during deployment")
        
        if any('proxy' in warning.lower() for warning in all_warnings):
            deployment_warnings.append("Corporate proxy may affect connectivity")
        
        report = {
            'test_summary': {
                'overall_success': self.overall_success,
                'deployment_ready': deployment_ready,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'skipped_tests': skipped_tests,
                'test_duration_seconds': duration
            },
            'test_results': self.test_results,
            'deployment_assessment': {
                'ready': deployment_ready,
                'blockers': deployment_blockers,
                'warnings': deployment_warnings,
                'recommendations': all_recommendations
            },
            'configuration_used': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report if configured
        if self.config['tests']['save_report']:
            report_file = self.config['tests']['report_file']
            try:
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"[INFO] Detailed report saved to {report_file}")
            except Exception as e:
                print(f"[WARN] Could not save report to {report_file}: {e}")
        
        return report
    
    def print_final_summary(self, report: Dict[str, Any]):
        """
        Print final test summary.
        """
        print("\n" + "="*70)
        print("UNIVERSAL PRE-FLIGHT TEST RESULTS")
        print("="*70)
        
        summary = report['test_summary']
        print(f"Overall Status: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}")
        print(f"Test Duration: {summary['test_duration_seconds']:.1f}s")
        print(f"Tests: {summary['passed_tests']} passed, {summary['failed_tests']} failed, {summary['skipped_tests']} skipped")
        
        assessment = report['deployment_assessment']
        
        if assessment['ready']:
            print("\n🚀 DEPLOYMENT READY")
            print("All critical services validated - system ready for TidyLLM deployment")
        else:
            print("\n❌ DEPLOYMENT NOT READY")
            print("Critical issues must be resolved before deployment:")
            for blocker in assessment['blockers']:
                print(f"  - {blocker}")
        
        if assessment['warnings']:
            print("\n⚠️ DEPLOYMENT WARNINGS:")
            for warning in assessment['warnings']:
                print(f"  - {warning}")
        
        if assessment['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in assessment['recommendations']:
                print(f"  - {rec}")
        
        print("\n" + "="*70)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run the complete universal pre-flight test suite.
        """
        self.start_time = time.time()
        self.print_test_header()
        
        try:
            # Run tests in sequence
            self.run_environment_detection()
            self.run_credential_discovery()
            self.run_aws_service_validation()
            self.run_database_validation()
            self.run_mlflow_validation()
            self.run_integration_validation()
            
            # Generate final report
            report = self.generate_final_report()
            
            # Print summary
            if self.config['tests']['detailed_output']:
                self.print_final_summary(report)
            
            return report
            
        except KeyboardInterrupt:
            print("\n[CANCELLED] Pre-flight tests cancelled by user")
            return {'test_summary': {'overall_success': False, 'cancelled': True}}
        except Exception as e:
            print(f"\n[ERROR] Pre-flight tests failed with error: {e}")
            return {'test_summary': {'overall_success': False, 'error': str(e)}}

# Main execution and convenience functions

def run_universal_preflight(config_file: Optional[str] = None) -> bool:
    """
    Run universal pre-flight tests and return success status.
    
    This is the main function to call from onboarding scripts.
    """
    test_runner = UniversalPreflightTest(config_file)
    report = test_runner.run_all_tests()
    return report.get('test_summary', {}).get('overall_success', False)

def run_quick_preflight() -> bool:
    """
    Run quick pre-flight test with minimal configuration (AWS only).
    """
    # Create minimal config for AWS-only testing
    config = {
        'aws': {'enabled': True, 'region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')},
        'postgresql': {'enabled': False},
        'mlflow': {'enabled': False},
        'tests': {'detailed_output': False, 'save_report': False}
    }
    
    test_runner = UniversalPreflightTest()
    test_runner.config = config
    report = test_runner.run_all_tests()
    return report.get('test_summary', {}).get('overall_success', False)

if __name__ == "__main__":
    """
    Command-line interface for universal pre-flight tests.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TidyLLM Universal Pre-Flight Test Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', '-c',
        help="Configuration file (YAML format)"
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help="Run quick AWS-only tests"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_preflight()
    else:
        success = run_universal_preflight(args.config)
    
    sys.exit(0 if success else 1)