#!/usr/bin/env python3
"""
TidyLLM Corporate Onboarding - Shipping Package Verifier
========================================================

Automated verification script that ensures the onboarding package
is complete and ready for corporate distribution.

This script validates:
- All required files are present
- Python modules import correctly
- No hardcoded credentials exist
- Basic functionality works
- Expected test patterns are met

Usage:
    python verify_shipping_package.py
    python verify_shipping_package.py --verbose
    python verify_shipping_package.py --test-integration
"""

import os
import sys
import yaml
import importlib.util
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

class ShippingPackageVerifier:
    """
    Comprehensive verification of the TidyLLM corporate onboarding package.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.base_path = Path(__file__).parent
        self.manifest_path = self.base_path / "shipping_manifest.yaml"
        self.manifest = self.load_manifest()
        self.results = []
        self.overall_success = True
        
    def load_manifest(self) -> Dict[str, Any]:
        """Load the shipping manifest."""
        try:
            with open(self.manifest_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[ERROR] Could not load shipping manifest: {e}")
            sys.exit(1)
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with optional verbose output."""
        if level == "ERROR" or self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def add_result(self, test_name: str, success: bool, message: str, details: Any = None):
        """Add a test result."""
        self.results.append({
            'test_name': test_name,
            'success': success,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
        if not success:
            self.overall_success = False
            
        status = "PASS" if success else "FAIL"
        self.log(f"{test_name}: {status} - {message}")
    
    def verify_file_presence(self) -> bool:
        """Verify all required files are present."""
        self.log("Verifying file presence according to manifest...")
        
        missing_files = []
        present_files = []
        
        # Check onboarding package files
        for file_info in self.manifest.get('onboarding_package', {}).get('core_files', []):
            file_path = self.base_path / file_info['name']
            if file_path.exists():
                present_files.append(file_info['name'])
                self.log(f"Found: {file_info['name']}", "DEBUG")
            else:
                missing_files.append(file_info['name'])
                self.log(f"Missing: {file_info['name']}", "WARN")
        
        # Check critical test files
        test_files = []
        for test_category in ['preflight_tests', 'core_tests', 'infrastructure_tests', 'integration_tests']:
            for test_info in self.manifest.get('test_suite', {}).get(test_category, []):
                test_path = Path(__file__).parent.parent / test_info['name']
                if test_path.exists():
                    test_files.append(test_info['name'])
                else:
                    missing_files.append(test_info['name'])
        
        success = len(missing_files) == 0
        message = f"Files check: {len(present_files)} onboarding files, {len(test_files)} test files"
        
        if missing_files:
            message += f", {len(missing_files)} missing"
            
        self.add_result("File Presence", success, message, {
            'present_files': present_files,
            'test_files': test_files,
            'missing_files': missing_files
        })
        
        return success
    
    def verify_python_imports(self) -> bool:
        """Verify Python modules can be imported."""
        self.log("Verifying Python module imports...")
        
        import_results = {}
        critical_modules = [
            'enhanced_session_validator',
            'universal_preflight',
            'enhanced_cli_onboarding',
            'config_generator'
        ]
        
        for module_name in critical_modules:
            try:
                module_path = self.base_path / f"{module_name}.py"
                if not module_path.exists():
                    import_results[module_name] = f"File not found: {module_path}"
                    continue
                
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                
                # Add current directory to sys.path temporarily
                original_path = sys.path.copy()
                if str(self.base_path) not in sys.path:
                    sys.path.insert(0, str(self.base_path))
                
                try:
                    spec.loader.exec_module(module)
                    import_results[module_name] = "SUCCESS"
                    self.log(f"Import SUCCESS: {module_name}", "DEBUG")
                except Exception as e:
                    import_results[module_name] = f"Import error: {str(e)}"
                    self.log(f"Import FAILED: {module_name} - {e}", "WARN")
                finally:
                    sys.path = original_path
                    
            except Exception as e:
                import_results[module_name] = f"Module loading error: {str(e)}"
        
        successful_imports = [m for m, result in import_results.items() if result == "SUCCESS"]
        failed_imports = [m for m, result in import_results.items() if result != "SUCCESS"]
        
        success = len(failed_imports) == 0
        message = f"Import check: {len(successful_imports)}/{len(critical_modules)} modules"
        
        self.add_result("Python Imports", success, message, import_results)
        return success
    
    def verify_no_hardcoded_credentials(self) -> bool:
        """Verify no hardcoded credentials exist in files."""
        self.log("Scanning for hardcoded credentials...")
        
        credential_patterns = [
            (r'aws_access_key_id\s*=\s*["\'][A-Z0-9]{20}["\']', 'AWS Access Key'),
            (r'aws_secret_access_key\s*=\s*["\'][A-Za-z0-9+/]{40}["\']', 'AWS Secret Key'),
            (r'password\s*=\s*["\'][^"\']{8,}["\']', 'Password'),
            (r'api_key\s*=\s*["\'][A-Za-z0-9]{20,}["\']', 'API Key'),
            (r'secret\s*=\s*["\'][A-Za-z0-9]{20,}["\']', 'Secret'),
            (r'token\s*=\s*["\'][A-Za-z0-9]{20,}["\']', 'Token')
        ]
        
        issues_found = []
        files_scanned = 0
        
        # Scan onboarding files
        for py_file in self.base_path.glob("*.py"):
            files_scanned += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, cred_type in credential_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Check if it's in a comment or example
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if any(match in line for match in matches):
                                # Skip if it's clearly an example or comment
                                if any(keyword in line.lower() for keyword in ['example', 'your_', 'placeholder', 'template', '#', 'TODO']):
                                    continue
                                issues_found.append({
                                    'file': py_file.name,
                                    'line': i + 1,
                                    'type': cred_type,
                                    'content': line.strip()[:100]  # First 100 chars
                                })
            except Exception as e:
                self.log(f"Could not scan {py_file}: {e}", "WARN")
        
        # Also scan YAML files
        for yaml_file in self.base_path.glob("*.yaml"):
            files_scanned += 1
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for actual values (not placeholders)
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if ':' in line and not line.strip().startswith('#'):
                        key, value = line.split(':', 1)
                        value = value.strip()
                        
                        # Skip obvious placeholders
                        if any(placeholder in value.lower() for placeholder in ['your_', 'placeholder', 'example', 'template', 'xxx']):
                            continue
                            
                        # Check for suspicious patterns
                        if re.match(r'^[A-Z0-9]{20}$', value):  # Looks like AWS key
                            issues_found.append({
                                'file': yaml_file.name,
                                'line': i + 1,
                                'type': 'Possible AWS Key',
                                'content': line.strip()
                            })
            except Exception as e:
                self.log(f"Could not scan {yaml_file}: {e}", "WARN")
        
        success = len(issues_found) == 0
        message = f"Credential scan: {files_scanned} files scanned, {len(issues_found)} issues"
        
        self.add_result("Credential Security", success, message, issues_found)
        return success
    
    def verify_basic_functionality(self) -> bool:
        """Test basic functionality of key modules."""
        self.log("Testing basic functionality...")
        
        functionality_results = {}
        
        # Test enhanced session validator
        try:
            sys.path.insert(0, str(self.base_path))
            from enhanced_session_validator import EnhancedSessionValidator
            
            validator = EnhancedSessionValidator(region='us-east-1')
            env_info = validator.detect_corporate_environment()
            
            functionality_results['enhanced_session_validator'] = {
                'status': 'SUCCESS',
                'proxy_detected': env_info.get('proxy_detected', False),
                'sso_configured': env_info.get('sso_configured', False),
                'aws_profiles': len(env_info.get('aws_profiles', []))
            }
            
        except Exception as e:
            functionality_results['enhanced_session_validator'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        # Test universal preflight
        try:
            from universal_preflight import UniversalPreflightTest
            
            test_runner = UniversalPreflightTest()
            # Just test initialization, not full run
            config = test_runner.config
            
            functionality_results['universal_preflight'] = {
                'status': 'SUCCESS',
                'config_loaded': bool(config),
                'aws_enabled': config.get('aws', {}).get('enabled', False)
            }
            
        except Exception as e:
            functionality_results['universal_preflight'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        # Test config generator
        try:
            from config_generator import create_template_config
            
            test_config = {
                'organization': 'TestOrg',
                'environment': 'test',
                'aws_region': 'us-east-1'
            }
            
            generated_config = create_template_config(test_config)
            
            functionality_results['config_generator'] = {
                'status': 'SUCCESS',
                'config_generated': bool(generated_config),
                'has_aws_section': 'aws' in generated_config,
                'has_security_section': 'security' in generated_config
            }
            
        except Exception as e:
            functionality_results['config_generator'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        successful_tests = [name for name, result in functionality_results.items() if result.get('status') == 'SUCCESS']
        success = len(successful_tests) == len(functionality_results)
        message = f"Functionality test: {len(successful_tests)}/{len(functionality_results)} modules working"
        
        self.add_result("Basic Functionality", success, message, functionality_results)
        return success
    
    def verify_test_pattern(self) -> bool:
        """Verify that tests show the expected pass/fail pattern."""
        self.log("Verifying expected test patterns...")
        
        test_results = {}
        
        # Test AWS connectivity (should pass if credentials available)
        aws_test_path = Path(__file__).parent.parent / "tests/00_preflight/test_aws_connectivity.py"
        if aws_test_path.exists():
            try:
                result = subprocess.run([
                    sys.executable, str(aws_test_path)
                ], capture_output=True, text=True, timeout=30, cwd=str(aws_test_path.parent.parent))
                
                test_results['aws_connectivity'] = {
                    'returncode': result.returncode,
                    'passed': result.returncode == 0,
                    'output_length': len(result.stdout) + len(result.stderr)
                }
                
            except subprocess.TimeoutExpired:
                test_results['aws_connectivity'] = {
                    'returncode': -1,
                    'passed': False,
                    'error': 'Timeout'
                }
            except Exception as e:
                test_results['aws_connectivity'] = {
                    'returncode': -1,
                    'passed': False,
                    'error': str(e)
                }
        else:
            test_results['aws_connectivity'] = {
                'returncode': -1,
                'passed': False,
                'error': 'Test file not found'
            }
        
        # Test enhanced session validator directly
        try:
            sys.path.insert(0, str(self.base_path))
            from enhanced_session_validator import quick_aws_validation
            
            validation_result = quick_aws_validation()
            test_results['enhanced_validation'] = {
                'passed': validation_result.get('overall_success', False),
                'services_tested': len(validation_result.get('service_validations', {})),
                'has_credentials': validation_result.get('credential_info') is not None
            }
            
        except Exception as e:
            test_results['enhanced_validation'] = {
                'passed': False,
                'error': str(e)
            }
        
        # Analyze pattern
        expected_passes = ['enhanced_validation']  # Should pass
        expected_failures = []  # May fail due to missing credentials
        
        pattern_correct = True
        pattern_details = {}
        
        for test_name, result in test_results.items():
            if test_name in expected_passes and not result.get('passed', False):
                pattern_correct = False
                pattern_details[test_name] = "Expected PASS but got FAIL"
            elif test_name in expected_failures and result.get('passed', False):
                pattern_correct = False
                pattern_details[test_name] = "Expected FAIL but got PASS"
            else:
                pattern_details[test_name] = "Expected result"
        
        message = f"Test pattern verification: {len([r for r in test_results.values() if r.get('passed', False)])}/{len(test_results)} tests passed"
        
        self.add_result("Test Pattern", pattern_correct, message, {
            'test_results': test_results,
            'pattern_analysis': pattern_details
        })
        
        return pattern_correct
    
    def verify_documentation_completeness(self) -> bool:
        """Verify documentation is complete."""
        self.log("Verifying documentation completeness...")
        
        readme_path = self.base_path / "README.md"
        doc_results = {
            'readme_exists': readme_path.exists(),
            'readme_size': 0,
            'required_sections': [],
            'missing_sections': []
        }
        
        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc_results['readme_size'] = len(content)
                    
                    # Check for required sections
                    required_sections = [
                        'Quick Start',
                        'Installation',
                        'Usage',
                        'Requirements',
                        'Features',
                        'Corporate',
                        'Security'
                    ]
                    
                    for section in required_sections:
                        if section.lower() in content.lower():
                            doc_results['required_sections'].append(section)
                        else:
                            doc_results['missing_sections'].append(section)
                            
            except Exception as e:
                doc_results['error'] = str(e)
        
        success = (doc_results['readme_exists'] and 
                  doc_results['readme_size'] > 1000 and  # At least 1KB
                  len(doc_results['missing_sections']) == 0)
        
        message = f"Documentation: README {doc_results['readme_size']} bytes, {len(doc_results['required_sections'])}/{len(doc_results['required_sections']) + len(doc_results['missing_sections'])} sections"
        
        self.add_result("Documentation", success, message, doc_results)
        return success
    
    def generate_verification_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report."""
        passed_tests = [r for r in self.results if r['success']]
        failed_tests = [r for r in self.results if not r['success']]
        
        report = {
            'verification_summary': {
                'overall_success': self.overall_success,
                'total_tests': len(self.results),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests),
                'verification_timestamp': datetime.now().isoformat(),
                'package_info': self.manifest.get('package_info', {})
            },
            'test_results': self.results,
            'recommendations': []
        }
        
        # Add recommendations based on failures
        for test in failed_tests:
            if test['test_name'] == 'File Presence':
                report['recommendations'].append("Ensure all required files are present before shipping")
            elif test['test_name'] == 'Python Imports':
                report['recommendations'].append("Fix import issues in Python modules")
            elif test['test_name'] == 'Credential Security':
                report['recommendations'].append("Remove or mask any hardcoded credentials")
            elif test['test_name'] == 'Basic Functionality':
                report['recommendations'].append("Fix functionality issues in core modules")
        
        if self.overall_success:
            report['recommendations'].append("Package appears ready for corporate distribution")
        else:
            report['recommendations'].append("Resolve failing tests before shipping")
        
        return report
    
    def run_all_verifications(self) -> Dict[str, Any]:
        """Run complete verification suite."""
        self.log("Starting comprehensive shipping package verification...")
        
        # Run all verification tests
        self.verify_file_presence()
        self.verify_python_imports()
        self.verify_no_hardcoded_credentials()
        self.verify_basic_functionality()
        self.verify_test_pattern()
        self.verify_documentation_completeness()
        
        # Generate report
        report = self.generate_verification_report()
        
        # Print summary
        print("\n" + "="*70)
        print("TIDYLLM CORPORATE ONBOARDING - SHIPPING VERIFICATION")
        print("="*70)
        
        summary = report['verification_summary']
        status = "✅ READY FOR SHIPPING" if summary['overall_success'] else "❌ NOT READY"
        print(f"Overall Status: {status}")
        print(f"Tests: {summary['passed_tests']} passed, {summary['failed_tests']} failed")
        
        if summary['failed_tests'] > 0:
            print("\n❌ FAILED TESTS:")
            for test in [r for r in self.results if not r['success']]:
                print(f"  - {test['test_name']}: {test['message']}")
        
        if report['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print("\n" + "="*70)
        
        return report

def main():
    """Main verification entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TidyLLM Corporate Onboarding Package Verifier"
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--save-report', '-s', help='Save verification report to file')
    
    args = parser.parse_args()
    
    verifier = ShippingPackageVerifier(verbose=args.verbose)
    report = verifier.run_all_verifications()
    
    if args.save_report:
        try:
            with open(args.save_report, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, indent=2)
            print(f"\n📋 Verification report saved to: {args.save_report}")
        except Exception as e:
            print(f"\n❌ Could not save report: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if report['verification_summary']['overall_success'] else 1)

if __name__ == "__main__":
    main()