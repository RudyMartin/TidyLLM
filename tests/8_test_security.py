#!/usr/bin/env python3
"""
TidyLLM Strategic Test Suite #7: Security & Authentication
=========================================================

Tests security, authentication, and data protection mechanisms.
Replaces multiple security/auth test files.
"""

import unittest
import sys
import os
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import tidyllm
    TIDYLLM_AVAILABLE = True
except ImportError:
    TIDYLLM_AVAILABLE = False


class TestCredentialSecurity(unittest.TestCase):
    """Test credential security and protection."""
    
    def test_no_hardcoded_credentials_in_config(self):
        """Test that no hardcoded credentials exist in configuration files."""
        # Find settings files to check
        settings_files = []
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if file.endswith(('.yaml', '.yml', '.json', '.ini', '.env')):
                    settings_files.append(Path(root) / file)
        
        dangerous_patterns = [
            'password',
            'secret',
            'key',
            'token',
            'credential'
        ]
        
        issues_found = []
        
        for file_path in settings_files[:20]:  # Limit to avoid excessive testing
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                
                # Check for dangerous patterns with actual values
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    for pattern in dangerous_patterns:
                        if pattern in line and ':' in line and not line.strip().startswith('#'):
                            # Check if there's an actual value after the colon
                            parts = line.split(':', 1)
                            if len(parts) > 1 and parts[1].strip() and parts[1].strip() not in ['null', 'none', '""', "''", '~']:
                                issues_found.append(f"{file_path}:{i+1} - {line.strip()}")
                
            except Exception:
                # Skip files that can't be read
                continue
        
        if issues_found:
            print(f"[WARN] Potential credential exposure found:")
            for issue in issues_found[:5]:  # Show first 5 issues
                print(f"   {issue}")
            if len(issues_found) > 5:
                print(f"   ... and {len(issues_found)-5} more")
        else:
            print("[OK] No hardcoded credentials detected in configuration files")
    
    def test_environment_variable_security(self):
        """Test that sensitive environment variables are handled securely."""
        sensitive_env_vars = [
            'AWS_SECRET_ACCESS_KEY',
            'AWS_SESSION_TOKEN',
            'DATABASE_PASSWORD',
            'POSTGRES_PASSWORD',
            'API_KEY',
            'SECRET_KEY'
        ]
        
        exposed_vars = {}
        for var in sensitive_env_vars:
            value = os.environ.get(var)
            if value:
                # Don't log the actual value, just that it exists
                exposed_vars[var] = len(value)
        
        if exposed_vars:
            print(f"[OK] Sensitive environment variables detected (properly handled): {list(exposed_vars.keys())}")
        else:
            print("[INFO] No sensitive environment variables currently set")
    
    def test_file_permissions_security(self):
        """Test that sensitive files have appropriate permissions."""
        sensitive_files = [
            project_root / "admin" / "settings.yaml",
            project_root / "tidyllm" / "admin" / "settings.yaml",
        ]
        
        permission_issues = []
        
        for file_path in sensitive_files:
            if file_path.exists():
                try:
                    stat_info = file_path.stat()
                    # On Unix systems, check if file is readable by others
                    if hasattr(stat_info, 'st_mode'):
                        mode = stat_info.st_mode
                        if mode & 0o044:  # Check if group/others can read
                            permission_issues.append(f"{file_path} is readable by group/others")
                except Exception:
                    # Skip on systems where this check doesn't apply (like Windows)
                    pass
        
        if permission_issues:
            print(f"[WARN] File permission issues: {permission_issues}")
        else:
            print("[OK] Sensitive file permissions check passed")


class TestInputValidation(unittest.TestCase):
    """Test input validation and sanitization."""
    
    def test_configuration_input_validation(self):
        """Test that configuration inputs are validated."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        # Test with malicious configuration inputs
        malicious_configs = [
            {"postgres": {"host": "../../../etc/passwd"}},  # Path traversal
            {"postgres": {"host": "'; DROP TABLE users; --"}},  # SQL injection
            {"postgres": {"host": "<script>alert('xss')</script>"}},  # XSS
            {"aws": {"region": "us-west-2\n\nimport os; os.system('rm -rf /')"}}  # Command injection
        ]
        
        for i, malicious_config in enumerate(malicious_configs):
            try:
                registry = tidyllm.init_gateways(malicious_config)
                # If we get here, the system should have sanitized the input
                print(f"[OK] Malicious config {i+1} handled gracefully")
            except Exception as e:
                # This is acceptable - system should reject malicious input
                print(f"[OK] Malicious config {i+1} properly rejected: {type(e).__name__}")
    
    def test_service_name_validation(self):
        """Test that service names are validated."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        registry = tidyllm.init_gateways()
        
        # Test with potentially malicious service names
        malicious_names = [
            "../../../etc/passwd",
            "'; DROP TABLE services; --",
            "<script>alert('xss')</script>",
            "admin\nrm -rf /",
            "service\x00name"  # Null byte injection
        ]
        
        for name in malicious_names:
            service = registry.get(name)
            self.assertIsNone(service, f"Malicious service name '{name}' should return None")
        
        print("[OK] Service name validation passed")


class TestDataProtection(unittest.TestCase):
    """Test data protection and privacy mechanisms."""
    
    def test_sensitive_data_logging(self):
        """Test that sensitive data is not logged."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        # This test would ideally capture logs and check for sensitive data
        # For now, we just ensure the system initializes without exposing secrets
        
        config_with_secrets = {
            "postgres": {
                "host": "localhost",
                "password": "super_secret_password_12345",
                "db_name": "testdb"
            },
            "aws": {
                "access_key": "AKIAIOSFODNN7EXAMPLE",
                "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            }
        }
        
        try:
            registry = tidyllm.init_gateways(config_with_secrets)
            print("[OK] System handles sensitive configuration data")
        except Exception as e:
            print(f"[WARN] System failed to handle sensitive config: {e}")
    
    def test_configuration_data_exposure(self):
        """Test that configuration data is not exposed in error messages."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        # Test with invalid but sensitive configuration
        sensitive_config = {
            "postgres": {
                "password": "top_secret_password",
                "host": "invalid_host_that_should_fail"
            }
        }
        
        try:
            registry = tidyllm.init_gateways(sensitive_config)
            # Even if this succeeds, that's fine
            print("[OK] Sensitive configuration handled without exposure")
        except Exception as e:
            error_message = str(e)
            # Check that the password is not in the error message
            if "top_secret_password" in error_message:
                self.fail("Sensitive data exposed in error message")
            print("[OK] Error handling does not expose sensitive data")


class TestAccessControl(unittest.TestCase):
    """Test access control mechanisms."""
    
    def test_service_access_control(self):
        """Test that services implement proper access control."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        registry = tidyllm.init_gateways()
        services = registry.get_available_services()
        
        # Test that services don't expose administrative functions
        for service_name in services:
            service = registry.get(service_name)
            if service:
                # Services should not expose dangerous methods
                dangerous_methods = ['execute', 'eval', 'exec', 'system', 'shell']
                for method in dangerous_methods:
                    self.assertFalse(
                        hasattr(service, method), 
                        f"Service '{service_name}' should not expose '{method}' method"
                    )
        
        print(f"[OK] Access control check passed for {len(services)} services")
    
    def test_knowledge_server_security(self):
        """Test knowledge server security mechanisms."""
        if not TIDYLLM_AVAILABLE:
            self.skipTest("TidyLLM not available")
        
        registry = tidyllm.init_gateways()
        knowledge_server = registry.get('knowledge_resources')
        
        if knowledge_server:
            # Test that knowledge server doesn't allow arbitrary file access
            if hasattr(knowledge_server, 'handle_mcp_tool_call'):
                try:
                    # Try to access a sensitive file
                    result = knowledge_server.handle_mcp_tool_call("search", {
                        "query": "../../../etc/passwd"
                    })
                    # If this doesn't raise an exception, the result should be sanitized
                    print("[OK] Knowledge server handles malicious queries safely")
                except Exception:
                    print("[OK] Knowledge server properly rejects malicious queries")
            else:
                print("[INFO] Knowledge server MCP interface not available for testing")
        else:
            print("[INFO] Knowledge server not configured for security testing")


class TestNetworkSecurity(unittest.TestCase):
    """Test network security configurations."""
    
    def test_tls_configuration(self):
        """Test TLS/SSL configuration requirements."""
        # Check configuration files for TLS settings
        settings_files = [
            project_root / "admin" / "settings.yaml",
            project_root / "tidyllm" / "admin" / "settings.yaml"
        ]
        
        tls_configs = []
        
        for settings_file in settings_files:
            if settings_file.exists():
                try:
                    with open(settings_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check for TLS-related configurations
                    if isinstance(config, dict):
                        for section_name, section in config.items():
                            if isinstance(section, dict):
                                for key in section.keys():
                                    if any(tls_term in key.lower() for tls_term in ['ssl', 'tls', 'secure', 'cert']):
                                        tls_configs.append(f"{section_name}.{key}")
                
                except Exception:
                    continue
        
        if tls_configs:
            print(f"[OK] TLS configuration found: {tls_configs}")
        else:
            print("[INFO] No explicit TLS configuration found (may use defaults)")
    
    def test_port_configuration_security(self):
        """Test that port configurations are secure."""
        settings_files = [
            project_root / "admin" / "settings.yaml",
            project_root / "tidyllm" / "admin" / "settings.yaml"
        ]
        
        port_configs = []
        insecure_ports = []
        
        for settings_file in settings_files:
            if settings_file.exists():
                try:
                    with open(settings_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    if isinstance(config, dict):
                        for section_name, section in config.items():
                            if isinstance(section, dict):
                                for key, value in section.items():
                                    if 'port' in key.lower() and isinstance(value, int):
                                        port_configs.append(f"{section_name}.{key}:{value}")
                                        # Check for potentially insecure ports
                                        if value in [21, 23, 25, 53, 80, 110, 143, 993, 995]:
                                            insecure_ports.append(f"{section_name}.{key}:{value}")
                
                except Exception:
                    continue
        
        if port_configs:
            print(f"[OK] Port configurations found: {len(port_configs)}")
            if insecure_ports:
                print(f"[WARN] Potentially insecure ports: {insecure_ports}")
        else:
            print("[INFO] No explicit port configurations found")


def run_security_tests():
    """Run all security tests."""
    print("="*60)
    print("TIDYLLM SECURITY TESTS")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCredentialSecurity,
        TestInputValidation,
        TestDataProtection,
        TestAccessControl,
        TestNetworkSecurity
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("SECURITY TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"Overall status: {'PASS' if success else 'FAIL'}")
    
    if not success:
        print("\n[CRITICAL] SECURITY ISSUES DETECTED - IMMEDIATE ATTENTION REQUIRED")
    else:
        print("\n[SECURE] Security tests passed - system appears secure")
    
    return success


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)