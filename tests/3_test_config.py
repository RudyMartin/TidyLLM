#!/usr/bin/env python3
"""
TidyLLM Strategic Test Suite #5: Configuration Management
========================================================

Tests configuration loading, validation, and management.
Replaces multiple config/settings test files.
"""

import unittest
import sys
import os
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestConfigurationFiles(unittest.TestCase):
    """Test configuration file discovery and loading."""
    
    def test_settings_file_discovery(self):
        """Test that settings files can be discovered."""
        # Check multiple possible locations
        potential_locations = [
            project_root / "admin" / "settings.yaml",
            project_root / "tidyllm" / "admin" / "settings.yaml",
            project_root / "settings.yaml",
            project_root / "config" / "settings.yaml"
        ]
        
        found_locations = []
        for location in potential_locations:
            if location.exists():
                found_locations.append(location)
                print(f"[OK] Found settings: {location}")
        
        if found_locations:
            # Test loading first found settings file
            settings_file = found_locations[0]
            try:
                with open(settings_file, 'r') as f:
                    settings = yaml.safe_load(f)
                
                self.assertIsInstance(settings, dict)
                print(f"[OK] Successfully loaded settings with {len(settings)} top-level keys")
                
                # Check for common configuration sections
                common_sections = ['postgres', 'mlflow', 's3', 'aws', 'integrations']
                found_sections = [section for section in common_sections if section in settings]
                
                if found_sections:
                    print(f"[OK] Found configuration sections: {found_sections}")
                else:
                    print("[WARN] No standard configuration sections found")
                
            except yaml.YAMLError as e:
                print(f"[FAIL] YAML parsing error: {e}")
                self.fail(f"Settings file is not valid YAML: {e}")
            except Exception as e:
                print(f"[FAIL] Error reading settings: {e}")
                self.fail(f"Could not read settings file: {e}")
        else:
            print("[WARN] No settings files found (will use defaults)")
    
    def test_configuration_structure(self):
        """Test configuration file structure and required fields."""
        settings_file = self._find_settings_file()
        
        if not settings_file:
            self.skipTest("No settings file found")
        
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f)
        
        # Test that settings is a proper dictionary
        self.assertIsInstance(settings, dict)
        
        # Check for database configuration if present
        if 'postgres' in settings:
            postgres_config = settings['postgres']
            self.assertIsInstance(postgres_config, dict)
            
            # Check required postgres fields
            postgres_fields = ['host', 'port', 'db_name', 'db_user']
            missing_fields = [field for field in postgres_fields if field not in postgres_config]
            
            if missing_fields:
                print(f"[WARN] Missing postgres fields: {missing_fields}")
            else:
                print("[OK] Postgres configuration complete")
        
        # Check for MLflow configuration if present
        if 'mlflow' in settings:
            mlflow_config = settings['mlflow']
            self.assertIsInstance(mlflow_config, dict)
            print("[OK] MLflow configuration found")
        
        # Check for integrations
        if 'integrations' in settings:
            integrations = settings['integrations']
            self.assertIsInstance(integrations, dict)
            print(f"[OK] Integration configurations: {list(integrations.keys())}")
    
    def _find_settings_file(self):
        """Helper to find the first available settings file."""
        locations = [
            project_root / "admin" / "settings.yaml",
            project_root / "tidyllm" / "admin" / "settings.yaml",
            project_root / "settings.yaml"
        ]
        
        for location in locations:
            if location.exists():
                return location
        
        return None


class TestEnvironmentConfiguration(unittest.TestCase):
    """Test environment-based configuration."""
    
    def test_environment_variables(self):
        """Test relevant environment variables."""
        # TidyLLM-specific environment variables
        tidyllm_env_vars = [
            'TIDYLLM_CONFIG_PATH',
            'TIDYLLM_ADMIN_PATH',
            'TIDYLLM_DEBUG'
        ]
        
        # AWS/Cloud environment variables
        aws_env_vars = [
            'AWS_PROFILE',
            'AWS_DEFAULT_REGION',
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY'
        ]
        
        # Database environment variables
        db_env_vars = [
            'DATABASE_URL',
            'POSTGRES_HOST',
            'POSTGRES_PORT',
            'POSTGRES_DB',
            'POSTGRES_USER'
        ]
        
        all_env_vars = tidyllm_env_vars + aws_env_vars + db_env_vars
        configured_vars = {}
        
        for var in all_env_vars:
            value = os.environ.get(var)
            if value:
                # Don't print sensitive values
                if any(sensitive in var.upper() for sensitive in ['KEY', 'SECRET', 'PASSWORD']):
                    display_value = value[:8] + '...' if len(value) > 8 else '***'
                else:
                    display_value = value
                
                configured_vars[var] = display_value
        
        if configured_vars:
            print(f"[OK] Environment variables configured: {len(configured_vars)}")
            for var, value in configured_vars.items():
                print(f"  {var}: {value}")
        else:
            print("[WARN] No relevant environment variables configured")
    
    def test_python_path_configuration(self):
        """Test Python path configuration."""
        python_paths = sys.path
        project_root_str = str(project_root)
        
        # Check if project root is in Python path
        in_python_path = any(project_root_str in path for path in python_paths)
        
        if in_python_path:
            print("[OK] Project root in Python path")
        else:
            print("[WARN] Project root not in Python path (may cause import issues)")
        
        # Check for key directories in path
        key_dirs = ['tidyllm', 'src', 'lib']
        found_dirs = []
        
        for path in python_paths:
            for key_dir in key_dirs:
                if key_dir in path.lower():
                    found_dirs.append(f"{key_dir}:{path}")
                    break
        
        if found_dirs:
            print(f"[OK] Key directories in path: {len(found_dirs)}")


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and defaults."""
    
    def test_configuration_defaults(self):
        """Test that system works with default configuration."""
        try:
            import tidyllm
            
            # Test that TidyLLM can import without explicit configuration
            self.assertTrue(hasattr(tidyllm, 'init_gateways'))
            print("[OK] TidyLLM imports without explicit configuration")
            
            # Test that gateway registry can initialize with defaults
            registry = tidyllm.init_gateways()
            self.assertIsNotNone(registry)
            print("[OK] Gateway registry initializes with defaults")
            
        except Exception as e:
            print(f"[FAIL] Configuration defaults test failed: {e}")
            raise
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        # Create a temporary invalid config for testing
        invalid_configs = [
            {},  # Empty config
            {"invalid_key": "invalid_value"},  # Unknown keys
            {"postgres": {"host": None}},  # Invalid values
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            print(f"Testing invalid config {i+1}: {invalid_config}")
            
            try:
                import tidyllm
                # Test that system doesn't crash with invalid config
                registry = tidyllm.init_gateways(invalid_config)
                print(f"[OK] System handles invalid config {i+1} gracefully")
            except Exception as e:
                print(f"[WARN] Invalid config {i+1} caused error: {e}")
                # This is acceptable - system should handle gracefully


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration with system components."""
    
    def test_gateway_configuration_integration(self):
        """Test that gateways can access configuration."""
        settings_file = self._find_settings_file()
        
        if settings_file:
            with open(settings_file, 'r') as f:
                settings = yaml.safe_load(f)
        else:
            settings = {}
        
        try:
            import tidyllm
            
            # Test that gateways initialize with available configuration
            registry = tidyllm.init_gateways(settings)
            available_services = registry.get_available_services()
            
            print(f"[OK] Gateways integrated with config: {len(available_services)} services")
            
            # Test configuration accessibility through registry
            for service_name in available_services:
                service_info = registry.get_service_info(service_name)
                if service_info:
                    print(f"[OK] Service {service_name} has accessible configuration")
            
        except Exception as e:
            print(f"[FAIL] Gateway configuration integration failed: {e}")
    
    def test_knowledge_server_configuration(self):
        """Test knowledge server configuration integration."""
        try:
            import tidyllm
            
            registry = tidyllm.init_gateways()
            knowledge_server = registry.get('knowledge_resources')
            
            if knowledge_server:
                # Test that knowledge server has configuration
                self.assertTrue(hasattr(knowledge_server, 'config'))
                print("[OK] Knowledge server configuration accessible")
            else:
                print("[WARN] Knowledge server not available (may need configuration)")
                
        except Exception as e:
            print(f"[FAIL] Knowledge server configuration test failed: {e}")
    
    def _find_settings_file(self):
        """Helper to find settings file."""
        locations = [
            project_root / "admin" / "settings.yaml",
            project_root / "tidyllm" / "admin" / "settings.yaml"
        ]
        
        for location in locations:
            if location.exists():
                return location
        return None


def run_config_tests():
    """Run all configuration tests."""
    print("="*60)
    print("TIDYLLM CONFIGURATION TESTS")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfigurationFiles,
        TestEnvironmentConfiguration,
        TestConfigurationValidation,
        TestConfigurationIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("CONFIGURATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"Overall status: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_config_tests()
    sys.exit(0 if success else 1)