#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM Settings Test - Priority Test #0

This test runs first (0_) to force validation of admin/settings.yaml
and ensure all required configuration fields are properly loaded.
Critical for application startup and configuration validation.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidyllm.settings_loader import SettingsLoader, Settings

class TestSettingsLoader:
    """Test suite for TidyLLM Settings Loader"""
    
    @pytest.fixture
    def admin_settings_path(self):
        """Fixture providing path to admin/settings.yaml"""
        return Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
    
    @pytest.fixture
    def settings_loader(self, admin_settings_path):
        """Fixture providing initialized SettingsLoader"""
        return SettingsLoader(str(admin_settings_path))
    
    def test_admin_settings_file_exists(self, admin_settings_path):
        """Test that admin/settings.yaml file exists"""
        assert admin_settings_path.exists(), f"Admin settings file missing: {admin_settings_path}"
        assert admin_settings_path.is_file(), f"Admin settings path is not a file: {admin_settings_path}"
        assert admin_settings_path.stat().st_size > 0, "Admin settings file is empty"
    
    def test_settings_loader_initialization(self, admin_settings_path):
        """Test SettingsLoader can initialize with admin settings"""
        loader = SettingsLoader(str(admin_settings_path))
        assert loader is not None
        assert loader.settings is not None
        assert isinstance(loader.settings, Settings)
        assert loader.settings_path == admin_settings_path
    
    def test_required_s3_configuration_fields(self, settings_loader):
        """Test that all required S3 fields are present"""
        s3_config = settings_loader.get_s3_config()
        
        # Test S3 config is loaded
        assert s3_config is not None, "S3 configuration not loaded"
        assert isinstance(s3_config, dict), "S3 configuration is not a dictionary"
        
        # Test required S3 fields
        required_s3_fields = ['region', 'bucket', 'prefix']
        for field in required_s3_fields:
            assert field in s3_config, f"Required S3 field missing: {field}"
            assert s3_config[field] is not None, f"S3 field {field} is None"
            assert s3_config[field] != "", f"S3 field {field} is empty string"
        
        # Test specific S3 values
        assert s3_config['region'] == 'us-east-1', f"S3 region should be us-east-1, got: {s3_config['region']}"
        assert s3_config['bucket'] == 'nsc-mvp1', f"S3 bucket should be nsc-mvp1, got: {s3_config['bucket']}"
        assert s3_config['prefix'] == 'pages/', f"S3 prefix should be pages/, got: {s3_config['prefix']}"
        
        # Test optional S3 fields
        optional_s3_fields = ['connection_timeout', 'max_retries']
        for field in optional_s3_fields:
            if field in s3_config:
                assert isinstance(s3_config[field], (int, float)), f"S3 {field} should be numeric"
    
    def test_required_postgres_configuration_fields(self, settings_loader):
        """Test that all required PostgreSQL fields are present"""
        postgres_config = settings_loader.get_postgres_config()
        
        # Test PostgreSQL config is loaded
        assert postgres_config is not None, "PostgreSQL configuration not loaded"
        assert isinstance(postgres_config, dict), "PostgreSQL configuration is not a dictionary"
        
        # Test required PostgreSQL fields
        required_postgres_fields = ['host', 'port', 'db_name', 'db_user', 'db_password']
        for field in required_postgres_fields:
            assert field in postgres_config, f"Required PostgreSQL field missing: {field}"
            assert postgres_config[field] is not None, f"PostgreSQL field {field} is None"
            assert postgres_config[field] != "", f"PostgreSQL field {field} is empty string"
        
        # Test specific PostgreSQL values
        assert postgres_config['host'].endswith('.rds.amazonaws.com'), "PostgreSQL host should be RDS endpoint"
        assert postgres_config['port'] == 5432, f"PostgreSQL port should be 5432, got: {postgres_config['port']}"
        assert postgres_config['db_name'] == 'vectorqa', f"Database name should be vectorqa, got: {postgres_config['db_name']}"
        assert postgres_config['db_user'] == 'vectorqa_user', f"Database user should be vectorqa_user, got: {postgres_config['db_user']}"
        
        # Test optional PostgreSQL fields
        optional_postgres_fields = ['ssl_mode', 'connection_pool_size', 'max_retries', 'retry_delay']
        for field in optional_postgres_fields:
            if field in postgres_config:
                if field in ['connection_pool_size', 'max_retries']:
                    assert isinstance(postgres_config[field], int), f"PostgreSQL {field} should be integer"
                elif field == 'retry_delay':
                    assert isinstance(postgres_config[field], (int, float)), f"PostgreSQL {field} should be numeric"
    
    def test_required_aws_configuration_fields(self, settings_loader):
        """Test that all required AWS fields are present"""
        aws_config = settings_loader.settings.aws
        
        # Test AWS config is loaded
        assert aws_config is not None, "AWS configuration not loaded"
        assert isinstance(aws_config, dict), "AWS configuration is not a dictionary"
        
        # Test required AWS fields
        required_aws_fields = ['region', 'bedrock']
        for field in required_aws_fields:
            assert field in aws_config, f"Required AWS field missing: {field}"
            assert aws_config[field] is not None, f"AWS field {field} is None"
        
        # Test AWS region
        assert aws_config['region'] == 'us-east-1', f"AWS region should be us-east-1, got: {aws_config['region']}"
        
        # Test Bedrock configuration exists
        bedrock_config = aws_config['bedrock']
        assert isinstance(bedrock_config, dict), "Bedrock configuration should be a dictionary"
        assert 'region' in bedrock_config, "Bedrock region missing"
        assert 'default_model' in bedrock_config, "Bedrock default_model missing"
    
    def test_all_top_level_sections_present(self, settings_loader):
        """Test that all expected top-level configuration sections are present"""
        expected_sections = [
            'postgres', 's3', 'aws', 'retry', 'cache', 'validation',
            'batch', 'metrics', 'logging', 'cost_optimization',
            'security', 'performance', 'development', 'integrations',
            'environments'
        ]
        
        for section in expected_sections:
            section_data = getattr(settings_loader.settings, section, None)
            assert section_data is not None, f"Required section missing: {section}"
            assert isinstance(section_data, dict), f"Section {section} should be a dictionary"
    
    def test_integrations_configuration(self, settings_loader):
        """Test integrations configuration contains expected fields"""
        integrations_config = settings_loader.get_integrations_config()
        
        assert integrations_config is not None, "Integrations configuration not loaded"
        assert isinstance(integrations_config, dict), "Integrations configuration is not a dictionary"
        
        # Test expected integration types
        expected_integrations = ['mlflow', 'wandb', 'prometheus']
        for integration in expected_integrations:
            assert integration in integrations_config, f"Integration missing: {integration}"
            assert isinstance(integrations_config[integration], dict), f"Integration {integration} should be a dictionary"
    
    def test_settings_validation(self, settings_loader):
        """Test that settings validation passes"""
        is_valid = settings_loader.validate_settings()
        assert is_valid is True, "Settings validation should pass"
    
    def test_environment_detection(self, settings_loader):
        """Test environment detection works correctly"""
        assert settings_loader.environment is not None, "Environment should be detected"
        assert isinstance(settings_loader.environment, str), "Environment should be a string"
        assert settings_loader.environment.lower() in ['development', 'production', 'testing'], f"Unknown environment: {settings_loader.environment}"
    
    def test_model_configuration_access(self, settings_loader):
        """Test model configuration can be accessed"""
        # Test getting configuration for a known model
        claude_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        model_config = settings_loader.get_model_config(claude_model)
        
        assert model_config is not None, "Model configuration should not be None"
        assert isinstance(model_config, dict), "Model configuration should be a dictionary"
        assert 'max_tokens' in model_config, "Model config should have max_tokens"
        assert 'temperature' in model_config, "Model config should have temperature"
    
    def test_retry_configuration(self, settings_loader):
        """Test retry configuration is properly loaded"""
        retry_config = settings_loader.get_retry_config()
        
        assert retry_config is not None, "Retry configuration not loaded"
        assert isinstance(retry_config, dict), "Retry configuration is not a dictionary"
        
        required_retry_fields = ['max_retries', 'base_delay', 'max_delay', 'backoff_factor']
        for field in required_retry_fields:
            assert field in retry_config, f"Required retry field missing: {field}"
            assert isinstance(retry_config[field], (int, float)), f"Retry field {field} should be numeric"
    
    def test_cache_configuration(self, settings_loader):
        """Test cache configuration is properly loaded"""
        cache_config = settings_loader.get_cache_config()
        
        assert cache_config is not None, "Cache configuration not loaded"
        assert isinstance(cache_config, dict), "Cache configuration is not a dictionary"
        
        required_cache_fields = ['enabled', 'cache_dir', 'expiration_hours']
        for field in required_cache_fields:
            assert field in cache_config, f"Required cache field missing: {field}"
        
        assert isinstance(cache_config['enabled'], bool), "Cache enabled should be boolean"
        assert isinstance(cache_config['expiration_hours'], (int, float)), "Cache expiration should be numeric"
    
    def test_logging_configuration(self, settings_loader):
        """Test logging configuration is properly loaded"""
        logging_config = settings_loader.get_logging_config()
        
        assert logging_config is not None, "Logging configuration not loaded"
        assert isinstance(logging_config, dict), "Logging configuration is not a dictionary"
        
        required_logging_fields = ['level', 'format']
        for field in required_logging_fields:
            assert field in logging_config, f"Required logging field missing: {field}"
        
        # Test log level is valid
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert logging_config['level'].upper() in valid_log_levels, f"Invalid log level: {logging_config['level']}"
    
    def test_metrics_configuration(self, settings_loader):
        """Test metrics configuration is properly loaded"""
        metrics_config = settings_loader.get_metrics_config()
        
        assert metrics_config is not None, "Metrics configuration not loaded"
        assert isinstance(metrics_config, dict), "Metrics configuration is not a dictionary"
        assert 'enabled' in metrics_config, "Metrics enabled field missing"
        assert isinstance(metrics_config['enabled'], bool), "Metrics enabled should be boolean"
    
    @patch.dict(os.environ, {'TIDYLLM_ENV': 'testing'})
    def test_environment_override(self):
        """Test environment can be overridden via environment variable"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        loader = SettingsLoader(str(admin_settings_path))
        assert loader.environment == 'testing', f"Environment should be testing, got: {loader.environment}"
    
    def test_missing_settings_file_handling(self):
        """Test graceful handling of missing settings file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_file = Path(temp_dir) / "missing.yaml"
            
            with pytest.raises(FileNotFoundError):
                SettingsLoader(str(missing_file))
    
    def test_settings_summary_generation(self, settings_loader):
        """Test that settings summary can be generated without errors"""
        # This test ensures print_summary doesn't crash
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            settings_loader.print_summary()
        
        summary_output = f.getvalue()
        assert len(summary_output) > 0, "Settings summary should produce output"
        assert "TidyLLM Settings Summary" in summary_output, "Summary should contain title"
        assert "PostgreSQL:" in summary_output, "Summary should contain PostgreSQL section"
        assert "S3:" in summary_output, "Summary should contain S3 section"

class TestSettingsIntegration:
    """Integration tests for settings with other TidyLLM components"""
    
    def test_settings_can_configure_boto3(self):
        """Test that S3 settings can be used to configure boto3"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        loader = SettingsLoader(str(admin_settings_path))
        s3_config = loader.get_s3_config()
        
        # Test that we can extract boto3 configuration
        assert 'region' in s3_config, "Region needed for boto3 configuration"
        
        # Mock boto3 configuration
        boto3_config = {
            'region_name': s3_config['region'],
            'config': {
                'connect_timeout': s3_config.get('connection_timeout', 30),
                'retries': {'max_attempts': s3_config.get('max_retries', 3)}
            }
        }
        
        assert boto3_config['region_name'] == 'us-east-1'
        assert boto3_config['config']['connect_timeout'] == 30
        assert boto3_config['config']['retries']['max_attempts'] == 3
    
    def test_settings_can_configure_postgres_connection(self):
        """Test that PostgreSQL settings can be used for database connections"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        loader = SettingsLoader(str(admin_settings_path))
        postgres_config = loader.get_postgres_config()
        
        # Test that we can construct a connection string
        connection_params = {
            'host': postgres_config['host'],
            'port': postgres_config['port'],
            'dbname': postgres_config['db_name'],
            'user': postgres_config['db_user'],
            'password': postgres_config['db_password']
        }
        
        # Construct connection string
        conn_string = f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}"
        
        assert conn_string.startswith('postgresql://'), "Should generate valid PostgreSQL connection string"
        assert 'vectorqa-cluster' in conn_string, "Should contain correct RDS endpoint"
        assert ':5432/' in conn_string, "Should contain correct port"

def test_priority_settings_load():
    """Priority test to ensure settings can be loaded (runs first due to 0_ prefix)"""
    admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
    
    # This test MUST pass for the application to function
    assert admin_settings_path.exists(), f"CRITICAL: Admin settings missing at {admin_settings_path}"
    
    loader = SettingsLoader(str(admin_settings_path))
    assert loader is not None, "CRITICAL: SettingsLoader failed to initialize"
    
    # Test critical configurations
    s3_config = loader.get_s3_config()
    postgres_config = loader.get_postgres_config()
    
    assert s3_config, "CRITICAL: S3 configuration missing"
    assert postgres_config, "CRITICAL: PostgreSQL configuration missing"
    
    print("SUCCESS: Priority settings test passed - configuration is valid")

if __name__ == "__main__":
    # Run the priority test directly
    test_priority_settings_load()
    print("PASSED: Settings can be loaded successfully from admin/settings.yaml")