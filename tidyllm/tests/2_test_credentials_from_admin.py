#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 2: Credentials from Admin Settings

Verifies that credentials and configurations are properly loaded
from admin/settings.yaml and can be used for service connections.
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidyllm.settings_loader import SettingsLoader

class TestCredentialsFromAdmin:
    """Test suite for verifying credentials from admin settings"""
    
    @pytest.fixture
    def settings_loader(self):
        """Fixture providing initialized SettingsLoader"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        return SettingsLoader(str(admin_settings_path))
    
    def test_admin_settings_file_accessible(self, settings_loader):
        """Test that admin settings file is accessible and loaded"""
        assert settings_loader is not None, "SettingsLoader should be initialized"
        assert settings_loader.settings is not None, "Settings should be loaded"
        
        settings_path = settings_loader.settings_path
        assert settings_path.exists(), f"Settings file should exist: {settings_path}"
        print(f"✅ Admin settings loaded from: {settings_path}")
    
    def test_s3_credentials_configuration(self, settings_loader):
        """Test S3 configuration from admin settings"""
        s3_config = settings_loader.get_s3_config()
        
        assert s3_config is not None, "S3 configuration should be loaded"
        assert isinstance(s3_config, dict), "S3 config should be a dictionary"
        
        # Test required S3 fields
        required_fields = ['region', 'bucket', 'prefix']
        for field in required_fields:
            assert field in s3_config, f"S3 config missing required field: {field}"
            assert s3_config[field] is not None, f"S3 {field} should not be None"
            print(f"✅ S3 {field}: {s3_config[field]}")
        
        # Test S3 connection parameters
        optional_fields = ['connection_timeout', 'max_retries']
        for field in optional_fields:
            if field in s3_config:
                print(f"✅ S3 {field}: {s3_config[field]}")
        
        # Verify S3 region is valid AWS region
        assert s3_config['region'].startswith('us-'), f"Expected US region, got: {s3_config['region']}"
        
        return s3_config
    
    def test_postgresql_credentials_configuration(self, settings_loader):
        """Test PostgreSQL configuration from admin settings"""
        postgres_config = settings_loader.get_postgres_config()
        
        assert postgres_config is not None, "PostgreSQL configuration should be loaded"
        assert isinstance(postgres_config, dict), "PostgreSQL config should be a dictionary"
        
        # Test required PostgreSQL fields
        required_fields = ['host', 'port', 'db_name', 'db_user', 'db_password']
        for field in required_fields:
            assert field in postgres_config, f"PostgreSQL config missing required field: {field}"
            assert postgres_config[field] is not None, f"PostgreSQL {field} should not be None"
            
            # Don't print password, but verify it exists
            if 'password' in field.lower():
                print(f"✅ PostgreSQL {field}: ***hidden***")
            else:
                print(f"✅ PostgreSQL {field}: {postgres_config[field]}")
        
        # Test connection parameters
        assert postgres_config['port'] == 5432, f"Expected PostgreSQL port 5432, got: {postgres_config['port']}"
        assert postgres_config['host'].endswith('.rds.amazonaws.com'), "Expected AWS RDS endpoint"
        
        # Test optional fields
        optional_fields = ['ssl_mode', 'connection_pool_size', 'max_retries']
        for field in optional_fields:
            if field in postgres_config:
                print(f"✅ PostgreSQL {field}: {postgres_config[field]}")
        
        return postgres_config
    
    def test_aws_bedrock_credentials_configuration(self, settings_loader):
        """Test AWS Bedrock configuration from admin settings"""
        aws_config = settings_loader.settings.aws
        
        assert aws_config is not None, "AWS configuration should be loaded"
        assert isinstance(aws_config, dict), "AWS config should be a dictionary"
        
        # Test AWS region
        assert 'region' in aws_config, "AWS region should be configured"
        print(f"✅ AWS Region: {aws_config['region']}")
        
        # Test KMS key (if available)
        if 'kms_key_id' in aws_config:
            kms_key = aws_config['kms_key_id']
            assert kms_key.startswith('arn:aws:kms:'), f"Invalid KMS key format: {kms_key}"
            print(f"✅ KMS Key configured: {kms_key[:50]}...")
        
        # Test Bedrock configuration
        bedrock_config = aws_config.get('bedrock', {})
        if bedrock_config:
            print(f"✅ Bedrock region: {bedrock_config.get('region', 'Not set')}")
            print(f"✅ Bedrock default model: {bedrock_config.get('default_model', 'Not set')}")
        
        return aws_config
    
    def test_environment_variables_from_settings(self, settings_loader):
        """Test that settings can set environment variables"""
        s3_config = settings_loader.get_s3_config()
        aws_config = settings_loader.settings.aws
        
        # Check if AWS region is properly configured
        expected_region = aws_config.get('region', s3_config.get('region'))
        if expected_region:
            print(f"✅ Expected AWS region from settings: {expected_region}")
        
        # Test environment variable setting (simulated)
        test_env_vars = {
            'AWS_DEFAULT_REGION': expected_region,
            'TIDYLLM_ENV': 'testing'
        }
        
        for var_name, expected_value in test_env_vars.items():
            if expected_value:
                print(f"✅ Can set {var_name}={expected_value}")
    
    def test_connection_string_generation(self, settings_loader):
        """Test generating connection strings from settings"""
        postgres_config = settings_loader.get_postgres_config()
        s3_config = settings_loader.get_s3_config()
        
        # Test PostgreSQL connection string generation
        if postgres_config:
            conn_params = {
                'host': postgres_config['host'],
                'port': postgres_config['port'],
                'dbname': postgres_config['db_name'],
                'user': postgres_config['db_user'],
                'password': postgres_config['db_password']
            }
            
            conn_string = (
                f"postgresql://{conn_params['user']}:{conn_params['password']}@"
                f"{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
            )
            
            assert conn_string.startswith('postgresql://'), "Should generate valid PostgreSQL URL"
            assert str(postgres_config['port']) in conn_string, "Should contain port number"
            print(f"✅ PostgreSQL connection string format: postgresql://user:***@host:port/db")
        
        # Test S3 configuration for boto3
        if s3_config:
            boto3_config = {
                'region_name': s3_config['region'],
                'bucket_name': s3_config['bucket'],
                'prefix': s3_config['prefix']
            }
            
            print(f"✅ S3 boto3 config ready: region={boto3_config['region_name']}, bucket={boto3_config['bucket_name']}")
    
    def test_mlflow_integration_settings(self, settings_loader):
        """Test MLflow integration settings from admin config"""
        integrations_config = settings_loader.get_integrations_config()
        
        assert integrations_config is not None, "Integrations config should be loaded"
        
        # Test MLflow configuration
        mlflow_config = integrations_config.get('mlflow', {})
        if mlflow_config:
            print(f"✅ MLflow enabled: {mlflow_config.get('enabled', False)}")
            if mlflow_config.get('tracking_uri'):
                print(f"✅ MLflow tracking URI configured")
            print(f"✅ MLflow experiment: {mlflow_config.get('experiment_name', 'Not set')}")
        
        # Test other integrations
        wandb_config = integrations_config.get('wandb', {})
        if wandb_config:
            print(f"✅ WandB enabled: {wandb_config.get('enabled', False)}")
        
        prometheus_config = integrations_config.get('prometheus', {})
        if prometheus_config:
            print(f"✅ Prometheus enabled: {prometheus_config.get('enabled', False)}")
    
    def test_settings_validation_comprehensive(self, settings_loader):
        """Test comprehensive settings validation"""
        validation_result = settings_loader.validate_settings()
        assert validation_result is True, "Settings should pass comprehensive validation"
        print("✅ All settings passed comprehensive validation")
    
    def test_credentials_security_check(self, settings_loader):
        """Test that credentials are handled securely"""
        postgres_config = settings_loader.get_postgres_config()
        
        # Verify password is not empty or default
        if postgres_config and 'db_password' in postgres_config:
            password = postgres_config['db_password']
            assert len(password) > 8, "Password should be reasonably long"
            assert password != 'password', "Password should not be default"
            assert password != 'admin', "Password should not be admin"
            print("✅ Database password appears to be secure")
        
        # Check that sensitive data is not logged
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            settings_loader.print_summary()
        
        summary_output = f.getvalue()
        assert 'db_password' not in summary_output, "Password should not appear in summary"
        print("✅ Sensitive data not exposed in logs")

def test_priority_credentials_check():
    """Priority test to ensure credentials can be loaded"""
    try:
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        loader = SettingsLoader(str(admin_settings_path))
        
        s3_config = loader.get_s3_config()
        postgres_config = loader.get_postgres_config()
        
        assert s3_config, "CRITICAL: S3 credentials not loaded"
        assert postgres_config, "CRITICAL: PostgreSQL credentials not loaded"
        
        print("SUCCESS: Credentials loaded from admin settings")
    except Exception as e:
        pytest.fail(f"CRITICAL: Failed to load credentials from admin: {e}")

if __name__ == "__main__":
    # Run the priority test directly
    test_priority_credentials_check()
    print("PASSED: Credentials successfully loaded from admin/settings.yaml")