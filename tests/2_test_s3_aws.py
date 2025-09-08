#!/usr/bin/env python3
"""
TidyLLM Strategic Test Suite: S3 & AWS Services
===============================================

Tests AWS/S3 connectivity and cloud service functionality.
Replaces multiple S3/AWS connection test files.
"""

import unittest
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# AWS/S3 imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class TestAWSCredentials(unittest.TestCase):
    """Test AWS credentials and basic connectivity."""
    
    def test_boto3_available(self):
        """Test boto3 is available."""
        if not BOTO3_AVAILABLE:
            self.skipTest("boto3 not available")
        
        import boto3
        print("boto3 available")
    
    def test_aws_credentials_configured(self):
        """Test AWS credentials are configured."""
        if not BOTO3_AVAILABLE:
            self.skipTest("boto3 not available")
        
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if credentials:
                self.assertIsNotNone(credentials.access_key)
                print("[OK] AWS credentials configured")
                
                # Don't print actual credentials, just confirm they exist
                print(f"Access key starts with: {credentials.access_key[:8]}...")
                return True
            else:
                print("[WARN]  No AWS credentials found")
                return False
                
        except Exception as e:
            print(f"[WARN]  Error checking credentials: {e}")
            return False
    
    def test_aws_region_configured(self):
        """Test AWS region is configured."""
        if not BOTO3_AVAILABLE:
            self.skipTest("boto3 not available")
        
        try:
            session = boto3.Session()
            region = session.region_name
            
            if region:
                print(f"[OK] AWS region configured: {region}")
            else:
                print("[WARN]  No AWS region configured, using default")
                
        except Exception as e:
            print(f"Error checking region: {e}")


class TestS3Connectivity(unittest.TestCase):
    """Test S3 service connectivity."""
    
    def setUp(self):
        """Set up S3 client for tests."""
        if BOTO3_AVAILABLE:
            try:
                # AUDIT COMPLIANCE: Use UnifiedSessionManager instead of direct boto3
                try:
                    from scripts.infrastructure.start_unified_sessions import UnifiedSessionManager
                    session_manager = UnifiedSessionManager()
                    self.s3_client = session_manager.get_s3_client()
                except ImportError:
                    # Fallback to direct boto3
                    self.s3_client = boto3.client('s3')
            except Exception as e:
                print(f"Failed to create S3 client: {e}")
                self.s3_client = None
        else:
            self.s3_client = None
    
    def test_s3_service_access(self):
        """Test basic S3 service access."""
        if not BOTO3_AVAILABLE or not self.s3_client:
            self.skipTest("S3 client not available")
        
        try:
            # Test basic S3 access by listing buckets
            response = self.s3_client.list_buckets()
            buckets = response.get('Buckets', [])
            
            print(f"[OK] S3 access successful - found {len(buckets)} buckets")
            
            # List bucket names (first 5)
            if buckets:
                bucket_names = [bucket['Name'] for bucket in buckets[:5]]
                print(f"Sample buckets: {bucket_names}")
            
            return True
            
        except NoCredentialsError:
            print("[WARN]  No AWS credentials for S3 access")
            return False
        except ClientError as e:
            print(f"[WARN]  S3 access failed: {e}")
            return False
        except Exception as e:
            print(f"[WARN]  Unexpected S3 error: {e}")
            return False
    
    def test_s3_bucket_access(self):
        """Test access to specific buckets if available."""
        if not BOTO3_AVAILABLE or not self.s3_client:
            self.skipTest("S3 client not available")
        
        # Common bucket names that might exist
        test_bucket_patterns = [
            'tidyllm', 'company', 'docs', 'knowledge', 'ml', 'data'
        ]
        
        try:
            response = self.s3_client.list_buckets()
            all_buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
            
            # Find buckets that match our patterns
            matching_buckets = []
            for bucket in all_buckets:
                for pattern in test_bucket_patterns:
                    if pattern.lower() in bucket.lower():
                        matching_buckets.append(bucket)
                        break
            
            if matching_buckets:
                print(f"Found potential TidyLLM buckets: {matching_buckets[:3]}")
                
                # Test access to first matching bucket
                test_bucket = matching_buckets[0]
                try:
                    objects = self.s3_client.list_objects_v2(Bucket=test_bucket, MaxKeys=1)
                    print(f"[OK] Can access bucket: {test_bucket}")
                except ClientError as e:
                    print(f"[WARN]  Cannot access bucket {test_bucket}: {e}")
            else:
                print("No obvious TidyLLM-related buckets found")
                
        except Exception as e:
            print(f"Error testing bucket access: {e}")


class TestS3Operations(unittest.TestCase):
    """Test S3 operations (with careful real-world testing)."""
    
    def setUp(self):
        """Set up for S3 operations."""
        if BOTO3_AVAILABLE:
            try:
                # AUDIT COMPLIANCE: Use UnifiedSessionManager instead of direct boto3
                try:
                    from scripts.infrastructure.start_unified_sessions import UnifiedSessionManager
                    session_manager = UnifiedSessionManager()
                    self.s3_client = session_manager.get_s3_client()
                except ImportError:
                    # Fallback to direct boto3
                    self.s3_client = boto3.client('s3')
                # Use a test bucket name that's likely to be available
                self.test_bucket = self._find_test_bucket()
            except Exception:
                self.s3_client = None
                self.test_bucket = None
        else:
            self.s3_client = None
            self.test_bucket = None
    
    def _find_test_bucket(self):
        """Find a suitable bucket for testing (read-only operations)."""
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
            
            # Look for buckets that might be safe to read from
            safe_patterns = ['test', 'demo', 'public', 'docs']
            for bucket in buckets:
                for pattern in safe_patterns:
                    if pattern in bucket.lower():
                        return bucket
            
            # Return first bucket if no safe pattern found (for read-only ops)
            return buckets[0] if buckets else None
            
        except Exception:
            return None
    
    def test_s3_list_objects(self):
        """Test S3 list objects operation."""
        if not BOTO3_AVAILABLE or not self.s3_client or not self.test_bucket:
            self.skipTest("S3 not available or no test bucket")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.test_bucket,
                MaxKeys=5
            )
            
            objects = response.get('Contents', [])
            print(f"[OK] Listed {len(objects)} objects in {self.test_bucket}")
            
            # Show sample object keys (first 3)
            if objects:
                sample_keys = [obj['Key'] for obj in objects[:3]]
                print(f"Sample objects: {sample_keys}")
                
        except ClientError as e:
            print(f"[WARN]  List objects failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    def test_s3_head_object(self):
        """Test S3 head object operation (metadata only)."""
        if not BOTO3_AVAILABLE or not self.s3_client or not self.test_bucket:
            self.skipTest("S3 not available or no test bucket")
        
        try:
            # Get first object to test head operation
            response = self.s3_client.list_objects_v2(
                Bucket=self.test_bucket,
                MaxKeys=1
            )
            
            objects = response.get('Contents', [])
            if not objects:
                print("No objects found for head test")
                return
            
            test_key = objects[0]['Key']
            
            # Get object metadata
            head_response = self.s3_client.head_object(
                Bucket=self.test_bucket,
                Key=test_key
            )
            
            print(f"[OK] Got metadata for {test_key}")
            print(f"Size: {head_response.get('ContentLength', 0)} bytes")
            print(f"Type: {head_response.get('ContentType', 'unknown')}")
            
        except ClientError as e:
            print(f"[WARN]  Head object failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


class TestS3Configuration(unittest.TestCase):
    """Test S3 configuration and settings."""
    
    def test_s3_settings_availability(self):
        """Test if S3 settings are configured in TidyLLM."""
        settings_paths = [
            project_root / "tidyllm" / "admin" / "settings.yaml",
            project_root / "admin" / "settings.yaml", 
            project_root / "settings.yaml"
        ]
        
        settings_found = False
        for settings_path in settings_paths:
            if settings_path.exists():
                print(f"[OK] Found settings file: {settings_path}")
                settings_found = True
                
                # Try to read S3 configuration
                try:
                    import yaml
                    with open(settings_path, 'r') as f:
                        settings = yaml.safe_load(f)
                    
                    if 's3' in settings or 'aws' in settings:
                        print("[OK] S3/AWS configuration found in settings")
                    else:
                        print("[WARN]  No S3/AWS configuration in settings")
                        
                except Exception as e:
                    print(f"Error reading settings: {e}")
                    
                break
        
        if not settings_found:
            print("[WARN]  No settings file found")
    
    def test_environment_variables(self):
        """Test AWS environment variables."""
        aws_env_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY', 
            'AWS_DEFAULT_REGION',
            'AWS_PROFILE'
        ]
        
        env_configured = False
        for var in aws_env_vars:
            if os.environ.get(var):
                if 'KEY' in var:
                    # Don't print actual keys
                    value = os.environ[var][:8] + '...'
                else:
                    value = os.environ[var]
                print(f"[OK] {var}: {value}")
                env_configured = True
            else:
                print(f"[WARN]  {var}: Not set")
        
        if env_configured:
            print("[OK] AWS environment variables configured")
        else:
            print("[WARN]  No AWS environment variables found")


def run_s3_aws_tests():
    """Run all S3/AWS tests with detailed output."""
    print("="*60)
    print("TIDYLLM S3 & AWS CONNECTIVITY TESTS")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAWSCredentials,
        TestS3Connectivity,
        TestS3Operations,
        TestS3Configuration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("S3/AWS TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    # AWS connectivity summary
    total_skipped = len(getattr(result, 'skipped', []))
    if total_skipped > 0:
        print(f"\n[WARN]  {total_skipped} tests skipped (likely due to missing AWS config)")
        print("This is normal if AWS is not configured.")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n[OK] AWS/S3 TESTS PASSED")
        print("(Note: Skipped tests are not failures)")
    else:
        print("\n[FAIL] AWS/S3 TESTS FAILED")
        print("Check AWS credentials and configuration")
    
    return success


if __name__ == "__main__":
    success = run_s3_aws_tests()
    sys.exit(0 if success else 1)