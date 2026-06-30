#!/usr/bin/env python3
"""
AWS Session Restart Utility
===========================

Restarts AWS sessions to resolve authentication issues and token expiration.
Handles both environment variables and boto3 session reinitialization.

Usage:
    python restart_aws_session.py
    python restart_aws_session.py --verify
    python restart_aws_session.py --clear-cache
"""

import os
import sys
import boto3
import time
from pathlib import Path
from datetime import datetime
import json

class AWSSessionManager:
    """Manages AWS session restart and verification"""
    
    def __init__(self):
        # REMOVED: Hardcoded credentials cleaned (key rotated 2026-06-30)
        self.credentials = {
            'AWS_ACCESS_KEY_ID': os.environ.get('AWS_ACCESS_KEY_ID', ''),
            'AWS_SECRET_ACCESS_KEY': os.environ.get('AWS_SECRET_ACCESS_KEY', ''),
            'AWS_DEFAULT_REGION': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        }
        
    def restart_session(self, clear_cache=True, verify=True):
        """Restart AWS session completely"""
        
        print("="*50)
        print("AWS SESSION RESTART UTILITY")
        print("="*50)
        
        # Step 1: Clear environment variables
        print("[STEP 1] Clearing existing AWS environment variables...")
        self._clear_aws_env_vars()
        
        # Step 2: Clear boto3 session cache
        if clear_cache:
            print("[STEP 2] Clearing boto3 session cache...")
            self._clear_boto3_cache()
        
        # Step 3: Set fresh environment variables
        print("[STEP 3] Setting fresh AWS credentials...")
        self._set_fresh_credentials()
        
        # Step 4: Initialize new boto3 session
        print("[STEP 4] Initializing new boto3 session...")
        new_session = self._init_new_session()
        
        # Step 5: Verify session works
        if verify:
            print("[STEP 5] Verifying AWS session...")
            verification_result = self._verify_session(new_session)
            
            if verification_result['success']:
                print("[SUCCESS] AWS session restart complete!")
                print(f"User ARN: {verification_result.get('user_arn', 'Unknown')}")
                print(f"Account: {verification_result.get('account', 'Unknown')}")
                return True
            else:
                print("[FAILED] AWS session verification failed!")
                print(f"Error: {verification_result.get('error', 'Unknown error')}")
                return False
        
        print("[COMPLETE] AWS session restart complete (verification skipped)")
        return True
    
    def _clear_aws_env_vars(self):
        """Clear existing AWS environment variables"""
        aws_vars = [
            'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION',
            'AWS_SESSION_TOKEN', 'AWS_SECURITY_TOKEN', 'AWS_PROFILE',
            'AWS_CONFIG_FILE', 'AWS_SHARED_CREDENTIALS_FILE'
        ]
        
        cleared_count = 0
        for var in aws_vars:
            if var in os.environ:
                del os.environ[var]
                cleared_count += 1
                print(f"  Cleared: {var}")
        
        print(f"  Total cleared: {cleared_count} environment variables")
    
    def _clear_boto3_cache(self):
        """Clear boto3 cached sessions and credentials"""
        try:
            # Clear the default session
            boto3.DEFAULT_SESSION = None
            
            # Clear any cached credentials
            if hasattr(boto3.session, '_session'):
                boto3.session._session = None
            
            print("  boto3 cache cleared successfully")
        except Exception as e:
            print(f"  Warning: Could not clear boto3 cache: {e}")
    
    def _set_fresh_credentials(self):
        """Set fresh AWS credentials in environment"""
        for key, value in self.credentials.items():
            os.environ[key] = value
            print(f"  Set: {key}")
        
        print("  Fresh credentials loaded")
    
    def _init_new_session(self):
        """Initialize completely new boto3 session"""
        try:
            # Create new session with explicit credentials
            session = boto3.Session(
                aws_access_key_id=self.credentials['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=self.credentials['AWS_SECRET_ACCESS_KEY'],
                region_name=self.credentials['AWS_DEFAULT_REGION']
            )
            
            print("  New boto3 session created")
            return session
            
        except Exception as e:
            print(f"  Error creating session: {e}")
            return None
    
    def _verify_session(self, session):
        """Verify AWS session is working"""
        if not session:
            return {'success': False, 'error': 'No session provided'}
        
        try:
            # Test 1: Get caller identity
            sts_client = session.client('sts')
            identity = sts_client.get_caller_identity()
            
            # Test 2: List S3 buckets (basic permission test)
            s3_client = session.client('s3')
            response = s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
            
            print(f"  Identity check: SUCCESS")
            print(f"  User ARN: {identity.get('Arn', 'Unknown')}")
            print(f"  Account ID: {identity.get('Account', 'Unknown')}")
            print(f"  S3 access: SUCCESS ({len(buckets)} buckets accessible)")
            
            return {
                'success': True,
                'user_arn': identity.get('Arn'),
                'account': identity.get('Account'),
                'buckets': buckets
            }
            
        except Exception as e:
            print(f"  Verification failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_s3_operations(self, bucket_name='nsc-mvp1'):
        """Test basic S3 operations with restarted session"""
        
        print(f"\n[S3_TEST] Testing S3 operations on bucket: {bucket_name}")
        
        try:
            s3_client = boto3.client('s3')
            
            # Test 1: Head bucket (basic access)
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"  Bucket access: SUCCESS")
            
            # Test 2: Try to list objects (may fail due to permissions)
            try:
                response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                print(f"  List objects: SUCCESS")
            except Exception as e:
                print(f"  List objects: DENIED ({str(e)[:50]}...)")
            
            # Test 3: Generate presigned URL (should work)
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': 'test-key'},
                ExpiresIn=3600
            )
            print(f"  Presigned URL generation: SUCCESS")
            
            return True
            
        except Exception as e:
            print(f"  S3 operations failed: {e}")
            return False
    
    def create_session_status_report(self):
        """Create detailed session status report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'environment_variables': {},
            'session_status': {},
            's3_access': {}
        }
        
        # Check environment variables
        aws_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
        for var in aws_env_vars:
            value = os.environ.get(var, 'NOT_SET')
            # Mask sensitive values
            if 'SECRET' in var:
                value = '***MASKED***' if value != 'NOT_SET' else 'NOT_SET'
            report['environment_variables'][var] = value
        
        # Test session
        try:
            session = boto3.Session()
            sts_client = session.client('sts')
            identity = sts_client.get_caller_identity()
            
            report['session_status'] = {
                'status': 'SUCCESS',
                'user_arn': identity.get('Arn'),
                'account_id': identity.get('Account'),
                'user_id': identity.get('UserId')
            }
        except Exception as e:
            report['session_status'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        # Test S3 access
        try:
            s3_client = boto3.client('s3')
            buckets_response = s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in buckets_response.get('Buckets', [])]
            
            report['s3_access'] = {
                'status': 'SUCCESS',
                'accessible_buckets': buckets,
                'bucket_count': len(buckets)
            }
        except Exception as e:
            report['s3_access'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        return report
    
    def save_session_report(self, report):
        """Save session status report"""
        report_path = Path("aws_session_status.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Session report saved: {report_path}")
        return report_path

def main():
    """Main function with command line options"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='AWS Session Restart Utility')
    parser.add_argument('--verify', action='store_true', help='Verify session after restart')
    parser.add_argument('--clear-cache', action='store_true', default=True, help='Clear boto3 cache (default: True)')
    parser.add_argument('--no-clear-cache', action='store_true', help='Skip clearing boto3 cache')
    parser.add_argument('--test-s3', action='store_true', help='Test S3 operations')
    parser.add_argument('--report', action='store_true', help='Generate session status report')
    parser.add_argument('--bucket', default='nsc-mvp1', help='S3 bucket for testing (default: nsc-mvp1)')
    
    args = parser.parse_args()
    
    # Handle cache clearing logic
    clear_cache = args.clear_cache and not args.no_clear_cache
    verify = args.verify or args.test_s3  # Auto-verify if testing S3
    
    # Initialize session manager
    session_manager = AWSSessionManager()
    
    # Generate report if requested
    if args.report:
        print("Generating AWS session status report...")
        report = session_manager.create_session_status_report()
        session_manager.save_session_report(report)
        
        print(f"\\nCurrent Session Status:")
        print(f"  AWS Identity: {report['session_status'].get('status', 'Unknown')}")
        print(f"  S3 Access: {report['s3_access'].get('status', 'Unknown')}")
        
        if not args.verify and not args.test_s3:
            return
    
    # Restart session
    success = session_manager.restart_session(
        clear_cache=clear_cache,
        verify=verify
    )
    
    # Test S3 operations if requested
    if args.test_s3 and success:
        session_manager.test_s3_operations(bucket_name=args.bucket)
    
    if success:
        print(f"\\n{('='*50)}")
        print("AWS SESSION READY FOR USE")
        print("="*50)
        print("You can now use boto3, AWS CLI, and TidyLLM S3 operations")
    else:
        print(f"\\n{('='*50)}")
        print("AWS SESSION RESTART FAILED")
        print("="*50)
        print("Check your credentials and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()