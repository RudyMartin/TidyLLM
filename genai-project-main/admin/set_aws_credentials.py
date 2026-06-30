#!/usr/bin/env python3
"""
TidyLLM AWS Credentials Setup (Python)
======================================
Sets AWS credentials in the current Python environment
"""

import os
import sys

def set_aws_credentials():
    """Set AWS credentials as environment variables."""
    
    print("Setting TidyLLM AWS credentials...")
    
    # Set AWS environment variables
    # REMOVED: Hardcoded credentials cleaned (key rotated 2026-06-30)
    print("ERROR: Hardcoded credentials have been removed from this script.")
    print("Source credentials externally before running.")
    sys.exit(1)
    
    print("AWS credentials set for current Python session:")
    print(f"  Access Key: {os.environ['AWS_ACCESS_KEY_ID'][:10]}...")
    print(f"  Region: {os.environ['AWS_DEFAULT_REGION']}")
    
    # Test credentials
    try:
        import boto3
        
        print("\nTesting AWS connectivity...")
        
        # Test S3
        s3 = boto3.client('s3')
        buckets = s3.list_buckets()
        print(f"[SUCCESS] S3 connection successful - {len(buckets['Buckets'])} buckets accessible")
        
        # Test identity
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"[SUCCESS] AWS Identity confirmed - Account: {identity['Account']}")
        print(f"[SUCCESS] User: {identity['Arn'].split('/')[-1]}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] AWS connectivity test failed: {e}")
        return False

def main():
    """Main function."""
    success = set_aws_credentials()
    
    if success:
        print("\n" + "="*50)
        print("AWS CREDENTIALS SUCCESSFULLY CONFIGURED")
        print("TidyLLM can now use real AWS services")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("AWS CREDENTIAL SETUP FAILED")
        print("Check your network connection and credentials")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()