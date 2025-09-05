#!/usr/bin/env python3
"""Test S3 bucket access and permissions"""

from dotenv import load_dotenv

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

load_dotenv()

import boto3
from botocore.exceptions import ClientError

def test_bucket_access():
    s3 = boto3.client('s3')
    bucket_name = s3_config["bucket"]
    
    print(f"Testing access to bucket: {bucket_name}")
    print("=" * 50)
    
    # Test 1: Check if bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"SUCCESS: Bucket '{bucket_name}' exists and is accessible")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"ERROR: Bucket '{bucket_name}' does not exist")
        elif error_code == '403':
            print(f"ERROR: Access denied to bucket '{bucket_name}'")
            print("   You need s3:ListBucket permission for this bucket")
        else:
            print(f"ERROR: Error accessing bucket: {e}")
        return False
    
    # Test 2: List objects in bucket
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix='papers/', MaxKeys=5)
        count = response.get('KeyCount', 0)
        print(f"SUCCESS: Can list objects - found {count} objects in papers/ prefix")
    except ClientError as e:
        print(f"ERROR: Cannot list objects: {e}")
    
    # Test 3: Test upload permission
    test_key = 'papers/test-permission-check.txt'
    try:
        s3.put_object(Bucket=bucket_name, Key=test_key, Body=b'test')
        print(f"SUCCESS: Can upload objects to {bucket_name}/papers/")
        # Clean up test file
        s3.delete_object(Bucket=bucket_name, Key=test_key)
        print(f"SUCCESS: Can delete objects from {bucket_name}/papers/")
    except ClientError as e:
        print(f"ERROR: Cannot upload objects: {e}")
        print("   You need s3:PutObject permission for this bucket")
    
    return True

if __name__ == '__main__':
    test_bucket_access()