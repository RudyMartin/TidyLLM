#!/usr/bin/env python3
"""Test access to dsai-2025-asu bucket"""

from dotenv import load_dotenv
load_dotenv()

import boto3

s3 = boto3.client('s3')
bucket = 'dsai-2025-asu'

print(f'Testing access to {bucket} bucket...')
print('=' * 50)

try:
    # List objects in papers/ prefix
    response = s3.list_objects_v2(Bucket=bucket, Prefix='papers/', MaxKeys=20)
    count = response.get('KeyCount', 0)
    print(f'SUCCESS: Found {count} objects in papers/ prefix')
    
    if response.get('Contents'):
        print('\nFirst 10 files:')
        for obj in response['Contents'][:10]:
            size_mb = obj['Size'] / (1024 * 1024)
            print(f'  - {obj["Key"]} ({size_mb:.2f} MB)')
    else:
        print('No files found in papers/ prefix')
        
    # Test upload permission
    print('\nTesting upload permission...')
    test_key = 'papers/test-permission.txt'
    s3.put_object(Bucket=bucket, Key=test_key, Body=b'test')
    print(f'SUCCESS: Can upload to {bucket}/papers/')
    
    # Clean up
    s3.delete_object(Bucket=bucket, Key=test_key)
    print(f'SUCCESS: Can delete from {bucket}/papers/')
    
except Exception as e:
    print(f'ERROR: {e}')