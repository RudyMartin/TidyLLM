#!/usr/bin/env python3
"""Check bucket status and ownership"""

from dotenv import load_dotenv
load_dotenv()

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')
bucket_name = 'nsc-mvp1'

print(f"Checking status of bucket: {bucket_name}")
print("=" * 50)

# Check if bucket exists globally (S3 bucket names are globally unique)
try:
    s3.head_bucket(Bucket=bucket_name)
    print(f"SUCCESS: Bucket '{bucket_name}' exists and you have access")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        print(f"RESULT: Bucket '{bucket_name}' does not exist")
        print("\nYou can create this bucket since the name is available")
        
        # Try to create the bucket
        try:
            print(f"\nAttempting to create bucket '{bucket_name}'...")
            if s3.meta.region_name == 'us-east-1':
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': s3.meta.region_name}
                )
            print(f"SUCCESS: Created bucket '{bucket_name}'")
        except Exception as create_error:
            print(f"ERROR: Could not create bucket: {create_error}")
            
    elif error_code == '403':
        print(f"RESULT: Bucket '{bucket_name}' exists but you don't have access")
        print("\nThis means either:")
        print("  1. The bucket is owned by another AWS account")
        print("  2. The bucket is in your account but IAM permissions are missing")
        print("  3. There's a bucket policy blocking access")
        print("\nRecommended actions:")
        print("  - Use a different bucket name (e.g., 'nsc-mvp1-yourusername')")
        print("  - Or request access from the bucket owner")
    else:
        print(f"ERROR: Unexpected error: {e}")

# List your accessible buckets to confirm
print("\n" + "=" * 50)
print("Your accessible buckets:")
response = s3.list_buckets()
for bucket in response['Buckets']:
    print(f"  - {bucket['Name']}")
    
print(f"\nTotal accessible buckets: {len(response['Buckets'])}")