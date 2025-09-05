#!/usr/bin/env python3
"""List accessible S3 buckets"""

from dotenv import load_dotenv
load_dotenv()

import boto3

s3 = boto3.client('s3')
response = s3.list_buckets()

print('Accessible S3 buckets:')
print('=' * 50)
for bucket in response['Buckets']:
    print(f'  - {bucket["Name"]}')
    
print(f'\nTotal: {len(response["Buckets"])} buckets')

# Check if nsc-mvp1 is in the list
bucket_names = [b['Name'] for b in response['Buckets']]
if 'nsc-mvp1' in bucket_names:
    print('\nSUCCESS: nsc-mvp1 bucket is in your accessible buckets list')
else:
    print('\nWARNING: nsc-mvp1 bucket is NOT in your accessible buckets list')
    print('This means either:')
    print('  1. The bucket does not exist')
    print('  2. The bucket is in a different AWS account')
    print('  3. You need additional permissions to see this bucket')