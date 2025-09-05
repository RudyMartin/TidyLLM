#!/usr/bin/env python3
"""List accessible S3 buckets"""

from dotenv import load_dotenv

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

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
if s3_config["bucket"] in bucket_names:
    print('\nSUCCESS: nsc-mvp1 bucket is in your accessible buckets list')
else:
    print('\nWARNING: nsc-mvp1 bucket is NOT in your accessible buckets list')
    print('This means either:')
    print('  1. The bucket does not exist')
    print('  2. The bucket is in a different AWS account')
    print('  3. You need additional permissions to see this bucket')