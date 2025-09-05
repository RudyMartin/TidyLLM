#!/usr/bin/env python3
"""
Test AWS S3 Connection with credentials from admin folder
"""

import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def setup_aws_credentials():
    """Set AWS credentials from admin config"""
    os.environ['AWS_ACCESS_KEY_ID'] = 'REMOVED_AWS_KEY'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'REMOVED_AWS_SECRET'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    print("[OK] AWS credentials configured:")
    print(f"  Access Key: {os.environ['AWS_ACCESS_KEY_ID'][:10]}...")
    print(f"  Region: {os.environ['AWS_DEFAULT_REGION']}")

def test_s3_connection():
    """Test S3 connection"""
    try:
        s3 = boto3.client('s3')
        response = s3.list_buckets()
        
        buckets = response.get('Buckets', [])
        print(f"\n[OK] S3 connection successful!")
        print(f"  Found {len(buckets)} buckets")
        
        if buckets:
            print("\n  Buckets available:")
            for bucket in buckets[:5]:  # Show first 5
                print(f"    - {bucket['Name']}")
            if len(buckets) > 5:
                print(f"    ... and {len(buckets) - 5} more")
                
        # Test specific bucket if nsc-mvp1 exists
        if any(b['Name'] == 'nsc-mvp1' for b in buckets):
            print("\n[OK] Target bucket 'nsc-mvp1' is accessible")
            
            # List some objects in the bucket
            try:
                objects = s3.list_objects_v2(Bucket='nsc-mvp1', MaxKeys=5)
                if 'Contents' in objects:
                    print(f"  Sample objects in nsc-mvp1:")
                    for obj in objects['Contents'][:5]:
                        print(f"    - {obj['Key']} ({obj['Size']} bytes)")
            except Exception as e:
                print(f"  Could not list objects: {e}")
                
        return True
        
    except NoCredentialsError:
        print("[ERROR] No AWS credentials found")
        return False
    except ClientError as e:
        print(f"[ERROR] AWS client error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def main():
    print("=" * 60)
    print("AWS S3 CONNECTION TEST")
    print("=" * 60)
    
    # Setup credentials
    setup_aws_credentials()
    
    # Test connection
    success = test_s3_connection()
    
    if success:
        print("\n" + "=" * 60)
        print("[SUCCESS] AWS session is ready!")
        print("You can now run:")
        print("  python scripts/production_tracking_drop_zones.py")
        print("  python scripts/unified_drop_zones.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[FAILED] AWS session could not be established")
        print("Check your credentials in tidyllm/admin/set_aws_env.bat")
        print("=" * 60)
    
    return success

if __name__ == "__main__":
    exit(0 if main() else 1)