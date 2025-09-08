#!/usr/bin/env python3
"""

# Centralized AWS credential management
import sys
from pathlib import Path

# Add admin directory to path for credential loading
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()
Test AWS S3 Connection with credentials from admin folder
"""

import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def setup_aws_credentials():
    """Set AWS credentials from admin config"""
    
    
    
    
    print("[OK] AWS credentials configured:")
    print(f"  Access Key: {os.environ['AWS_ACCESS_KEY_ID'][:10]}...")
    print(f"  Region: {os.environ['AWS_DEFAULT_REGION']}")

def test_s3_connection():
    """Test S3 connection"""
    try:
        # AUDIT COMPLIANCE: Use UnifiedSessionManager instead of direct boto3
        try:
            from scripts.infrastructure.start_unified_sessions import UnifiedSessionManager
            session_manager = UnifiedSessionManager()
            s3 = session_manager.get_s3_client()
        except ImportError:
            # Fallback to direct boto3
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
        if any(b['Name'] == s3_config["bucket"] for b in buckets):
            print("\n[OK] Target bucket s3_config["bucket"] is accessible")
            
            # List some objects in the bucket
            try:
                objects = s3.list_objects_v2(Bucket=s3_config["bucket"], MaxKeys=5)
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