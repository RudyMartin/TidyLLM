#!/usr/bin/env python
"""
List Available S3 Buckets
=========================
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.start_unified_sessions import UnifiedSessionManager

def list_buckets():
    """List available S3 buckets"""
    
    print("=" * 50)
    print("AVAILABLE S3 BUCKETS")
    print("=" * 50)
    
    try:
        session_mgr = UnifiedSessionManager()
        s3_client = session_mgr.get_s3_client()
        
        response = s3_client.list_buckets()
        
        print(f"Found {len(response['Buckets'])} buckets:")
        
        for bucket in response['Buckets']:
            bucket_name = bucket['Name']
            created = bucket['CreationDate'].strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\nBucket: {bucket_name}")
            print(f"Created: {created}")
            
            # Try to check access
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                print(f"Access: [OK] - Can read/write")
                
                # List first few objects
                try:
                    objects = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
                    if 'Contents' in objects:
                        print(f"Objects: {objects['KeyCount']} (showing first 5)")
                        for obj in objects['Contents']:
                            print(f"  - {obj['Key']}")
                    else:
                        print("Objects: Empty bucket")
                except Exception as e:
                    print(f"Objects: Cannot list - {e}")
                    
            except Exception as e:
                print(f"Access: [DENIED] - {e}")
                
    except Exception as e:
        print(f"Error listing buckets: {e}")

if __name__ == "__main__":
    list_buckets()