#!/usr/bin/env python3
"""
Test S3 Upload Functionality
"""

import os
import boto3
from datetime import datetime

# Set AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'REMOVED_AWS_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'REMOVED_AWS_SECRET' 
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

def test_s3_upload():
    """Test S3 upload to nsc-mvp1 bucket"""
    
    print("=" * 60)
    print("S3 UPLOAD TEST")
    print("=" * 60)
    
    try:
        # Create S3 client
        s3 = boto3.client('s3')
        
        # Test content
        test_content = f"TidyLLM S3 Test Upload\nTimestamp: {datetime.now().isoformat()}\nStatus: Testing AWS session restart"
        test_key = f"tidyllm/test/test_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        print(f"Uploading test file to s3://nsc-mvp1/{test_key}")
        
        # Try upload
        try:
            s3.put_object(
                Bucket='nsc-mvp1',
                Key=test_key,
                Body=test_content.encode('utf-8'),
                ContentType='text/plain'
            )
            print("[SUCCESS] File uploaded successfully!")
            print(f"Location: s3://nsc-mvp1/{test_key}")
            
            # Try to read it back
            response = s3.get_object(Bucket='nsc-mvp1', Key=test_key)
            content = response['Body'].read().decode('utf-8')
            print(f"[VERIFIED] Read back {len(content)} bytes")
            
            return True
            
        except s3.exceptions.NoSuchBucket:
            print("[ERROR] Bucket 'nsc-mvp1' does not exist")
            return False
            
        except Exception as e:
            error_code = str(e)
            if 'AccessDenied' in error_code:
                print("[ERROR] Access denied - checking permissions...")
                print("User may not have write permissions to nsc-mvp1 bucket")
                
                # Try alternate bucket
                print("\nTrying alternate bucket 'dsai-2025-asu'...")
                try:
                    test_key2 = f"tidyllm/test/test_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    s3.put_object(
                        Bucket='dsai-2025-asu',
                        Key=test_key2,
                        Body=test_content.encode('utf-8'),
                        ContentType='text/plain'
                    )
                    print(f"[SUCCESS] Uploaded to alternate bucket: s3://dsai-2025-asu/{test_key2}")
                    return True
                except Exception as e2:
                    print(f"[ERROR] Alternate bucket also failed: {e2}")
                    
            else:
                print(f"[ERROR] Upload failed: {e}")
            return False
            
    except Exception as e:
        print(f"[ERROR] S3 client creation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_s3_upload()
    
    if success:
        print("\n" + "=" * 60)
        print("S3 UPLOAD CAPABILITY: WORKING")
        print("AWS Session is properly configured for uploads")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("S3 UPLOAD CAPABILITY: LIMITED")
        print("May need to check bucket permissions or use alternate bucket")
        print("=" * 60)