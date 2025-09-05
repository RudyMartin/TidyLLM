#!/usr/bin/env python
"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

Check S3 MVR Setup - Verify AWS session and S3 folders
======================================================

Restarts AWS session and checks/creates necessary S3 folders for MVR processing.
"""

import sys
import boto3
from pathlib import Path
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))

def check_s3_mvr_setup():
    """Restart AWS session and verify S3 structure"""
    
    print("=" * 70)
    print("S3 MVR SETUP CHECK")
    print("=" * 70)
    
    # Import UnifiedSessionManager for official session management
    try:
        from scripts.start_unified_sessions import UnifiedSessionManager
        print("[OK] UnifiedSessionManager imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import UnifiedSessionManager: {e}")
        return False
    
    print("\n[STEP 1] Restarting AWS Session...")
    print("-" * 40)
    
    try:
        # Initialize new session through UnifiedSessionManager
        session_mgr = UnifiedSessionManager()
        s3_client = session_mgr.get_s3_client()
        
        # Test connection
        response = s3_client.list_buckets()
        print(f"[OK] AWS session established")
        print(f"[OK] Found {len(response['Buckets'])} S3 buckets")
        
    except Exception as e:
        print(f"[ERROR] Failed to establish AWS session: {e}")
        print("\n[TIP] Check your AWS credentials:")
        print("  - AWS_ACCESS_KEY_ID environment variable")
        print("  - AWS_SECRET_ACCESS_KEY environment variable")
        print("  - AWS_DEFAULT_REGION environment variable")
        return False
    
    print("\n[STEP 2] Checking S3 Bucket: nsc-mvp1")
    print("-" * 40)
    
    bucket_name = s3_config["bucket"]
    
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"[OK] Bucket '{bucket_name}' exists and is accessible")
        
    except s3_client.exceptions.NoSuchBucket:
        print(f"[ERROR] Bucket '{bucket_name}' does not exist")
        print("[ACTION] Creating bucket...")
        try:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': 'us-east-1'}
            )
            print(f"[OK] Bucket '{bucket_name}' created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create bucket: {e}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Cannot access bucket '{bucket_name}': {e}")
        return False
    
    print("\n[STEP 3] Checking/Creating MVR Folder Structure")
    print("-" * 40)
    
    # Define required folder structure
    required_folders = [
        "mvr-raw/",           # Raw MVR documents
        "mvr-processed/",     # Processed documents
        "mvr-embeddings/",    # Document embeddings
        "mvr-reports/",       # Generated reports
        "mvr-reports/compliance/",
        "mvr-reports/intelligence/",
        "mvr-reports/knowledge/",
        "mvr-metadata/",      # Document metadata
        "dropzones/",         # Drop zone uploads
        "prompts/",           # Prompt templates
        "prompts/favorites/"  # Favorite prompts
    ]
    
    # Check and create folders
    for folder in required_folders:
        try:
            # List objects with folder prefix
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=folder,
                MaxKeys=1
            )
            
            if 'Contents' in response:
                print(f"[OK] Folder exists: s3://{bucket_name}/{folder}")
            else:
                # Create folder by uploading empty object
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=folder,
                    Body=b''
                )
                print(f"[CREATED] Folder: s3://{bucket_name}/{folder}")
                
        except Exception as e:
            print(f"[ERROR] Failed to check/create folder {folder}: {e}")
    
    print("\n[STEP 4] Uploading Prompt Templates")
    print("-" * 40)
    
    # Check if prompt templates need to be uploaded
    prompt_files = {
        "JB_Overview_Prompt.md": "qaz_20250321-main/src/assets/prompts/favorites/JB_Overview_Prompt.md",
        "comprehensive_whitepaper_analysis.md": "qaz_20250321-main/src/assets/prompts/favorites/comprehensive_whitepaper_analysis.md",
        "comprehensive_whitepaper_analysis_enhanced.md": "qaz_20250321-main/src/assets/prompts/favorites/comprehensive_whitepaper_analysis_enhanced.md",
        "toc_extraction_prompt.md": "qaz_20250321-main/src/assets/prompts/favorites/toc_extraction_prompt.md"
    }
    
    for prompt_name, local_path in prompt_files.items():
        s3_key = f"prompts/favorites/{prompt_name}"
        
        try:
            # Check if already exists
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            print(f"[EXISTS] s3://{bucket_name}/{s3_key}")
            
        except s3_client.exceptions.NoSuchKey:
            # Upload if exists locally
            if Path(local_path).exists():
                try:
                    s3_client.upload_file(local_path, bucket_name, s3_key)
                    print(f"[UPLOADED] {prompt_name} -> s3://{bucket_name}/{s3_key}")
                except Exception as e:
                    print(f"[ERROR] Failed to upload {prompt_name}: {e}")
            else:
                print(f"[SKIP] Local file not found: {local_path}")
                
        except Exception as e:
            print(f"[ERROR] Failed to check {s3_key}: {e}")
    
    print("\n[STEP 5] Verifying Permissions")
    print("-" * 40)
    
    # Test permissions by creating and deleting a test object
    test_key = f"test/permission_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        # Test upload
        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=b'Permission test',
            ServerSideEncryption='AES256'
        )
        print(f"[OK] Upload permission verified")
        
        # Test read
        response = s3_client.get_object(Bucket=bucket_name, Key=test_key)
        print(f"[OK] Read permission verified")
        
        # Test delete
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print(f"[OK] Delete permission verified")
        
    except Exception as e:
        print(f"[ERROR] Permission check failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("S3 MVR SETUP COMPLETE")
    print("=" * 70)
    
    print("\n[SUMMARY]")
    print(f"  Bucket: s3://{bucket_name}")
    print(f"  Folders: {len(required_folders)} folders ready")
    print(f"  Prompts: {len(prompt_files)} templates available")
    print(f"  Status: READY FOR MVR PROCESSING")
    
    print("\n[NEXT STEPS]")
    print("  1. Drop MVR document in: drop_zones/input/")
    print("  2. Ensure query in: drop_zones/queries/mvr_compliance_review.md")
    print("  3. Run: python scripts/unified_drop_zones.py")
    
    return True

if __name__ == "__main__":
    success = check_s3_mvr_setup()
    sys.exit(0 if success else 1)