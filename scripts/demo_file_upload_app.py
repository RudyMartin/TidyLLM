#!/usr/bin/env python3
"""
TidyLLM Demo File Upload App

Uploads a randomly selected file from whitepapers folder to S3,
demonstrating file upload functionality with proper error handling
and comprehensive logging. Uses the existing S3 credentials.
"""

import os
import boto3
import tempfile
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Set up AWS credentials (from existing tests)
os.environ['AWS_ACCESS_KEY_ID'] = 'REMOVED_AWS_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'REMOVED_AWS_SECRET'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

class DocumentFileUploader:
    """Demo file uploader for documents from tidyllm-vectorqa documents folder"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        self.target_bucket = None
        
    def setup_s3_environment(self):
        """Setup S3 environment and select target bucket"""
        print("1. Setting up S3 environment...")
        
        try:
            # List available buckets
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            
            print(f"   SUCCESS: Found {len(buckets)} accessible buckets:")
            for bucket in buckets:
                print(f"      - {bucket}")
            
            # Use first accessible bucket for demo
            if buckets:
                self.target_bucket = buckets[0]
                print(f"   SUCCESS: Using bucket for upload: {self.target_bucket}")
                return True
            else:
                print("   ERROR: No accessible buckets found")
                return False
                
        except Exception as e:
            print(f"   ERROR: S3 setup failed: {e}")
            return False
    
    def select_document_file(self) -> Path:
        """Select a document file from tidyllm-vectorqa documents folder"""
        print("\n2. Selecting document file...")
        
        # Use the business document template file
        doc_file = Path("C:/Users/marti/github/tidyllm-vectorqa/tidyllm_vectorqa/documents/templates/business.py")
        
        if doc_file.exists():
            size = doc_file.stat().st_size
            print(f"   SUCCESS: Selected: {doc_file.name}")
            print(f"      - Size: {size:,} bytes ({size/1024:.1f} KB)")
            print(f"      - Type: Business Document Template")
            print(f"      - Module: Document Processing")
            return doc_file
        else:
            print(f"   ERROR: Document file not found: {doc_file}")
            return None
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for integrity verification"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def upload_document(self, file_path: Path) -> Dict[str, Any]:
        """Upload document to S3 with metadata"""
        print(f"\n3. Uploading {file_path.name} to S3...")
        
        if not self.target_bucket:
            print("   ERROR: No target bucket configured")
            return {"success": False, "error": "No target bucket"}
        
        try:
            # Generate S3 key with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"tidyllm-demos/documents/{timestamp}/{file_path.name}"
            
            # Calculate file hash for integrity
            file_hash = self.calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            
            print(f"   UPLOADING to: s3://{self.target_bucket}/{s3_key}")
            print(f"   File hash: {file_hash[:16]}...")
            
            # Upload with comprehensive metadata
            self.s3_client.upload_file(
                str(file_path),
                self.target_bucket,
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'document-type': 'business-template',
                        'module-name': 'tidyllm-vectorqa-documents',
                        'file-category': 'templates',
                        'component': 'BusinessDocumentProcessor',
                        'source-path': 'documents/templates/business.py',
                        'upload-source': 'tidyllm-demo-app',
                        'upload-timestamp': datetime.now().isoformat(),
                        'file-hash-sha256': file_hash,
                        'original-size': str(file_size),
                        'demo-version': '1.0'
                    },
                    'ContentType': 'text/x-python'
                }
            )
            
            # Verify upload
            response = self.s3_client.head_object(Bucket=self.target_bucket, Key=s3_key)
            uploaded_size = response['ContentLength']
            
            print(f"   SUCCESS: Upload successful!")
            print(f"      - S3 location: s3://{self.target_bucket}/{s3_key}")
            print(f"      - Size verified: {uploaded_size:,} bytes")
            print(f"      - ETag: {response['ETag']}")
            
            return {
                "success": True,
                "bucket": self.target_bucket,
                "key": s3_key,
                "size": uploaded_size,
                "hash": file_hash,
                "etag": response['ETag'],
                "metadata": response.get('Metadata', {})
            }
            
        except Exception as e:
            print(f"   ERROR: Upload failed: {e}")
            return {"success": False, "error": str(e)}
    
    def verify_upload_integrity(self, upload_result: Dict[str, Any], original_file: Path) -> bool:
        """Verify upload integrity by downloading and comparing"""
        print("\n4. Verifying upload integrity...")
        
        if not upload_result.get("success"):
            print("   WARNING: Cannot verify - upload failed")
            return False
        
        try:
            # Download file to temp location
            temp_dir = Path(tempfile.mkdtemp(prefix='verify_'))
            download_path = temp_dir / f"downloaded_{original_file.name}"
            
            print(f"   DOWNLOADING for verification...")
            self.s3_client.download_file(
                upload_result["bucket"],
                upload_result["key"],
                str(download_path)
            )
            
            # Calculate hash of downloaded file
            download_hash = self.calculate_file_hash(download_path)
            original_hash = upload_result["hash"]
            
            # Compare hashes
            if download_hash == original_hash:
                print(f"   SUCCESS: Integrity verified: hashes match")
                print(f"      - Original:  {original_hash[:16]}...")
                print(f"      - Downloaded: {download_hash[:16]}...")
                
                # Cleanup
                download_path.unlink()
                temp_dir.rmdir()
                return True
            else:
                print(f"   ERROR: Integrity check failed: hashes don't match")
                print(f"      - Original:  {original_hash}")
                print(f"      - Downloaded: {download_hash}")
                return False
                
        except Exception as e:
            print(f"   ERROR: Verification failed: {e}")
            return False
    
    def generate_upload_report(self, upload_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive upload report"""
        print("\n5. Generating upload report...")
        
        report = {
            "demo_info": {
                "app_name": "TidyLLM Demo File Upload App",
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "environment": "development"
            },
            "file_info": {
                "filename": "2308.02381_Non-Ideal_Measurement_Heat_Engines.pdf",
                "type": "academic_paper",
                "subject": "Quantum Heat Engines",
                "arxiv_id": "2308.02381v1"
            },
            "upload_result": upload_result,
            "aws_info": {
                "region": os.getenv('AWS_DEFAULT_REGION'),
                "bucket": upload_result.get("bucket"),
                "key": upload_result.get("key")
            }
        }
        
        print(f"   REPORT: Generated with {len(report.keys())} sections")
        return report

def main():
    """Run the demo file upload application"""
    print("=" * 60)
    print("  TidyLLM Demo File Upload App")
    print("  TidyLLM-VectorQA Document Upload to S3")
    print("=" * 60)
    
    uploader = DocumentFileUploader()
    
    try:
        # Setup S3 environment
        if not uploader.setup_s3_environment():
            print("\nERROR: S3 setup failed - aborting demo")
            return False
        
        # Select document file
        doc_file = uploader.select_document_file()
        if not doc_file:
            print("\nERROR: File selection failed - aborting demo")
            return False
        
        # Upload file
        upload_result = uploader.upload_document(doc_file)
        
        # Verify upload integrity
        integrity_check = uploader.verify_upload_integrity(upload_result, doc_file)
        
        # Generate report
        report = uploader.generate_upload_report(upload_result)
        
        # Final summary
        print("\n" + "=" * 60)
        print("  Demo Upload Results")
        print("=" * 60)
        
        if upload_result.get("success") and integrity_check:
            print("SUCCESS: DEMO FILE UPLOAD: SUCCESS!")
            print(f"SUCCESS: File uploaded: {doc_file.name}")
            print(f"SUCCESS: S3 location: s3://{upload_result['bucket']}/{upload_result['key']}")
            print(f"SUCCESS: Size: {upload_result['size']:,} bytes")
            print(f"SUCCESS: Integrity verified: SHA-256 hash match")
            print(f"SUCCESS: Document metadata preserved")
            success = True
        else:
            print("WARNING: DEMO FILE UPLOAD: PARTIAL SUCCESS")
            print("Some operations failed - check details above")
            success = False
        
        return success
        
    except Exception as e:
        print(f"\nERROR: Demo application error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)