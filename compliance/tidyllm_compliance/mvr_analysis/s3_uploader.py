#!/usr/bin/env python3
"""
S3 MVR Uploader for tidyllm-compliance
======================================

Upload MVR documents and prompts to S3 for processing.
Integrated with tidyllm-compliance framework for regulatory workflow.

Features:
- Batch upload of multiple documents
- Automatic file type detection and validation
- S3 path organization for efficient processing
- Upload progress tracking and error handling
- Integration with compliance validation pipeline

Part of tidyllm-compliance: Professional regulatory compliance platform
"""

import sys
from pathlib import Path

# Add tidyllm admin directory for credential management
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent.parent.parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()

import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import mimetypes

# Import UnifiedSessionManager for audit-compliant session management
try:
    from tidyllm.infrastructure.session.unified import UnifiedSessionManager
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    # Fallback to direct boto3 import if UnifiedSessionManager not available
    import boto3
    UNIFIED_SESSION_AVAILABLE = False

class S3MVRUploader:
    """
    Upload MVR documents and analysis prompts to S3.
    Organizes files for efficient processing by MVR analysis pipeline.
    """
    
    def __init__(self, 
                 bucket_name: str = "nsc-mvp1",
                 base_prefix: str = "mvr_analysis"):
        
        self.bucket_name = bucket_name
        self.base_prefix = base_prefix
        # AUDIT COMPLIANCE: Use UnifiedSessionManager instead of direct boto3
        if UNIFIED_SESSION_AVAILABLE:
            print("[UPLOADER] Using UnifiedSessionManager for audit-compliant S3 access")
            self.session_manager = UnifiedSessionManager()
            self.s3_client = self.session_manager.get_s3_client()
        else:
            print("[UPLOADER] NO FALLBACK - UnifiedSessionManager is required")
            raise RuntimeError("S3Uploader: UnifiedSessionManager is required for S3 access")
        
        # S3 path organization
        self.paths = {
            'raw': f"{base_prefix}/raw/",
            'prompts': f"{base_prefix}/prompts/",
            'metadata': f"{base_prefix}/metadata/"
        }
        
        # Supported file types
        self.supported_document_types = ['.txt', '.md', '.pdf', '.docx', '.doc']
        self.supported_prompt_types = ['.md', '.txt']
        
        print(f"[MVR_UPLOADER] Initialized for bucket: {bucket_name}")
        print(f"[MVR_UPLOADER] Base prefix: {base_prefix}")
    
    def upload_mvr_document(self, 
                          local_file_path: str, 
                          s3_filename: Optional[str] = None,
                          metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Upload a single MVR document to S3.
        
        Args:
            local_file_path: Path to local MVR document
            s3_filename: Custom S3 filename (optional)
            metadata: Additional metadata tags
            
        Returns:
            Upload result with S3 location and metadata
        """
        print(f"\n[UPLOAD] Uploading MVR document: {local_file_path}")
        
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"File not found: {local_file_path}")
        
        # Validate file type
        if not self._is_supported_document(local_file_path):
            raise ValueError(f"Unsupported document type. Supported: {self.supported_document_types}")
        
        # Generate S3 key
        if not s3_filename:
            s3_filename = os.path.basename(local_file_path)
        
        s3_key = f"{self.paths['raw']}{s3_filename}"
        
        # Prepare metadata
        upload_metadata = {
            'upload_timestamp': datetime.now(timezone.utc).isoformat(),
            'original_filename': os.path.basename(local_file_path),
            'file_type': 'mvr_document',
            'content_type': self._get_content_type(local_file_path)
        }
        
        if metadata:
            upload_metadata.update(metadata)
        
        # Upload file
        try:
            with open(local_file_path, 'rb') as file:
                self.s3_client.upload_fileobj(
                    file,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={
                        'Metadata': upload_metadata,
                        'ContentType': upload_metadata['content_type']
                    }
                )
            
            file_size = os.path.getsize(local_file_path)
            
            result = {
                'status': 'success',
                's3_location': f"s3://{self.bucket_name}/{s3_key}",
                's3_key': s3_key,
                'file_size_bytes': file_size,
                'upload_metadata': upload_metadata,
                'uploaded_at': datetime.now(timezone.utc).isoformat()
            }
            
            print(f"[SUCCESS] Uploaded to: {result['s3_location']}")
            print(f"[SIZE] File size: {file_size:,} bytes")
            
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'error_message': str(e),
                'local_file_path': local_file_path,
                'attempted_s3_key': s3_key
            }
            print(f"[ERROR] Upload failed: {e}")
            return error_result
    
    def upload_analysis_prompt(self, 
                             local_prompt_path: str,
                             s3_filename: Optional[str] = None,
                             metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Upload analysis prompt to S3.
        
        Args:
            local_prompt_path: Path to local prompt file
            s3_filename: Custom S3 filename (optional)
            metadata: Additional metadata tags
            
        Returns:
            Upload result with S3 location and metadata
        """
        print(f"\n[UPLOAD] Uploading analysis prompt: {local_prompt_path}")
        
        if not os.path.exists(local_prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {local_prompt_path}")
        
        # Validate file type
        if not self._is_supported_prompt(local_prompt_path):
            raise ValueError(f"Unsupported prompt type. Supported: {self.supported_prompt_types}")
        
        # Generate S3 key
        if not s3_filename:
            s3_filename = os.path.basename(local_prompt_path)
        
        s3_key = f"{self.paths['prompts']}{s3_filename}"
        
        # Prepare metadata
        upload_metadata = {
            'upload_timestamp': datetime.now(timezone.utc).isoformat(),
            'original_filename': os.path.basename(local_prompt_path),
            'file_type': 'analysis_prompt',
            'content_type': self._get_content_type(local_prompt_path)
        }
        
        if metadata:
            upload_metadata.update(metadata)
        
        # Upload file
        try:
            with open(local_prompt_path, 'rb') as file:
                self.s3_client.upload_fileobj(
                    file,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={
                        'Metadata': upload_metadata,
                        'ContentType': upload_metadata['content_type']
                    }
                )
            
            file_size = os.path.getsize(local_prompt_path)
            
            result = {
                'status': 'success',
                's3_location': f"s3://{self.bucket_name}/{s3_key}",
                's3_key': s3_key,
                'file_size_bytes': file_size,
                'upload_metadata': upload_metadata,
                'uploaded_at': datetime.now(timezone.utc).isoformat()
            }
            
            print(f"[SUCCESS] Uploaded prompt to: {result['s3_location']}")
            print(f"[SIZE] File size: {file_size:,} bytes")
            
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'error_message': str(e),
                'local_file_path': local_prompt_path,
                'attempted_s3_key': s3_key
            }
            print(f"[ERROR] Prompt upload failed: {e}")
            return error_result
    
    def batch_upload_documents(self, 
                             document_paths: List[str],
                             metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Upload multiple MVR documents in batch.
        
        Args:
            document_paths: List of local file paths
            metadata: Common metadata for all uploads
            
        Returns:
            Batch upload results with individual file status
        """
        print(f"\n{'='*50}")
        print(f"BATCH UPLOAD - {len(document_paths)} documents")
        print(f"{'='*50}")
        
        results = {
            'batch_id': datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'),
            'total_files': len(document_paths),
            'successful_uploads': 0,
            'failed_uploads': 0,
            'upload_results': [],
            'started_at': datetime.now(timezone.utc).isoformat()
        }
        
        for i, doc_path in enumerate(document_paths, 1):
            print(f"\n[BATCH] Processing {i}/{len(document_paths)}: {os.path.basename(doc_path)}")
            
            try:
                upload_result = self.upload_mvr_document(doc_path, metadata=metadata)
                
                if upload_result['status'] == 'success':
                    results['successful_uploads'] += 1
                else:
                    results['failed_uploads'] += 1
                    
                results['upload_results'].append(upload_result)
                
            except Exception as e:
                results['failed_uploads'] += 1
                results['upload_results'].append({
                    'status': 'error',
                    'error_message': str(e),
                    'local_file_path': doc_path
                })
                print(f"[ERROR] Batch upload failed for {doc_path}: {e}")
        
        results['completed_at'] = datetime.now(timezone.utc).isoformat()
        
        print(f"\n{'='*50}")
        print(f"BATCH UPLOAD COMPLETE")
        print(f"Successful: {results['successful_uploads']}/{results['total_files']}")
        print(f"Failed: {results['failed_uploads']}/{results['total_files']}")
        print(f"{'='*50}")
        
        return results
    
    def list_uploaded_files(self, file_type: str = 'all') -> Dict[str, Any]:
        """
        List files already uploaded to S3.
        
        Args:
            file_type: 'documents', 'prompts', or 'all'
            
        Returns:
            Dictionary with file listings
        """
        try:
            listings = {'documents': [], 'prompts': []}
            
            if file_type in ['documents', 'all']:
                # List documents
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=self.paths['raw']
                )
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        if not obj['Key'].endswith('/'):
                            listings['documents'].append({
                                'key': obj['Key'],
                                's3_location': f"s3://{self.bucket_name}/{obj['Key']}",
                                'size_bytes': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                                'filename': obj['Key'].split('/')[-1]
                            })
            
            if file_type in ['prompts', 'all']:
                # List prompts
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=self.paths['prompts']
                )
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        if not obj['Key'].endswith('/'):
                            listings['prompts'].append({
                                'key': obj['Key'],
                                's3_location': f"s3://{self.bucket_name}/{obj['Key']}",
                                'size_bytes': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                                'filename': obj['Key'].split('/')[-1]
                            })
            
            return {
                'status': 'success',
                'bucket': self.bucket_name,
                'base_prefix': self.base_prefix,
                'listings': listings,
                'total_documents': len(listings['documents']),
                'total_prompts': len(listings['prompts']),
                'retrieved_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'retrieved_at': datetime.now(timezone.utc).isoformat()
            }
    
    def _is_supported_document(self, file_path: str) -> bool:
        """Check if file type is supported for MVR documents."""
        return any(file_path.lower().endswith(ext) for ext in self.supported_document_types)
    
    def _is_supported_prompt(self, file_path: str) -> bool:
        """Check if file type is supported for prompts."""
        return any(file_path.lower().endswith(ext) for ext in self.supported_prompt_types)
    
    def _get_content_type(self, file_path: str) -> str:
        """Determine MIME content type for file."""
        content_type, _ = mimetypes.guess_type(file_path)
        return content_type or 'application/octet-stream'

# Example usage and testing
def demo_mvr_upload():
    """
    Demonstrate S3 MVR upload functionality.
    """
    uploader = S3MVRUploader(
        bucket_name="nsc-mvp1",
        base_prefix="mvr_analysis"
    )
    
    # Show current uploaded files
    print("\nCurrent uploaded files:")
    listings = uploader.list_uploaded_files()
    
    if listings['status'] == 'success':
        print(f"Documents: {listings['total_documents']}")
        print(f"Prompts: {listings['total_prompts']}")
        
        # Show some examples
        if listings['listings']['documents']:
            print("\nExample documents:")
            for doc in listings['listings']['documents'][:3]:
                print(f"  - {doc['filename']} ({doc['size_bytes']:,} bytes)")
        
        if listings['listings']['prompts']:
            print("\nExample prompts:")
            for prompt in listings['listings']['prompts'][:3]:
                print(f"  - {prompt['filename']} ({prompt['size_bytes']:,} bytes)")
    else:
        print(f"Failed to list files: {listings.get('error_message')}")
    
    print("\nNote: To upload files, use:")
    print("uploader.upload_mvr_document('/path/to/document.txt')")
    print("uploader.upload_analysis_prompt('/path/to/prompt.md')")

if __name__ == "__main__":
    demo_mvr_upload()