"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

Knowledge Systems S3 Manager
============================

CLEAN MIGRATION TO UNIFIEDSESSIONMANAGER
========================================

This S3Manager properly delegates to UnifiedSessionManager for all operations.
Maintains compatibility with knowledge systems while following TidyLLM constraints.
"""

import os
import json
import logging
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import tempfile

# Import UnifiedSessionManager from same module
from .unified import UnifiedSessionManager

logger = logging.getLogger("s3_manager")

@dataclass
class S3Config:
    """S3 configuration settings"""
    region: str = "us-east-1"
    default_bucket: Optional[str] = None
    default_prefix: Optional[str] = None

@dataclass  
class UploadResult:
    """Result of S3 upload operation"""
    success: bool
    s3_key: Optional[str] = None
    bucket: Optional[str] = None
    file_size: Optional[int] = None
    upload_duration: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class S3Manager:
    """
    CLEAN MIGRATION: Knowledge Systems S3 Manager using UnifiedSessionManager
    
    All S3 operations delegated to official UnifiedSessionManager.
    Maintains knowledge systems compatibility.
    """
    
    def __init__(self, config: S3Config = None, auto_discover: bool = True):
        """Initialize S3Manager with UnifiedSessionManager"""
        
        self.config = config or S3Config()
        
        # Use official UnifiedSessionManager
        self.session_mgr = UnifiedSessionManager()
        
        logger.info("[OK] S3Manager initialized with UnifiedSessionManager")
    
    def get_s3_client(self):
        """Get S3 client via UnifiedSessionManager"""
        return self.session_mgr.get_s3_client()
    
    def upload_file(self, file_path: Union[str, Path], bucket: str = None, 
                   s3_key: str = None, metadata: Dict[str, str] = None) -> UploadResult:
        """Upload file to S3 via UnifiedSessionManager"""
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            # Use defaults if not provided
            bucket = bucket or self.config.default_bucket or s3_config["bucket"]
            
            # Generate S3 key if not provided
            if not s3_key:
                prefix = self.config.default_prefix or ""
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                s3_key = f"{prefix}knowledge-systems-{timestamp}-{file_path.name}"
            
            # Use UnifiedSessionManager for upload
            success = self.session_mgr.upload_to_s3(bucket, s3_key, str(file_path))
            
            upload_duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                return UploadResult(
                    success=True,
                    s3_key=s3_key,
                    bucket=bucket,
                    file_size=file_path.stat().st_size,
                    upload_duration=upload_duration,
                    metadata={'via': 'UnifiedSessionManager'}
                )
            else:
                return UploadResult(
                    success=False,
                    error="UnifiedSessionManager upload failed"
                )
                
        except Exception as e:
            return UploadResult(
                success=False,
                error=f"Upload failed: {e}"
            )
    
    def download_file(self, bucket: str, s3_key: str, local_path: Union[str, Path] = None) -> Dict[str, Any]:
        """Download file from S3 via UnifiedSessionManager"""
        try:
            # Use UnifiedSessionManager to download
            content = self.session_mgr.download_from_s3(bucket, s3_key)
            
            if local_path:
                with open(local_path, 'wb') as f:
                    if isinstance(content, str):
                        f.write(content.encode('utf-8'))
                    else:
                        f.write(content)
                
                return {
                    'success': True,
                    'local_path': str(local_path),
                    'via': 'UnifiedSessionManager'
                }
            else:
                return {
                    'success': True,
                    'content': content,
                    'via': 'UnifiedSessionManager'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'via': 'UnifiedSessionManager'
            }
    
    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict[str, Any]]:
        """List objects in S3 bucket via UnifiedSessionManager"""
        try:
            s3_client = self.session_mgr.get_s3_client()
            
            objects = []
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            
            for obj in response.get('Contents', []):
                objects.append({
                    'Key': obj['Key'],
                    'Size': obj['Size'],
                    'LastModified': obj['LastModified'],
                    'ETag': obj['ETag']
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"List objects failed: {e}")
            return []
    
    def delete_object(self, bucket: str, s3_key: str) -> bool:
        """Delete object from S3 via UnifiedSessionManager"""
        try:
            s3_client = self.session_mgr.get_s3_client()
            s3_client.delete_object(Bucket=bucket, Key=s3_key)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

def get_s3_manager(config: S3Config = None) -> S3Manager:
    """Get S3Manager instance (knowledge systems compatibility)"""
    return S3Manager(config)

def reset_s3_manager():
    """Reset S3Manager (knowledge systems compatibility)"""
    # No-op since UnifiedSessionManager handles state
    pass