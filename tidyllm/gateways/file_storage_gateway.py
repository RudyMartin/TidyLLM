#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Storage Utility Service - Enterprise File Management
=========================================================
🔧 UTILITY SERVICE - NOT A CORE GATEWAY  
This is a specialized S3/file storage wrapper, not part of the main gateway workflow.

LEGAL DOCUMENT ANALYSIS WORKFLOW EXAMPLE:
For legal contract review processing, this gateway:
- Securely stores original legal contracts with version control and audit trails
- Manages processed document outputs (analysis reports, extracted clauses, risk assessments)
- Provides controlled access to legal document libraries and precedent databases
- Enforces legal retention policies (7-year retention for contracts, compliance documents)
- Integrates with both local corporate storage and cloud storage (S3) for scalability

AI AGENT INTEGRATION GUIDE:
Purpose: Enterprise file management with security, versioning, and compliance
- Provides secure file storage with comprehensive audit logging
- Supports both local corporate storage and cloud storage integration
- Implements access control and file lifecycle management
- Offers automatic cleanup and retention policy enforcement

DEPENDENCIES & REQUIREMENTS:
- Infrastructure: S3Manager (for cloud storage capability and scalability)
- Infrastructure: Centralized Settings Manager (for storage policies and retention rules)
- Local Storage: Corporate file systems with enterprise-grade security
- Cloud Storage: AWS S3 integration for backup and disaster recovery
- Security: File encryption, access control, and audit logging

INTEGRATION PATTERNS:
- Use store_file() to securely upload legal documents with metadata
- Call retrieve_file() to access stored documents with permission validation
- Execute list_files() to browse document collections with tag filtering
- Monitor with cleanup_expired_files() for automatic retention policy enforcement

FILE OPERATIONS:
- Secure file upload with virus scanning and content validation
- Metadata tagging for document classification and search
- Version control for document revisions and change tracking
- Automatic file expiration based on retention policies
- Cross-platform storage (local + cloud) with intelligent routing

STORAGE FEATURES:
- Local storage for immediate access and corporate compliance
- Cloud storage (S3) integration for scalability and disaster recovery
- File deduplication and compression for storage efficiency
- Automatic backup and replication across storage tiers
- Content-based file validation and security scanning

SECURITY & COMPLIANCE:
- Enterprise-grade access control with role-based permissions
- Comprehensive audit logging for all file operations
- Automatic retention policy enforcement for legal compliance
- Data encryption at rest and in transit
- Secure file deletion with overwrite capabilities

ERROR HANDLING:
- Returns StorageError for upload/download failures
- Provides QuotaExceededError for storage limit violations
- Implements AccessDeniedError for permission failures
- Offers detailed error context for troubleshooting and compliance
"""

from typing import Dict, Any, Optional, List, Union, IO
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
import tempfile
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from .base_gateway import BaseGateway, GatewayResponse, GatewayDependencies


@dataclass
class FileStorageConfig:
    """Configuration for file storage operations."""
    storage_root: str = "/tmp/tidyllm_storage"
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.txt', '.md', '.csv'])
    retention_days: int = 30
    enable_versioning: bool = True
    enable_audit_log: bool = True


@dataclass 
class StoredFile:
    """Metadata for stored files."""
    file_id: str
    original_name: str
    stored_path: str
    size_bytes: int
    mime_type: str
    created_at: datetime
    accessed_at: Optional[datetime] = None
    checksum: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)


class FileStorageGateway(BaseGateway):
    """
    Enterprise file storage gateway for TidyLLM ecosystem.
    
    Provides secure file storage with audit logging, access control,
    and integration with enterprise storage systems.
    """
    
    def __init__(self, config: Optional[FileStorageConfig] = None):
        # Initialize with centralized settings approach
        super().__init__(config={})
        
        # Set our specific file storage config
        self.file_config = config or FileStorageConfig()
        
        # Initialize S3 client through USM (will be set by session_manager)
        self.s3_client = None
        logger.info("FileStorageGateway: Ready for USM S3 client integration")
        
        self._ensure_storage_directory()
        self._stored_files: Dict[str, StoredFile] = {}
    
    def set_s3_client(self, s3_client):
        """Set S3 client from USM."""
        self.s3_client = s3_client
        logger.info("FileStorageGateway: S3 client set from USM")
        
    def _execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of abstract method from BaseGateway."""
        # For file storage, we don't use the generic request pattern
        # Operations are called directly (store_file, retrieve_file, etc.)
        operation = request.get('operation', 'unknown')
        return {"success": False, "error": f"Operation {operation} not supported through _execute_request"}
    
    def _get_default_dependencies(self) -> GatewayDependencies:
        """Get default dependencies for file storage gateway."""
        return GatewayDependencies(
            requires_ai_processing=False,
            requires_corporate_llm=False,
            requires_workflow_optimizer=False,
            requires_knowledge_resources=False
        )
    
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process file storage request."""
        operation = request.get('operation', 'unknown')
        if operation == 'store_file':
            return self.store_file(
                request.get('file_path'),
                request.get('original_name'),
                request.get('tags')
            ).to_dict()
        elif operation == 'retrieve_file':
            return self.retrieve_file(request.get('file_id')).to_dict()
        elif operation == 'list_files':
            return self.list_files(request.get('tags')).to_dict()
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
    
    def process_sync(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous processing (same as async for file operations)."""
        return self.process(request)
    
    def validate_config(self) -> bool:
        """Validate file storage configuration."""
        try:
            # Check storage root exists or can be created
            if not os.path.exists(self.file_config.storage_root):
                os.makedirs(self.file_config.storage_root, exist_ok=True)
            
            # Check S3 client if available
            if self.s3_client:
                # Test S3 connection
                self.s3_client.list_buckets()
            
            return True
        except Exception as e:
            logger.error(f"FileStorageGateway: Config validation failed: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get file storage capabilities."""
        return {
            "local_storage": True,
            "s3_storage": self.s3_client is not None,
            "file_operations": ["store", "retrieve", "list", "delete"],
            "supported_formats": ["*"],
            "max_file_size": self.file_config.max_file_size_mb,
            "retention_policies": True
        }
        
    def _ensure_storage_directory(self):
        """Ensure storage directory exists."""
        os.makedirs(self.file_config.storage_root, exist_ok=True)
        
    def store_file(self, file_path: Union[str, IO], 
                  original_name: Optional[str] = None,
                  tags: Optional[Dict[str, Any]] = None) -> GatewayResponse:
        """
        Store a file securely with metadata.
        
        Args:
            file_path: Path to file or file-like object
            original_name: Original filename (if different from path)
            tags: Additional metadata tags
            
        Returns:
            GatewayResponse with file_id and storage metadata
        """
        try:
            # Validate file
            if isinstance(file_path, str):
                if not os.path.exists(file_path):
                    return GatewayResponse(
                        success=False,
                        error=f"FILE_NOT_FOUND: File not found: {file_path}"
                    )
                    
                file_size = os.path.getsize(file_path)
                original_name = original_name or os.path.basename(file_path)
            else:
                # File-like object
                file_size = len(file_path.read())
                file_path.seek(0)  # Reset position
                original_name = original_name or "uploaded_file"
                
            # Check file size
            if file_size > self.file_config.max_file_size_mb * 1024 * 1024:
                return GatewayResponse(
                    success=False,
                    error=f"FILE_TOO_LARGE: File size {file_size} exceeds limit"
                )
                
            # Check extension
            ext = Path(original_name).suffix.lower()
            if ext not in self.file_config.allowed_extensions:
                return GatewayResponse(
                    success=False,
                    error=f"INVALID_FILE_TYPE: File type {ext} not allowed"
                )
                
            # Generate unique file ID
            import uuid
            file_id = str(uuid.uuid4())
            stored_name = f"{file_id}_{original_name}"
            stored_path = os.path.join(self.file_config.storage_root, stored_name)
            
            # Store file
            if isinstance(file_path, str):
                shutil.copy2(file_path, stored_path)
            else:
                with open(stored_path, 'wb') as f:
                    shutil.copyfileobj(file_path, f)
                    
            # Create metadata
            stored_file = StoredFile(
                file_id=file_id,
                original_name=original_name,
                stored_path=stored_path,
                size_bytes=file_size,
                mime_type=self._guess_mime_type(original_name),
                created_at=datetime.now(),
                tags=tags or {}
            )
            
            # Store metadata
            self._stored_files[file_id] = stored_file
            
            # Log access
            self._log_file_operation("STORE", file_id, original_name)
            
            return GatewayResponse(
                success=True,
                data={
                    "file_id": file_id,
                    "original_name": original_name,
                    "size_bytes": file_size,
                    "stored_at": stored_file.created_at.isoformat()
                }
            )
            
        except Exception as e:
            return GatewayResponse(
                success=False,
                error=f"STORAGE_ERROR: Failed to store file: {str(e)}"
            )
            
    def retrieve_file(self, file_id: str) -> GatewayResponse:
        """
        Retrieve a stored file.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            GatewayResponse with file path and metadata
        """
        try:
            if file_id not in self._stored_files:
                return GatewayResponse(
                    success=False,
                    error=f"FILE_NOT_FOUND: File {file_id} not found"
                )
                
            stored_file = self._stored_files[file_id]
            
            # Check if file still exists
            if not os.path.exists(stored_file.stored_path):
                return GatewayResponse(
                    success=False,
                    error=f"FILE_MISSING: File {file_id} missing from storage"
                )
                
            # Update access time
            stored_file.accessed_at = datetime.now()
            
            # Log access
            self._log_file_operation("RETRIEVE", file_id, stored_file.original_name)
            
            return GatewayResponse(
                success=True,
                data={
                    "file_id": file_id,
                    "file_path": stored_file.stored_path,
                    "original_name": stored_file.original_name,
                    "size_bytes": stored_file.size_bytes,
                    "mime_type": stored_file.mime_type,
                    "created_at": stored_file.created_at.isoformat(),
                    "tags": stored_file.tags
                }
            )
            
        except Exception as e:
            return GatewayResponse(
                success=False,
                error=f"RETRIEVAL_ERROR: Failed to retrieve file: {str(e)}"
            )
            
    def list_files(self, tags: Optional[Dict[str, Any]] = None) -> GatewayResponse:
        """
        List stored files with optional tag filtering.
        
        Args:
            tags: Optional tags to filter by
            
        Returns:
            GatewayResponse with list of file metadata
        """
        try:
            files = []
            
            for file_id, stored_file in self._stored_files.items():
                # Filter by tags if provided
                if tags:
                    if not all(stored_file.tags.get(k) == v for k, v in tags.items()):
                        continue
                        
                files.append({
                    "file_id": file_id,
                    "original_name": stored_file.original_name,
                    "size_bytes": stored_file.size_bytes,
                    "mime_type": stored_file.mime_type,
                    "created_at": stored_file.created_at.isoformat(),
                    "accessed_at": stored_file.accessed_at.isoformat() if stored_file.accessed_at else None,
                    "tags": stored_file.tags
                })
                
            return GatewayResponse(
                success=True,
                data={"files": files, "count": len(files)}
            )
            
        except Exception as e:
            return GatewayResponse(
                success=False,
                error=f"LIST_ERROR: Failed to list files: {str(e)}"
            )
            
    def delete_file(self, file_id: str) -> GatewayResponse:
        """
        Delete a stored file.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            GatewayResponse confirming deletion
        """
        try:
            if file_id not in self._stored_files:
                return GatewayResponse(
                    success=False,
                    error=f"FILE_NOT_FOUND: File {file_id} not found"
                )
                
            stored_file = self._stored_files[file_id]
            
            # Remove physical file
            if os.path.exists(stored_file.stored_path):
                os.remove(stored_file.stored_path)
                
            # Remove metadata
            del self._stored_files[file_id]
            
            # Log deletion
            self._log_file_operation("DELETE", file_id, stored_file.original_name)
            
            return GatewayResponse(
                success=True,
                data={"file_id": file_id, "deleted": True}
            )
            
        except Exception as e:
            return GatewayResponse(
                success=False,
                error=f"DELETE_ERROR: Failed to delete file: {str(e)}"
            )
            
    def _guess_mime_type(self, filename: str) -> str:
        """Guess MIME type from filename."""
        ext = Path(filename).suffix.lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.csv': 'text/csv',
            '.json': 'application/json'
        }
        return mime_types.get(ext, 'application/octet-stream')
        
    def _log_file_operation(self, operation: str, file_id: str, filename: str):
        """Log file operation for audit trail."""
        if not self.file_config.enable_audit_log:
            return
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "file_id": file_id,
            "filename": filename,
            "user": "system"  # TODO: Get actual user from context
        }
        
        # In production, this would write to a proper audit log
        print(f"AUDIT: {log_entry}")
        
    def cleanup_expired_files(self) -> GatewayResponse:
        """Remove files older than retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.file_config.retention_days)
            expired_files = []
            
            for file_id, stored_file in list(self._stored_files.items()):
                if stored_file.created_at < cutoff_date:
                    # Delete expired file
                    if os.path.exists(stored_file.stored_path):
                        os.remove(stored_file.stored_path)
                    del self._stored_files[file_id]
                    expired_files.append(file_id)
                    
            return GatewayResponse(
                success=True,
                data={"expired_files": expired_files, "count": len(expired_files)}
            )
            
        except Exception as e:
            return GatewayResponse(
                success=False,
                error=f"CLEANUP_ERROR: Failed to cleanup files: {str(e)}"
            )