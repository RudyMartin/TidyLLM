#!/usr/bin/env python3
"""
S3 MVR Monitor for tidyllm-compliance
====================================

Monitors S3 bucket for new MVR documents and triggers processing automatically.
Integrated with tidyllm-compliance framework for complete regulatory workflow.

Features:
- Continuous S3 monitoring with configurable intervals
- Automatic processing trigger for new documents
- Integrated YRSN validation and compliance checking
- Processing state tracking via S3 markers
- Error handling and retry logic

Part of tidyllm-compliance: Professional regulatory compliance platform
"""

import sys
from pathlib import Path

# Add tidyllm admin directory for credential management
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent.parent.parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()

import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from .s3_first_processor import S3FirstMVRProcessor

# Import UnifiedSessionManager for audit-compliant session management
try:
    from tidyllm.infrastructure.session.unified import UnifiedSessionManager
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    # Fallback to direct boto3 import if UnifiedSessionManager not available
    import boto3
    UNIFIED_SESSION_AVAILABLE = False

class S3MVRMonitor:
    """
    Continuous monitoring of S3 bucket for new MVR documents.
    Automatically triggers processing when new documents are detected.
    """
    
    def __init__(self, 
                 bucket_name: str = "nsc-mvp1",
                 base_prefix: str = "mvr_analysis",
                 monitor_interval: int = 60,
                 enable_compliance_validation: bool = True):
        
        self.bucket_name = bucket_name
        self.base_prefix = base_prefix
        self.monitor_interval = monitor_interval
        # AUDIT COMPLIANCE: Use UnifiedSessionManager instead of direct boto3
        if UNIFIED_SESSION_AVAILABLE:
            print("[MONITOR] Using UnifiedSessionManager for audit-compliant S3 access")
            self.session_manager = UnifiedSessionManager()
            self.s3_client = self.session_manager.get_s3_client()
        else:
            print("[MONITOR] NO FALLBACK - UnifiedSessionManager is required")
            raise RuntimeError("S3Monitor: UnifiedSessionManager is required for S3 access")
        
        # Initialize MVR processor with compliance validation
        self.processor = S3FirstMVRProcessor(
            bucket_name=bucket_name,
            base_prefix=base_prefix,
            enable_yrsn_validation=enable_compliance_validation,
            enable_evidence_validation=enable_compliance_validation
        )
        
        # S3 paths
        self.paths = {
            'raw': f"{base_prefix}/raw/",
            'processed': f"{base_prefix}/processed/",
            'prompts': f"{base_prefix}/prompts/",
            'markers': f"{base_prefix}/.markers/"
        }
        
        print(f"[MVR_MONITOR] Initialized monitor for bucket: {bucket_name}")
        print(f"[MVR_MONITOR] Monitoring interval: {monitor_interval} seconds")
        print(f"[MVR_MONITOR] Compliance validation: {'enabled' if enable_compliance_validation else 'disabled'}")
    
    def start_monitoring(self, max_iterations: Optional[int] = None) -> None:
        """
        Start continuous monitoring for new MVR documents.
        
        Args:
            max_iterations: Maximum monitoring cycles (None for infinite)
        """
        print(f"\n{'='*60}")
        print(f"STARTING S3 MVR MONITORING")
        print(f"Bucket: s3://{self.bucket_name}/{self.base_prefix}/")
        print(f"Interval: {self.monitor_interval} seconds")
        print(f"{'='*60}\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                
                print(f"[MONITOR] Cycle {iteration} - {current_time}")
                
                # Check for new documents
                new_documents = self._scan_for_new_documents()
                
                if new_documents:
                    print(f"[FOUND] {len(new_documents)} new document(s) to process")
                    
                    for doc_info in new_documents:
                        try:
                            self._process_document(doc_info)
                        except Exception as e:
                            print(f"[ERROR] Failed to process {doc_info['key']}: {e}")
                            # Continue with other documents
                else:
                    print(f"[MONITOR] No new documents found")
                
                # Check if we should stop
                if max_iterations and iteration >= max_iterations:
                    print(f"[MONITOR] Completed {max_iterations} monitoring cycles")
                    break
                
                # Wait for next cycle
                print(f"[MONITOR] Waiting {self.monitor_interval} seconds until next check...\n")
                time.sleep(self.monitor_interval)
                
        except KeyboardInterrupt:
            print(f"\n[MONITOR] Monitoring stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\n[ERROR] Monitor crashed: {e}")
            raise
    
    def _scan_for_new_documents(self) -> List[Dict[str, Any]]:
        """Scan S3 for new MVR documents that haven't been processed."""
        try:
            # List objects in raw directory
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.paths['raw']
            )
            
            if 'Contents' not in response:
                return []
            
            new_documents = []
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                
                # Skip directories and non-document files
                if s3_key.endswith('/') or not self._is_document_file(s3_key):
                    continue
                
                # Check if already processed
                if not self._is_already_processed(s3_key):
                    new_documents.append({
                        'key': s3_key,
                        'last_modified': obj['LastModified'],
                        'size': obj['Size']
                    })
            
            return new_documents
            
        except Exception as e:
            print(f"[ERROR] Failed to scan for new documents: {e}")
            return []
    
    def _is_document_file(self, s3_key: str) -> bool:
        """Check if S3 key represents a document file."""
        document_extensions = ['.txt', '.md', '.pdf', '.docx', '.doc']
        return any(s3_key.lower().endswith(ext) for ext in document_extensions)
    
    def _is_already_processed(self, s3_key: str) -> bool:
        """Check if document has already been processed by looking for marker."""
        # Create marker key based on document key
        marker_key = self._get_marker_key(s3_key)
        
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=marker_key)
            return True  # Marker exists, document already processed
        except self.s3_client.exceptions.ClientError:
            return False  # Marker doesn't exist, document not processed
    
    def _get_marker_key(self, document_key: str) -> str:
        """Generate marker key for tracking processed documents."""
        # Extract filename from document key
        filename = document_key.split('/')[-1]
        return f"{self.paths['markers']}{filename}.processed"
    
    def _process_document(self, doc_info: Dict[str, Any]) -> None:
        """
        Process a single MVR document with compliance validation.
        
        Args:
            doc_info: Document information from S3 scan
        """
        s3_key = doc_info['key']
        print(f"\n[PROCESSING] Starting analysis of: {s3_key}")
        
        try:
            # Find appropriate prompt for this document
            prompt_key = self._find_prompt_for_document(s3_key)
            
            if not prompt_key:
                print(f"[WARNING] No suitable prompt found for {s3_key} - using default")
                prompt_key = f"{self.paths['prompts']}default_mvr_prompt.md"
            
            # Process through MVR processor with compliance validation
            result = self.processor.process_mvr_document(s3_key, prompt_key)
            
            # Create processing marker
            self._create_processing_marker(s3_key, result)
            
            # Log results
            compliance_status = result.get('compliance_validation', {}).get('overall_compliance_status', 'unknown')
            print(f"[SUCCESS] Processed {s3_key}")
            print(f"[COMPLIANCE] Status: {compliance_status}")
            print(f"[REPORT] Location: {result.get('report_location', 'unknown')}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {s3_key}: {e}")
            # Create error marker
            self._create_error_marker(s3_key, str(e))
            raise
    
    def _find_prompt_for_document(self, document_key: str) -> Optional[str]:
        """Find appropriate analysis prompt for document type."""
        try:
            # List available prompts
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.paths['prompts']
            )
            
            if 'Contents' not in response:
                return None
            
            # Look for specific prompt types
            available_prompts = [obj['Key'] for obj in response['Contents'] 
                               if obj['Key'].endswith('.md')]
            
            # Simple matching logic (can be enhanced)
            for prompt_key in available_prompts:
                if 'overview' in prompt_key.lower() or 'jb_overview' in prompt_key.lower():
                    return prompt_key
            
            # Return first available prompt if no specific match
            return available_prompts[0] if available_prompts else None
            
        except Exception as e:
            print(f"[WARNING] Failed to find prompt for {document_key}: {e}")
            return None
    
    def _create_processing_marker(self, document_key: str, result: Dict[str, Any]) -> None:
        """Create S3 marker indicating document has been processed."""
        marker_key = self._get_marker_key(document_key)
        
        marker_data = {
            'document_key': document_key,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'process_id': result.get('process_id'),
            'compliance_status': result.get('compliance_validation', {}).get('overall_compliance_status'),
            'report_location': result.get('report_location'),
            'processing_time_seconds': result.get('processing_time_seconds')
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=marker_key,
                Body=json.dumps(marker_data, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
            print(f"[MARKER] Created processing marker: {marker_key}")
        except Exception as e:
            print(f"[WARNING] Failed to create processing marker: {e}")
    
    def _create_error_marker(self, document_key: str, error_message: str) -> None:
        """Create S3 marker indicating document processing failed."""
        marker_key = self._get_marker_key(document_key) + '.error'
        
        error_data = {
            'document_key': document_key,
            'error_at': datetime.now(timezone.utc).isoformat(),
            'error_message': error_message,
            'status': 'processing_failed'
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=marker_key,
                Body=json.dumps(error_data, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
            print(f"[MARKER] Created error marker: {marker_key}")
        except Exception as e:
            print(f"[WARNING] Failed to create error marker: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and statistics."""
        try:
            # Count processed documents
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.paths['markers']
            )
            
            processed_count = len(response.get('Contents', []))
            
            # Count pending documents  
            new_documents = self._scan_for_new_documents()
            pending_count = len(new_documents)
            
            return {
                'monitoring_active': True,
                'bucket': self.bucket_name,
                'base_prefix': self.base_prefix,
                'processed_documents': processed_count,
                'pending_documents': pending_count,
                'monitor_interval_seconds': self.monitor_interval,
                'last_check': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'monitoring_active': False,
                'error': str(e),
                'last_check': datetime.now(timezone.utc).isoformat()
            }

# Example usage and testing
def demo_mvr_monitoring():
    """
    Demonstrate S3 MVR monitoring functionality.
    """
    monitor = S3MVRMonitor(
        bucket_name="nsc-mvp1",
        base_prefix="mvr_analysis",
        monitor_interval=30,  # Check every 30 seconds for demo
        enable_compliance_validation=True
    )
    
    # Show current status
    status = monitor.get_monitoring_status()
    print(f"\nMonitoring Status:")
    print(f"- Processed documents: {status.get('processed_documents', 0)}")
    print(f"- Pending documents: {status.get('pending_documents', 0)}")
    
    # Start monitoring (run for 3 cycles as demo)
    try:
        monitor.start_monitoring(max_iterations=3)
    except Exception as e:
        print(f"Monitoring demo failed: {e}")
        print(f"Note: This requires actual S3 bucket access and documents")

if __name__ == "__main__":
    demo_mvr_monitoring()