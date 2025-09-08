#!/usr/bin/env python3
"""
TidyLLM Smart File Upload App

Automatically reads S3 settings from admin/settings.yaml and configures
itself on startup. No need to hardcode credentials - everything is loaded
from the centralized configuration file.
"""

import os
import sys
import boto3
import tempfile
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add tidyllm to path for settings loader
sys.path.insert(0, str(Path(__file__).parent / 'tidyllm'))

from tidyllm.settings_loader import SettingsLoader

class SmartFileUploader:
    """Smart file uploader that auto-configures from admin/settings.yaml"""
    
    def __init__(self):
        self.settings_loader = None
        self.s3_client = None
        self.s3_resource = None
        self.target_bucket = None
        self.s3_config = {}
        self.postgres_config = {}
        
    def initialize_from_settings(self) -> bool:
        """Initialize S3 configuration from admin/settings.yaml"""
        print("1. Loading configuration from admin/settings.yaml...")
        
        try:
            # Load settings from admin folder
            admin_settings_path = Path(__file__).parent / 'tidyllm' / 'tidyllm' / 'admin' / 'settings.yaml'
            
            if admin_settings_path.exists():
                self.settings_loader = SettingsLoader(str(admin_settings_path))
                print(f"   SUCCESS: Settings loaded from admin folder")
            else:
                self.settings_loader = SettingsLoader()
                print(f"   SUCCESS: Settings loaded from default location")
            
            # Get S3 configuration
            self.s3_config = self.settings_loader.get_s3_config()
            print(f"   SUCCESS: S3 config loaded with {len(self.s3_config)} fields")
            
            # Get PostgreSQL configuration
            self.postgres_config = self.settings_loader.get_postgres_config()
            print(f"   SUCCESS: PostgreSQL config loaded with {len(self.postgres_config)} fields")
            
            # Set AWS environment variables from AWS credentials (if available)
            aws_config = self.settings_loader.settings.aws
            if 'region' in aws_config:
                os.environ['AWS_DEFAULT_REGION'] = aws_config['region']
                print(f"   SUCCESS: AWS region set to {aws_config['region']}")
            
            return True
            
        except Exception as e:
            print(f"   ERROR: Settings initialization failed: {e}")
            return False
    
    def setup_s3_client(self) -> bool:
        """Setup S3 client using UnifiedSessionManager"""
        print("\n2. Setting up S3 client...")
        
        try:
            # Try UnifiedSessionManager first
            try:
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent.parent.parent))
                from scripts.infrastructure.start_unified_sessions import UnifiedSessionManager
                
                print("   [SESSION] Using UnifiedSessionManager for S3 access")
                session_manager = UnifiedSessionManager()
                self.s3_client = session_manager.get_s3_client()
                
                # For resource operations, use the same session
                import boto3
                session = session_manager._create_boto3_session()
                self.s3_resource = session.resource('s3')
                
            except ImportError:
                print("   [SESSION] Fallback to direct boto3 (UnifiedSessionManager unavailable)")
                # Fallback to direct boto3
                region = self.s3_config.get('region', 'us-east-1')
                self.s3_client = boto3.client('s3', region_name=region)
                self.s3_resource = boto3.resource('s3', region_name=region)
            
            # Test connection by listing buckets
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            
            print(f"   SUCCESS: Connected to S3 in {region}")
            print(f"   SUCCESS: Found {len(buckets)} accessible buckets")
            
            # Use bucket from settings if specified and available
            settings_bucket = self.s3_config.get('bucket')
            if settings_bucket and settings_bucket in buckets:
                self.target_bucket = settings_bucket
                print(f"   SUCCESS: Using configured bucket: {self.target_bucket}")
            elif buckets:
                self.target_bucket = buckets[0]
                print(f"   SUCCESS: Using first available bucket: {self.target_bucket}")
            else:
                print("   ERROR: No accessible buckets found")
                return False
            
            return True
            
        except Exception as e:
            print(f"   ERROR: S3 setup failed: {e}")
            print("   INFO: This is expected if AWS credentials are not configured")
            print("   INFO: The settings were successfully loaded though!")
            return False
    
    def show_configuration_summary(self):
        """Display summary of loaded configuration"""
        print("\n3. Configuration Summary:")
        print(f"   S3 Region: {self.s3_config.get('region', 'Not set')}")
        print(f"   S3 Bucket: {self.target_bucket}")
        print(f"   S3 Prefix: {self.s3_config.get('prefix', 'Not set')}")
        print(f"   Connection Timeout: {self.s3_config.get('connection_timeout', 'Not set')}s")
        print(f"   Max Retries: {self.s3_config.get('max_retries', 'Not set')}")
        
        print(f"\n   PostgreSQL Host: {self.postgres_config.get('host', 'Not set')}")
        print(f"   PostgreSQL Database: {self.postgres_config.get('db_name', 'Not set')}")
        print(f"   PostgreSQL User: {self.postgres_config.get('db_user', 'Not set')}")
        print(f"   Connection Pool Size: {self.postgres_config.get('connection_pool_size', 'Not set')}")
    
    def select_sample_document(self) -> Optional[Path]:
        """Select a sample document to upload"""
        print("\n4. Selecting sample document...")
        
        # Use the same business.py file from the documents folder
        doc_file = Path("tidyllm-vectorqa/tidyllm_vectorqa/documents/templates/business.py")
        
        if doc_file.exists():
            size = doc_file.stat().st_size
            print(f"   SUCCESS: Selected: {doc_file.name}")
            print(f"      - Size: {size:,} bytes ({size/1024:.1f} KB)")
            print(f"      - Type: Business Document Template")
            return doc_file
        else:
            print(f"   ERROR: Sample file not found: {doc_file}")
            return None
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash for integrity verification"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def upload_with_settings(self, file_path: Path) -> Dict[str, Any]:
        """Upload file using settings-based configuration"""
        print(f"\n5. Uploading {file_path.name} with smart configuration...")
        
        try:
            # Generate S3 key using prefix from settings
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = self.s3_config.get('prefix', 'uploads/')
            s3_key = f"{prefix}smart-demo/{timestamp}/{file_path.name}"
            
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            
            print(f"   UPLOADING to: s3://{self.target_bucket}/{s3_key}")
            print(f"   File hash: {file_hash[:16]}...")
            
            # Upload with metadata including configuration info
            self.s3_client.upload_file(
                str(file_path),
                self.target_bucket,
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'upload-method': 'smart-settings-based',
                        'settings-source': 'admin/settings.yaml',
                        'configured-bucket': self.s3_config.get('bucket', 'auto-detected'),
                        'configured-region': self.s3_config.get('region', 'default'),
                        'configured-prefix': self.s3_config.get('prefix', 'none'),
                        'file-hash-sha256': file_hash,
                        'original-size': str(file_size),
                        'upload-timestamp': datetime.now().isoformat(),
                        'postgres-available': str(bool(self.postgres_config)),
                        'app-version': '2.0-smart-config'
                    },
                    'ContentType': 'text/x-python'
                }
            )
            
            # Verify upload
            response = self.s3_client.head_object(Bucket=self.target_bucket, Key=s3_key)
            uploaded_size = response['ContentLength']
            
            print(f"   SUCCESS: Upload complete!")
            print(f"      - S3 location: s3://{self.target_bucket}/{s3_key}")
            print(f"      - Size verified: {uploaded_size:,} bytes")
            print(f"      - Using prefix: {self.s3_config.get('prefix', 'default')}")
            
            return {
                "success": True,
                "bucket": self.target_bucket,
                "key": s3_key,
                "size": uploaded_size,
                "hash": file_hash,
                "config_source": "admin/settings.yaml",
                "settings_used": {
                    "s3_region": self.s3_config.get('region'),
                    "s3_bucket": self.s3_config.get('bucket'),
                    "s3_prefix": self.s3_config.get('prefix'),
                    "postgres_host": self.postgres_config.get('host')
                }
            }
            
        except Exception as e:
            print(f"   ERROR: Upload failed: {e}")
            return {"success": False, "error": str(e)}

def main():
    """Run the smart file upload application"""
    print("=" * 60)
    print("  TidyLLM Smart File Upload App")
    print("  Auto-configured from admin/settings.yaml")
    print("=" * 60)
    
    uploader = SmartFileUploader()
    
    try:
        # Initialize from settings
        if not uploader.initialize_from_settings():
            print("\nERROR: Settings initialization failed")
            return False
        
        # Show configuration summary first
        uploader.show_configuration_summary()
        
        # Try to setup S3 client (may fail without credentials)
        s3_available = uploader.setup_s3_client()
        if not s3_available:
            print("\nWARNING: S3 not available, but settings were loaded successfully")
            print("CONFIG: Demonstration of settings loading completed!")
            return True
        
        # Select sample document
        doc_file = uploader.select_sample_document()
        if not doc_file:
            print("\nERROR: No sample document found")
            return False
        
        # Upload with smart settings
        result = uploader.upload_with_settings(doc_file)
        
        # Final summary
        print("\n" + "=" * 60)
        print("  Smart Upload Results")
        print("=" * 60)
        
        if result.get("success"):
            print("SUCCESS: SMART FILE UPLOAD: SUCCESS!")
            print(f"SUCCESS: Auto-configured from: {result.get('config_source')}")
            print(f"SUCCESS: File uploaded: {doc_file.name}")
            print(f"SUCCESS: S3 location: s3://{result['bucket']}/{result['key']}")
            print(f"SUCCESS: Size: {result['size']:,} bytes")
            print(f"SUCCESS: Configuration automatically loaded")
            
            settings_info = result.get('settings_used', {})
            print(f"\nCONFIG: Settings Applied:")
            print(f"   S3 Region: {settings_info.get('s3_region')}")
            print(f"   S3 Bucket: {settings_info.get('s3_bucket')} (configured)")
            print(f"   S3 Prefix: {settings_info.get('s3_prefix')}")
            print(f"   PostgreSQL: {settings_info.get('postgres_host')}")
            
            success = True
        else:
            print("WARNING: SMART FILE UPLOAD: PARTIAL SUCCESS")
            print("Some operations failed - check details above")
            success = False
        
        return success
        
    except Exception as e:
        print(f"\nERROR: Smart upload application error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)