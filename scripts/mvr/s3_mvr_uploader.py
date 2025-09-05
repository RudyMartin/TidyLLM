#!/usr/bin/env python
"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

S3 MVR Uploader - CONSTRAINTS COMPLIANT
========================================

Upload MVR documents and prompts directly to S3 for processing.
NO local processing, NO local storage - pure S3-first workflow.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.start_unified_sessions import UnifiedSessionManager
from datetime import datetime
import os


class S3MVRUploader:
    """
    S3-First MVR document uploader
    
    Uploads MVR documents and prompt templates directly to S3
    for constraints-compliant processing.
    """
    
    def __init__(self):
        print("[S3-UPLOADER] Initializing...")
        
        # Use UnifiedSessionManager (constraints-compliant)
        self.session_mgr = UnifiedSessionManager()
        
        # S3 configuration
        self.bucket = s3_config["bucket"]
        self.paths = {
            "mvr_raw": build_s3_path("mvr_analysis", "raw/"),
            "prompts": build_s3_path("mvr_analysis", "prompts/"),
            "processed": build_s3_path("mvr_analysis", "processed/")
        }
        
        print(f"[S3-UPLOADER] Target bucket: {self.bucket}")
        print(f"[S3-UPLOADER] Session: UnifiedSessionManager")
    
    def upload_mvr_document(self, local_file_path: str, document_name: str = None) -> str:
        """
        Upload MVR document to S3 for processing
        
        Args:
            local_file_path: Path to local MVR file
            document_name: Optional custom name
            
        Returns:
            S3 key of uploaded document
        """
        
        if not Path(local_file_path).exists():
            raise FileNotFoundError(f"MVR file not found: {local_file_path}")
        
        # Generate S3 key
        if not document_name:
            document_name = Path(local_file_path).name
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"{self.paths['mvr_raw']}{timestamp}_{document_name}"
        
        print(f"\n[UPLOAD] MVR Document:")
        print(f"   Local: {local_file_path}")
        print(f"   S3: s3://{self.bucket}/{s3_key}")
        
        try:
            # Upload using UnifiedSessionManager
            success = self.session_mgr.upload_to_s3(
                bucket=self.bucket,
                key=s3_key,
                file_path=local_file_path
            )
            
            if success:
                print(f"   Status: [SUCCESS] Document uploaded")
                
                # Log upload to MLflow
                self.session_mgr.log_mlflow_experiment({
                    "operation": "mvr_upload",
                    "document": document_name,
                    "s3_location": f"s3://{self.bucket}/{s3_key}",
                    "file_size": os.path.getsize(local_file_path),
                    "timestamp": timestamp
                })
                
                return s3_key
            else:
                raise Exception("Upload failed")
                
        except Exception as e:
            print(f"   Status: [FAILED] {e}")
            raise
    
    def upload_prompt_template(self, local_prompt_path: str, prompt_name: str = None) -> str:
        """
        Upload prompt template to S3
        
        Args:
            local_prompt_path: Path to local prompt file
            prompt_name: Optional custom name
            
        Returns:
            S3 key of uploaded prompt
        """
        
        if not Path(local_prompt_path).exists():
            raise FileNotFoundError(f"Prompt file not found: {local_prompt_path}")
        
        # Generate S3 key
        if not prompt_name:
            prompt_name = Path(local_prompt_path).name
        
        s3_key = f"{self.paths['prompts']}{prompt_name}"
        
        print(f"\n[UPLOAD] Prompt Template:")
        print(f"   Local: {local_prompt_path}")
        print(f"   S3: s3://{self.bucket}/{s3_key}")
        
        try:
            # Upload using UnifiedSessionManager
            success = self.session_mgr.upload_to_s3(
                bucket=self.bucket,
                key=s3_key,
                file_path=local_prompt_path
            )
            
            if success:
                print(f"   Status: [SUCCESS] Prompt uploaded")
                return s3_key
            else:
                raise Exception("Upload failed")
                
        except Exception as e:
            print(f"   Status: [FAILED] {e}")
            raise
    
    def setup_jb_overview_prompt(self) -> str:
        """Upload JB_Overview_Prompt for compliance validation"""
        
        prompt_path = "qaz_20250321-main/src/assets/prompts/favorites/JB_Overview_Prompt.md"
        
        if Path(prompt_path).exists():
            return self.upload_prompt_template(prompt_path, "JB_Overview_Prompt.md")
        else:
            raise FileNotFoundError(f"JB_Overview_Prompt not found at: {prompt_path}")
    
    def list_uploaded_files(self):
        """List files in S3 bucket"""
        
        print(f"\n[S3-LIST] Files in s3://{self.bucket}/:")
        print("-" * 60)
        
        try:
            s3_client = self.session_mgr.get_s3_client()
            
            # List MVR documents
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.paths['mvr_raw']
            )
            
            print("MVR Documents:")
            if 'Contents' in response:
                for obj in response['Contents']:
                    size_mb = obj['Size'] / (1024 * 1024)
                    modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M')
                    print(f"   {obj['Key']} ({size_mb:.1f}MB, {modified})")
            else:
                print("   No MVR documents found")
            
            # List prompts
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.paths['prompts']
            )
            
            print("\nPrompt Templates:")
            if 'Contents' in response:
                for obj in response['Contents']:
                    size_kb = obj['Size'] / 1024
                    modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M')
                    print(f"   {obj['Key']} ({size_kb:.1f}KB, {modified})")
            else:
                print("   No prompts found")
                
        except Exception as e:
            print(f"   [ERROR] Cannot list files: {e}")


def main():
    """Demo S3 MVR uploader"""
    
    print("=" * 70)
    print("S3-FIRST MVR UPLOADER - CONSTRAINTS COMPLIANT")
    print("=" * 70)
    
    uploader = S3MVRUploader()
    
    print("\n[STEP 1] Setting up JB_Overview_Prompt...")
    try:
        prompt_key = uploader.setup_jb_overview_prompt()
        print(f"✅ Prompt ready: s3://{uploader.bucket}/{prompt_key}")
    except Exception as e:
        print(f"❌ Prompt setup failed: {e}")
        return
    
    print(f"\n[STEP 2] Ready to upload MVR documents")
    print("Example usage:")
    print("   mvr_key = uploader.upload_mvr_document('path/to/your/mvr.pdf')")
    print("   # Document will be uploaded to S3 and ready for processing")
    
    print(f"\n[STEP 3] Process with S3-first processor:")
    print("   from scripts.s3_first_mvr_processor import S3FirstMVRProcessor")
    print("   processor = S3FirstMVRProcessor()")
    print("   result = processor.process_mvr_s3_to_s3(")
    print(f"       '{uploader.bucket}',")
    print("       mvr_key,")
    print(f"       '{prompt_key}'")
    print("   )")
    
    print(f"\n[ARCHITECTURE] S3-First Workflow:")
    print("1. Upload MVR → S3")
    print("2. Upload Prompt → S3")  
    print("3. Process S3 → S3 (streaming)")
    print("4. Results → S3 (no local storage)")
    print("5. Track → PostgreSQL (direct)")
    
    # Show current S3 contents
    uploader.list_uploaded_files()


if __name__ == "__main__":
    main()