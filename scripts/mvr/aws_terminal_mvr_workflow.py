#!/usr/bin/env python
"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

AWS Terminal MVR Workflow - CONSTRAINTS COMPLIANT
=================================================

Terminal-based workflow for:
1. Upload MVR documents to S3 folders
2. Process documents in cloud (S3 -> S3)
3. Download generated reports from S3

Usage:
    python scripts/aws_terminal_mvr_workflow.py upload document.pdf
    python scripts/aws_terminal_mvr_workflow.py process
    python scripts/aws_terminal_mvr_workflow.py download reports/
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.start_unified_sessions import UnifiedSessionManager
from scripts.s3_first_mvr_processor import S3FirstMVRProcessor
import json
import argparse
from datetime import datetime


class AWSTerminalMVRWorkflow:
    """Terminal-based S3-first MVR workflow"""
    
    def __init__(self):
        print("[AWS-TERMINAL] Initializing S3-First MVR Workflow...")
        
        # Use UnifiedSessionManager (constraints-compliant)
        self.session_mgr = UnifiedSessionManager()
        
        # S3 configuration
        self.bucket = s3_config["bucket"]
        self.prefix = "mvr_analysis"
        
        # Folder structure
        self.folders = {
            "raw": f"{self.prefix}/raw/",
            "prompts": f"{self.prefix}/prompts/", 
            "reports": f"{self.prefix}/reports/",
            "embeddings": f"{self.prefix}/embeddings/",
            "metadata": f"{self.prefix}/metadata/"
        }
        
        print(f"   Bucket: {self.bucket}")
        print(f"   Prefix: {self.prefix}/")
        print(f"   Session: UnifiedSessionManager")
    
    def upload_file_to_folder(self, local_file: str, folder_type: str) -> str:
        """Upload file to specific S3 folder"""
        
        if not Path(local_file).exists():
            raise FileNotFoundError(f"File not found: {local_file}")
        
        if folder_type not in self.folders:
            raise ValueError(f"Unknown folder type: {folder_type}. Options: {list(self.folders.keys())}")
        
        # Generate S3 key
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = Path(local_file).name
        s3_key = f"{self.folders[folder_type]}{timestamp}_{filename}"
        
        print(f"\n[UPLOAD] {folder_type.upper()} File:")
        print(f"   Local: {local_file}")
        print(f"   S3: s3://{self.bucket}/{s3_key}")
        
        try:
            # Upload using UnifiedSessionManager
            success = self.session_mgr.upload_to_s3(
                bucket=self.bucket,
                key=s3_key,
                file_path=local_file
            )
            
            if success:
                print(f"   Status: [SUCCESS]")
                
                # Log to MLflow
                self.session_mgr.log_mlflow_experiment({
                    "operation": f"terminal_upload_{folder_type}",
                    "file": filename,
                    "s3_location": f"s3://{self.bucket}/{s3_key}",
                    "timestamp": timestamp
                })
                
                return s3_key
            else:
                raise Exception("Upload failed")
                
        except Exception as e:
            print(f"   Status: [FAILED] {e}")
            raise
    
    def setup_prompt_templates(self):
        """Upload all prompt templates to S3"""
        
        print(f"\n[SETUP] Uploading prompt templates...")
        
        prompt_files = [
            ("qaz_20250321-main/src/assets/prompts/favorites/JB_Overview_Prompt.md", "JB_Overview_Prompt.md"),
            ("qaz_20250321-main/src/assets/prompts/favorites/comprehensive_whitepaper_analysis.md", "comprehensive_whitepaper_analysis.md"),
            ("qaz_20250321-main/src/assets/prompts/favorites/toc_extraction_prompt.md", "toc_extraction_prompt.md")
        ]
        
        uploaded_prompts = []
        
        for local_path, prompt_name in prompt_files:
            if Path(local_path).exists():
                try:
                    s3_key = f"{self.folders['prompts']}{prompt_name}"
                    
                    # Check if already exists
                    s3_client = self.session_mgr.get_s3_client()
                    try:
                        s3_client.head_object(Bucket=self.bucket, Key=s3_key)
                        print(f"   [EXISTS] {prompt_name}")
                        uploaded_prompts.append(s3_key)
                    except s3_client.exceptions.NoSuchKey:
                        # Upload prompt
                        success = self.session_mgr.upload_to_s3(
                            bucket=self.bucket,
                            key=s3_key,
                            file_path=local_path
                        )
                        if success:
                            print(f"   [UPLOADED] {prompt_name}")
                            uploaded_prompts.append(s3_key)
                        else:
                            print(f"   [FAILED] {prompt_name}")
                            
                except Exception as e:
                    print(f"   [ERROR] {prompt_name}: {e}")
            else:
                print(f"   [NOT FOUND] {local_path}")
        
        return uploaded_prompts
    
    def list_s3_folder(self, folder_type: str):
        """List files in S3 folder"""
        
        if folder_type not in self.folders:
            raise ValueError(f"Unknown folder: {folder_type}")
        
        print(f"\n[LIST] Files in s3://{self.bucket}/{self.folders[folder_type]}:")
        print("-" * 60)
        
        try:
            s3_client = self.session_mgr.get_s3_client()
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.folders[folder_type]
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    size_mb = obj['Size'] / (1024 * 1024)
                    modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M')
                    key = obj['Key'].replace(self.folders[folder_type], '')
                    print(f"   {key} ({size_mb:.1f}MB, {modified})")
            else:
                print("   No files found")
                
        except Exception as e:
            print(f"   [ERROR] Cannot list folder: {e}")
    
    def process_latest_mvr(self):
        """Process the latest MVR document in raw folder"""
        
        print(f"\n[PROCESS] Finding latest MVR document...")
        
        try:
            s3_client = self.session_mgr.get_s3_client()
            
            # Find latest MVR document
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.folders['raw']
            )
            
            if 'Contents' not in response:
                print("   [ERROR] No MVR documents found in raw folder")
                return None
            
            # Get latest file
            latest_file = max(response['Contents'], key=lambda x: x['LastModified'])
            mvr_key = latest_file['Key']
            
            print(f"   Found: {mvr_key}")
            
            # Use JB_Overview_Prompt
            prompt_key = f"{self.folders['prompts']}JB_Overview_Prompt.md"
            
            print(f"   Using prompt: {prompt_key}")
            
            # Process using S3FirstMVRProcessor
            processor = S3FirstMVRProcessor()
            result = processor.process_mvr_s3_to_s3(
                source_bucket=self.bucket,
                mvr_key=mvr_key,
                prompt_key=prompt_key
            )
            
            return result
            
        except Exception as e:
            print(f"   [ERROR] Processing failed: {e}")
            return None
    
    def download_folder_contents(self, folder_type: str, local_dir: str = None):
        """Download all files from S3 folder to local directory"""
        
        if folder_type not in self.folders:
            raise ValueError(f"Unknown folder: {folder_type}")
        
        # Set default local directory
        if not local_dir:
            local_dir = f"./downloaded_{folder_type}"
        
        Path(local_dir).mkdir(exist_ok=True)
        
        print(f"\n[DOWNLOAD] Downloading {folder_type} files to {local_dir}/")
        
        try:
            s3_client = self.session_mgr.get_s3_client()
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.folders[folder_type]
            )
            
            if 'Contents' not in response:
                print("   [INFO] No files to download")
                return []
            
            downloaded_files = []
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                filename = Path(s3_key).name
                local_file = Path(local_dir) / filename
                
                print(f"   Downloading: {filename}")
                
                try:
                    s3_client.download_file(self.bucket, s3_key, str(local_file))
                    downloaded_files.append(str(local_file))
                    print(f"      -> {local_file}")
                except Exception as e:
                    print(f"      [FAILED] {e}")
            
            print(f"\n   [SUCCESS] Downloaded {len(downloaded_files)} files")
            return downloaded_files
            
        except Exception as e:
            print(f"   [ERROR] Download failed: {e}")
            return []


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="AWS Terminal MVR Workflow - S3-First Processing"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload file to S3 folder')
    upload_parser.add_argument('file', help='Local file to upload')
    upload_parser.add_argument('folder', choices=['raw', 'prompts'], 
                              help='S3 folder type')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup prompt templates')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List files in S3 folder')
    list_parser.add_argument('folder', choices=['raw', 'prompts', 'reports', 'embeddings', 'metadata'],
                            help='Folder to list')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process latest MVR document')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download folder contents')
    download_parser.add_argument('folder', choices=['reports', 'embeddings', 'metadata'],
                                help='Folder to download')
    download_parser.add_argument('--local-dir', help='Local directory for downloads')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("=" * 70)
    print("AWS TERMINAL MVR WORKFLOW - S3-FIRST PROCESSING")
    print("=" * 70)
    
    workflow = AWSTerminalMVRWorkflow()
    
    try:
        if args.command == 'upload':
            s3_key = workflow.upload_file_to_folder(args.file, args.folder)
            print(f"\n[SUCCESS] File uploaded to: s3://{workflow.bucket}/{s3_key}")
            
        elif args.command == 'setup':
            prompts = workflow.setup_prompt_templates()
            print(f"\n[SUCCESS] Setup complete. {len(prompts)} prompts ready.")
            
        elif args.command == 'list':
            workflow.list_s3_folder(args.folder)
            
        elif args.command == 'process':
            result = workflow.process_latest_mvr()
            if result:
                print(f"\n[SUCCESS] Processing complete: {result['process_id']}")
                print(f"   Status: {result['status']}")
                if result['status'] == 'completed':
                    print("   Generated reports:")
                    for report_type, location in result['outputs'].items():
                        print(f"      {report_type}: {location}")
            else:
                print(f"\n[FAILED] Processing failed")
                
        elif args.command == 'download':
            files = workflow.download_folder_contents(args.folder, args.local_dir)
            print(f"\n[SUCCESS] Downloaded {len(files)} files")
            
    except Exception as e:
        print(f"\n[ERROR] Command failed: {e}")


if __name__ == "__main__":
    main()