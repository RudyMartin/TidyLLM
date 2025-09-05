#!/usr/bin/env python
"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

S3 MVR Monitor - Automatic Processing Trigger
============================================

Monitors S3 for new MVR files and automatically processes them.
CONSTRAINTS COMPLIANT - No local storage, S3-first architecture.

Two modes:
1. Polling Mode - Checks S3 periodically for new files
2. Manual Trigger Mode - Process specific files on command
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.start_unified_sessions import UnifiedSessionManager
from scripts.s3_first_mvr_processor import S3FirstMVRProcessor
import json
import time
from datetime import datetime, timedelta
import argparse


class S3MVRMonitor:
    """
    S3-First MVR File Monitor and Processor
    
    Watches S3 bucket for new MVR files and automatically processes them
    using JB_Overview_Prompt for compliance validation.
    """
    
    def __init__(self):
        print("[S3-MONITOR] Initializing S3 MVR Monitor...")
        
        # Use UnifiedSessionManager (constraints-compliant)
        self.session_mgr = UnifiedSessionManager()
        self.processor = S3FirstMVRProcessor()
        
        # S3 configuration
        self.bucket = s3_config["bucket"]
        self.raw_prefix = build_s3_path("mvr_analysis", "raw/")
        self.prompts_prefix = build_s3_path("mvr_analysis", "prompts/")
        self.processed_marker_prefix = build_s3_path("mvr_analysis", "processed_markers/")
        
        # Processing state (stored in S3, not locally)
        self.processed_files = set()
        
        print(f"   Bucket: {self.bucket}")
        print(f"   Monitoring: {self.raw_prefix}")
        print(f"   Architecture: S3-First (no local storage)")
    
    def load_processed_files_from_s3(self):
        """Load list of already processed files from S3 (not local storage)"""
        
        try:
            s3_client = self.session_mgr.get_s3_client()
            
            # List processed markers in S3
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.processed_marker_prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract original file name from marker
                    marker_key = obj['Key']
                    original_file = marker_key.replace(self.processed_marker_prefix, '').replace('_processed.marker', '')
                    self.processed_files.add(f"{self.raw_prefix}{original_file}")
                    
            print(f"   Loaded {len(self.processed_files)} processed file markers from S3")
            
        except Exception as e:
            print(f"   [WARN] Could not load processed files list: {e}")
    
    def mark_file_as_processed(self, mvr_key: str, process_result: dict):
        """Mark file as processed by creating marker in S3 (not local storage)"""
        
        try:
            # Create processed marker in S3
            original_filename = mvr_key.replace(self.raw_prefix, '')
            marker_key = f"{self.processed_marker_prefix}{original_filename}_processed.marker"
            
            marker_data = {
                "original_file": mvr_key,
                "processed_at": datetime.now().isoformat(),
                "process_id": process_result.get('process_id'),
                "status": process_result.get('status'),
                "outputs": process_result.get('outputs', {})
            }
            
            s3_client = self.session_mgr.get_s3_client()
            s3_client.put_object(
                Bucket=self.bucket,
                Key=marker_key,
                Body=json.dumps(marker_data, indent=2).encode(),
                ServerSideEncryption='AES256'
            )
            
            # Add to in-memory set
            self.processed_files.add(mvr_key)
            
            print(f"   [MARKED] File marked as processed: {original_filename}")
            
        except Exception as e:
            print(f"   [ERROR] Failed to mark file as processed: {e}")
    
    def scan_for_new_files(self) -> list:
        """Scan S3 for new MVR files that haven't been processed"""
        
        try:
            s3_client = self.session_mgr.get_s3_client()
            
            # List all files in raw folder
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.raw_prefix
            )
            
            if 'Contents' not in response:
                return []
            
            # Find unprocessed files
            new_files = []
            
            for obj in response['Contents']:
                file_key = obj['Key']
                
                # Skip if already processed
                if file_key in self.processed_files:
                    continue
                
                # Skip if not a document file
                if not any(file_key.lower().endswith(ext) for ext in ['.pdf', '.docx', '.doc', '.txt']):
                    continue
                
                # Check if file is "settled" (not actively being uploaded)
                # File should be at least 30 seconds old
                if datetime.now() - obj['LastModified'].replace(tzinfo=None) < timedelta(seconds=30):
                    continue
                
                new_files.append({
                    'key': file_key,
                    'size': obj['Size'],
                    'modified': obj['LastModified']
                })
            
            return new_files
            
        except Exception as e:
            print(f"   [ERROR] Failed to scan for files: {e}")
            return []
    
    def process_file_automatically(self, file_info: dict) -> dict:
        """Process a single file automatically"""
        
        mvr_key = file_info['key']
        filename = mvr_key.replace(self.raw_prefix, '')
        
        print(f"\n[AUTO-PROCESS] Processing: {filename}")
        print(f"   S3 Key: {mvr_key}")
        print(f"   Size: {file_info['size']:,} bytes")
        
        try:
            # Use JB_Overview_Prompt for compliance validation
            prompt_key = f"{self.prompts_prefix}JB_Overview_Prompt.md"
            
            # Process using S3FirstMVRProcessor
            result = self.processor.process_mvr_s3_to_s3(
                source_bucket=self.bucket,
                mvr_key=mvr_key,
                prompt_key=prompt_key
            )
            
            if result['status'] == 'completed':
                print(f"   [SUCCESS] Processing completed: {result['process_id']}")
                print(f"   Reports generated:")
                for report_type, location in result['outputs'].items():
                    print(f"      {report_type}: {location}")
                
                # Mark as processed in S3
                self.mark_file_as_processed(mvr_key, result)
                
                # Log success to MLflow
                self.session_mgr.log_mlflow_experiment({
                    "operation": "auto_mvr_processing",
                    "file": filename,
                    "process_id": result['process_id'],
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                })
                
            else:
                print(f"   [FAILED] Processing failed: {result.get('error', 'Unknown error')}")
                
                # Log failure to MLflow  
                self.session_mgr.log_mlflow_experiment({
                    "operation": "auto_mvr_processing",
                    "file": filename,
                    "status": "failed",
                    "error": result.get('error', 'Unknown error'),
                    "timestamp": datetime.now().isoformat()
                })
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"   [ERROR] Processing failed: {error_msg}")
            
            # Log error to MLflow
            self.session_mgr.log_mlflow_experiment({
                "operation": "auto_mvr_processing",
                "file": filename,
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            return {"status": "error", "error": error_msg}
    
    def monitor_continuously(self, scan_interval: int = 60, max_duration: int = 3600):
        """
        Continuously monitor S3 for new files and process them
        
        Args:
            scan_interval: Seconds between S3 scans
            max_duration: Maximum monitoring time in seconds
        """
        
        print(f"\n[MONITOR] Starting continuous monitoring...")
        print(f"   Scan interval: {scan_interval} seconds")
        print(f"   Max duration: {max_duration} seconds ({max_duration//60} minutes)")
        print(f"   Watching: s3://{self.bucket}/{self.raw_prefix}")
        print(f"   Press Ctrl+C to stop")
        
        # Load existing processed files
        self.load_processed_files_from_s3()
        
        start_time = time.time()
        files_processed = 0
        
        try:
            while time.time() - start_time < max_duration:
                print(f"\n[SCAN] Scanning for new MVR files...")
                
                new_files = self.scan_for_new_files()
                
                if new_files:
                    print(f"   Found {len(new_files)} new file(s)")
                    
                    for file_info in new_files:
                        result = self.process_file_automatically(file_info)
                        if result['status'] in ['completed', 'success']:
                            files_processed += 1
                        
                        # Small delay between files
                        time.sleep(5)
                else:
                    print(f"   No new files found")
                
                print(f"   Next scan in {scan_interval} seconds...")
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            print(f"\n[STOPPED] Monitoring stopped by user")
        
        elapsed = time.time() - start_time
        print(f"\n[SUMMARY] Monitoring completed:")
        print(f"   Duration: {elapsed:.1f} seconds")
        print(f"   Files processed: {files_processed}")
        print(f"   Status: Ready for more files")
    
    def process_specific_file(self, filename: str):
        """Process a specific file by name"""
        
        mvr_key = f"{self.raw_prefix}{filename}"
        
        print(f"\n[MANUAL] Processing specific file: {filename}")
        
        try:
            # Check if file exists
            s3_client = self.session_mgr.get_s3_client()
            s3_client.head_object(Bucket=self.bucket, Key=mvr_key)
            
            # Get file info
            response = s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=mvr_key
            )
            
            if 'Contents' in response and len(response['Contents']) > 0:
                file_info = {
                    'key': mvr_key,
                    'size': response['Contents'][0]['Size'],
                    'modified': response['Contents'][0]['LastModified']
                }
                
                return self.process_file_automatically(file_info)
            else:
                raise Exception("File not found in S3")
                
        except Exception as e:
            print(f"   [ERROR] Cannot process file: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="S3 MVR Monitor - Automatic Processing Trigger"
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Monitoring modes')
    
    # Continuous monitoring
    monitor_parser = subparsers.add_parser('monitor', help='Continuously monitor S3')
    monitor_parser.add_argument('--interval', type=int, default=60,
                               help='Scan interval in seconds (default: 60)')
    monitor_parser.add_argument('--duration', type=int, default=3600,
                               help='Max monitoring duration in seconds (default: 3600)')
    
    # Process specific file
    process_parser = subparsers.add_parser('process', help='Process specific file')
    process_parser.add_argument('filename', help='Filename in raw folder to process')
    
    # Status check
    status_parser = subparsers.add_parser('status', help='Check monitoring status')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("S3 MVR MONITOR - AUTOMATIC PROCESSING TRIGGER")
    print("=" * 70)
    
    monitor = S3MVRMonitor()
    
    if not args.mode:
        print("\n[USAGE] Available commands:")
        print("   python scripts/s3_mvr_monitor.py monitor")
        print("   python scripts/s3_mvr_monitor.py process filename.pdf")  
        print("   python scripts/s3_mvr_monitor.py status")
        return
    
    try:
        if args.mode == 'monitor':
            monitor.monitor_continuously(args.interval, args.duration)
            
        elif args.mode == 'process':
            result = monitor.process_specific_file(args.filename)
            if result['status'] in ['completed', 'success']:
                print(f"\n[SUCCESS] File processed successfully")
            else:
                print(f"\n[FAILED] Processing failed: {result.get('error', 'Unknown error')}")
                
        elif args.mode == 'status':
            monitor.load_processed_files_from_s3()
            new_files = monitor.scan_for_new_files()
            
            print(f"\n[STATUS] S3 MVR Monitor Status:")
            print(f"   Bucket: s3://{monitor.bucket}/{monitor.raw_prefix}")
            print(f"   Processed files: {len(monitor.processed_files)}")
            print(f"   New files pending: {len(new_files)}")
            
            if new_files:
                print(f"   Pending files:")
                for file_info in new_files:
                    filename = file_info['key'].replace(monitor.raw_prefix, '')
                    print(f"      - {filename} ({file_info['size']:,} bytes)")
                    
    except Exception as e:
        print(f"\n[ERROR] Monitor failed: {e}")


if __name__ == "__main__":
    main()