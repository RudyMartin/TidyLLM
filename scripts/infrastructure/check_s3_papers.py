#!/usr/bin/env python3
"""

# Centralized AWS credential management
import sys
from pathlib import Path

# Add admin directory to path for credential loading
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()
Check S3 bucket for papers count and details
"""
import boto3
import os
from datetime import datetime

def check_s3_papers():
    # Set AWS credentials
    
    
    
    
    try:
        print("="*80)
        print("S3 BUCKET ANALYSIS: nsc-mvp1/papers/papers/")
        print("="*80)
        
        s3 = boto3.client('s3')
        bucket = s3_config["bucket"]
        prefix = 'papers/papers/'
        
        print(f"Bucket: {bucket}")
        print(f"Prefix: {prefix}")
        print()
        
        # List objects with pagination
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        total_files = 0
        total_size = 0
        file_types = {}
        sample_files = []
        
        print("Scanning S3 bucket...")
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    total_files += 1
                    total_size += obj['Size']
                    
                    # Track file types
                    key = obj['Key']
                    if '.' in key:
                        ext = key.split('.')[-1].lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
                    
                    # Collect sample files (first 15)
                    if len(sample_files) < 15:
                        sample_files.append({
                            'key': key,
                            'size': obj['Size'],
                            'modified': obj['LastModified']
                        })
        
        # Display results
        print(f"\n" + "-"*60)
        print("SUMMARY")
        print("-"*60)
        print(f"Total Files: {total_files:,}")
        print(f"Total Size: {total_size:,} bytes ({total_size / (1024*1024):.1f} MB)")
        
        if file_types:
            print(f"\nFile Types:")
            for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  .{ext}: {count:,} files")
        
        if sample_files:
            print(f"\n" + "-"*60)
            print("SAMPLE FILES (first 15)")
            print("-"*60)
            for file_info in sample_files:
                size_mb = file_info['size'] / (1024*1024)
                modified = file_info['modified'].strftime("%Y-%m-%d %H:%M:%S")
                filename = file_info['key'].split('/')[-1]  # Just filename
                print(f"{filename:<50} {size_mb:>8.2f} MB  {modified}")
        
        # Check for specific patterns
        print(f"\n" + "-"*60)
        print("ANALYSIS")
        print("-"*60)
        
        pdf_count = file_types.get('pdf', 0)
        txt_count = file_types.get('txt', 0)
        json_count = file_types.get('json', 0)
        
        print(f"Research Papers (PDF): {pdf_count:,}")
        print(f"Text Files: {txt_count:,}")
        print(f"JSON Metadata: {json_count:,}")
        
        if total_files > 0:
            avg_size = total_size / total_files / (1024*1024)
            print(f"Average File Size: {avg_size:.1f} MB")
        
        return total_files, total_size, file_types
        
    except Exception as e:
        print(f"ERROR: Failed to access S3 bucket: {e}")
        return 0, 0, {}

if __name__ == "__main__":
    files, size, types = check_s3_papers()
    print(f"\n" + "="*80)
    print(f"RESULT: {files:,} files found in s3://nsc-mvp1/papers/papers/")
    print("="*80)