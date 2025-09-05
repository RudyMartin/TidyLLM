#!/usr/bin/env python3
"""
Flow Agreement: SOP PDFs Upload
==============================
Pushes SOP PDFs to S3://nsc-mvp1/knowledge_base/sop/
"""

import os
import boto3
from pathlib import Path

def upload_sop_pdfs():
    """Upload SOP PDFs to S3 - Flow Agreement"""
    
    print("=" * 50)
    print("FLOW AGREEMENT: SOP PDF UPLOAD")
    print("=" * 50)
    
    # Set AWS credentials explicitly
    os.environ['AWS_ACCESS_KEY_ID'] = 'REMOVED_AWS_KEY'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'REMOVED_AWS_SECRET'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    # Local SOP folder (sorted PDFs)
    sop_folder = Path("knowledge_base/sop")
    if not sop_folder.exists():
        print("[ERROR] No sop folder found")
        return False
    
    # Get existing session (no new boto3!)
    try:
        s3 = boto3.client('s3')
        print("[OK] Using existing AWS session")
    except Exception as e:
        print(f"[ERROR] Session issue: {e}")
        return False
    
    # Upload each PDF
    bucket = 'nsc-mvp1'
    s3_prefix = 'knowledge_base/sop/'
    
    uploaded = 0
    for pdf_file in sop_folder.glob("*.pdf"):
        try:
            s3_key = f"{s3_prefix}{pdf_file.name}"
            s3.upload_file(str(pdf_file), bucket, s3_key)
            print(f"[UPLOAD] {pdf_file.name} -> s3://{bucket}/{s3_key}")
            uploaded += 1
        except Exception as e:
            print(f"[ERROR] {pdf_file.name}: {e}")
    
    print(f"[COMPLETE] {uploaded} SOP PDFs uploaded")
    return uploaded > 0

if __name__ == "__main__":
    success = upload_sop_pdfs()
    exit(0 if success else 1)