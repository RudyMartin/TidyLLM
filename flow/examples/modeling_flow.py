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
Flow Agreement: Modeling PDFs Upload
===================================
Pushes modeling PDFs to S3://nsc-mvp1/knowledge_base/modeling/
"""

import os
import boto3
from pathlib import Path

def upload_modeling_pdfs():
    """Upload modeling PDFs to S3 - Flow Agreement"""
    
    print("=" * 50)
    print("FLOW AGREEMENT: MODELING PDF UPLOAD")
    print("=" * 50)
    
    # Set AWS credentials explicitly
    
    
    
    
    # Local modeling folder (sorted PDFs)
    modeling_folder = Path(build_s3_path("knowledge_base", "modeling"))
    if not modeling_folder.exists():
        print("[ERROR] No modeling folder found")
        return False
    
    # Get S3 client through UnifiedSessionManager
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from tidyllm.infrastructure.session.unified import UnifiedSessionManager
        
        print("[SESSION] Using UnifiedSessionManager for S3 access")
        session_manager = UnifiedSessionManager()
        s3 = session_manager.get_s3_client()
        print("[OK] Using unified AWS session")
    except ImportError:
        print("[SESSION] NO FALLBACK - UnifiedSessionManager is required")
        raise RuntimeError("ModelingFlow: UnifiedSessionManager is required for S3 access")
    except Exception as e:
        print(f"[ERROR] Session issue: {e}")
        return False
    
    # Upload each PDF
    bucket = s3_config["bucket"]
    s3_prefix = build_s3_path("knowledge_base", "modeling/")
    
    uploaded = 0
    for pdf_file in modeling_folder.glob("*.pdf"):
        try:
            s3_key = f"{s3_prefix}{pdf_file.name}"
            s3.upload_file(str(pdf_file), bucket, s3_key)
            print(f"[UPLOAD] {pdf_file.name} -> s3://{bucket}/{s3_key}")
            uploaded += 1
        except Exception as e:
            print(f"[ERROR] {pdf_file.name}: {e}")
    
    print(f"[COMPLETE] {uploaded} modeling PDFs uploaded")
    return uploaded > 0

if __name__ == "__main__":
    success = upload_modeling_pdfs()
    exit(0 if success else 1)