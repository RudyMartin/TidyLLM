#!/usr/bin/env python3
"""

# Centralized AWS credential management
import sys
from pathlib import Path

# Use proper package imports instead of sys.path manipulation
try:
    from ...admin.credential_loader import set_aws_environment
except ImportError:
    # Graceful fallback for standalone package
    def set_aws_environment():
        print("[INFO] AWS credential setup not available in standalone mode")
        print("Please configure AWS credentials manually via environment variables")

# Load AWS credentials using centralized system
set_aws_environment()
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
    
    
    
    
    # Local SOP folder (sorted PDFs)
    sop_folder = Path(build_s3_path("knowledge_base", "sop"))
    if not sop_folder.exists():
        print("[ERROR] No sop folder found")
        return False
    
    # Get S3 client through UnifiedSessionManager
    try:
        import sys
        from pathlib import Path
        # future_fix: remove parent.parent.parent.parent sys.path manipulation and use relative imports
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from tidyllm.infrastructure.session.unified import UnifiedSessionManager
        
        print("[SESSION] Using UnifiedSessionManager for S3 access")
        session_manager = UnifiedSessionManager()
        s3 = session_manager.get_s3_client()
        print("[OK] Using unified AWS session")
    except ImportError:
        print("[SESSION] NO FALLBACK - UnifiedSessionManager is required")
        raise RuntimeError("SOPFlow: UnifiedSessionManager is required for S3 access")
    except Exception as e:
        print(f"[ERROR] Session issue: {e}")
        return False
    
    # Upload each PDF
    bucket = s3_config["bucket"]
    s3_prefix = build_s3_path("knowledge_base", "sop/")
    
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