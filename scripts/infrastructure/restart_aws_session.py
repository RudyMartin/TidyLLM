#!/usr/bin/env python
"""
Restart AWS Session - CLI Utility
================================

Manually restart AWS session when credentials go cold or expire.

Usage:
    python scripts/restart_aws_session.py
    python scripts/restart_aws_session.py --test
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.start_unified_sessions import UnifiedSessionManager
import argparse


def restart_aws_session():
    """Restart AWS session and validate connectivity"""
    
    print("=" * 60)
    print("AWS SESSION RESTART UTILITY")
    print("=" * 60)
    
    try:
        print("[RESTART] Initializing new UnifiedSessionManager...")
        session_mgr = UnifiedSessionManager()
        
        print("[TEST] Testing S3 connectivity...")
        s3_client = session_mgr.get_s3_client()
        
        if not s3_client:
            raise Exception("S3 client not available after restart")
        
        # Test with list_buckets
        buckets_response = s3_client.list_buckets()
        buckets = buckets_response['Buckets']
        
        print(f"[SUCCESS] AWS session restarted successfully!")
        print(f"[SUCCESS] Found {len(buckets)} accessible bucket(s):")
        
        for bucket in buckets:
            print(f"   - {bucket['Name']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Session restart failed: {e}")
        print("[ERROR] Please check AWS credentials:")
        print("   1. Run 'aws configure' to update credentials")
        print("   2. Check environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        print("   3. Verify IAM permissions")
        return False


def test_mvr_processor():
    """Test MVR processor initialization after session restart"""
    
    print("\n" + "=" * 60)
    print("TESTING MVR PROCESSOR WITH RESTARTED SESSION")
    print("=" * 60)
    
    try:
        from scripts.s3_first_mvr_processor import S3FirstMVRProcessor
        
        print("[TEST] Initializing S3FirstMVRProcessor...")
        processor = S3FirstMVRProcessor()
        
        print("[SUCCESS] MVR processor initialized successfully!")
        print(f"[SUCCESS] Bucket configured: {processor.s3_config['bucket']}")
        print(f"[SUCCESS] Gateways available: {processor.gateways is not None}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] MVR processor test failed: {e}")
        return False


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Restart AWS Session - CLI Utility"
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Test MVR processor after session restart')
    
    args = parser.parse_args()
    
    # Step 1: Restart AWS session
    session_success = restart_aws_session()
    
    if not session_success:
        print(f"\n[FAILED] Cannot proceed without valid AWS session")
        sys.exit(1)
    
    # Step 2: Test MVR processor if requested
    if args.test:
        processor_success = test_mvr_processor()
        
        if processor_success:
            print(f"\n[READY] AWS session and MVR processor are both ready!")
            print(f"[READY] You can now process MVR documents")
        else:
            print(f"\n[WARNING] AWS session works but MVR processor has issues")
            sys.exit(1)
    else:
        print(f"\n[READY] AWS session restarted successfully!")
        print(f"[READY] Use --test flag to verify MVR processor")


if __name__ == "__main__":
    main()