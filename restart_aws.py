#!/usr/bin/env python3
"""
AWS Restart using existing admin utilities
==========================================

Simple script to restart AWS sessions using the existing tidyllm/admin infrastructure.
Uses bucket-level operations as required for corporate environments.
"""

import os
import sys
import subprocess

def restart_aws():
    """Restart AWS using existing admin utilities"""
    
    print("=" * 60)
    print("AWS RESTART USING ADMIN UTILITIES")
    print("=" * 60)
    
    print("Restarting AWS using existing admin infrastructure...")
    
    # Use the existing, working admin restart script
    print("\nRunning tidyllm/admin/restart_aws_session.py...")
    
    try:
        result = subprocess.run([
            sys.executable, "tidyllm/admin/restart_aws_session.py", "--verify"
        ], capture_output=True, text=True, timeout=120)
        
        # Show the output
        if result.stdout:
            print("\nAdmin restart output:")
            print(result.stdout)
        
        if result.stderr:
            print("\nAdmin restart errors:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("AWS RESTART COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print("AWS credentials are now configured and verified")
            print("UnifiedSessionManager can now be used for S3 operations")
            print("Use bucket-level operations (upload/download/list_objects)")
            print("Do NOT use account-level operations (list_buckets)")
            return True
        else:
            print(f"\nAdmin restart failed with code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Admin restart script failed: {e}")
        return False

if __name__ == "__main__":
    success = restart_aws()
    
    if not success:
        print("\n" + "=" * 60)
        print("AWS RESTART FAILED")
        print("=" * 60)
        print("Check the admin utilities in tidyllm/admin/")
        sys.exit(1)