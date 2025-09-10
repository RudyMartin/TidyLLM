#!/usr/bin/env python3
"""
AWS Configuration Checker for TidyLLM MCP Server
===============================================

This script helps diagnose AWS configuration issues with the TidyLLM MCP server.

Usage:
    python scripts/check_aws_config.py
"""

import sys
import os
from pathlib import Path

# Add tidyllm to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_aws_credentials():
    """Check for AWS credentials in various locations."""
    print("=== AWS Credentials Check ===")
    
    found_methods = []
    
    # 1. Environment variables
    if 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_ACCESS_KEY' in os.environ:
        found_methods.append("Environment variables")
        print("[OK] AWS credentials found in environment variables")
        print(f"     AWS_ACCESS_KEY_ID: {os.environ['AWS_ACCESS_KEY_ID'][:10]}...")
    else:
        print("[MISSING] AWS environment variables not set")
        print("     Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    
    # 2. AWS credentials file
    creds_file = Path.home() / ".aws" / "credentials"
    if creds_file.exists():
        found_methods.append("~/.aws/credentials")
        print(f"[OK] AWS credentials file found: {creds_file}")
        try:
            with open(creds_file) as f:
                content = f.read()
                if '[default]' in content:
                    print("     [default] profile found")
                else:
                    print("     [WARNING] No [default] profile found")
        except Exception as e:
            print(f"     [WARNING] Could not read file: {e}")
    else:
        print(f"[MISSING] AWS credentials file not found: {creds_file}")
    
    # 3. AWS config file
    config_file = Path.home() / ".aws" / "config" 
    if config_file.exists():
        print(f"[OK] AWS config file found: {config_file}")
        try:
            with open(config_file) as f:
                content = f.read()
                if 'region' in content:
                    print("     Region configuration found")
        except Exception as e:
            print(f"     [WARNING] Could not read file: {e}")
    else:
        print(f"[INFO] AWS config file not found: {config_file}")
    
    # 4. IAM role (for EC2)
    if 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI' in os.environ or 'AWS_CONTAINER_CREDENTIALS_FULL_URI' in os.environ:
        found_methods.append("Container credentials")
        print("[OK] Container/ECS credentials detected")
    
    return found_methods


def test_boto3_connection():
    """Test boto3 S3 connection."""
    print("\n=== Boto3 Connection Test ===")
    
    try:
        import boto3
        print("[OK] Boto3 imported successfully")
        
        # Test S3 client creation via UnifiedSessionManager
        try:
            from tidyllm.infrastructure.session.unified import UnifiedSessionManager
            session_mgr = UnifiedSessionManager()
            s3 = session_mgr.get_s3_client()
            print("[OK] S3 client created via UnifiedSessionManager")
            
            # Test actual AWS connection
            try:
                response = s3.list_buckets()
                buckets = response.get('Buckets', [])
                print(f"[OK] AWS connection successful - found {len(buckets)} buckets")
                
                if len(buckets) > 0:
                    print("     Sample buckets:")
                    for bucket in buckets[:3]:
                        print(f"     - {bucket['Name']}")
                        
                return True, None
                
            except Exception as e:
                print(f"[ERROR] AWS connection failed: {e}")
                return False, str(e)
                
        except Exception as e:
            print(f"[ERROR] Cannot create S3 client: {e}")
            return False, str(e)
            
    except ImportError as e:
        print(f"[ERROR] Cannot import boto3: {e}")
        return False, str(e)


def test_unified_session_manager():
    """Test UnifiedSessionManager connection."""
    print("\n=== UnifiedSessionManager Test ===")
    
    try:
        from tidyllm.infrastructure.session.unified import UnifiedSessionManager
        print("[OK] UnifiedSessionManager imported")
        
        try:
            usm = UnifiedSessionManager()
            print("[OK] UnifiedSessionManager instance created")
            
            try:
                s3_client = usm.get_s3_client()
                print("[OK] S3 client obtained from UnifiedSessionManager")
                
                try:
                    response = s3_client.list_buckets()
                    buckets = response.get('Buckets', [])
                    print(f"[OK] UnifiedSessionManager AWS connection works - {len(buckets)} buckets")
                    return True, None
                    
                except Exception as e:
                    print(f"[ERROR] UnifiedSessionManager AWS connection failed: {e}")
                    return False, str(e)
                    
            except Exception as e:
                print(f"[ERROR] Cannot get S3 client from UnifiedSessionManager: {e}")
                return False, str(e)
                
        except Exception as e:
            print(f"[ERROR] Cannot create UnifiedSessionManager: {e}")
            return False, str(e)
            
    except ImportError as e:
        print(f"[ERROR] Cannot import UnifiedSessionManager: {e}")
        return False, str(e)


def provide_recommendations(creds_found, boto3_works, usm_works):
    """Provide recommendations based on test results."""
    print("\n=== Recommendations ===")
    
    if not creds_found:
        print("❌ NO AWS CREDENTIALS FOUND")
        print("\nTo fix this, choose one option:")
        print("1. Set environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your_access_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("   export AWS_DEFAULT_REGION=us-east-1")
        print("\n2. Or create ~/.aws/credentials file:")
        print("   [default]")
        print("   aws_access_key_id = your_access_key")
        print("   aws_secret_access_key = your_secret_key")
        print("   region = us-east-1")
        
    elif usm_works:
        print("✅ CONFIGURATION WORKING")
        print("Your AWS configuration is working with UnifiedSessionManager.")
        print("The TidyLLM MCP server should work properly.")
        
    elif boto3_works:
        print("⚠️  PARTIAL CONFIGURATION")
        print("Boto3 works but UnifiedSessionManager has issues.")
        print("The MCP server will work but may have reduced functionality.")
        
    else:
        print("❌ CONFIGURATION ISSUES")
        print("Found credentials but neither boto3 nor UnifiedSessionManager work.")
        print("Check your AWS credentials and permissions.")
        
    print(f"\nFound credential methods: {', '.join(creds_found) if creds_found else 'None'}")


def main():
    """Main diagnostic function."""
    print("TidyLLM MCP Server - AWS Configuration Checker")
    print("=" * 50)
    
    # Check credentials
    creds_found = check_aws_credentials()
    
    # Test connections
    boto3_works, boto3_error = test_boto3_connection()
    usm_works, usm_error = test_unified_session_manager()
    
    # Provide recommendations
    provide_recommendations(creds_found, boto3_works, usm_works)
    
    # Show current MCP server behavior
    print("\n=== MCP Server Behavior ===")
    if usm_works:
        print("✅ MCP server will use real S3 data")
    elif creds_found:
        print("⚠️  MCP server will attempt S3 but may fall back to mock data")
    else:
        print("📝 MCP server will use mock data (demo mode)")
        
    print("\nTo test the MCP server:")
    print("  python scripts/run_mcp_server.py")
    print("  python scripts/demo_mcp_integration.py")


if __name__ == "__main__":
    main()