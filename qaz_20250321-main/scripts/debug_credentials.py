#!/usr/bin/env python3
"""
Debug script to check credential loading
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def debug_credentials():
    """Debug credential loading"""
    print("🔍 Debugging Credential Loading")
    print("=" * 40)
    
    # Check environment variables
    print("\n📋 Environment Variables:")
    print(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'NOT SET')}")
    print(f"DB_HOST: {os.getenv('DB_HOST', 'NOT SET')}")
    print(f"DB_USER: {os.getenv('DB_USER', 'NOT SET')}")
    print(f"DB_PASSWORD: {os.getenv('DB_PASSWORD', 'NOT SET')}")
    
    # Try to import credential manager
    try:
        from backend.config.credential_manager import credential_manager
        print("\n✅ Credential manager imported successfully")
        
        # Get database config
        db_config = credential_manager.get_database_config()
        print(f"\n📊 Database config from credential manager:")
        print(f"URL: {db_config.get('url', 'NOT SET')}")
        
        # Check credential manager status
        print(f"\n🔐 Credential manager status:")
        credential_manager.print_status()
        
    except Exception as e:
        print(f"\n❌ Error importing credential manager: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_credentials()
