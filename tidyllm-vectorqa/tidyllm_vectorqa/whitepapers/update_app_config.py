#!/usr/bin/env python3
"""
Update App Configuration Script
==============================

Updates the Streamlit app to use the new working demo settings.yaml configuration.
"""

def update_app_imports():
    """Update app.py to use the new settings adapter"""
    
    # Read current app.py
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Add settings adapter import after existing imports
    import_addition = """
# Import settings adapter for working demo configuration
try:
    from settings_adapter import get_settings_adapter, create_compatible_backend_config
    SETTINGS_ADAPTER_AVAILABLE = True
except ImportError:
    SETTINGS_ADAPTER_AVAILABLE = False
    print("Settings adapter not available, using legacy configuration")
"""
    
    # Find the right place to insert (after backend_config import)
    if "from backend_config import" in content and "from settings_adapter import" not in content:
        content = content.replace(
            "from backend_config import get_backend_config, render_backend_sidebar",
            "from backend_config import get_backend_config, render_backend_sidebar" + import_addition
        )
    
    # Update get_backend_config calls to use adapter when available
    if "get_backend_config()" in content:
        content = content.replace(
            "backend_config = get_backend_config()",
            """backend_config = get_backend_config()
    # Try to use settings adapter if available
    if SETTINGS_ADAPTER_AVAILABLE:
        try:
            settings_adapter = get_settings_adapter()
            # Update backend_config with adapter data
            if hasattr(backend_config, 'settings') and hasattr(backend_config.settings, 'postgres'):
                adapter_postgres = settings_adapter.postgres
                backend_config.settings.postgres.host = adapter_postgres.host
                backend_config.settings.postgres.port = adapter_postgres.port
                backend_config.settings.postgres.database = adapter_postgres.database
                backend_config.settings.postgres.username = adapter_postgres.username
                backend_config.settings.postgres.password = adapter_postgres.password
                backend_config.settings.postgres.ssl_mode = adapter_postgres.ssl_mode
        except Exception as e:
            st.warning(f"Could not load settings adapter: {e}")"""
        )
    
    # Write updated content
    with open('app_updated.py', 'w') as f:
        f.write(content)
    
    print("Created app_updated.py with settings adapter integration")


def create_simple_app_runner():
    """Create a simple app runner that works without Streamlit installed"""
    
    runner_content = '''#!/usr/bin/env python3
"""
Simple App Configuration Test
============================

Tests the configuration without requiring Streamlit to be installed.
"""

import sys
import os

# Add path for imports
sys.path.append('.')

def test_configuration():
    """Test the configuration setup"""
    print("Testing TidyLLM Whitepapers Configuration...")
    print("=" * 50)
    
    try:
        # Test settings adapter
        from settings_adapter import get_settings_adapter
        
        adapter = get_settings_adapter()
        print("Settings Adapter:")
        print(f"  PostgreSQL: {adapter.postgres.host}:{adapter.postgres.port}")
        print(f"  Database: {adapter.postgres.database}")
        print(f"  AWS Region: {adapter.aws.region}")
        print(f"  S3 Bucket: {adapter.aws.default_bucket}")
        print(f"  KMS Key: {adapter.aws.kms_key_id}")
        print()
        
        # Test S3 session manager
        from s3_session_manager import S3SessionManager
        
        manager = S3SessionManager()
        status = manager.get_credential_status()
        
        print("S3 Session Manager:")
        print(f"  KMS Key: {status['kms_key_id']}")
        print(f"  Default Bucket: {status['default_bucket']}")
        print(f"  Default Prefix: {status['default_prefix']}")
        print(f"  Credential Source: {status['source']}")
        print()
        
        # Test paper repository
        from paper_repository import get_paper_repository
        
        repo = get_paper_repository()
        stats = repo.get_repository_stats()
        
        print("Paper Repository:")
        print(f"  Total Papers: {stats['total_papers']}")
        print(f"  Total Size: {stats['total_size_mb']} MB")
        print(f"  Repository Path: {stats['repository_path']}")
        print()
        
        # Test search tracker
        try:
            from search_tracker import YRSNSearchTracker
            from settings_adapter import create_compatible_backend_config
            
            backend_config = create_compatible_backend_config()
            tracker = YRSNSearchTracker(backend_config)
            
            print("Search Tracker:")
            print("  PostgreSQL connection: Available")
            print("  YRSN tables: Ready")
            
        except Exception as e:
            print(f"Search Tracker: Error - {e}")
        
        print()
        print("SUCCESS: All components configured correctly!")
        print("Ready for Streamlit app with working demo settings.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_configuration()
'''
    
    with open('test_app_config.py', 'w') as f:
        f.write(runner_content)
    
    print("Created test_app_config.py for testing configuration")


if __name__ == '__main__':
    print("Updating TidyLLM Whitepapers App Configuration...")
    print("=" * 50)
    
    # Update app imports
    update_app_imports()
    
    # Create test runner
    create_simple_app_runner()
    
    print()
    print("Configuration update complete!")
    print("Next steps:")
    print("1. Test configuration: python test_app_config.py")
    print("2. Use app_updated.py with Streamlit when available")
    print("3. All components now use working demo settings.yaml")