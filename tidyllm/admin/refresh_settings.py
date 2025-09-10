#!/usr/bin/env python3
"""
Command-line utility to refresh TidyLLM settings.
Use this when you've added new values to settings.yaml and need to refresh the cache.
"""

import sys
import os
from pathlib import Path

# Add the tidyllm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def refresh_settings_cli():
    """Command-line interface for refreshing settings."""
    print("TidyLLM Settings Refresh Utility")
    print("=" * 40)
    
    try:
        from tidyllm.infrastructure.settings_manager import get_settings_manager, refresh_settings
        
        print("🔍 Step 1: Get Settings Manager...")
        settings_manager = get_settings_manager()
        print(f"✅ Settings Manager found")
        print(f"   - Settings file: {settings_manager.settings_file}")
        
        print("🔍 Step 2: Check current settings...")
        current_settings = settings_manager.get_settings()
        print(f"   - Current sections: {len(current_settings)}")
        print(f"   - Sections: {list(current_settings.keys())}")
        
        print("🔍 Step 3: Refresh settings from YAML file...")
        refresh_success = refresh_settings()
        
        if refresh_success:
            print("✅ Settings refreshed successfully!")
            
            print("🔍 Step 4: Verify refreshed settings...")
            refreshed_settings = settings_manager.get_settings()
            print(f"   - Refreshed sections: {len(refreshed_settings)}")
            print(f"   - Sections: {list(refreshed_settings.keys())}")
            
            # Show key configuration values
            print("\n📋 Key Configuration Values:")
            system_config = settings_manager.get_system_config()
            print(f"   - Root Path: {system_config.get('root_path')}")
            print(f"   - Corporate Mode: {system_config.get('corporate_mode')}")
            
            workflow_config = settings_manager.get_workflow_optimizer_config()
            print(f"   - Workflow Optimization Level: {workflow_config.get('optimization_level')}")
            print(f"   - DAG Manager Enabled: {workflow_config.get('enable_dag_manager')}")
            
            gateways_config = settings_manager.get_gateways_config()
            print(f"   - Gateways Configured: {len(gateways_config)}")
            for gateway_name, config in gateways_config.items():
                print(f"     - {gateway_name}: enabled={config.get('enabled')}, timeout={config.get('timeout')}s")
            
            print("\n🎉 Settings refresh completed successfully!")
            print("✅ All TidyLLM components will now use the updated configuration")
            
        else:
            print("❌ Failed to refresh settings")
            print("   - Check if settings.yaml file exists and is readable")
            print("   - Verify file permissions")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error refreshing settings: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_help():
    """Show help information."""
    print("""
TidyLLM Settings Refresh Utility

Usage:
    python refresh_settings.py [options]

Options:
    --help, -h          Show this help message
    --status            Show current settings status
    --refresh           Refresh settings from YAML file (default)

Examples:
    python refresh_settings.py                    # Refresh settings
    python refresh_settings.py --status          # Show current status
    python refresh_settings.py --help            # Show help

When to use:
    - After adding new configuration values to settings.yaml
    - When changing root_path or other system settings
    - After modifying gateway configurations
    - When updating workflow optimizer settings

The refresh utility updates the in-memory cache so all TidyLLM components
will immediately use the new configuration without restarting.
""")

def show_status():
    """Show current settings status."""
    print("TidyLLM Settings Status")
    print("=" * 30)
    
    try:
        from tidyllm.infrastructure.settings_manager import get_settings_manager
        
        settings_manager = get_settings_manager()
        print(f"Settings File: {settings_manager.settings_file}")
        print(f"Is Loaded: {settings_manager.is_loaded}")
        
        if settings_manager.is_loaded:
            settings = settings_manager.get_settings()
            print(f"Sections: {len(settings)}")
            print(f"Available Sections: {', '.join(settings.keys())}")
            
            # Show key values
            system_config = settings_manager.get_system_config()
            print(f"\nSystem Configuration:")
            print(f"  Root Path: {system_config.get('root_path')}")
            print(f"  Corporate Mode: {system_config.get('corporate_mode')}")
            
    except Exception as e:
        print(f"❌ Error getting status: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--help', '-h']:
            show_help()
        elif arg == '--status':
            show_status()
        elif arg == '--refresh':
            refresh_settings_cli()
        else:
            print(f"Unknown option: {arg}")
            show_help()
    else:
        # Default: refresh settings
        refresh_settings_cli()
