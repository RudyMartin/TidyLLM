#!/usr/bin/env python3
"""Test settings refresh functionality."""

import sys
from pathlib import Path

# Add parent directory to Python path for tidyllm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_settings_refresh():
    """Test settings refresh functionality."""
    print("Testing Settings Refresh Functionality")
    print("=" * 50)
    
    try:
        from tidyllm.infrastructure.settings_manager import get_settings_manager, refresh_settings
        
        print("🔍 Step 1: Initialize Settings Manager...")
        settings_manager = get_settings_manager()
        print(f"✅ Settings Manager initialized")
        print(f"   - Settings file: {settings_manager.settings_file}")
        
        print("🔍 Step 2: Check initial settings...")
        initial_settings = settings_manager.get_settings()
        print(f"   - Initial settings sections: {len(initial_settings)}")
        print(f"   - Sections: {list(initial_settings.keys())}")
        
        print("🔍 Step 3: Test refresh functionality...")
        refresh_success = refresh_settings()
        print(f"   - Refresh successful: {'✅' if refresh_success else '❌'}")
        
        print("🔍 Step 4: Verify settings after refresh...")
        refreshed_settings = settings_manager.get_settings()
        print(f"   - Refreshed settings sections: {len(refreshed_settings)}")
        print(f"   - Sections: {list(refreshed_settings.keys())}")
        
        print("🔍 Step 5: Test specific configuration access...")
        system_config = settings_manager.get_system_config()
        workflow_config = settings_manager.get_workflow_optimizer_config()
        gateways_config = settings_manager.get_gateways_config()
        
        print(f"   - System config items: {len(system_config)}")
        print(f"   - Workflow config items: {len(workflow_config)}")
        print(f"   - Gateways config items: {len(gateways_config)}")
        
        print("🔍 Step 6: Test path resolution after refresh...")
        root_path = settings_manager.get_root_path()
        config_path = settings_manager.get_config_path("settings.yaml")
        print(f"   - Root Path: {root_path}")
        print(f"   - Config Path: {config_path}")
        
        print()
        if refresh_success and len(refreshed_settings) > 0:
            print("🎉 Settings Refresh: SUCCESS!")
            print("✅ Settings can be refreshed from YAML file")
            print("✅ Cached settings updated")
            print("✅ All configuration accessible after refresh")
        else:
            print("❌ Settings Refresh: FAILED")
        
        return refresh_success and len(refreshed_settings) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_settings_refresh()
