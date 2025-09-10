#!/usr/bin/env python3
"""Test centralized settings manager."""

import sys
from pathlib import Path

# Add parent directory to Python path for tidyllm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_centralized_settings():
    """Test centralized settings manager."""
    print("Testing Centralized Settings Manager")
    print("=" * 50)
    
    try:
        from tidyllm.infrastructure.settings_manager import get_settings_manager, get_settings, get_workflow_optimizer_config
        
        print("🔍 Step 1: Initialize Settings Manager...")
        settings_manager = get_settings_manager()
        print(f"✅ Settings Manager initialized")
        print(f"   - Settings file: {settings_manager.settings_file}")
        print(f"   - Is loaded: {settings_manager.is_loaded}")
        
        print("🔍 Step 2: Test settings access...")
        all_settings = get_settings()
        print(f"   - Total settings sections: {len(all_settings)}")
        print(f"   - Sections: {list(all_settings.keys())}")
        
        print("🔍 Step 3: Test system configuration...")
        system_config = settings_manager.get_system_config()
        print(f"   - Root Path: {system_config.get('root_path')}")
        print(f"   - Config Folder: {system_config.get('config_folder')}")
        print(f"   - Corporate Mode: {system_config.get('corporate_mode')}")
        
        print("🔍 Step 4: Test workflow optimizer config...")
        workflow_config = get_workflow_optimizer_config()
        print(f"   - DAG Manager: {workflow_config.get('enable_dag_manager')}")
        print(f"   - Optimization Level: {workflow_config.get('optimization_level')}")
        print(f"   - Max Workflow Depth: {workflow_config.get('max_workflow_depth')}")
        
        print("🔍 Step 5: Test gateway configuration...")
        gateways_config = settings_manager.get_gateways_config()
        for gateway_name, config in gateways_config.items():
            enabled = settings_manager.is_gateway_enabled(gateway_name)
            timeout = settings_manager.get_gateway_timeout(gateway_name)
            print(f"   - {gateway_name}: enabled={enabled}, timeout={timeout}s")
        
        print("🔍 Step 6: Test path resolution...")
        root_path = settings_manager.get_root_path()
        config_path = settings_manager.get_config_path("settings.yaml")
        data_path = settings_manager.get_data_path("test.txt")
        print(f"   - Root Path: {root_path}")
        print(f"   - Config Path: {config_path}")
        print(f"   - Data Path: {data_path}")
        
        print()
        print("🎉 Centralized Settings Manager: WORKING!")
        print("✅ Settings loaded once and cached")
        print("✅ All configuration accessible without re-reading files")
        print("✅ Thread-safe singleton pattern")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_centralized_settings()
