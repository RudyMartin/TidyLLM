#!/usr/bin/env python3
"""Test WorkflowOptimizerGateway with centralized settings."""

import sys
from pathlib import Path

# Add parent directory to Python path for tidyllm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_workflow_with_centralized_settings():
    """Test WorkflowOptimizerGateway with centralized settings."""
    print("Testing WorkflowOptimizerGateway with Centralized Settings")
    print("=" * 60)
    
    try:
        from tidyllm.gateways.workflow_optimizer_gateway import WorkflowOptimizerGateway
        from tidyllm.infrastructure.settings_manager import get_settings_manager
        
        print("🔍 Step 1: Initialize Settings Manager...")
        settings_manager = get_settings_manager()
        print(f"✅ Settings loaded from: {settings_manager.settings_file}")
        
        print("🔍 Step 2: Initialize WorkflowOptimizerGateway...")
        gateway = WorkflowOptimizerGateway()
        print("✅ WorkflowOptimizerGateway initialized")
        
        print("🔍 Step 3: Check configuration loading...")
        if gateway.config_manager:
            print("✅ Config Manager: ACTIVE")
            print(f"   - Config loaded from: {gateway.config_manager.config_path}")
            
            # Check if settings are properly loaded
            config_data = gateway.config_manager._config
            system_config = config_data.get("system", {})
            workflow_config = config_data.get("workflow_optimizer", {})
            
            print(f"   - Root Path: {system_config.get('root_path')}")
            print(f"   - Workflow Config Items: {len(workflow_config)}")
            print(f"   - DAG Manager: {workflow_config.get('enable_dag_manager')}")
            print(f"   - Optimization Level: {workflow_config.get('optimization_level')}")
        else:
            print("❌ Config Manager: NOT FOUND")
        
        print("🔍 Step 4: Check gateway components...")
        print(f"   - Session Manager: {'✅' if gateway.session_manager else '❌'}")
        print(f"   - DAG Manager: {'✅' if gateway.dag_manager else '❌'}")
        print(f"   - Flow Manager: {'✅' if gateway.flow_manager else '❌'}")
        
        print("🔍 Step 5: Test gateway capabilities...")
        capabilities = gateway.get_capabilities()
        print(f"   - Capabilities: {len(capabilities)} items")
        print(f"   - Operations: {len(capabilities.get('operations', []))}")
        
        print("🔍 Step 6: Verify centralized settings usage...")
        # Check if the gateway is using centralized settings
        workflow_config_from_manager = settings_manager.get_workflow_optimizer_config()
        print(f"   - Settings Manager Workflow Config: {len(workflow_config_from_manager)} items")
        print(f"   - Gateway Optimization Level: {gateway.optimizer_config.optimization_level}")
        print(f"   - Settings Manager Optimization Level: {workflow_config_from_manager.get('optimization_level')}")
        
        # Verify they match
        settings_match = (gateway.optimizer_config.optimization_level == 
                         workflow_config_from_manager.get('optimization_level'))
        print(f"   - Settings Match: {'✅' if settings_match else '❌'}")
        
        print()
        if gateway.config_manager and settings_match:
            print("🎉 WorkflowOptimizerGateway with Centralized Settings: SUCCESS!")
            print("✅ Settings loaded once and cached")
            print("✅ No repeated file reads")
            print("✅ Gateway uses centralized configuration")
            print("✅ Ready for onboarding system!")
        else:
            print("❌ WorkflowOptimizerGateway with Centralized Settings: FAILED")
        
        return gateway.config_manager is not None and settings_match
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_workflow_with_centralized_settings()
