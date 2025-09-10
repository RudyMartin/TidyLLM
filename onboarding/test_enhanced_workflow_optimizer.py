#!/usr/bin/env python3
"""Test the enhanced WorkflowOptimizerGateway with DomainRAG integration."""

import sys
from pathlib import Path

# Add parent directory to Python path for tidyllm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_enhanced_workflow_optimizer():
    """Test the enhanced WorkflowOptimizerGateway with DomainRAG integration."""
    print("Testing Enhanced WorkflowOptimizerGateway with DomainRAG Integration...")
    print("=" * 80)
    
    try:
        from tidyllm.workflow_optimizer import HierarchicalDAGManager
        from tidyllm.gateways.workflow_optimizer_gateway import WorkflowOptimizerGateway
        
        print("🚀 Initializing Enhanced WorkflowOptimizerGateway...")
        
        # Initialize the enhanced gateway
        gateway = WorkflowOptimizerGateway()
        
        print("✅ Enhanced WorkflowOptimizerGateway initialized successfully!")
        print()
        
        # Check if real components are available
        if hasattr(gateway, 'dag_manager') and gateway.dag_manager:
            dag_manager = gateway.dag_manager
            
            print("🧠 AI Manager Agent Capabilities:")
            print("-" * 40)
            agent_capabilities = dag_manager.agent_capabilities
            
            for capability, value in agent_capabilities.items():
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    print(f"{status} {capability}: {value}")
                elif isinstance(value, list):
                    print(f"📋 {capability}: {', '.join(value)}")
                elif isinstance(value, dict):
                    print(f"📊 {capability}: {len(value)} items")
                    for key, info in value.items():
                        if isinstance(info, dict) and 'document_count' in info:
                            print(f"   - {key}: {info['document_count']} documents")
                else:
                    print(f"ℹ️  {capability}: {value}")
            
            print()
            
            # Test workflow creation and optimization
            print("🔄 Testing Workflow Creation and Optimization...")
            print("-" * 50)
            
            # Create a test workflow from dropzone
            test_workflow = dag_manager.create_workflow_from_dropzone(
                dropzone_path="test_dropzone/",
                workflow_id="test_workflow_001"
            )
            
            print(f"✅ Created workflow: {test_workflow.dag_id}")
            print(f"   - Nodes: {len(test_workflow.nodes)}")
            print(f"   - Edges: {len(test_workflow.edges)}")
            print(f"   - Source: {test_workflow.metadata.get('source', 'unknown')}")
            print()
            
            # Test workflow optimization with DomainRAG
            print("🎯 Testing Workflow Optimization with DomainRAG...")
            print("-" * 50)
            
            optimization_result = dag_manager.optimize_workflow("test_workflow_001")
            
            print(f"✅ Workflow optimization completed!")
            print(f"   - Performance Gain: {optimization_result['performance_gain']:.1f}%")
            print(f"   - Optimizations Applied: {len(optimization_result['optimizations'])}")
            print(f"   - Bottlenecks Resolved: {optimization_result['bottlenecks_resolved']}")
            print()
            
            # Show optimization details
            print("📋 Optimization Details:")
            print("-" * 30)
            for i, optimization in enumerate(optimization_result['optimizations'], 1):
                print(f"{i}. {optimization}")
            print()
            
            # Show metadata
            metadata = optimization_result['optimization_metadata']
            print("🔍 Optimization Metadata:")
            print("-" * 30)
            print(f"- Agent-Based: {metadata.get('agent_based', False)}")
            print(f"- AI Analysis Used: {metadata.get('ai_analysis_used', False)}")
            print(f"- Chat Interface Used: {metadata.get('chat_interface_used', False)}")
            print(f"- Worker Registry Used: {metadata.get('worker_registry_used', False)}")
            print()
            
            # Show infrastructure components
            components = metadata.get('infrastructure_components', {})
            print("🏗️  Infrastructure Components:")
            print("-" * 30)
            for component, available in components.items():
                status = "✅" if available else "❌"
                print(f"{status} {component}: {available}")
            print()
            
            # Show performance metrics
            metrics = dag_manager.get_performance_metrics()
            print("📊 Performance Metrics:")
            print("-" * 30)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"- {metric}: {value:.2f}")
                else:
                    print(f"- {metric}: {value}")
            print()
            
            print("🎉 Enhanced WorkflowOptimizerGateway with DomainRAG Integration Test Complete!")
            print("=" * 80)
            
        else:
            print("❌ DAG Manager not available - workflow optimization components not loaded")
            print("   This indicates the workflow_optimizer module is not properly integrated")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_workflow_optimizer()
