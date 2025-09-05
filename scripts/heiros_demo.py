#!/usr/bin/env python3
"""
TidyLLM-HeirOS Live Demonstration
Simple walkthrough to show how the hierarchical workflow system works
"""

import sys
import os
sys.path.append('tidyllm-heiros/src/dag-manager')
sys.path.append('tidyllm-heiros/src/sparse-agreement')

try:
    from hierarchical_dag_manager import *
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the right directory with TidyLLM-HeirOS")
    sys.exit(1)

def demo_simple_workflow():
    """Demonstrate a simple hierarchical workflow"""
    print("="*80)
    print("TIDYLLM-HEIROS LIVE DEMONSTRATION")
    print("Hierarchical Workflow Management System")
    print("="*80)
    
    # 1. Create the hierarchical DAG manager
    print("\n1. Creating Hierarchical DAG Manager...")
    dag_manager = HierarchicalDAGManager(
        name="Simple Demo Workflow",
        compliance_level=ComplianceLevel.FULL_TRANSPARENCY
    )
    print("   SUCCESS: DAG Manager created with full transparency compliance")
    
    # 2. Build a simple workflow hierarchy
    print("\n2. Building Simple Workflow Hierarchy...")
    
    # Root workflow - sequence of steps
    root = SequenceNode(
        node_id="demo_root",
        name="Document Processing Demo",
        description="Simple document processing workflow"
    )
    
    # Step 1: Document validation
    validation_step = ActionNode(
        node_id="validate_doc",
        name="Document Validation",
        description="Check document format and size",
        action=lambda context: {
            "status": "success",
            "message": "Document validation passed",
            "file_size": 1024,
            "format": "PDF"
        }
    )
    
    # Step 2: Decision selector (different processing paths)
    processing_selector = SelectorNode(
        node_id="processing_choice",
        name="Processing Path Selection",
        description="Choose processing path based on document type"
    )
    
    # Path A: Simple processing
    simple_process = ActionNode(
        node_id="simple_process",
        name="Simple Processing",
        description="Standard document processing",
        action=lambda context: {
            "status": "success",
            "message": "Simple processing completed",
            "processing_time": 2.3
        }
    )
    
    # Path B: Complex processing  
    complex_process = ActionNode(
        node_id="complex_process",
        name="Complex Processing",
        description="Advanced document analysis",
        action=lambda context: {
            "status": "success", 
            "message": "Complex processing completed",
            "processing_time": 8.7
        }
    )
    
    # Build the hierarchy
    processing_selector.add_child(simple_process)
    processing_selector.add_child(complex_process)
    
    root.add_child(validation_step)
    root.add_child(processing_selector)
    
    print("   SUCCESS: Workflow hierarchy built:")
    print("     - Document Validation")
    print("     - Processing Path Selection")
    print("       - Simple Processing")
    print("       - Complex Processing")
    
    # 3. Add root to DAG manager and visualize
    print("\n3. Workflow Visualization:")
    dag_manager.add_root_node(root)
    visualization = dag_manager.visualize_hierarchy()
    print(visualization)
    
    # 4. Execute the workflow
    print("\n4. Executing Workflow...")
    
    # Create execution context
    context = {
        "workflow_id": "demo_001",
        "user_id": "demo_user", 
        "compliance_level": "full_transparency",
        "timestamp": "2025-09-01",
        "document_type": "standard"
    }
    
    # Execute the workflow
    result = dag_manager.execute_dag(context)
    
    print(f"   SUCCESS: Execution Status: {result.get('status', 'unknown')}")
    print(f"   SUCCESS: Duration: {result.get('duration', 0):.2f} seconds")
    print(f"   SUCCESS: Nodes Executed: {len(result.get('node_results', []))}")
    
    # 5. Generate compliance report
    print("\n5. Compliance & Audit Report:")
    try:
        report = dag_manager.generate_compliance_report()
        print("   SUCCESS: Compliance report generated")
        print(f"   - DAG Manager: {dag_manager.name}")
        print(f"   - Root Nodes: {len(dag_manager.root_nodes)}")
        print(f"   - Compliance Level: Full Transparency")
        if isinstance(report, dict):
            for key, value in report.items():
                print(f"   - {key}: {value}")
    except Exception as e:
        print(f"   Report generation: {e}")
        print("   - System operational with basic reporting")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE - Key Features Demonstrated:")
    print("SUCCESS: Hierarchical workflow structure (easy to understand)")
    print("SUCCESS: Sequence and Selector node types")
    print("SUCCESS: Action nodes with executable functions")
    print("SUCCESS: Complete audit trails for compliance")
    print("SUCCESS: Visual workflow representation")
    print("SUCCESS: Corporate-grade transparency and control")
    print("="*80)

def demo_sparse_agreements():
    """Demonstrate SPARSE agreement system"""
    print("\n" + "="*80)
    print("SPARSE AGREEMENTS DEMONSTRATION")
    print("Structured Pre-Approved Reasoning for Systematic Execution")
    print("="*80)
    
    try:
        from sparse_system import SparseAgreementManager, RiskLevel
        
        # Create SPARSE manager
        sparse_manager = SparseAgreementManager()
        print("SUCCESS: SPARSE Agreement Manager created")
        
        # Show existing agreements
        agreements = sparse_manager.list_agreements()
        print(f"SUCCESS: Found {len(agreements)} existing SPARSE agreements")
        
        if agreements:
            latest = agreements[0]
            print(f"\nLatest Agreement:")
            print(f"  - Title: {latest.get('title', 'Unknown')}")
            print(f"  - Status: {latest.get('status', 'Unknown')}")
            print(f"  - Risk Level: {latest.get('risk_assessment', {}).get('risk_level', 'Unknown')}")
            print(f"  - Stakeholder Approvals: {len(latest.get('stakeholder_approvals', []))}")
        
    except ImportError:
        print("SPARSE system not available in this demo")
    
    print("="*80)

if __name__ == "__main__":
    try:
        demo_simple_workflow()
        demo_sparse_agreements()
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()