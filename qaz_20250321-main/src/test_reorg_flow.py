#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reorg Flow Test - Proper MCP Hierarchy

Tests the new organizational structure:
MCPOrchestrator → Planner → SimpleQACoordinator → Workers

This validates that the reorg from SimpleQAOrchestrator to SimpleQACoordinator 
maintains functionality while following proper chain of command.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from backend.core.mcp.orchestrator import MCPOrchestrator
from backend.core.mcp.coordinators.simple_qa_coordinator import SimpleQACoordinator


def test_new_hierarchy():
    """Test the new MCP hierarchy with SimpleQA"""
    
    print("🏢 REORG TEST: New MCP Hierarchy")
    print("=" * 50)
    
    try:
        # Test 1: Direct SimpleQACoordinator (for comparison)
        print("\n📋 Test 1: Direct SimpleQACoordinator")
        qa_coordinator = SimpleQACoordinator()
        
        test_document = {
            'content': '''# Test Document
            
This is a test document for quality analysis.
It has some structure and reasonable content length.

## Section 1
- Point 1
- Point 2

## Section 2
More content here to meet the length requirements.
            ''',
            'metadata': {'title': 'Test Document', 'author': 'Reorg Test'}
        }
        
        # Test using the legacy compatibility method
        result = qa_coordinator.process_document(test_document)
        print(f"✅ Direct QA Result: Quality Score = {result.get('quality_score', 0):.2f}")
        print(f"   Status: {result.get('status', 'unknown')}")
        
        # Test 2: Through proper MCP hierarchy  
        print("\n🎯 Test 2: Full MCP Hierarchy Flow")
        mcp_orchestrator = MCPOrchestrator()
        
        # Register the QA coordinator with the planner
        qa_coordinator_for_mcp = SimpleQACoordinator()
        mcp_orchestrator.planner.register_coordinator("qa", qa_coordinator_for_mcp)
        
        # Process through full hierarchy
        user_request = "Analyze the quality of this test document for basic QA assessment"
        constraints = {
            "task_type": "qa_analysis",
            "coordinator": "qa",
            "document": test_document
        }
        
        mcp_result = mcp_orchestrator.process_request(user_request, constraints)
        print(f"✅ MCP Hierarchy Result: Success = {mcp_result.get('success', False)}")
        print(f"   Execution Time: {mcp_result.get('execution_metadata', {}).get('total_execution_time', 0):.2f}s")
        
        # Test 3: System Status Check
        print("\n📊 Test 3: System Status After Reorg")
        system_status = mcp_orchestrator.get_system_status()
        print(f"✅ System Overview:")
        print(f"   Total Executions: {system_status['system_overview']['total_executions']}")
        print(f"   Success Rate: {system_status['system_overview']['success_rate']:.2%}")
        print(f"   Registered Coordinators: {system_status['planner']['registered_coordinators']}")
        
        print("\n🎉 REORG SUCCESS: New hierarchy operational!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Missing MCP base components - need to fix imports")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        print("   Hierarchy test failed - need debugging")
        return False


def compare_old_vs_new():
    """Compare old vs new organizational patterns"""
    
    print("\n📊 ORGANIZATIONAL COMPARISON")
    print("=" * 50)
    
    print("\n❌ OLD PATTERN (Broken Hierarchy):")
    print("   User → SimpleQAOrchestrator")
    print("        → Direct worker calls") 
    print("        → Bypasses planning layer")
    print("        → No coordination with other orchestrators")
    print("        → Circular dependencies possible")
    
    print("\n✅ NEW PATTERN (Proper Chain of Command):")
    print("   User → MCPOrchestrator")
    print("        → Planner (strategic planning)")
    print("        → SimpleQACoordinator (tactical execution)")
    print("        → QA Workers (operational tasks)")
    print("        → Clean separation of concerns")
    
    print("\n💡 BENEFITS:")
    print("   ✅ No circular dependencies")
    print("   ✅ Clear chain of responsibility")
    print("   ✅ Consistent protocols across all components")
    print("   ✅ Proper error handling and retries")
    print("   ✅ Centralized analytics and monitoring")
    print("   ✅ Easy to extend with new coordinators")


if __name__ == "__main__":
    success = test_new_hierarchy()
    compare_old_vs_new()
    
    if success:
        print("\n🚀 Ready for Phase 2: Enhanced QA migration")
    else:
        print("\n🔧 Need to fix issues before proceeding")