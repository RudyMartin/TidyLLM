#!/usr/bin/env python3
"""
Single Working FLOW Entry Point Demo
===================================

This demonstrates the ONE working FLOW system without scattered code.

Usage:
    python flow_demo.py
    python flow_demo.py "[Integration Test]"
"""

import sys
from pathlib import Path

# Add tidyllm to path
sys.path.insert(0, str(Path(__file__).parent / 'tidyllm'))

def main():
    """Demo the working FLOW system."""
    
    print("=" * 60)
    print("SINGLE WORKING FLOW SYSTEM DEMO")
    print("=" * 60)
    
    try:
        from tidyllm.flow import execute_flow_command, FlowAgreementManager
        
        # Get command from args or use default
        command = sys.argv[1] if len(sys.argv) > 1 else "[Integration Test]"
        
        print(f"Executing FLOW command: {command}")
        print("-" * 60)
        
        # Execute the flow
        result = execute_flow_command(command, context={"demo": True})
        
        # Show results
        if "error" in result:
            print("❌ FLOW FAILED")
            print(f"Error: {result['error']}")
            print("\nAvailable commands:")
            for cmd in result.get('available_agreements', []):
                print(f"  {cmd}")
        else:
            print("✅ FLOW EXECUTED")
            print(f"Action: {result.get('action', 'N/A')}")
            print(f"Mode: {result.get('execution_mode', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            
            if 'result' in result:
                print("\nResult:")
                if isinstance(result['result'], dict):
                    for key, value in result['result'].items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {result['result']}")
        
        print("\n" + "=" * 60)
        print("AVAILABLE FLOWS:")
        print("=" * 60)
        
        # Show all available flows
        manager = FlowAgreementManager()
        available = manager.get_available_agreements()
        
        for i, flow in enumerate(available, 1):
            print(f"{i:2d}. {flow}")
        
        print(f"\nTotal: {len(available)} FLOW commands available")
        
    except Exception as e:
        print(f"❌ FLOW SYSTEM ERROR: {e}")
        print("\nThis means the scattered FLOW code has import issues.")
        print("Need to consolidate the working pieces.")

if __name__ == "__main__":
    main()