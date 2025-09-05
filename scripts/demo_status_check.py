#!/usr/bin/env python3
"""
Demo Status Check - Test if TidyLLM demos work

Checks the status of all demo components without running full demos.
"""

import sys
import os
import importlib

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"SUCCESS: {description}")
        return True
    except ImportError as e:
        print(f"FAILED: {description} - {e}")
        return False
    except Exception as e:
        print(f"ERROR: {description} - {e}")
        return False

def main():
    print("=== TidyLLM Demo Status Check ===")
    print("Checking if all demo components can be imported...\n")
    
    # Add tidyllm to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'tidyllm'))
    
    status = {}
    
    # Core TidyLLM
    status['core'] = check_import('tidyllm', 'Core TidyLLM package')
    
    # Gateway components
    status['gateway'] = check_import('tidyllm.gateway', 'Gateway integration')
    status['dspy_wrapper'] = check_import('tidyllm.dspy_wrapper', 'DSPy wrapper')
    status['dspy_gateway'] = check_import('tidyllm.dspy_gateway_backend', 'DSPy Gateway backend')
    
    # Demo components (check individually)
    print("\n--- Demo Components ---")
    
    # Change to tidyllm directory for relative imports
    original_cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(__file__), 'tidyllm'))
    
    try:
        status['sparse'] = check_import('sparse_agreements', 'SPARSE agreements system')
        status['error_tracking'] = check_import('error_tracker', 'Error tracking system')  
        status['demo_protection'] = check_import('demo_protection', 'Demo protection system')
        status['connection_manager'] = check_import('connection_manager', 'Connection manager')
    finally:
        os.chdir(original_cwd)
    
    # External dependencies
    print("\n--- External Dependencies ---")
    status['streamlit'] = check_import('streamlit', 'Streamlit (for web interface)')
    status['mlflow'] = check_import('mlflow', 'MLflow (for Gateway backend)')
    status['dspy'] = check_import('dspy', 'DSPy (for AI programming)')
    status['yaml'] = check_import('yaml', 'YAML (for configuration)')
    
    # Summary
    print("\n=== Demo Status Summary ===")
    total = len(status)
    working = sum(status.values())
    
    print(f"Components working: {working}/{total} ({working/total*100:.1f}%)")
    
    if working == total:
        print("SUCCESS: All components are working!")
        print("Demos should run without import errors.")
    elif working >= total * 0.8:
        print("MOSTLY WORKING: Most components are working.")
        print("Some demos may have issues with failed components.")
    else:
        print("ISSUES: Many components have import problems.")
        print("Demos will likely fail. Check missing dependencies.")
    
    print("\n--- Component Status Details ---")
    for component, works in status.items():
        symbol = "SUCCESS" if works else "FAILED"
        print(f"{symbol}: {component}")
    
    return working == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)