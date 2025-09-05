#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naked Call Test Runner

Simple script to run the naked call test suite and provide clear output.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run the naked call test suite"""
    print("🚀 Naked Call Test Runner")
    print("=" * 50)
    
    try:
        # Import and run the test suite
        from tests.test_naked_calls import run_naked_call_tests
        
        success = run_naked_call_tests()
        
        if success:
            print("\n🎉 All naked call tests passed!")
            print("✅ Your codebase is properly centralized!")
            return 0
        else:
            print("\n⚠️  Some naked call tests failed!")
            print("🔧 Please fix the issues above to ensure proper centralization.")
            return 1
            
    except ImportError as e:
        print(f"❌ Error importing test suite: {e}")
        print("💡 Make sure you're running from the project root directory")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
