#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Test Script

This script tests if the environment is correctly set up for the MVR Streamlit demo.
"""

import sys
import os

def main():
    """Test the environment"""
    print("🔍 Testing Environment for MVR Streamlit Demo")
    print("=" * 50)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info[0] < 3:
        print("❌ CRITICAL ERROR: Python 2 detected!")
        print("🚫 This demo requires Python 3.11+ in conda environment 'py311'")
        return False
    
    if sys.version_info[0] == 3 and sys.version_info[1] < 11:
        print("❌ WARNING: Python version < 3.11 detected!")
        print("💡 Recommended: Use Python 3.11+ in conda environment 'py311'")
    else:
        print("✅ Python version OK")
    
    # Check conda environment
    conda_env = os.getenv('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"🐍 Conda environment: {conda_env}")
        if conda_env == "py311":
            print("✅ Correct conda environment detected")
        else:
            print(f"❌ Wrong conda environment: {conda_env}")
            print("💡 Please run: conda activate py311")
            return False
    else:
        print("❌ No conda environment detected")
        print("💡 Please run: conda activate py311")
        return False
    
    # Check Streamlit
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not installed")
        print("💡 Please install: pip install streamlit plotly pandas")
        return False
    
    # Check Plotly
    try:
        import plotly
        print(f"✅ Plotly version: {plotly.__version__}")
    except ImportError:
        print("❌ Plotly not installed")
        print("💡 Please install: pip install plotly")
        return False
    
    # Check Pandas
    try:
        import pandas
        print(f"✅ Pandas version: {pandas.__version__}")
    except ImportError:
        print("❌ Pandas not installed")
        print("💡 Please install: pip install pandas")
        return False
    
    print("=" * 50)
    print("✅ Environment test passed!")
    print("🚀 Ready to run MVR Streamlit Demo")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)


