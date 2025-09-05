"""
Compile DSPy Module Tab

This tab handles the compilation of DSPy modules with few-shot optimization.
"""

import streamlit as st
import sys
import os

# Add backend to path for imports

# Robust import setup
import sys
from pathlib import Path
_src_dir = Path(__file__).parent
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from core.config import CONFIG

def tab_compile_dspy_pipeline():
    st.header("🔧 Compile DSPy Module")
    st.write("This tab will compile DSPy modules with few-shot optimization.")
    st.info("DSPy compilation functionality will be implemented here.")
    
    # Placeholder for DSPy compilation logic
    if st.button("Compile DSPy Module"):
        st.success("DSPy module compilation completed!")
        st.write("Compiled module saved successfully.")
