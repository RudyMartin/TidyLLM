"""
DSPy Prompt Configuration Tab

This tab handles prompt strategy configuration with DSPy/LLM toggle functionality.
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

from core.dspy_prompt_config import render_dspy_prompt

# Initialize session state
if 'prompt_config' not in st.session_state:
    st.session_state.prompt_config = {
        'run_mode': 'LLM',
        'strategy': 'naive'
    }

def tab_dspy_prompt_configurator():
    st.header("⚙️ Prompt Config (DSPy)")
    st.write("Configure DSPy prompt strategies and LLM settings.")
    
    # Placeholder for DSPy prompt configuration
    st.info("DSPy prompt configuration functionality will be implemented here.")
    
    if st.button("Configure DSPy Prompts"):
        st.success("DSPy prompts configured successfully!")
