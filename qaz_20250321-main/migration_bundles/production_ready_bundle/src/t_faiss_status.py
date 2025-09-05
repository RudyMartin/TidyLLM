"""
FAISS and Model Status Tab

This tab displays FAISS index status and model information.
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

# Initialize session state
if 'faiss_status' not in st.session_state:
    st.session_state.faiss_status = None

def tab_faiss_status():
    st.header("🔍 FAISS & Model Status")
    
    # Display configuration information
    st.subheader("Configuration")
    st.json(CONFIG)
    
    # Display model status
    st.subheader("Model Status")
    st.info("Model status information will be displayed here.")
    
    # Placeholder for FAISS status
    st.subheader("FAISS Index Status")
    st.info("FAISS index status will be displayed here.")
