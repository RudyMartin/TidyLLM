"""
Main Streamlit Application Entry Point for VectorQA Sage

This module serves as the main entry point for the VectorQA Sage Streamlit application,
providing a unified interface for all QA evaluation and optimization features.

The application is organized into tabs that handle different aspects of the QA pipeline:
from data upload and normalization to model evaluation and DSPy compilation.

TODO - Add comprehensive error handling for tab loading failures
TODO - Add session state management and persistence
TODO - Add user authentication and authorization
TODO - Add application configuration management
TODO - Add real-time status monitoring
"""

# Environment setup
from config.setup import setup_env

# Robust import setup
import sys
from pathlib import Path
_src_dir = Path(__file__).parent
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

setup_env()

import streamlit as st

# Import tab modules using absolute imports
from t_upload import tab_upload_normalize
from t_split import tab_split_dataset
from t_edit import tab_edit_examples
from t_dspy_config import tab_dspy_prompt_configurator
from t_evaluate import tab_evaluate_models
from t_faiss_status import tab_faiss_status
from t_compile_dspy import tab_compile_dspy_pipeline
from t_dashboard import tab_evaluation_dashboard

st.set_page_config(page_title="VectorQA Sage", layout="wide")
st.sidebar.title("📚 VectorQA Sage")
tab = st.sidebar.radio("Select a tab", [
    "8. Dashboard (Topic Accuracy)",
    "1. Upload & Normalize",
    "2. Split Dataset",
    "3. Edit Examples",
    "4. Prompt Config (DSPy)",
    "5. Evaluate Models",
    "6. FAISS & Model Status",
    "7. Compile DSPy Module"
])

if tab.startswith("1"):
    tab_upload_normalize()
elif tab.startswith("2"):
    tab_split_dataset()
elif tab.startswith("3"):
    tab_edit_examples()
elif tab.startswith("4"):
    tab_dspy_prompt_configurator()
elif tab.startswith("5"):
    tab_evaluate_models()
elif tab.startswith("6"):
    tab_faiss_status()
elif tab.startswith("7"):
    tab_compile_dspy_pipeline()
elif tab.startswith("8"):
    tab_evaluation_dashboard()
