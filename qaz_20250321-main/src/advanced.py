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

st.set_page_config(
    page_title="VectorQA Sage - Advanced Interface",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("📚 VectorQA Sage")
st.sidebar.markdown("### Advanced Interface")

# Create a more intuitive navigation
tab_options = {
    "📊 Dashboard": "dashboard",
    "📤 Upload & Normalize": "upload",
    "✂️ Split Dataset": "split", 
    "✏️ Edit Examples": "edit",
    "⚙️ Prompt Config (DSPy)": "config",
    "🔍 Evaluate Models": "evaluate",
    "🗄️ FAISS & Model Status": "faiss",
    "🔧 Compile DSPy Module": "compile"
}

selected_tab = st.sidebar.selectbox(
    "Select a tab:",
    list(tab_options.keys()),
    index=0  # Start with Dashboard
)

# Add some helpful info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Info")
st.sidebar.info("This is the advanced interface with all VectorQA Sage features.")

# Navigation logic
tab_key = tab_options[selected_tab]

if tab_key == "dashboard":
    tab_evaluation_dashboard()
elif tab_key == "upload":
    tab_upload_normalize()
elif tab_key == "split":
    tab_split_dataset()
elif tab_key == "edit":
    tab_edit_examples()
elif tab_key == "config":
    tab_dspy_prompt_configurator()
elif tab_key == "evaluate":
    tab_evaluate_models()
elif tab_key == "faiss":
    tab_faiss_status()
elif tab_key == "compile":
    tab_compile_dspy_pipeline()
