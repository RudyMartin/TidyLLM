"""
TidyLLM Onboarding System - Main Application
============================================

Clean, normalized Streamlit application for TidyLLM corporate onboarding.
"""

import streamlit as st
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import core components
from core.session_manager import init_streamlit_session_state
from ui.components.sidebar import render_sidebar
from ui.pages import (
    render_connection_page,
    render_chat_page,
    render_knowledge_page,
    render_workflows_page,
    render_testing_page,
    render_dashboard_page
)

# Streamlit configuration
st.set_page_config(
    page_title="TidyLLM Corporate Onboarding",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
    }
    
    .status-success {
        color: #27ae60;
        font-weight: bold;
    }
    
    .status-error {
        color: #e74c3c;
        font-weight: bold;
    }
    
    .status-warning {
        color: #f39c12;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point."""
    
    # Initialize session state
    init_streamlit_session_state()
    
    # Render header
    st.markdown('<div class="main-header">🏢 TidyLLM Corporate Onboarding System</div>', unsafe_allow_html=True)
    
    # Render sidebar
    selected_page = render_sidebar()
    
    # Render main content based on selection
    if selected_page == "Connection Config":
        render_connection_page()
    elif selected_page == "Chat Test":
        render_chat_page()
    elif selected_page == "Knowledge Management":
        render_knowledge_page()
    elif selected_page == "Workflows":
        render_workflows_page()
    elif selected_page == "Test Workflow":
        render_testing_page()
    elif selected_page == "Dashboard":
        render_dashboard_page()
    else:
        st.error(f"Unknown page: {selected_page}")

if __name__ == "__main__":
    main()
