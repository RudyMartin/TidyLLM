#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced QA Document Processing Demo - Simplified Version
Streamlit interface for smart QA document processing without external dependencies
"""

# Environment setup
from config.setup import setup_env
setup_env()

import streamlit as st
import zipfile
import tempfile
from typing import List, Dict, Any
from datetime import datetime
import json
import re

def extract_zip_safely(zip_file, max_size_mb=50, max_files=100):
    """Safely extract ZIP file with security checks"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_files = []
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir():
                        continue
                    
                    file_path = file_info.filename
                    content = zip_ref.read(file_path)
                    
                    extracted_files.append({
                        'name': file_path,
                        'content': content,
                        'type': 'unknown',
                        'size': len(content)
                    })
            
            return extracted_files
    except Exception as e:
        return f"Error extracting ZIP file: {str(e)}"

def validate_review_id(review_id):
    """Validate Review ID format"""
    if not review_id:
        return False, "Review ID is required"
    
    # Check format REVXXXXX
    pattern = r'^REV\d{5}$'
    if not re.match(pattern, review_id):
        return False, "Review ID must be in format REVXXXXX (e.g., REV00001)"
    
    return True, "Valid Review ID format"

def mock_document_processing(file_content, file_name):
    """Mock document processing for demo purposes"""
    # Simulate processing time
    import time
    time.sleep(1)
    
    # Generate mock results
    return {
        'success': True,
        'file_name': file_name,
        'processing_time': 1.2,
        'extracted_text': f"Sample extracted text from {file_name}",
        'key_topics': ['revenue', 'compliance', 'performance'],
        'claims': [
            'Revenue increased by 15% year-over-year',
            'Compliance score improved to 95%',
            'System performance maintained 99.9% uptime'
        ],
        'evidence': [
            'Financial reports show consistent growth',
            'Audit results indicate strong compliance',
            'Monitoring data confirms high performance'
        ],
        'live_context': {
            'available': True,
            'events': 27,
            'relevance_score': 0.85,
            'temporal_insights': [
                'Document correlates with recent market events',
                'Claims align with current business metrics',
                'Performance data matches live monitoring'
            ]
        }
    }

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Enhanced QA Document Processing",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Enhanced QA Document Processing")
    st.markdown("### Smart Document Processing with Live Context Integration")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Main App", use_container_width=True):
            st.switch_page("main.py")
    
    with col2:
        if st.button("📋 Basic QA Demo", use_container_width=True):
            st.switch_page("qa_demo.py")
    
    with col3:
        if st.button("📊 MCP Dashboard", use_container_width=True):
            st.switch_page("mcp_dashboard.py")
    
    st.markdown("---")
    
    # Initialize session state
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Review ID input
    review_id = st.sidebar.text_input(
        "Review ID",
        value="REV00001",
        help="Enter a review ID in format REVXXXXX"
    )
    
    # Validate review ID
    is_valid, validation_message = validate_review_id(review_id)
    if not is_valid:
        st.sidebar.error(validation_message)
    else:
        st.sidebar.success(validation_message)
    
    # User preferences
    st.sidebar.subheader("🎯 User Preferences")
    
    include_live_context = st.sidebar.checkbox(
        "Include Live Context",
        value=True,
        help="Enable real-time data integration"
    )
    
    processing_mode = st.sidebar.selectbox(
        "Processing Mode",
        ["Standard", "Enhanced", "Comprehensive"],
        help="Select the level of document analysis"
    )
    
    # Main content area
    st.header("📄 Document Upload & Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document file",
        type=['pdf', 'txt', 'docx', 'zip'],
        help="Upload a document for processing"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show file details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col2:
            st.metric("File Type", uploaded_file.type or "Unknown")
        with col3:
            st.metric("Processing Mode", processing_mode)
        
        # Process button
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                # Mock processing
                result = mock_document_processing(uploaded_file.read(), uploaded_file.name)
                st.session_state.workflow_result = result
            
            st.success("✅ Document processing completed!")
    
    # Display results
    if st.session_state.workflow_result:
        result = st.session_state.workflow_result
        
        st.header("📊 Processing Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Processing Time", f"{result['processing_time']:.1f}s")
        with col2:
            st.metric("Key Topics", len(result['key_topics']))
        with col3:
            st.metric("Claims Found", len(result['claims']))
        with col4:
            st.metric("Evidence Items", len(result['evidence']))
        
        # Key topics
        st.subheader("🎯 Key Topics Identified")
        for topic in result['key_topics']:
            st.write(f"• {topic}")
        
        # Claims
        st.subheader("📋 Claims Extracted")
        for i, claim in enumerate(result['claims'], 1):
            st.write(f"{i}. {claim}")
        
        # Evidence
        st.subheader("🔍 Supporting Evidence")
        for i, evidence in enumerate(result['evidence'], 1):
            st.write(f"{i}. {evidence}")
        
        # Live context
        if include_live_context and result['live_context']['available']:
            st.subheader("🔗 Live Context Integration")
            
            live_ctx = result['live_context']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Live Events", live_ctx['events'])
            with col2:
                st.metric("Relevance Score", f"{live_ctx['relevance_score']:.2f}")
            with col3:
                st.metric("Context Status", "✅ Available")
            
            st.subheader("💡 Temporal Insights")
            for insight in live_ctx['temporal_insights']:
                st.write(f"• {insight}")
        
        # Raw results
        with st.expander("🔧 Technical Details"):
            st.json(result)
    
    # Demo information
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Demo Information")
    st.sidebar.info("""
    This is a simplified demo version that shows:
    
    • Document upload and processing
    • Key topic extraction
    • Claims and evidence identification
    • Live context integration
    • User preference management
    
    For full functionality, use the command-line demos.
    """)

if __name__ == "__main__":
    main()
