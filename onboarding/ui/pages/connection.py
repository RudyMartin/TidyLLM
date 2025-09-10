"""
TidyLLM Onboarding Connection Config Page
=========================================

Connection configuration and validation page.
"""

import streamlit as st
import time
from core.validator import ConnectionValidator

def render_connection_page():
    """Render the connection configuration page."""
    
    st.markdown('<div class="section-header">🚨 CRITICAL: AWS Connection Required</div>', unsafe_allow_html=True)
    
    st.error("""
    **⚠️ NOTHING WORKS WITHOUT AWS CONNECTION**
    
    All TidyLLM gateways and features require AWS connectivity:
    - **S3** - Document storage and retrieval
    - **Bedrock** - AI model access and processing  
    - **STS** - Security token service
    - **PostgreSQL** - Vector database
    
    **Configure connections below to enable the entire system.**
    """)
    
    # Connection validator
    validator = ConnectionValidator()
    
    # Test buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Test AWS Connectivity", use_container_width=True):
            with st.spinner("Testing AWS connectivity..."):
                aws_results = validator.validate_aws_connectivity()
                st.session_state.aws_results = aws_results
    
    with col2:
        if st.button("🗄️ Test Database", use_container_width=True):
            with st.spinner("Testing database connectivity..."):
                db_results = validator.validate_database_connectivity()
                st.session_state.db_results = db_results
    
    with col3:
        if st.button("🚪 Test Gateways", use_container_width=True):
            with st.spinner("Testing gateways..."):
                gateway_results = validator.validate_gateways()
                st.session_state.gateway_results = gateway_results
    
    # Run all tests button
    if st.button("🔄 Run All Tests", use_container_width=True):
        with st.spinner("Running comprehensive tests..."):
            all_results = validator.run_full_validation()
            st.session_state.all_results = all_results
    
    # Display results
    if hasattr(st.session_state, 'aws_results'):
        st.markdown("### AWS Connectivity Results")
        display_aws_results(st.session_state.aws_results)
    
    if hasattr(st.session_state, 'db_results'):
        st.markdown("### Database Connectivity Results")
        display_db_results(st.session_state.db_results)
    
    if hasattr(st.session_state, 'gateway_results'):
        st.markdown("### Gateway Results")
        display_gateway_results(st.session_state.gateway_results)
    
    if hasattr(st.session_state, 'all_results'):
        st.markdown("### Complete Test Results")
        display_all_results(st.session_state.all_results)

def display_aws_results(results):
    """Display AWS connectivity results."""
    for service, result in results.items():
        if service == 'error':
            st.error(f"Error: {result}")
            continue
            
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if result['status'] == 'success':
                st.success(f"✅ {service.upper()}: {result['message']}")
            else:
                st.error(f"❌ {service.upper()}: {result['message']}")
        
        with col2:
            st.metric("Latency", f"{result['latency']:.1f}ms")
        
        with col3:
            st.metric("Status", result['status'].title())

def display_db_results(results):
    """Display database connectivity results."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if results['status'] == 'success':
            st.success(f"✅ PostgreSQL: {results['message']}")
        else:
            st.error(f"❌ PostgreSQL: {results['message']}")
    
    with col2:
        st.metric("Latency", f"{results['latency']:.1f}ms")
    
    with col3:
        st.metric("Status", results['status'].title())

def display_gateway_results(results):
    """Display gateway results."""
    for gateway, result in results.items():
        if gateway == 'error':
            st.error(f"Error: {result}")
            continue
            
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if result['status'] == 'success':
                st.success(f"✅ {gateway}: {result['message']}")
            else:
                st.error(f"❌ {gateway}: {result['message']}")
        
        with col2:
            st.metric("Latency", f"{result['latency']:.1f}ms")
        
        with col3:
            status_color = "🟢" if result['status'] == 'success' else "🔴"
            st.metric("Status", f"{status_color} {result['status'].title()}")

def display_all_results(results):
    """Display complete test results."""
    st.json(results)
