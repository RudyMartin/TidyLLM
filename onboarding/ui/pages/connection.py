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
    
    st.markdown('<div class="section-header">🚨 CRITICAL: System Configuration Required</div>', unsafe_allow_html=True)
    
    # Root Path Configuration Section
    st.markdown("### 📁 Root Path Configuration")
    st.info("""
    **Configure the base path for TidyLLM operations.**
    
    This is especially important for corporate environments with deep folder structures.
    The system will use this path to locate configuration files, data, and logs.
    """)
    
    # Get current root path from settings
    try:
        import yaml
        import os
        from pathlib import Path
        
        # Try to load current settings
        settings_path = Path(__file__).parent.parent.parent.parent / "tidyllm" / "admin" / "settings.yaml"
        current_root_path = ""
        
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f) or {}
            current_root_path = settings.get("system", {}).get("root_path", "")
        
        # Root path input
        col1, col2 = st.columns([3, 1])
        with col1:
            root_path = st.text_input(
                "Root Path", 
                value=current_root_path,
                placeholder="C:/Users/username/projects or /home/user/projects",
                help="Base directory for all TidyLLM operations"
            )
        
        with col2:
            if st.button("Auto-Detect", help="Automatically detect root path"):
                # Auto-detect logic
                current_dir = os.getcwd()
                if "tidyllm" in current_dir:
                    # Find parent directory containing tidyllm
                    parts = Path(current_dir).parts
                    for i, part in enumerate(parts):
                        if part == "tidyllm":
                            root_parts = parts[:i]
                            root_path = str(Path(*root_parts)) if root_parts else "."
                            st.rerun()
        
        # Update settings if changed
        if root_path and root_path != current_root_path:
            if st.button("Update Root Path", type="primary"):
                try:
                    # Load current settings
                    if settings_path.exists():
                        with open(settings_path, 'r') as f:
                            settings = yaml.safe_load(f) or {}
                    else:
                        settings = {}
                    
                    # Update system configuration
                    if "system" not in settings:
                        settings["system"] = {}
                    
                    settings["system"]["root_path"] = root_path
                    settings["system"]["config_folder"] = "tidyllm/admin"
                    settings["system"]["data_folder"] = "tidyllm/data"
                    settings["system"]["logs_folder"] = "tidyllm/logs"
                    
                    # Save settings
                    with open(settings_path, 'w') as f:
                        yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
                    
                    st.success(f"✅ Root path updated to: {root_path}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error updating root path: {e}")
        
        # Show current configuration
        if current_root_path:
            st.success(f"✅ Current root path: `{current_root_path}`")
        else:
            st.warning("⚠️ No root path configured")
        
        # Refresh settings button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Refresh Settings", help="Reload settings from YAML file"):
                try:
                    from tidyllm.infrastructure.settings_manager import refresh_settings
                    if refresh_settings():
                        st.success("✅ Settings refreshed from YAML file!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to refresh settings")
                except Exception as e:
                    st.error(f"❌ Error refreshing settings: {e}")
    
    except Exception as e:
        st.error(f"❌ Error loading configuration: {e}")
    
    st.markdown("---")
    
    st.markdown("### 🚨 AWS Connection Required")
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
