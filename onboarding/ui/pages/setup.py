"""
TidyLLM Onboarding Setup Page
============================

Complete system setup and service initialization page.
"""

import streamlit as st
import yaml
import os
from pathlib import Path
from typing import Dict, Any

def render_setup_page():
    """Render the complete system setup page."""
    
    st.markdown('<div class="section-header">🚀 TidyLLM System Setup</div>', unsafe_allow_html=True)
    
    st.info("""
    **Complete system configuration and service initialization.**
    
    This page will guide you through setting up all required services:
    - AWS credentials and services (S3, Bedrock)
    - Database configuration
    - Workflow settings
    - Service initialization and testing
    """)
    
    # Get settings path
    settings_path = Path(__file__).parent.parent.parent.parent / "tidyllm" / "admin" / "settings.yaml"
    
    # Load current settings
    current_settings = {}
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                current_settings = yaml.safe_load(f) or {}
        except Exception as e:
            st.error(f"❌ Error loading settings: {e}")
    
    # Create tabs for different configuration sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔑 AWS Setup", 
        "🗄️ Database Setup", 
        "⚙️ Workflow Setup", 
        "🧪 Service Testing", 
        "🚀 Initialize System"
    ])
    
    with tab1:
        render_aws_setup_tab(current_settings, settings_path)
    
    with tab2:
        render_database_setup_tab(current_settings, settings_path)
    
    with tab3:
        render_workflow_setup_tab(current_settings, settings_path)
    
    with tab4:
        render_service_testing_tab()
    
    with tab5:
        render_system_initialization_tab()

def render_aws_setup_tab(current_settings: Dict[str, Any], settings_path: Path):
    """Render AWS configuration tab."""
    st.markdown("### 🔑 AWS Services Configuration")
    
    st.info("""
    **Configure AWS credentials and services.**
    
    Required for: S3 storage, Bedrock AI models, STS authentication
    """)
    
    # Get current AWS config
    aws_config = current_settings.get("aws", {})
    api_keys = current_settings.get("api_keys", {})
    
    # AWS credentials
    col1, col2 = st.columns(2)
    
    with col1:
        aws_access_key = st.text_input(
            "AWS Access Key ID",
            value=aws_config.get("access_key_id", api_keys.get("aws_access_key_id", "")),
            type="password",
            help="Your AWS access key ID"
        )
        
        aws_region = st.selectbox(
            "AWS Region",
            options=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
            index=0 if aws_config.get("region", "us-east-1") == "us-east-1" else 1,
            help="AWS region for your services"
        )
    
    with col2:
        aws_secret_key = st.text_input(
            "AWS Secret Access Key",
            value=aws_config.get("secret_access_key", api_keys.get("aws_secret_access_key", "")),
            type="password",
            help="Your AWS secret access key"
        )
        
        # Show current region
        if aws_config.get("region"):
            st.info(f"Current region: `{aws_config.get('region')}`")
    
    # S3 Configuration
    st.markdown("#### 📦 S3 Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        s3_bucket = st.text_input(
            "S3 Default Bucket",
            value=current_settings.get("s3", {}).get("default_bucket", ""),
            help="Default S3 bucket for document storage"
        )
    
    with col2:
        s3_prefix = st.text_input(
            "S3 Default Prefix",
            value=current_settings.get("s3", {}).get("default_prefix", "tidyllm/"),
            help="Default S3 prefix for organized storage"
        )
    
    # Bedrock Configuration
    st.markdown("#### 🤖 Bedrock Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        bedrock_model = st.selectbox(
            "Default Bedrock Model",
            options=[
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "anthropic.claude-3-opus-20240229-v1:0"
            ],
            index=0,
            help="Default model for AI processing"
        )
    
    with col2:
        st.info("Bedrock models are region-specific. Ensure your region supports the selected model.")
    
    # Save AWS configuration
    if st.button("💾 Save AWS Configuration", type="primary"):
        if aws_access_key and aws_secret_key:
            try:
                # Update settings
                if "aws" not in current_settings:
                    current_settings["aws"] = {}
                if "api_keys" not in current_settings:
                    current_settings["api_keys"] = {}
                if "s3" not in current_settings:
                    current_settings["s3"] = {}
                if "bedrock" not in current_settings:
                    current_settings["bedrock"] = {}
                
                # AWS credentials
                current_settings["aws"]["access_key_id"] = aws_access_key
                current_settings["aws"]["secret_access_key"] = aws_secret_key
                current_settings["aws"]["region"] = aws_region
                current_settings["aws"]["default_region"] = aws_region
                
                # API keys (for compatibility)
                current_settings["api_keys"]["aws_access_key_id"] = aws_access_key
                current_settings["api_keys"]["aws_secret_access_key"] = aws_secret_key
                
                # S3 configuration
                current_settings["s3"]["default_bucket"] = s3_bucket
                current_settings["s3"]["default_prefix"] = s3_prefix
                
                # Bedrock configuration
                current_settings["bedrock"]["default_model"] = bedrock_model
                current_settings["bedrock"]["region"] = aws_region
                
                # Save to file
                with open(settings_path, 'w') as f:
                    yaml.dump(current_settings, f, default_flow_style=False, sort_keys=False)
                
                st.success("✅ AWS configuration saved successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error saving AWS configuration: {e}")
        else:
            st.warning("⚠️ Please enter both AWS Access Key ID and Secret Access Key")

def render_database_setup_tab(current_settings: Dict[str, Any], settings_path: Path):
    """Render database configuration tab."""
    st.markdown("### 🗄️ Database Configuration")
    
    st.info("""
    **Configure PostgreSQL database connection.**
    
    Required for: Vector storage, knowledge management, workflow persistence
    """)
    
    # Get current database config
    postgres_config = current_settings.get("postgres", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        db_host = st.text_input(
            "Database Host",
            value=postgres_config.get("host", ""),
            help="PostgreSQL server hostname or IP"
        )
        
        db_port = st.number_input(
            "Database Port",
            value=postgres_config.get("port", 5432),
            min_value=1,
            max_value=65535,
            help="PostgreSQL server port"
        )
        
        db_name = st.text_input(
            "Database Name",
            value=postgres_config.get("database", "tidyllm"),
            help="Database name"
        )
    
    with col2:
        db_username = st.text_input(
            "Database Username",
            value=postgres_config.get("username", ""),
            help="Database username"
        )
        
        db_password = st.text_input(
            "Database Password",
            value=postgres_config.get("password", ""),
            type="password",
            help="Database password"
        )
        
        # Connection test
        if st.button("🧪 Test Database Connection"):
            with st.spinner("Testing database connection..."):
                try:
                    import psycopg2
                    conn = psycopg2.connect(
                        host=db_host,
                        port=db_port,
                        database=db_name,
                        user=db_username,
                        password=db_password
                    )
                    conn.close()
                    st.success("✅ Database connection successful!")
                except Exception as e:
                    st.error(f"❌ Database connection failed: {e}")
    
    # Save database configuration
    if st.button("💾 Save Database Configuration", type="primary"):
        if db_host and db_username and db_password:
            try:
                # Update settings
                if "postgres" not in current_settings:
                    current_settings["postgres"] = {}
                
                current_settings["postgres"]["host"] = db_host
                current_settings["postgres"]["port"] = db_port
                current_settings["postgres"]["database"] = db_name
                current_settings["postgres"]["username"] = db_username
                current_settings["postgres"]["password"] = db_password
                
                # Save to file
                with open(settings_path, 'w') as f:
                    yaml.dump(current_settings, f, default_flow_style=False, sort_keys=False)
                
                st.success("✅ Database configuration saved successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error saving database configuration: {e}")
        else:
            st.warning("⚠️ Please enter database host, username, and password")

def render_workflow_setup_tab(current_settings: Dict[str, Any], settings_path: Path):
    """Render workflow configuration tab."""
    st.markdown("### ⚙️ Workflow Configuration")
    
    st.info("""
    **Configure workflow optimization settings.**
    
    Required for: Workflow analysis, optimization, and management
    """)
    
    # Get current workflow config
    workflow_config = current_settings.get("workflow_optimizer", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        workflow_enabled = st.checkbox(
            "Enable Workflow Optimizer",
            value=workflow_config.get("enabled", True),
            help="Enable workflow optimization features"
        )
        
        analysis_mode = st.selectbox(
            "Analysis Mode",
            options=["full", "analysis_only", "monitoring_only"],
            index=1 if workflow_config.get("analysis_mode", "analysis_only") == "analysis_only" else 0,
            help="Workflow analysis mode"
        )
    
    with col2:
        max_workers = st.number_input(
            "Maximum Workers",
            value=workflow_config.get("max_workers", 4),
            min_value=1,
            max_value=16,
            help="Maximum number of workflow workers"
        )
        
        cache_size = st.number_input(
            "Cache Size (MB)",
            value=workflow_config.get("cache_size_mb", 100),
            min_value=10,
            max_value=1000,
            help="Workflow cache size in megabytes"
        )
    
    # Advanced settings
    with st.expander("🔧 Advanced Workflow Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            timeout_seconds = st.number_input(
                "Timeout (seconds)",
                value=workflow_config.get("timeout_seconds", 300),
                min_value=30,
                max_value=3600,
                help="Workflow operation timeout"
            )
        
        with col2:
            retry_attempts = st.number_input(
                "Retry Attempts",
                value=workflow_config.get("retry_attempts", 3),
                min_value=0,
                max_value=10,
                help="Number of retry attempts for failed operations"
            )
    
    # Save workflow configuration
    if st.button("💾 Save Workflow Configuration", type="primary"):
        try:
            # Update settings
            if "workflow_optimizer" not in current_settings:
                current_settings["workflow_optimizer"] = {}
            
            current_settings["workflow_optimizer"]["enabled"] = workflow_enabled
            current_settings["workflow_optimizer"]["analysis_mode"] = analysis_mode
            current_settings["workflow_optimizer"]["max_workers"] = max_workers
            current_settings["workflow_optimizer"]["cache_size_mb"] = cache_size
            current_settings["workflow_optimizer"]["timeout_seconds"] = timeout_seconds
            current_settings["workflow_optimizer"]["retry_attempts"] = retry_attempts
            
            # Save to file
            with open(settings_path, 'w') as f:
                yaml.dump(current_settings, f, default_flow_style=False, sort_keys=False)
            
            st.success("✅ Workflow configuration saved successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error saving workflow configuration: {e}")

def render_service_testing_tab():
    """Render service testing tab."""
    st.markdown("### 🧪 Service Testing")
    
    st.info("""
    **Test all configured services and connections.**
    
    Verify that all services are properly configured and accessible.
    """)
    
    # Test buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔑 Test AWS Services", use_container_width=True):
            with st.spinner("Testing AWS services..."):
                try:
                    from core.validator import ConnectionValidator
                    validator = ConnectionValidator()
                    aws_results = validator.validate_aws_connectivity()
                    st.session_state.aws_test_results = aws_results
                except Exception as e:
                    st.error(f"❌ AWS test failed: {e}")
        
        if st.button("🗄️ Test Database", use_container_width=True):
            with st.spinner("Testing database..."):
                try:
                    from core.validator import ConnectionValidator
                    validator = ConnectionValidator()
                    db_results = validator.validate_database_connectivity()
                    st.session_state.db_test_results = db_results
                except Exception as e:
                    st.error(f"❌ Database test failed: {e}")
    
    with col2:
        if st.button("⚙️ Test Workflow Services", use_container_width=True):
            with st.spinner("Testing workflow services..."):
                try:
                    from core.validator import ConnectionValidator
                    validator = ConnectionValidator()
                    workflow_results = validator.validate_workflow_services()
                    st.session_state.workflow_test_results = workflow_results
                except Exception as e:
                    st.error(f"❌ Workflow test failed: {e}")
        
        if st.button("🧪 Test All Services", use_container_width=True):
            with st.spinner("Testing all services..."):
                try:
                    from core.validator import ConnectionValidator
                    validator = ConnectionValidator()
                    all_results = validator.validate_all_services()
                    st.session_state.all_test_results = all_results
                except Exception as e:
                    st.error(f"❌ All services test failed: {e}")
    
    # Display test results
    if hasattr(st.session_state, 'aws_test_results'):
        display_test_results("AWS Services", st.session_state.aws_test_results)
    
    if hasattr(st.session_state, 'db_test_results'):
        display_test_results("Database", st.session_state.db_test_results)
    
    if hasattr(st.session_state, 'workflow_test_results'):
        display_test_results("Workflow Services", st.session_state.workflow_test_results)
    
    if hasattr(st.session_state, 'all_test_results'):
        display_test_results("All Services", st.session_state.all_test_results)

def render_system_initialization_tab():
    """Render system initialization tab."""
    st.markdown("### 🚀 System Initialization")
    
    st.info("""
    **Initialize and start all TidyLLM services.**
    
    This will start all configured services and make them available for use.
    """)
    
    # Service status
    st.markdown("#### 📊 Service Status")
    
    try:
        from core.session_manager import SessionManager
        
        # Get session manager
        session_manager = SessionManager.get_instance()
        if session_manager:
            st.success("✅ Session Manager: Active")
        else:
            st.error("❌ Session Manager: Inactive")
        
        # Get gateways
        gateways = SessionManager.get_gateways()
        if gateways:
            st.success(f"✅ Gateways: {len(gateways)} active")
            for name, gateway in gateways.items():
                if gateway:
                    st.success(f"  ✅ {name}")
                else:
                    st.error(f"  ❌ {name}")
        else:
            st.error("❌ Gateways: None active")
        
        # Get services
        services = SessionManager.get_services()
        if services:
            st.success(f"✅ Services: {len(services)} active")
            for name, service in services.items():
                if service:
                    st.success(f"  ✅ {name}")
                else:
                    st.error(f"  ❌ {name}")
        else:
            st.error("❌ Services: None active")
    
    except Exception as e:
        st.error(f"❌ Error checking service status: {e}")
    
    # Initialization buttons
    st.markdown("#### 🚀 Initialize Services")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Services", use_container_width=True):
            with st.spinner("Refreshing services..."):
                try:
                    from tidyllm.infrastructure.settings_manager import refresh_settings
                    if refresh_settings():
                        st.success("✅ Settings refreshed!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to refresh settings")
                except Exception as e:
                    st.error(f"❌ Error refreshing services: {e}")
    
    with col2:
        if st.button("🚀 Initialize All", use_container_width=True):
            with st.spinner("Initializing all services..."):
                try:
                    # Force reinitialize session manager
                    from core.session_manager import SessionManager
                    SessionManager._instance = None  # Reset singleton
                    session_manager = SessionManager.get_instance()
                    
                    if session_manager:
                        st.success("✅ All services initialized successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to initialize services")
                except Exception as e:
                    st.error(f"❌ Error initializing services: {e}")
    
    with col3:
        if st.button("🧪 Test System", use_container_width=True):
            with st.spinner("Testing complete system..."):
                try:
                    from core.validator import ConnectionValidator
                    validator = ConnectionValidator()
                    system_results = validator.validate_complete_system()
                    st.session_state.system_test_results = system_results
                except Exception as e:
                    st.error(f"❌ System test failed: {e}")
    
    # Display system test results
    if hasattr(st.session_state, 'system_test_results'):
        display_test_results("Complete System", st.session_state.system_test_results)

def display_test_results(title: str, results: Dict[str, Any]):
    """Display test results in a formatted way."""
    st.markdown(f"#### 📊 {title} Test Results")
    
    if isinstance(results, dict):
        for service, result in results.items():
            if isinstance(result, dict):
                status = result.get("status", "unknown")
                message = result.get("message", "No message")
                
                if status == "success":
                    st.success(f"✅ {service}: {message}")
                elif status == "corporate_restricted":
                    st.warning(f"⚠️ {service}: {message}")
                elif status == "permission_error":
                    st.error(f"❌ {service}: {message}")
                else:
                    st.error(f"❌ {service}: {message}")
            else:
                st.info(f"ℹ️ {service}: {result}")
    else:
        st.info(f"ℹ️ {title}: {results}")
