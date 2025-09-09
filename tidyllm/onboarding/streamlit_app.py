"""
TidyLLM Corporate Onboarding Wizard
==================================

Interactive Streamlit application for setting up TidyLLM in corporate environments.

This wizard guides IT administrators through:
1. Environment detection
2. Credential configuration  
3. AWS connectivity testing
4. Configuration file generation
5. Deployment validation

Run with: streamlit run tidyllm/onboarding/streamlit_app.py
"""

import streamlit as st
import yaml
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .session_validator import CorporateSessionManager, test_full_aws_stack
from .config_generator import create_template_config, validate_aws_setup

# Configure Streamlit page
st.set_page_config(
    page_title="TidyLLM Corporate Onboarding",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'config' not in st.session_state:
        st.session_state.config = {}
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = {}
    if 'corporate_session' not in st.session_state:
        st.session_state.corporate_session = CorporateSessionManager()

def show_header():
    """Display the application header."""
    st.title("🚀 TidyLLM Corporate Onboarding Wizard")
    st.markdown("""
    Welcome to the TidyLLM corporate deployment wizard. This tool will guide you through
    setting up TidyLLM in your corporate environment with proper security and compliance.
    """)
    
    # Progress indicator
    steps = ["Environment Detection", "AWS Credentials", "Service Validation", "Configuration", "Deployment"]
    cols = st.columns(len(steps))
    
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        with col:
            if i < st.session_state.step:
                st.success(f"✅ {step_name}")
            elif i == st.session_state.step:
                st.info(f"🔄 {step_name}")
            else:
                st.write(f"⏳ {step_name}")

def step_1_environment_detection():
    """Step 1: Detect corporate environment settings."""
    st.header("Step 1: Environment Detection")
    
    st.markdown("""
    First, let's detect your corporate environment settings. This helps us understand
    any special configuration needed for your network.
    """)
    
    if st.button("🔍 Detect Environment", type="primary"):
        with st.spinner("Detecting corporate environment..."):
            env_info = st.session_state.corporate_session.detect_corporate_environment()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Network Settings")
                if env_info.has_proxy:
                    st.warning("🌐 Corporate proxy detected")
                    with st.expander("Proxy Details"):
                        st.json(env_info.proxy_settings)
                else:
                    st.success("✅ No proxy configuration needed")
                
                if env_info.ca_bundle_path:
                    st.info(f"🔒 Custom CA bundle found: {env_info.ca_bundle_path}")
                else:
                    st.info("🔒 Using system CA certificates")
            
            with col2:
                st.subheader("AWS Environment")
                if env_info.iam_role_available:
                    st.success("✅ EC2 IAM role detected")
                    st.info("You can use IAM roles instead of access keys")
                else:
                    st.info("💻 Running on local/non-EC2 environment")
                
                if env_info.current_aws_profile:
                    st.info(f"📋 AWS Profile: {env_info.current_aws_profile}")
                else:
                    st.info("📋 No AWS profile configured")
            
            # Test network connectivity
            st.subheader("Network Connectivity Test")
            connectivity = st.session_state.corporate_session.test_network_connectivity()
            
            for service, is_connected in connectivity.items():
                if is_connected:
                    st.success(f"✅ {service.title()} service reachable")
                else:
                    st.error(f"❌ {service.title()} service unreachable")
            
            # Store results
            st.session_state.validation_results['environment'] = {
                'env_info': env_info,
                'connectivity': connectivity
            }
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Next: AWS Credentials →", type="primary"):
            st.session_state.step = 2
            st.rerun()

def step_2_aws_credentials():
    """Step 2: Configure AWS credentials."""
    st.header("Step 2: AWS Credentials Configuration")
    
    st.markdown("""
    Configure your AWS credentials for accessing Bedrock and S3. Choose the method
    that works best in your corporate environment.
    """)
    
    # Credential input methods
    auth_method = st.radio(
        "Choose authentication method:",
        ["Environment Variables", "Direct Input", "AWS Profile", "IAM Role (EC2 only)"],
        help="Select how you want to provide AWS credentials"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if auth_method == "Environment Variables":
            st.code("""
# Set these environment variables:
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_SESSION_TOKEN="your-session-token"  # Optional
export AWS_DEFAULT_REGION="us-east-1"
            """)
            
            # Check if env vars are set
            env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
            all_set = all(os.environ.get(var) for var in env_vars)
            
            if all_set:
                st.success("✅ AWS environment variables are configured")
            else:
                missing = [var for var in env_vars if not os.environ.get(var)]
                st.warning(f"⚠️ Missing environment variables: {', '.join(missing)}")
        
        elif auth_method == "Direct Input":
            st.warning("⚠️ Only use this for testing. Use environment variables in production.")
            
            access_key = st.text_input("AWS Access Key ID", type="password")
            secret_key = st.text_input("AWS Secret Access Key", type="password")
            session_token = st.text_input("AWS Session Token (optional)", type="password")
            
            if access_key and secret_key:
                st.session_state.config['aws_credentials'] = {
                    'access_key_id': access_key,
                    'secret_access_key': secret_key,
                    'session_token': session_token if session_token else None
                }
        
        elif auth_method == "AWS Profile":
            profile_name = st.text_input("AWS Profile Name", value="default")
            st.info(f"Using AWS profile: {profile_name}")
            st.session_state.config['aws_credentials'] = {'profile': profile_name}
        
        else:  # IAM Role
            st.info("Using IAM role attached to this EC2 instance")
            st.session_state.config['aws_credentials'] = {'use_iam_role': True}
    
    with col2:
        # Region and basic settings
        region = st.selectbox(
            "AWS Region",
            ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1"],
            help="Choose the AWS region for Bedrock and S3 services"
        )
        
        st.session_state.config['aws_region'] = region
        
        # Test credentials button
        if st.button("🧪 Test AWS Credentials", type="secondary"):
            with st.spinner("Testing AWS credentials..."):
                try:
                    # Get credentials based on method
                    creds = st.session_state.config.get('aws_credentials', {})
                    
                    session = st.session_state.corporate_session.create_corporate_session(
                        access_key_id=creds.get('access_key_id'),
                        secret_access_key=creds.get('secret_access_key'),
                        session_token=creds.get('session_token'),
                        region=region,
                        profile=creds.get('profile')
                    )
                    
                    st.success("✅ AWS credentials validated successfully!")
                    
                    # Show identity information
                    try:
                        sts = session.client('sts')
                        identity = sts.get_caller_identity()
                        
                        st.info(f"**Account:** {identity.get('Account')}")
                        st.info(f"**User/Role:** {identity.get('Arn')}")
                    except:
                        pass
                    
                except Exception as e:
                    st.error(f"❌ AWS credential validation failed: {str(e)}")
                    st.error("Please check your credentials and network connectivity.")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back: Environment", type="secondary"):
            st.session_state.step = 1
            st.rerun()
    with col3:
        if st.button("Next: Validation →", type="primary"):
            st.session_state.step = 3
            st.rerun()

def step_3_service_validation():
    """Step 3: Validate AWS services."""
    st.header("Step 3: Service Validation")
    
    st.markdown("""
    Now let's test connectivity to all required AWS services and your database.
    This comprehensive test will identify any configuration issues.
    """)
    
    # Service configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("S3 Configuration")
        s3_bucket = st.text_input(
            "S3 Bucket Name (optional)",
            help="Leave empty to test general S3 access, or specify your bucket"
        )
        
        st.subheader("PostgreSQL Configuration")
        use_postgres = st.checkbox("Test PostgreSQL Connection")
        
        if use_postgres:
            pg_host = st.text_input("PostgreSQL Host")
            pg_port = st.number_input("PostgreSQL Port", value=5432)
            pg_database = st.text_input("Database Name")
            pg_username = st.text_input("Username")
            pg_password = st.text_input("Password", type="password")
    
    with col2:
        st.subheader("Validation Results")
        
        if st.button("🚀 Run Full Validation", type="primary"):
            with st.spinner("Running comprehensive validation..."):
                
                # Prepare test parameters
                creds = st.session_state.config.get('aws_credentials', {})
                region = st.session_state.config.get('aws_region', 'us-east-1')
                
                postgres_config = None
                if use_postgres and pg_host and pg_database and pg_username and pg_password:
                    postgres_config = {
                        'host': pg_host,
                        'port': pg_port,
                        'database': pg_database,
                        'username': pg_username,
                        'password': pg_password
                    }
                
                # Run comprehensive test
                results = test_full_aws_stack(
                    access_key_id=creds.get('access_key_id'),
                    secret_access_key=creds.get('secret_access_key'),
                    session_token=creds.get('session_token'),
                    region=region,
                    s3_bucket=s3_bucket if s3_bucket else None,
                    postgres_config=postgres_config
                )
                
                st.session_state.validation_results['services'] = results
                
                # Display results
                if results['overall_success']:
                    st.success("🎉 All services validated successfully!")
                else:
                    st.error("❌ Some validation tests failed")
                
                # Show detailed results
                with st.expander("🔍 Detailed Results", expanded=True):
                    
                    # AWS Session
                    if results['aws_session']['success']:
                        st.success(f"✅ AWS Session: {results['aws_session']['message']}")
                    else:
                        st.error(f"❌ AWS Session: {results['aws_session'].get('error', 'Failed')}")
                    
                    # Service validation
                    for service, result in results['service_validation'].items():
                        if result['success']:
                            st.success(f"✅ {service.title()}: {result['message']}")
                        else:
                            st.error(f"❌ {service.title()}: {result['message']}")
                
                # Show recommendations
                if results.get('recommendations'):
                    st.subheader("📋 Recommendations")
                    for rec in results['recommendations']:
                        st.warning(f"💡 {rec}")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back: Credentials", type="secondary"):
            st.session_state.step = 2
            st.rerun()
    with col3:
        if st.button("Next: Configuration →", type="primary"):
            st.session_state.step = 4
            st.rerun()

def step_4_configuration():
    """Step 4: Generate configuration file."""
    st.header("Step 4: Configuration Generation")
    
    st.markdown("""
    Generate your customized TidyLLM configuration file based on the validation results.
    This file will be ready for deployment in your corporate environment.
    """)
    
    # Organization details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Organization Settings")
        org_name = st.text_input("Organization Name", value="Your Company")
        deployment_env = st.selectbox(
            "Deployment Environment",
            ["production", "staging", "development"],
            index=0
        )
        
        # Model selection
        st.subheader("Model Configuration")
        default_model = st.selectbox(
            "Default Model",
            ["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"],
            help="Choose the default model for your organization"
        )
        
        # Security settings
        st.subheader("Security Settings")
        encrypt_cache = st.checkbox("Encrypt cache data", value=True)
        audit_logging = st.checkbox("Enable audit logging", value=True)
        sso_enabled = st.checkbox("Single Sign-On (SSO) required", value=True)
    
    with col2:
        st.subheader("Configuration Preview")
        
        # Generate configuration preview
        if st.button("🔧 Generate Configuration", type="primary"):
            
            # Collect all configuration data
            config_data = {
                'organization': org_name,
                'environment': deployment_env,
                'aws_region': st.session_state.config.get('aws_region', 'us-east-1'),
                'default_model': default_model,
                'security': {
                    'encrypt_cache': encrypt_cache,
                    'audit_logging': audit_logging,
                    'sso_enabled': sso_enabled
                }
            }
            
            # Add validation results
            if 'services' in st.session_state.validation_results:
                config_data['validation_results'] = st.session_state.validation_results['services']
            
            st.session_state.config['final_config'] = config_data
            
            # Show preview
            st.code(yaml.dump(config_data, default_flow_style=False), language='yaml')
    
    # Download configuration
    if 'final_config' in st.session_state.config:
        st.subheader("📥 Download Configuration")
        
        # Generate final settings.yaml
        final_config = create_template_config(st.session_state.config['final_config'])
        config_yaml = yaml.dump(final_config, default_flow_style=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download settings.yaml",
                data=config_yaml,
                file_name=f"tidyllm-settings-{datetime.now().strftime('%Y%m%d')}.yaml",
                mime="application/x-yaml",
                type="primary"
            )
        
        with col2:
            # Generate environment variables script
            env_script = generate_env_script(st.session_state.config)
            st.download_button(
                label="🔧 Download environment setup script",
                data=env_script,
                file_name="setup-environment.sh",
                mime="text/plain"
            )
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back: Validation", type="secondary"):
            st.session_state.step = 3
            st.rerun()
    with col3:
        if st.button("Next: Deployment →", type="primary"):
            st.session_state.step = 5
            st.rerun()

def step_5_deployment():
    """Step 5: Deployment instructions."""
    st.header("Step 5: Deployment Instructions")
    
    st.markdown("""
    🎉 **Congratulations!** Your TidyLLM configuration is ready for corporate deployment.
    Follow these instructions to deploy TidyLLM in your environment.
    """)
    
    # Deployment checklist
    st.subheader("📋 Deployment Checklist")
    
    with st.expander("🔧 1. Environment Setup", expanded=True):
        st.markdown("""
        **Required Actions:**
        1. Download the configuration files from Step 4
        2. Place `settings.yaml` in your TidyLLM installation directory
        3. Run the environment setup script to configure environment variables
        4. Ensure all network connectivity requirements are met
        """)
    
    with st.expander("🔒 2. Security Configuration"):
        st.markdown("""
        **Security Checklist:**
        - [ ] SSL/TLS certificates installed
        - [ ] Corporate CA certificates configured  
        - [ ] KMS encryption keys configured
        - [ ] Audit logging destination configured
        - [ ] SSO integration configured
        """)
    
    with st.expander("📊 3. Monitoring Setup"):
        st.markdown("""
        **Monitoring Requirements:**
        - [ ] Log aggregation configured
        - [ ] Health check endpoints monitored
        - [ ] Alert channels configured
        - [ ] Performance metrics collection enabled
        """)
    
    # Test deployment
    st.subheader("🧪 Test Your Deployment")
    
    if st.button("🚀 Test Deployment", type="primary"):
        st.code("""
# Test your TidyLLM deployment:

# 1. Test basic functionality
python -c "import tidyllm; print('TidyLLM import successful')"

# 2. Test AWS connectivity  
python -c "from tidyllm.onboarding import validate_corporate_environment; print(validate_corporate_environment())"

# 3. Run critical tests
python tests/run_critical_tests.py

# 4. Test API endpoints
curl http://localhost:8000/health
        """)
    
    # Support information
    st.subheader("💬 Support & Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Documentation:**
        - [TidyLLM Corporate Setup Guide](https://docs.tidyllm.com/corporate)
        - [Troubleshooting Guide](https://docs.tidyllm.com/troubleshooting)
        - [API Reference](https://docs.tidyllm.com/api)
        """)
    
    with col2:
        st.markdown("""
        **Support Channels:**
        - 📧 Email: support@tidyllm.com
        - 💬 Slack: #tidyllm-support
        - 🎫 Tickets: support.tidyllm.com
        """)
    
    # Final success message
    st.success("""
    🎉 **Setup Complete!** 
    
    Your TidyLLM corporate environment is configured and ready for deployment.
    All validation tests have passed and your configuration files are ready.
    """)
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back: Configuration", type="secondary"):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("🔄 Start Over", type="secondary"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def generate_env_script(config: Dict[str, Any]) -> str:
    """Generate environment setup script."""
    script = """#!/bin/bash
# TidyLLM Corporate Environment Setup Script
# Generated by TidyLLM Onboarding Wizard

echo "Setting up TidyLLM corporate environment..."

# AWS Configuration
export AWS_DEFAULT_REGION="{region}"

# Database Configuration (set your actual password)
export TIDYLLM_DB_PASSWORD="YOUR_ACTUAL_DB_PASSWORD"

# Optional: AWS credentials (if not using IAM roles)
# export AWS_ACCESS_KEY_ID="your-access-key"
# export AWS_SECRET_ACCESS_KEY="your-secret-key"
# export AWS_SESSION_TOKEN="your-session-token"

echo "Environment variables configured successfully!"
echo "Please update the placeholder values with your actual credentials."
""".format(
        region=config.get('aws_region', 'us-east-1')
    )
    
    return script

def main():
    """Main application function."""
    init_session_state()
    show_header()
    
    # Route to appropriate step
    if st.session_state.step == 1:
        step_1_environment_detection()
    elif st.session_state.step == 2:
        step_2_aws_credentials()
    elif st.session_state.step == 3:
        step_3_service_validation()
    elif st.session_state.step == 4:
        step_4_configuration()
    elif st.session_state.step == 5:
        step_5_deployment()
    
    # Sidebar with current status
    with st.sidebar:
        st.header("🔍 Current Status")
        
        if 'validation_results' in st.session_state:
            if 'environment' in st.session_state.validation_results:
                st.success("✅ Environment detected")
            
            if 'services' in st.session_state.validation_results:
                results = st.session_state.validation_results['services']
                if results.get('overall_success'):
                    st.success("✅ All services validated")
                else:
                    st.error("❌ Some validation failures")
        
        st.header("📚 Quick Help")
        st.markdown("""
        **Common Issues:**
        - Network connectivity problems
        - AWS credential configuration
        - Corporate proxy settings
        - SSL certificate issues
        """)

def run_onboarding_wizard():
    """Public function to run the onboarding wizard."""
    main()

if __name__ == "__main__":
    main()