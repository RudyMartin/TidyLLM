"""
TidyLLM Corporate Onboarding - Streamlit Web Interface
=====================================================

Web-based GUI for configuring TidyLLM in corporate environments.
Provides a user-friendly interface for IT administrators to set up
and validate TidyLLM configurations.

Usage:
    streamlit run streamlit_app.py
    
Or via CLI:
    python -m streamlit run streamlit_app.py
"""

import streamlit as st
import yaml
import os
import io
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

# Import our configuration generator
try:
    from config_generator import (
        create_template_config, 
        generate_dockerfile,
        generate_docker_compose,
        generate_kubernetes_manifests
    )
    from session_validator import test_full_aws_stack
except ImportError:
    st.error("Required modules not found. Ensure config_generator.py and session_validator.py are in the same directory.")
    st.stop()

# Configure Streamlit page
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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'step': 1,
        'config': {},
        'validation_results': {},
        'generated_config': None,
        'org_info': {},
        'security_config': {},
        'model_config': {},
        'env_validated': False,
        'aws_validated': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🏢 TidyLLM Corporate Onboarding</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p>Configure TidyLLM for your corporate environment with enterprise-grade security and compliance.</p>
    </div>
    """, unsafe_allow_html=True)

def render_progress_sidebar():
    """Render progress sidebar."""
    st.sidebar.markdown("## 📋 Setup Progress")
    
    steps = [
        ("Organization Info", 1),
        ("Security Settings", 2), 
        ("AI Model Config", 3),
        ("Environment Validation", 4),
        ("Generate Config", 5),
        ("Download Artifacts", 6)
    ]
    
    for step_name, step_num in steps:
        if step_num < st.session_state.step:
            st.sidebar.markdown(f"✅ {step_name}")
        elif step_num == st.session_state.step:
            st.sidebar.markdown(f"🔄 **{step_name}**")
        else:
            st.sidebar.markdown(f"⏳ {step_name}")
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.markdown("## 🚀 Quick Actions")
    if st.sidebar.button("🔄 Reset Configuration"):
        for key in ['config', 'generated_config', 'validation_results', 'org_info', 'security_config', 'model_config']:
            st.session_state[key] = {}
        st.session_state.step = 1
        st.rerun()
    
    if st.sidebar.button("📤 Load Template"):
        st.session_state.step = 5  # Skip to config generation
        st.rerun()

def render_step_1_organization():
    """Step 1: Organization Information."""
    st.markdown('<h2 class="step-header">Step 1: Organization Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        org_name = st.text_input(
            "Organization Name",
            value=st.session_state.org_info.get('organization', ''),
            placeholder="e.g., Acme Corporation",
            help="Your company or organization name"
        )
        
        environment = st.selectbox(
            "Deployment Environment",
            options=["production", "staging", "development", "corporate"],
            index=["production", "staging", "development", "corporate"].index(
                st.session_state.org_info.get('environment', 'production')
            ),
            help="Choose the deployment environment type"
        )
    
    with col2:
        aws_region = st.selectbox(
            "AWS Region",
            options=[
                "us-east-1", "us-west-2", "us-west-1", 
                "eu-west-1", "eu-west-2", "eu-central-1",
                "ap-southeast-1", "ap-southeast-2", "ap-northeast-1"
            ],
            index=0 if st.session_state.org_info.get('aws_region') is None else 
                  ["us-east-1", "us-west-2", "us-west-1", "eu-west-1", "eu-west-2", 
                   "eu-central-1", "ap-southeast-1", "ap-southeast-2", "ap-northeast-1"].index(
                      st.session_state.org_info.get('aws_region', 'us-east-1')
                  ),
            help="AWS region where your resources are located"
        )
        
        deployment_type = st.selectbox(
            "Deployment Type",
            options=["docker-compose", "kubernetes", "standalone"],
            help="How will you deploy TidyLLM?"
        )
    
    # Preview panel
    st.markdown("### 📋 Configuration Preview")
    preview_data = {
        "Organization": org_name,
        "Environment": environment,
        "AWS Region": aws_region,
        "Deployment": deployment_type
    }
    
    df = pd.DataFrame(list(preview_data.items()), columns=["Setting", "Value"])
    st.dataframe(df, hide_index=True)
    
    if st.button("Continue to Security Settings ➡️", type="primary"):
        st.session_state.org_info = {
            'organization': org_name,
            'environment': environment,
            'aws_region': aws_region,
            'deployment_type': deployment_type
        }
        st.session_state.step = 2
        st.rerun()

def render_step_2_security():
    """Step 2: Security Configuration."""
    st.markdown('<h2 class="step-header">Step 2: Security & Compliance</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Authentication")
        enable_sso = st.checkbox(
            "Enable Single Sign-On (SSO)",
            value=st.session_state.security_config.get('sso_enabled', True),
            help="Recommended for corporate environments"
        )
        
        if enable_sso:
            sso_provider = st.selectbox(
                "SSO Provider",
                options=["okta", "azure-ad", "ping", "saml", "other"],
                help="Select your organization's SSO provider"
            )
        else:
            sso_provider = None
            
        auth_method = st.selectbox(
            "Authentication Method",
            options=["oauth2", "ldap", "basic", "api-key"],
            help="Primary authentication method"
        )
    
    with col2:
        st.markdown("#### Data Protection")
        enable_encryption = st.checkbox(
            "Enable Data Encryption at Rest",
            value=st.session_state.security_config.get('encrypt_cache', True),
            help="Encrypt cached data and logs (required for corporate)"
        )
        
        enable_audit = st.checkbox(
            "Enable Audit Logging",
            value=st.session_state.security_config.get('audit_logging', True),
            help="Log all API requests for compliance"
        )
        
        mask_sensitive = st.checkbox(
            "Mask Sensitive Data in Logs",
            value=True,
            help="Automatically mask sensitive information"
        )
    
    st.markdown("#### Rate Limiting")
    col3, col4 = st.columns(2)
    with col3:
        requests_per_minute = st.number_input(
            "Requests per Minute",
            min_value=1, max_value=1000,
            value=30,
            help="Rate limit per user per minute"
        )
    with col4:
        requests_per_hour = st.number_input(
            "Requests per Hour", 
            min_value=10, max_value=10000,
            value=500,
            help="Rate limit per user per hour"
        )
    
    # Security compliance indicators
    st.markdown("### 🛡️ Security Compliance Status")
    compliance_checks = {
        "Data Encryption": enable_encryption,
        "Audit Logging": enable_audit,
        "Access Control": enable_sso or auth_method != "basic",
        "Rate Limiting": requests_per_minute <= 100,
        "Data Masking": mask_sensitive
    }
    
    cols = st.columns(5)
    for i, (check, status) in enumerate(compliance_checks.items()):
        with cols[i]:
            status_icon = "✅" if status else "⚠️"
            st.metric(check, status_icon)
    
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("⬅️ Back to Organization"):
            st.session_state.step = 1
            st.rerun()
    
    with col_next:
        if st.button("Continue to Model Config ➡️", type="primary"):
            st.session_state.security_config = {
                'sso_enabled': enable_sso,
                'sso_provider': sso_provider,
                'auth_method': auth_method,
                'encrypt_cache': enable_encryption,
                'audit_logging': enable_audit,
                'mask_sensitive_data': mask_sensitive,
                'rate_limit_per_minute': requests_per_minute,
                'rate_limit_per_hour': requests_per_hour
            }
            st.session_state.step = 3
            st.rerun()

def render_step_3_models():
    """Step 3: AI Model Configuration."""
    st.markdown('<h2 class="step-header">Step 3: AI Model Configuration</h2>', unsafe_allow_html=True)
    
    # Model selection
    st.markdown("#### Available Models")
    models = {
        "claude-3-sonnet": {
            "name": "Claude 3 Sonnet",
            "description": "Balanced performance and speed",
            "use_case": "General purpose, balanced cost/performance",
            "cost": "Medium",
            "speed": "Fast"
        },
        "claude-3-haiku": {
            "name": "Claude 3 Haiku", 
            "description": "Fastest responses, lower cost",
            "use_case": "Quick queries, high volume usage",
            "cost": "Low",
            "speed": "Very Fast"
        },
        "claude-3-opus": {
            "name": "Claude 3 Opus",
            "description": "Highest quality responses",
            "use_case": "Complex analysis, critical tasks",
            "cost": "High", 
            "speed": "Slower"
        }
    }
    
    # Create model comparison table
    model_data = []
    for model_id, info in models.items():
        model_data.append([
            info["name"],
            info["description"], 
            info["use_case"],
            info["cost"],
            info["speed"]
        ])
    
    df = pd.DataFrame(model_data, columns=["Model", "Description", "Best For", "Cost", "Speed"])
    st.dataframe(df, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_model = st.selectbox(
            "Default Model",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            help="Primary model for most requests"
        )
        
        enable_fallback = st.checkbox(
            "Enable Model Fallback",
            value=True,
            help="Automatically fallback to alternative models if primary fails"
        )
    
    with col2:
        max_tokens = st.number_input(
            "Max Tokens per Request",
            min_value=100, max_value=8000,
            value=2048,
            help="Maximum tokens per API request"
        )
        
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0, max_value=1.0,
            value=0.3,
            step=0.1,
            help="Higher values = more creative responses"
        )
    
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("⬅️ Back to Security"):
            st.session_state.step = 2
            st.rerun()
    
    with col_next:
        if st.button("Continue to Validation ➡️", type="primary"):
            st.session_state.model_config = {
                'default_model': default_model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'enable_fallback': enable_fallback,
                'model_aliases': {
                    'corporate-llm': default_model,
                    'quick-assist': 'claude-3-haiku',
                    'premium-llm': 'claude-3-opus',
                    'analyst-llm': 'claude-3-sonnet'
                }
            }
            st.session_state.step = 4
            st.rerun()

def render_step_4_validation():
    """Step 4: Environment Validation."""
    st.markdown('<h2 class="step-header">Step 4: Environment Validation</h2>', unsafe_allow_html=True)
    
    # Environment variables check
    st.markdown("#### Environment Variables")
    
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "TIDYLLM_DB_PASSWORD"]
    env_data = []
    for var in required_vars:
        present = bool(os.getenv(var))
        status = "✅ Set" if present else "❌ Missing"
        env_data.append([var, status])
    
    env_df = pd.DataFrame(env_data, columns=["Variable", "Status"])
    st.dataframe(env_df, hide_index=True)
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        st.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        st.info("Set these variables before deploying to production.")
    else:
        st.success("All required environment variables are set!")
    
    # AWS connectivity test
    st.markdown("#### AWS Connectivity Test")
    
    if st.button("🔍 Test AWS Connection", type="secondary"):
        with st.spinner("Testing AWS connectivity..."):
            aws_region = st.session_state.org_info.get('aws_region', 'us-east-1')
            
            try:
                aws_validation = test_full_aws_stack(region=aws_region)
                
                st.session_state.validation_results = {
                    'environment_variables': {var: bool(os.getenv(var)) for var in required_vars},
                    'aws_connectivity': aws_validation
                }
                st.session_state.aws_validated = True
                
                if aws_validation.get('overall_success', False):
                    st.success("AWS connectivity test passed!")
                else:
                    st.error("AWS connectivity test failed")
                    for rec in aws_validation.get('recommendations', []):
                        st.warning(f"- {rec}")
                        
            except Exception as e:
                st.error(f"AWS connectivity test failed: {e}")
    
    # Network requirements
    st.markdown("#### Network Requirements")
    aws_region = st.session_state.org_info.get('aws_region', 'us-east-1')
    
    network_reqs = [
        f"bedrock-runtime.{aws_region}.amazonaws.com:443",
        f"s3.{aws_region}.amazonaws.com:443", 
        "Your PostgreSQL database host:5432"
    ]
    
    network_df = pd.DataFrame(network_reqs, columns=["Required Endpoint"])
    st.dataframe(network_df, hide_index=True)
    
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("⬅️ Back to Model Config"):
            st.session_state.step = 3
            st.rerun()
    
    with col_next:
        if st.button("Generate Configuration ➡️", type="primary"):
            st.session_state.step = 5
            st.rerun()

def render_step_5_generate():
    """Step 5: Generate Configuration."""
    st.markdown('<h2 class="step-header">Step 5: Generate Configuration</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Generate Complete Configuration", type="primary"):
        with st.spinner("Generating configuration files..."):
            # Combine all collected configuration
            complete_config = {
                **st.session_state.org_info,
                **st.session_state.model_config,
                'security': st.session_state.security_config,
                'validation_results': st.session_state.validation_results
            }
            
            # Generate the final configuration
            final_config = create_template_config(complete_config)
            st.session_state.generated_config = final_config
            
        st.success("Configuration generated successfully!")
        st.session_state.step = 6
        st.rerun()
    
    # Show configuration preview if already generated
    if st.session_state.generated_config:
        st.markdown("### 📋 Configuration Preview")
        
        # Show key settings
        config = st.session_state.generated_config
        preview_settings = {
            "Organization": config.get('organization'),
            "Environment": config.get('environment'),
            "AWS Region": config.get('aws', {}).get('region'),
            "Default Model": config.get('aws', {}).get('bedrock', {}).get('default_model'),
            "Security": "Enabled" if config.get('security', {}).get('data', {}).get('encrypt_cache') else "Basic",
            "Audit Logging": "Enabled" if config.get('security', {}).get('audit', {}).get('enabled') else "Disabled"
        }
        
        preview_df = pd.DataFrame(list(preview_settings.items()), columns=["Setting", "Value"])
        st.dataframe(preview_df, hide_index=True)
        
        # Show YAML preview (truncated)
        st.markdown("### 📄 YAML Configuration (Preview)")
        yaml_preview = yaml.dump(config, default_flow_style=False, indent=2)
        st.code(yaml_preview[:2000] + "\n..." if len(yaml_preview) > 2000 else yaml_preview, language='yaml')

def render_step_6_download():
    """Step 6: Download Artifacts."""
    st.markdown('<h2 class="step-header">Step 6: Download Configuration</h2>', unsafe_allow_html=True)
    
    if not st.session_state.generated_config:
        st.error("No configuration generated. Please go back and generate configuration first.")
        return
    
    config = st.session_state.generated_config
    org_name = config.get('organization', 'company').lower()
    
    # Generate all deployment artifacts
    artifacts = {}
    
    # Main configuration
    artifacts['settings.yaml'] = yaml.dump(config, default_flow_style=False, indent=2)
    
    # Deployment files
    artifacts['Dockerfile'] = generate_dockerfile(config)
    artifacts['docker-compose.yml'] = generate_docker_compose(config)
    
    # Environment template
    artifacts['.env.template'] = generate_env_template(config)
    
    # Deployment instructions
    artifacts['DEPLOYMENT_INSTRUCTIONS.md'] = generate_deployment_instructions(config)
    
    # Create downloadable zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in artifacts.items():
            zip_file.writestr(filename, content)
    
    zip_buffer.seek(0)
    
    # Download buttons
    st.markdown("### 📦 Download Configuration Package")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📁 Download Complete Package (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"tidyllm-{org_name}-config-{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            type="primary"
        )
    
    with col2:
        st.download_button(
            label="📄 Download Settings YAML Only",
            data=artifacts['settings.yaml'],
            file_name="settings.yaml",
            mime="text/yaml"
        )
    
    # Individual file downloads
    st.markdown("### 📋 Individual Files")
    
    files_to_show = [
        ('docker-compose.yml', 'Docker Compose'),
        ('Dockerfile', 'Dockerfile'), 
        ('.env.template', 'Environment Template'),
        ('DEPLOYMENT_INSTRUCTIONS.md', 'Instructions')
    ]
    
    for filename, display_name in files_to_show:
        with st.expander(f"📄 {display_name}"):
            st.code(artifacts[filename], language='yaml' if filename.endswith('.yml') else 'bash')
            st.download_button(
                label=f"Download {display_name}",
                data=artifacts[filename],
                file_name=filename,
                key=f"download_{filename}"
            )
    
    # Next steps
    st.markdown("### 🚀 Next Steps")
    st.markdown("""
    1. **Download** the configuration package
    2. **Extract** files to your deployment server  
    3. **Edit** `.env.template` with your actual credentials
    4. **Review** `DEPLOYMENT_INSTRUCTIONS.md`
    5. **Deploy** using Docker Compose or Kubernetes
    6. **Test** the deployment with the health check endpoint
    """)
    
    if st.button("🔄 Start New Configuration"):
        # Reset all session state
        for key in list(st.session_state.keys()):
            if key not in ['step']:
                del st.session_state[key]
        st.session_state.step = 1
        st.rerun()

def generate_env_template(config: Dict[str, Any]) -> str:
    """Generate environment template file."""
    org_name = config.get('organization', 'company').lower()
    aws_region = config.get('aws', {}).get('region', 'us-east-1')
    
    return f"""# TidyLLM Corporate Environment Variables
# ========================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Organization: {config.get('organization', 'YourOrganization')}

# REQUIRED: Database Configuration
TIDYLLM_DB_PASSWORD=your_secure_database_password_here

# REQUIRED: AWS Credentials (if not using IAM roles)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_SESSION_TOKEN=your_aws_session_token_here_if_using_sts
AWS_DEFAULT_REGION={aws_region}

# Optional: Custom S3 bucket (defaults will be used if not set)
TIDYLLM_S3_BUCKET={org_name}-tidyllm-storage
TIDYLLM_AUDIT_BUCKET={org_name}-tidyllm-audit

# Optional: Database connection override
TIDYLLM_DB_HOST=your_postgres_host_here
TIDYLLM_DB_NAME=tidyllm_production
TIDYLLM_DB_USER=tidyllm_service

# Optional: KMS Key for encryption
TIDYLLM_KMS_KEY_ARN=arn:aws:kms:{aws_region}:account:key/your-key-id

# Optional: Logging configuration
TIDYLLM_LOG_LEVEL=INFO
TIDYLLM_LOG_FORMAT=json

# Optional: Security settings
TIDYLLM_ENCRYPT_CACHE=true
TIDYLLM_AUDIT_ENABLED=true
"""

def generate_deployment_instructions(config: Dict[str, Any]) -> str:
    """Generate deployment instructions."""
    return f"""# TidyLLM Corporate Deployment Instructions

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Organization: {config.get('organization', 'YourOrganization')}

## Quick Start

1. Extract the configuration package
2. Copy `.env.template` to `.env` and fill in your credentials
3. Run: `docker-compose up -d`
4. Test: `curl http://localhost:8000/health`

## Support

- Documentation: https://docs.tidyllm.ai
- Issues: https://github.com/tidyllm/tidyllm/issues
- Email: support@tidyllm.ai
"""

def main():
    """Main Streamlit app."""
    init_session_state()
    render_header()
    render_progress_sidebar()
    
    # Route to appropriate step
    if st.session_state.step == 1:
        render_step_1_organization()
    elif st.session_state.step == 2:
        render_step_2_security()
    elif st.session_state.step == 3:
        render_step_3_models()
    elif st.session_state.step == 4:
        render_step_4_validation()
    elif st.session_state.step == 5:
        render_step_5_generate()
    elif st.session_state.step == 6:
        render_step_6_download()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "TidyLLM Corporate Onboarding v1.0 | "
        "<a href='https://tidyllm.ai'>TidyLLM</a> | "
        "<a href='https://github.com/tidyllm/tidyllm'>GitHub</a>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()