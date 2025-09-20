"""
Setup Portal - Credential & Configuration Management

The central portal for reviewing, validating, and establishing system credentials
and configuration. Integrates with the new qa-shipping configuration system.

Port: 8511
Category: Infrastructure
Dependencies: database, aws
"""

import streamlit as st
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any
import json
import time

# Add the qa-shipping root to Python path
qa_shipping_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(qa_shipping_root))

try:
    from config.environment_manager import get_environment_manager, get_db_config, get_aws_config, get_mlflow_config
    from config.credential_validator import validate_all_credentials, quick_health_check
    from config.portal_config import get_portal_config_manager, get_portal_summary
except ImportError as e:
    st.error(f"Configuration system not available: {e}")
    st.stop()


def set_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Setup Portal - qa-shipping",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_header():
    """Render the portal header."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("⚙️ Setup Portal")
        st.markdown("**System Configuration & Credential Management**")
        st.markdown("---")


def render_system_status():
    """Render real-time system status."""
    st.subheader("🔍 System Status")

    # Create placeholders for real-time updates
    status_container = st.container()

    with status_container:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Environment", st.session_state.get('environment', 'Unknown'))

        with col2:
            health_status = st.session_state.get('health_status', 'Unknown')
            st.metric("Health", health_status,
                     delta="Healthy" if health_status == "OK" else "Issues detected")

        with col3:
            portal_count = st.session_state.get('active_portals', 0)
            st.metric("Active Portals", f"{portal_count}/7")


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_environment_summary():
    """Get environment summary with caching."""
    try:
        env_mgr = get_environment_manager()
        return env_mgr.get_environment_summary()
    except Exception as e:
        return {"error": str(e)}


def render_environment_config():
    """Render environment configuration section."""
    st.subheader("🌍 Environment Configuration")

    summary = get_environment_summary()

    if "error" in summary:
        st.error(f"Environment configuration error: {summary['error']}")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Current Environment:**")
        st.info(summary.get('environment', 'development'))

        st.write("**Configuration Sources:**")
        for component, source in summary.get('config_sources', {}).items():
            icon = "🟢" if source == "environment" else "🟡"
            st.write(f"{icon} {component}: {source}")

    with col2:
        st.write("**Component Validation:**")
        for component, valid in summary.get('validation_results', {}).items():
            status = "✅ Valid" if valid else "❌ Invalid"
            st.write(f"{status} {component}")


def render_credential_management():
    """Render credential management interface."""
    st.subheader("🔑 Credential Management")

    tab1, tab2, tab3 = st.tabs(["Environment Variables", "Validation", "Secure Input"])

    with tab1:
        render_environment_variables()

    with tab2:
        render_credential_validation()

    with tab3:
        render_secure_credential_input()


def render_environment_variables():
    """Render environment variable status."""
    st.write("**Required Environment Variables:**")

    required_vars = [
        ('DB_HOST', 'Database host address'),
        ('DB_PORT', 'Database port (default: 5432)'),
        ('DB_NAME', 'Database name'),
        ('DB_USERNAME', 'Database username'),
        ('DB_PASSWORD', 'Database password'),
    ]

    optional_vars = [
        ('AWS_ACCESS_KEY_ID', 'AWS access key'),
        ('AWS_SECRET_ACCESS_KEY', 'AWS secret key'),
        ('AWS_REGION', 'AWS region (default: us-east-1)'),
        ('MLFLOW_TRACKING_URI', 'MLflow tracking server'),
        ('ENVIRONMENT', 'Deployment environment'),
        ('LOG_LEVEL', 'Logging level (default: INFO)'),
    ]

    st.write("**Critical Variables:**")
    for var_name, description in required_vars:
        value = os.getenv(var_name)
        if value:
            st.success(f"✅ {var_name}: Set ({description})")
        else:
            st.error(f"❌ {var_name}: Not set - {description}")

    st.write("**Optional Variables:**")
    for var_name, description in optional_vars:
        value = os.getenv(var_name)
        if value:
            st.success(f"✅ {var_name}: Set ({description})")
        else:
            st.warning(f"⚠️ {var_name}: Not set - {description}")


def render_credential_validation():
    """Render credential validation interface."""
    st.write("**System Validation:**")

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🔍 Run Full Validation", type="primary"):
            with st.spinner("Validating system credentials..."):
                try:
                    # Run async validation
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    report = loop.run_until_complete(validate_all_credentials())
                    st.session_state['validation_report'] = report
                except Exception as e:
                    st.error(f"Validation failed: {e}")

        if st.button("⚡ Quick Health Check"):
            with st.spinner("Checking essential services..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    healthy = loop.run_until_complete(quick_health_check())
                    if healthy:
                        st.success("✅ Essential services healthy")
                    else:
                        st.error("❌ Essential services unavailable")
                except Exception as e:
                    st.error(f"Health check failed: {e}")

    with col2:
        if 'validation_report' in st.session_state:
            report = st.session_state['validation_report']
            render_validation_report(report)


def render_validation_report(report: Dict[str, Any]):
    """Render detailed validation report."""
    st.write("**Validation Results:**")

    # Overall status
    status = report.get('overall_status', 'unknown')
    if status == 'healthy':
        st.success(f"🎉 System Status: {status.upper()}")
    elif status == 'degraded':
        st.warning(f"⚠️ System Status: {status.upper()}")
    else:
        st.error(f"❌ System Status: {status.upper()}")

    # Summary
    summary = report.get('summary', {})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Checks", summary.get('total_checks', 0))
    with col2:
        st.metric("Successful", summary.get('successful', 0))
    with col3:
        st.metric("Errors", summary.get('errors', 0))

    # Detailed results
    st.write("**Component Details:**")
    for result in report.get('results', []):
        status_icon = {"success": "✅", "warning": "⚠️", "error": "❌"}
        icon = status_icon.get(result['status'], "❓")

        with st.expander(f"{icon} {result['component']} - {result['message']}"):
            st.write(f"**Status:** {result['status']}")
            st.write(f"**Message:** {result['message']}")
            if result.get('details'):
                st.write("**Details:**")
                st.json(result['details'])

    # Recommendations
    if report.get('recommendations'):
        st.write("**🔧 Recommendations:**")
        for rec in report['recommendations']:
            st.info(f"💡 {rec}")


def render_secure_credential_input():
    """Render secure credential input interface."""
    st.write("**Secure Credential Configuration:**")
    st.info("⚠️ For security, credentials are set via environment variables. Use your system's secure methods.")

    st.write("**Recommended Setup Methods:**")

    with st.expander("🪟 Windows Setup"):
        st.code("""
# Option 1: PowerShell (Current Session)
$env:DB_HOST="your_host"
$env:DB_PASSWORD="your_password"

# Option 2: Command Prompt (Current Session)
set DB_HOST=your_host
set DB_PASSWORD=your_password

# Option 3: System Environment Variables (Permanent)
# Go to: System Properties > Advanced > Environment Variables
        """, language="powershell")

    with st.expander("🐧 Linux/Mac Setup"):
        st.code("""
# Option 1: Current Session
export DB_HOST="your_host"
export DB_PASSWORD="your_password"

# Option 2: Add to ~/.bashrc or ~/.zshrc (Permanent)
echo 'export DB_HOST="your_host"' >> ~/.bashrc
echo 'export DB_PASSWORD="your_password"' >> ~/.bashrc

# Option 3: .env file (with python-dotenv)
# Create .env file in project root:
DB_HOST=your_host
DB_PASSWORD=your_password
        """, language="bash")

    with st.expander("🔐 Production Setup"):
        st.code("""
# AWS Secrets Manager
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Azure Key Vault
# Use Azure CLI or environment variables

# Docker Secrets
# Mount secrets as environment variables

# Kubernetes Secrets
# Use ConfigMaps and Secrets resources
        """, language="yaml")


def render_portal_management():
    """Render portal management interface."""
    st.subheader("🚀 Portal Management")

    try:
        summary = get_portal_summary()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Portals", summary['total_portals'])
        with col2:
            st.metric("Enabled", summary['enabled_portals'])
        with col3:
            st.metric("Disabled", summary['disabled_portals'])

        st.write("**Portal Registry:**")

        for portal in summary['portal_list']:
            with st.expander(f"{'🟢' if portal['enabled'] else '🔴'} {portal['title']} (Port {portal['port']})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Name:** {portal['name']}")
                    st.write(f"**Category:** {portal['category']}")
                    st.write(f"**Port:** {portal['port']}")

                with col2:
                    st.write(f"**Status:** {'Enabled' if portal['enabled'] else 'Disabled'}")
                    if portal['url']:
                        st.link_button("🔗 Open Portal", portal['url'])

                    if st.button(f"{'Disable' if portal['enabled'] else 'Enable'} {portal['name']}"):
                        st.info("Portal status management will be implemented in Phase 3")

    except Exception as e:
        st.error(f"Portal management error: {e}")


def render_health_endpoint():
    """Health check endpoint for monitoring."""
    if st.query_params.get('health') == 'true':
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            healthy = loop.run_until_complete(quick_health_check())

            return {
                "status": "healthy" if healthy else "unhealthy",
                "portal": "setup_portal",
                "port": 8511,
                "timestamp": time.time()
            }
        except Exception:
            return {
                "status": "error",
                "portal": "setup_portal",
                "port": 8511,
                "timestamp": time.time()
            }
    return None


def update_session_state():
    """Update session state with current system info."""
    try:
        summary = get_environment_summary()
        st.session_state['environment'] = summary.get('environment', 'Unknown')

        # Quick health check
        all_valid = all(summary.get('validation_results', {}).values())
        st.session_state['health_status'] = "OK" if all_valid else "Issues"

        # Portal count
        portal_summary = get_portal_summary()
        st.session_state['active_portals'] = portal_summary.get('enabled_portals', 0)

    except Exception:
        st.session_state['environment'] = 'Error'
        st.session_state['health_status'] = 'Error'
        st.session_state['active_portals'] = 0


def main():
    """Main portal application."""
    set_page_config()

    # Check for health endpoint
    health_response = render_health_endpoint()
    if health_response:
        st.json(health_response)
        return

    # Update session state
    update_session_state()

    # Render portal interface
    render_header()

    # Sidebar navigation
    with st.sidebar:
        st.title("🔧 Setup Menu")

        page = st.selectbox(
            "Select Section:",
            ["System Status", "Environment", "Credentials", "Portal Management"]
        )

        st.markdown("---")
        st.write("**Quick Actions:**")

        if st.button("🔄 Refresh Status"):
            st.rerun()

        if st.button("📊 Run Diagnostics"):
            st.session_state['run_diagnostics'] = True
            st.rerun()

        st.markdown("---")
        st.write("**Portal Info:**")
        st.write("Port: 8511")
        st.write("Category: Infrastructure")
        st.write("Dependencies: database, aws")

    # Main content based on page selection
    if page == "System Status":
        render_system_status()
        render_environment_config()

    elif page == "Environment":
        render_environment_config()

    elif page == "Credentials":
        render_credential_management()

    elif page == "Portal Management":
        render_portal_management()

    # Handle diagnostics
    if st.session_state.get('run_diagnostics'):
        st.markdown("---")
        st.subheader("🔬 System Diagnostics")

        with st.spinner("Running comprehensive diagnostics..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                report = loop.run_until_complete(validate_all_credentials())
                render_validation_report(report)
            except Exception as e:
                st.error(f"Diagnostics failed: {e}")

        st.session_state['run_diagnostics'] = False


if __name__ == "__main__":
    main()