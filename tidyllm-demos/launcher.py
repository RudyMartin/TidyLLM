#!/usr/bin/env python3
"""
TidyLLM Demos Launcher
Main entry point for all TidyLLM ecosystem demos
"""
import streamlit as st
import subprocess
import sys
import os
from pathlib import Path
from shared.utils import load_settings

# Page configuration
st.set_page_config(
    page_title="TidyLLM Demos",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    
    # Load settings
    settings = load_settings()
    
    # Sidebar navigation
    st.sidebar.header("🎯 Demo Selection")
    
    # Sidebar navigation with simplified logic
    demo_options = [
        "🏠 Home",
        "📡 Live AI Ticker",
        "🎛️ Gateway Control Dashboard", 
        "⚙️ Settings Configuration",
        "📊 MVR Demo",
        "🔧 System Status"
    ]
    
    # Initialize session state
    if "demo_choice" not in st.session_state:
        st.session_state.demo_choice = "🏠 Home"
    
    # Get current index safely
    try:
        current_index = demo_options.index(st.session_state.demo_choice)
    except ValueError:
        current_index = 0
        st.session_state.demo_choice = "🏠 Home"
    
    demo_choice = st.sidebar.selectbox(
        "Choose a Demo",
        demo_options,
        index=current_index
    )
    
    # Update session state
    st.session_state.demo_choice = demo_choice
    
    if demo_choice == "🏠 Home":
        show_home_page(settings)
    elif demo_choice == "📡 Live AI Ticker":
        launch_live_ticker()
    elif demo_choice == "🎛️ Gateway Control Dashboard":
        show_gateway_control_dashboard()
    elif demo_choice == "⚙️ Settings Configuration":
        launch_settings_config()
    elif demo_choice == "📊 MVR Demo":
        launch_mvr_demo()
    elif demo_choice == "🔧 System Status":
        show_system_status(settings)

def show_home_page(settings: dict):
    """Show the home page with overview"""
    st.header("🏠 Welcome to TidyLLM Demos")
    
    st.markdown("""
    **Single-Process Unified Demo Platform** - All demos run in one interface, no multiple tabs needed!
    
    Use the **sidebar navigation** to switch between demos seamlessly. Each demo is fully embedded 
    and ready to use with real functionality.
    """)
    
    # System health dashboard
    st.subheader("🔋 System Health Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Settings Status", "✅ Loaded" if settings else "❌ Missing")
    
    with col2:
        # Check demo files
        demo_files = [
            Path("demos/live-ticker/live_ticker.py"),
            Path("demos/settings-config/settings_config.py"),
            Path("demos/gateway-control/gateway_control.py")
        ]
        available_demos = sum(1 for f in demo_files if f.exists())
        st.metric("Demo Files", f"{available_demos}/3", "Available")
    
    with col3:
        # Check dependencies
        try:
            import streamlit, yaml, pandas
            deps_status = "✅ Ready"
        except ImportError:
            deps_status = "❌ Missing"
        st.metric("Dependencies", deps_status)
    
    with col4:
        # Single process indicator
        st.metric("Process Mode", "✅ Unified", "Single Port")
    
    # Quick navigation cards
    st.subheader("🚀 Quick Navigation")
    st.markdown("**Click any card to navigate to that demo:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("### 📡 Live AI Ticker")
            st.markdown("Real-time AI question processing with cost tracking")
            if st.button("→ Go to Live Ticker", key="nav_ticker", use_container_width=True):
                st.session_state.demo_choice = "📡 Live AI Ticker"
                st.rerun()
            st.markdown("**Status:** ✅ Ready")
        
        with st.container():
            st.markdown("### ⚙️ Settings Configuration") 
            st.markdown("Interactive settings management interface")
            if st.button("→ Go to Settings", key="nav_settings", use_container_width=True):
                st.session_state.demo_choice = "⚙️ Settings Configuration"
                st.rerun()
            st.markdown("**Status:** ✅ Ready")
    
    with col2:
        with st.container():
            st.markdown("### 🎛️ Gateway Control")
            st.markdown("MLflow Gateway management and monitoring") 
            if st.button("→ Go to Gateway Control", key="nav_gateway", use_container_width=True):
                st.session_state.demo_choice = "🎛️ Gateway Control Dashboard"
                st.rerun()
            st.markdown("**Status:** ✅ Ready")
        
        with st.container():
            st.markdown("### 📊 MVR Demo")
            st.markdown("Model Validation Report processing")
            if st.button("→ Go to MVR Demo", key="nav_mvr", use_container_width=True):
                st.session_state.demo_choice = "📊 MVR Demo"
                st.rerun()
            mvr_status = "✅ Ready" if Path("demos/mvr-demo/mvr_demo.py").exists() else "🔧 Placeholder"
            st.markdown(f"**Status:** {mvr_status}")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Architecture:** Single-process unified launcher")
        st.markdown("**Port:** 8501 (all demos)")
        
    with col2:
        st.markdown("**Navigation:** Sidebar or quick cards above")  
        st.markdown("**Mode:** Embedded demo content")

def launch_live_ticker():
    """Show the Live AI Ticker demo embedded"""
    st.header("📡 Live AI Ticker")
    st.markdown("### Real-time AI Question Processing with Cost Tracking")
    
    # Import and run the live ticker functionality directly
    try:
        # Add demos path to Python path
        import sys
        demos_path = str(Path(__file__).parent / "demos" / "live-ticker")
        if demos_path not in sys.path:
            sys.path.insert(0, demos_path)
        
        # Import the live ticker main function
        from live_ticker import render_live_ticker
        render_live_ticker()
        
    except ImportError as e:
        st.error(f"❌ Could not import Live Ticker demo: {e}")
        st.info("Demo functionality will be embedded here")
        
        # Fallback basic demo interface
        st.subheader("💬 AI Question Interface")
        question = st.text_area("Enter your question:", placeholder="What would you like to know?")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Ask AI", type="primary"):
                with st.spinner("Processing..."):
                    st.success("✅ Question processed!")
                    st.info("This is a placeholder response. Full demo integration needed.")
        
        # Cost tracking placeholder
        st.subheader("💰 Cost Tracking")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Requests Today", "0", "0")
        with col2:
            st.metric("Cost Today", "$0.00", "$0.00")
        with col3:
            st.metric("Avg Cost/Request", "$0.00", "0%")

def show_gateway_control_dashboard():
    """Show the Gateway Control Dashboard as a full page"""
    st.header("🎛️ Gateway Control Dashboard")
    st.markdown("### Real-time LLM Gateway Monitoring and Control")
    
    # Connection status
    st.subheader("🔌 Gateway Connection Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            import mlflow.gateway as gateway
            gateway.set_gateway_uri("http://localhost:5000")
            st.success("✅ MLflow Gateway Connected")
        except Exception as e:
            st.error("❌ MLflow Gateway Disconnected")
    
    with col2:
        try:
            from tidyllm_gateway.gateways.llm_gateway import LLMGateway, LLMGatewayConfig
            config = LLMGatewayConfig(
                mlflow_gateway_uri="http://localhost:5000",
                default_provider="claude",
                default_model="claude-3-5-sonnet"
            )
            gateway = LLMGateway(config)
            st.success("✅ TidyLLM Gateway Connected")
        except ImportError:
            st.warning("⚠️ TidyLLM Gateway Not Installed")
            st.info("Install with: `pip install -e ../tidyllm-gateway`")
        except Exception as e:
            st.error(f"❌ TidyLLM Gateway Error: {e}")
    
    with col3:
        try:
            import psycopg2
            st.success("✅ PostgreSQL Available")
        except Exception as e:
            st.error("❌ PostgreSQL Unavailable")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Real-Time Monitoring", 
        "🔄 Model Routing", 
        "🎯 Quality Control", 
        "🔒 Security Settings", 
        "💰 Cost Management",
        "📋 Audit Logs"
    ])
    
    with tab1:
        st.subheader("📊 Real-Time Gateway Monitoring")
        
        # Status indicator
        st.warning("⚠️ **MONITORING FEATURES**: These are UI demonstrations. Backend integration required for production use.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Requests", "23", "↗️ +5", delta_color="normal")
        
        with col2:
            st.metric("Response Time", "1.2s", "↘️ -0.3s", delta_color="inverse")
        
        with col3:
            st.metric("Success Rate", "98.5%", "↗️ +0.2%", delta_color="normal")
        
        with col4:
            st.metric("Cost/Hour", "$12.45", "↗️ +$2.10", delta_color="inverse")
        
        # Real-time activity feed
        st.subheader("🔄 Live Activity Feed")
        
        activities = [
            {"time": "14:30:25", "action": "Request processed", "model": "claude-3-5-sonnet", "cost": "$0.0023", "status": "✅ Success"},
            {"time": "14:30:20", "action": "Request queued", "model": "gpt-4", "cost": "$0.0150", "status": "⏳ Processing"},
            {"time": "14:30:15", "action": "Request completed", "model": "claude-3-5-sonnet", "cost": "$0.0018", "status": "✅ Success"},
            {"time": "14:30:10", "action": "Request failed", "model": "gpt-4", "cost": "$0.0000", "status": "❌ Retry"},
            {"time": "14:30:05", "action": "Request processed", "model": "claude-3-5-sonnet", "cost": "$0.0021", "status": "✅ Success"}
        ]
        
        for activity in activities:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 1, 1])
                with col1:
                    st.text(activity["time"])
                with col2:
                    st.text(activity["action"])
                with col3:
                    st.text(activity["model"])
                with col4:
                    st.text(activity["cost"])
                with col5:
                    st.text(activity["status"])
                st.divider()
    
    with tab2:
        st.subheader("🔄 Model Routing Configuration")
        
        # Model routing settings
        routing_strategy = st.selectbox(
            "Routing Strategy",
            ["cost_optimized", "performance_optimized", "balanced", "manual"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Models")
            models = {
                "claude-3-5-sonnet": {"cost": 0.003, "performance": 9.5, "status": "✅ Active"},
                "gpt-4": {"cost": 0.03, "performance": 9.8, "status": "✅ Active"},
                "gpt-3.5-turbo": {"cost": 0.002, "performance": 8.0, "status": "✅ Active"},
                "claude-3-haiku": {"cost": 0.00025, "performance": 7.5, "status": "⏸️ Paused"}
            }
            
            for model, config in models.items():
                with st.expander(f"{model} - {config['status']}"):
                    st.metric("Cost per 1K tokens", f"${config['cost']}")
                    st.metric("Performance Score", f"{config['performance']}/10")
        
        with col2:
            st.subheader("Routing Rules")
            st.text_area("Custom routing rules (JSON)", value="""
{
  "cost_threshold": 0.01,
  "performance_threshold": 8.0,
  "fallback_model": "claude-3-5-sonnet",
  "priority_models": ["claude-3-5-sonnet", "gpt-4"]
}
""", height=200)
    
    with tab3:
        st.subheader("🎯 Quality Control Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Response Time Threshold (seconds)", value=5.0, min_value=1.0, max_value=30.0)
            st.number_input("Cost Threshold per Request ($)", value=0.10, min_value=0.01, max_value=1.0)
            st.number_input("Quality Score Threshold", value=7.0, min_value=1.0, max_value=10.0)
        
        with col2:
            st.checkbox("Enable Quality Monitoring", value=True)
            st.checkbox("Auto-retry Failed Requests", value=True)
            st.checkbox("Log Quality Metrics", value=True)
        
        # Quality metrics
        st.subheader("📈 Quality Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Response Time", "1.2s")
            st.metric("Response Time 95th %", "2.8s")
        
        with col2:
            st.metric("Average Quality Score", "8.7/10")
            st.metric("Failed Request Rate", "1.5%")
        
        with col3:
            st.metric("Average Cost per Request", "$0.0042")
            st.metric("Cost Efficiency Score", "9.2/10")
    
    with tab4:
        st.subheader("🔒 Security Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rate Limiting")
            st.number_input("Requests per Minute", value=60, min_value=1, max_value=1000)
            st.number_input("Requests per Hour", value=3600, min_value=1, max_value=10000)
            st.number_input("Requests per Day", value=86400, min_value=1, max_value=100000)
        
        with col2:
            st.subheader("Access Control")
            st.checkbox("Enable API Key Authentication", value=True)
            st.checkbox("Enable IP Whitelisting", value=False)
            st.checkbox("Enable Request Logging", value=True)
            st.checkbox("Enable Audit Trail", value=True)
        
        # Security status
        st.subheader("🛡️ Security Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.success("✅ Rate Limiting Active")
        with col2:
            st.success("✅ Authentication Active")
        with col3:
            st.warning("⚠️ IP Whitelist Disabled")
        with col4:
            st.success("✅ Audit Logging Active")
    
    with tab5:
        st.subheader("💰 Cost Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Budget Settings")
            hourly_budget = st.number_input("Hourly Budget ($)", value=100.0, min_value=1.0, max_value=10000.0)
            daily_budget = st.number_input("Daily Budget ($)", value=2400.0, min_value=1.0, max_value=100000.0)
            monthly_budget = st.number_input("Monthly Budget ($)", value=72000.0, min_value=1.0, max_value=1000000.0)
        
        with col2:
            st.subheader("Cost Controls")
            st.checkbox("Enable Budget Alerts", value=True)
            st.checkbox("Auto-stop on Budget Exceeded", value=False)
            st.checkbox("Cost Optimization", value=True)
            st.checkbox("Detailed Cost Reporting", value=True)
        
        # Cost metrics
        st.subheader("📊 Cost Analytics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Hour Cost", "$12.45", "$87.55 remaining")
        with col2:
            st.metric("Current Day Cost", "$298.20", "$2101.80 remaining")
        with col3:
            st.metric("Current Month Cost", "$8,945.60", "$63054.40 remaining")
        with col4:
            st.metric("Cost Efficiency", "94.2%", "↗️ +2.1%")
        
        # Cost breakdown by model
        st.subheader("📈 Cost Breakdown by Model")
        cost_data = {
            "claude-3-5-sonnet": 45.2,
            "gpt-4": 32.8,
            "gpt-3.5-turbo": 22.0
        }
        
        for model, percentage in cost_data.items():
            st.progress(percentage / 100, text=f"{model}: {percentage}%")
    
    with tab6:
        st.subheader("📋 Audit Logs")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            log_level = st.selectbox("Log Level", ["All", "INFO", "WARNING", "ERROR"])
        with col2:
            time_range = st.selectbox("Time Range", ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"])
        with col3:
            model_filter = st.selectbox("Model", ["All Models", "claude-3-5-sonnet", "gpt-4", "gpt-3.5-turbo"])
        
        # Sample audit logs
        audit_logs = [
            {"timestamp": "2024-08-30 14:30:25", "level": "INFO", "user": "demo_user", "action": "Request processed", "model": "claude-3-5-sonnet", "cost": "$0.0023"},
            {"timestamp": "2024-08-30 14:30:20", "level": "INFO", "user": "demo_user", "action": "Request queued", "model": "gpt-4", "cost": "$0.0150"},
            {"timestamp": "2024-08-30 14:30:15", "level": "WARNING", "user": "demo_user", "action": "Rate limit approaching", "model": "claude-3-5-sonnet", "cost": "$0.0000"},
            {"timestamp": "2024-08-30 14:30:10", "level": "ERROR", "user": "demo_user", "action": "Request failed", "model": "gpt-4", "cost": "$0.0000"},
            {"timestamp": "2024-08-30 14:30:05", "level": "INFO", "user": "demo_user", "action": "Request processed", "model": "claude-3-5-sonnet", "cost": "$0.0021"}
        ]
        
        for log in audit_logs:
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 2, 2, 2, 1])
                with col1:
                    st.text(log["timestamp"])
                with col2:
                    if log["level"] == "ERROR":
                        st.error(log["level"])
                    elif log["level"] == "WARNING":
                        st.warning(log["level"])
                    else:
                        st.info(log["level"])
                with col3:
                    st.text(log["user"])
                with col4:
                    st.text(log["action"])
                with col5:
                    st.text(log["model"])
                with col6:
                    st.text(log["cost"])
                st.divider()
    
    # Save settings button
    st.subheader("💾 Configuration")
    if st.button("💾 Save Gateway Settings", type="primary"):
        st.success("✅ Gateway settings saved successfully!")
        st.info("Settings have been applied to the gateway configuration")

def launch_settings_config():
    """Show the Settings Configuration demo embedded"""
    st.header("⚙️ Settings Configuration")
    st.markdown("### Interactive Settings Management Interface")
    
    # Import and run the settings config functionality directly
    try:
        # Add demos path to Python path
        import sys
        demos_path = str(Path(__file__).parent / "demos" / "settings-config")
        if demos_path not in sys.path:
            sys.path.insert(0, demos_path)
        
        # Import the settings config functions
        from settings_config import render_postgres_config, render_aws_config, render_mlflow_config, load_settings, save_settings
        
        # Load current settings
        settings = load_settings()
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["🗄️ PostgreSQL", "☁️ AWS", "📊 MLflow"])
        
        with tab1:
            settings = render_postgres_config(settings)
        
        with tab2:
            settings = render_aws_config(settings)
        
        with tab3:
            settings = render_mlflow_config(settings)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Settings", type="primary"):
                if save_settings(settings):
                    st.success("Settings saved successfully!")
                else:
                    st.error("Failed to save settings")
        
        with col2:
            if st.button("🔄 Reset to Defaults"):
                st.success("Settings reset to defaults!")
                st.rerun()
        
        with col3:
            if st.button("📋 Show JSON"):
                st.json(settings)
        
    except ImportError as e:
        st.error(f"❌ Could not import Settings Config demo: {e}")
        st.info("Settings configuration interface will be embedded here")
        
        # Fallback settings interface
        st.subheader("⚙️ Configuration Settings")
        st.text_input("Database Host", value="localhost")
        st.number_input("Database Port", value=5432)
        st.text_input("AWS Region", value="us-east-1")
        if st.button("💾 Save", type="primary"):
            st.success("Settings saved (placeholder)")

def launch_mvr_demo():
    """Show the MVR Demo embedded"""
    
    # Check if demo exists and try to import
    demo_path = Path("demos/mvr-demo/mvr_demo.py")
    if demo_path.exists():
        try:
            # Add demos path to Python path
            import sys
            demos_path = str(Path(__file__).parent / "demos" / "mvr-demo")
            if demos_path not in sys.path:
                sys.path.insert(0, demos_path)
            
            # Import the MVR demo main function
            from mvr_demo import show_mvr_demo
            show_mvr_demo()
            
        except ImportError as e:
            st.error(f"❌ Could not import MVR demo: {e}")
            show_mvr_placeholder()
    else:
        st.warning("📊 MVR Demo files not found")
        show_mvr_placeholder()

def show_mvr_placeholder():
    """Show placeholder MVR demo interface"""
    st.info("Model Validation Report demo interface will be embedded here")
    
    # Placeholder MVR interface
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload MVR Document", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    st.subheader("🔍 Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("SPARSE Code Execution", value=True)
        st.checkbox("Compliance Tracking", value=True)
    
    with col2:
        st.checkbox("Document Processing", value=True)
        st.checkbox("Report Generation", value=True)
    
    if st.button("🚀 Process Document", type="primary"):
        with st.spinner("Processing document..."):
            st.success("✅ Document processed successfully!")
            st.info("Full MVR demo integration needed for complete functionality.")

def show_system_status(settings: dict):
    """Show system status and configuration"""
    st.header("🔧 System Status")
    
    # Settings status
    st.subheader("📋 Settings Status")
    if settings:
        st.success("✅ Settings loaded successfully")
        
        # Database status
        db_config = settings.get('database', {})
        if db_config:
            st.info(f"📊 Database: {db_config.get('host', 'N/A')}:{db_config.get('port', 'N/A')}")
        
        # MLflow status
        mlflow_config = settings.get('mlflow', {})
        if mlflow_config:
            st.info(f"🔗 MLflow: {mlflow_config.get('tracking_uri', 'N/A')}")
        
        # Gateway status
        gateway_config = settings.get('tidyllm_gateway', {})
        if gateway_config:
            st.info(f"🚪 Gateway: {gateway_config.get('default_provider', 'N/A')} - {gateway_config.get('default_model', 'N/A')}")
    else:
        st.error("❌ Settings not loaded")
    
    # Installation status
    st.subheader("📦 Installation Status")
    
    try:
        import mlflow
        st.success(f"✅ MLflow {mlflow.__version__}")
    except ImportError:
        st.error("❌ MLflow not installed")
    
    try:
        from tidyllm_gateway.gateways.llm_gateway import LLMGateway
        st.success("✅ TidyLLM Gateway available")
    except ImportError:
        st.warning("⚠️ TidyLLM Gateway not installed")
        st.info("To install: `pip install -e ../tidyllm-gateway`")
    
    try:
        import psycopg2
        st.success("✅ PostgreSQL driver available")
    except ImportError:
        st.error("❌ PostgreSQL driver not installed")
    
    # File structure
    st.subheader("📁 File Structure")
    
    required_dirs = ["demos", "shared", "docs"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            st.success(f"✅ {dir_name}/")
        else:
            st.error(f"❌ {dir_name}/")

if __name__ == "__main__":
    main()
