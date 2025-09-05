#!/usr/bin/env python3
"""
LLM Gateway Control Dashboard
Uses MLflow to control and monitor the LLM gateway
"""
import streamlit as st
import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="LLM Gateway Control",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_gateway_settings() -> Dict[str, Any]:
    """Load gateway settings from demo app's settings.yaml"""
    try:
        settings_paths = [
            "../settings.yaml",  # Demo app's settings
            "settings.yaml",     # Fallback to local if needed
        ]
        for path in settings_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    settings = yaml.safe_load(f)
                    return settings.get("gateway", {})
        return create_default_gateway_settings()
    except Exception as e:
        st.error(f"Error loading gateway settings: {e}")
        return create_default_gateway_settings()

def create_default_gateway_settings() -> Dict[str, Any]:
    """Create default gateway settings"""
    return {
        "model_routing": {
            "enabled": True,
            "strategy": "cost_optimized",
            "fallback_model": "gpt-3.5-turbo",
            "models": {
                "gpt-4": {"cost_per_1k": 0.03, "performance_score": 9.5},
                "gpt-3.5-turbo": {"cost_per_1k": 0.002, "performance_score": 8.0},
                "claude-3": {"cost_per_1k": 0.015, "performance_score": 9.0}
            }
        },
        "quality_control": {
            "enabled": True,
            "response_time_threshold": 5.0,
            "cost_threshold": 0.10,
            "quality_score_threshold": 7.0
        },
        "security": {
            "enabled": True,
            "rate_limiting": {"requests_per_minute": 60},
            "audit_logging": True
        },
        "cost_management": {
            "enabled": True,
            "daily_budget": 100.0,
            "monthly_budget": 2000.0
        }
    }

def save_gateway_settings(gateway_settings: Dict[str, Any]) -> bool:
    """Save gateway settings to demo app's settings.yaml"""
    try:
        settings_path = "../settings.yaml"
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
        else:
            settings = {}
        
        settings["gateway"] = gateway_settings
        
        with open(settings_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving gateway settings: {e}")
        return False

def render_gateway_monitoring():
    """Render real-time gateway monitoring"""
    st.subheader("📊 Real-Time Gateway Monitoring")
    
    # Status indicator
    st.warning("⚠️ **MONITORING FEATURES**: These are UI demonstrations only. Backend integration required for production use.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Requests", "23", "↗️ +5", delta_color="normal")
    
    with col2:
        st.metric("Avg Response Time", "2.3s", "↘️ -0.5s", delta_color="normal")
    
    with col3:
        st.metric("Success Rate", "98.5%", "↗️ +0.2%", delta_color="normal")
    
    with col4:
        st.metric("Cost Today", "$45.20", "↗️ +$12.30", delta_color="normal")
    
    # Real-time metrics chart
    st.write("**Request Volume (Last Hour)**")
    st.info("📊 **DEMO DATA**: This chart shows sample data. Real monitoring requires MLflow backend integration.")

def render_model_routing_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Render model routing configuration"""
    st.subheader("🎯 Model Routing & Load Balancing")
    
    # Status indicators
    col_status1, col_status2 = st.columns([3, 1])
    with col_status1:
        st.success("✅ **CONFIGURATION**: Fully functional - saves to settings.yaml")
    with col_status2:
        st.warning("⚠️ **ROUTING LOGIC**: Backend implementation required")
    
    routing = settings.get("model_routing", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        routing["enabled"] = st.checkbox("Enable Model Routing", routing.get("enabled", True))
        routing["strategy"] = st.selectbox(
            "Routing Strategy",
            ["cost_optimized", "performance", "quality", "balanced"],
            index=["cost_optimized", "performance", "quality", "balanced"].index(routing.get("strategy", "cost_optimized"))
        )
        routing["fallback_model"] = st.selectbox(
            "Fallback Model",
            ["gpt-3.5-turbo", "gpt-4", "claude-3"],
            index=["gpt-3.5-turbo", "gpt-4", "claude-3"].index(routing.get("fallback_model", "gpt-3.5-turbo"))
        )
    
    with col2:
        st.write("**Model Configuration**")
        models = routing.get("models", {})
        
        for model_name, config in models.items():
            st.write(f"**{model_name}**")
            config["cost_per_1k"] = st.number_input(
                f"Cost per 1K tokens ({model_name})", 
                value=config.get("cost_per_1k", 0.0),
                min_value=0.0, max_value=1.0, step=0.001,
                key=f"cost_{model_name}"
            )
            config["performance_score"] = st.number_input(
                f"Performance Score ({model_name})", 
                value=config.get("performance_score", 0.0),
                min_value=0.0, max_value=10.0, step=0.1,
                key=f"perf_{model_name}"
            )
    
    settings["model_routing"] = routing
    return settings

def render_quality_control_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Render quality control configuration"""
    st.subheader("🔍 Quality Control & Monitoring")
    
    # Status indicators
    col_status1, col_status2 = st.columns([3, 1])
    with col_status1:
        st.success("✅ **CONFIGURATION**: Fully functional - saves to settings.yaml")
    with col_status2:
        st.warning("⚠️ **MONITORING**: Backend integration required")
    
    quality = settings.get("quality_control", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        quality["enabled"] = st.checkbox("Enable Quality Control", quality.get("enabled", True))
        quality["response_time_threshold"] = st.number_input(
            "Response Time Threshold (seconds)", 
            value=quality.get("response_time_threshold", 5.0),
            min_value=1.0, max_value=30.0, step=0.5
        )
        quality["cost_threshold"] = st.number_input(
            "Cost Threshold (USD per request)", 
            value=quality.get("cost_threshold", 0.10),
            min_value=0.01, max_value=1.0, step=0.01
        )
    
    with col2:
        quality["quality_score_threshold"] = st.number_input(
            "Quality Score Threshold", 
            value=quality.get("quality_score_threshold", 7.0),
            min_value=1.0, max_value=10.0, step=0.1
        )
        
        st.write("**Monitoring Features**")
        st.info("📊 **FUTURE FEATURES**: These require MLflow backend integration")
        st.checkbox("Drift Detection", True, disabled=True)
        st.checkbox("Performance Tracking", True, disabled=True)
        st.checkbox("Anomaly Detection", True, disabled=True)
    
    settings["quality_control"] = quality
    return settings

def render_security_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Render security configuration"""
    st.subheader("🔒 Security & Access Control")
    
    # Status indicators
    col_status1, col_status2 = st.columns([3, 1])
    with col_status1:
        st.success("✅ **CONFIGURATION**: Fully functional - saves to settings.yaml")
    with col_status2:
        st.warning("⚠️ **ENFORCEMENT**: Backend implementation required")
    
    security = settings.get("security", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        security["enabled"] = st.checkbox("Enable Security Controls", security.get("enabled", True))
        security["audit_logging"] = st.checkbox("Audit Logging", security.get("audit_logging", True))
    
    with col2:
        st.write("**Rate Limiting**")
        rate_limiting = security.get("rate_limiting", {})
        rate_limiting["requests_per_minute"] = st.number_input(
            "Requests per Minute", 
            value=rate_limiting.get("requests_per_minute", 60),
            min_value=1, max_value=1000
        )
        security["rate_limiting"] = rate_limiting
    
    settings["security"] = security
    return settings

def render_cost_management_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Render cost management configuration"""
    st.subheader("💰 Cost Management & Optimization")
    
    # Status indicators
    col_status1, col_status2 = st.columns([3, 1])
    with col_status1:
        st.success("✅ **CONFIGURATION**: Fully functional - saves to settings.yaml")
    with col_status2:
        st.warning("⚠️ **TRACKING**: Backend integration required")
    
    cost_mgmt = settings.get("cost_management", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        cost_mgmt["enabled"] = st.checkbox("Enable Cost Management", cost_mgmt.get("enabled", True))
        cost_mgmt["daily_budget"] = st.number_input(
            "Daily Budget (USD)", 
            value=cost_mgmt.get("daily_budget", 100.0),
            min_value=1.0, max_value=10000.0, step=1.0
        )
    
    with col2:
        cost_mgmt["monthly_budget"] = st.number_input(
            "Monthly Budget (USD)", 
            value=cost_mgmt.get("monthly_budget", 2000.0),
            min_value=10.0, max_value=100000.0, step=10.0
        )
        
        st.write("**Optimization Features**")
        st.info("📊 **FUTURE FEATURES**: These require MLflow backend integration")
        st.checkbox("Auto-route to Cheaper Models", True, disabled=True)
        st.checkbox("Cache Responses", True, disabled=True)
        st.checkbox("Batch Requests", True, disabled=True)
    
    settings["cost_management"] = cost_mgmt
    return settings

def main():
    """Main application"""
    st.title("🚀 LLM Gateway Control Dashboard")
    st.markdown("Control and monitor your LLM gateway using MLflow backend")
    
    # Production Status Banner
    st.info("""
    **📋 PRODUCTION STATUS**: 
    - ✅ **Configuration Management**: Fully functional - saves to settings.yaml
    - ⚠️ **Real-time Monitoring**: UI demonstration only - requires MLflow backend integration
    - 🚫 **Active Control**: Backend implementation required for production use
    """)
    
    # Load current gateway settings
    gateway_settings = load_gateway_settings()
    
    # Sidebar for navigation
    st.sidebar.title("Gateway Control Sections")
    
    sections = {
        "Model Routing": render_model_routing_config,
        "Quality Control": render_quality_control_config,
        "Security": render_security_config,
        "Cost Management": render_cost_management_config
    }
    
    selected_section = st.sidebar.selectbox("Select Section", list(sections.keys()))
    
    # Render selected section
    gateway_settings = sections[selected_section](gateway_settings)
    
    # Real-time monitoring
    if selected_section == "Model Routing":
        render_gateway_monitoring()
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Gateway Settings", type="primary"):
            if save_gateway_settings(gateway_settings):
                st.success("Gateway settings saved successfully!")
            else:
                st.error("Failed to save gateway settings")
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            gateway_settings = create_default_gateway_settings()
            st.success("Gateway settings reset to defaults!")
            st.rerun()
    
    with col3:
        if st.button("📋 Show JSON"):
            st.json(gateway_settings)
    
    # Show current settings file location
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Gateway Settings File:**")
    settings_file = "../settings.yaml"
    if os.path.exists(settings_file):
        st.sidebar.success(f"✅ {settings_file}")
    else:
        st.sidebar.warning(f"⚠️ {settings_file} (will be created)")
    
    # Implementation roadmap
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Implementation Roadmap**")
    st.sidebar.markdown("**Phase 1**: ✅ Configuration UI")
    st.sidebar.markdown("**Phase 2**: 🔄 MLflow Integration")
    st.sidebar.markdown("**Phase 3**: 📊 Real-time Monitoring")
    st.sidebar.markdown("**Phase 4**: 🎯 Active Control")

if __name__ == "__main__":
    main()
