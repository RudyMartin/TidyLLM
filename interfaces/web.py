"""
TidyLLM Web Interface
====================

Corporate-controlled web interface for TidyLLM using Streamlit.
Integrates with existing onboarding system for access control.

The onboarding system handles corporate access control, environment detection,
and gateway initialization. This web interface provides the runtime UI
once access is granted.

Usage:
    # Corporate environments use onboarding for access control:
    cd onboarding && streamlit run app.py
    
    # Direct access (bypasses corporate controls):
    python -m tidyllm.interfaces.web
"""

import streamlit as st
from typing import Dict, Any, Optional
import json
from datetime import datetime

from ..gateways.gateway_registry import get_global_registry


class TidyLLMWebInterface:
    """Streamlit-based web interface for TidyLLM."""
    
    def __init__(self):
        self.registry = get_global_registry()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.registry = self.registry
            
            # Auto-configure gateways on first load
            try:
                self.registry.auto_configure()
                st.session_state.gateway_status = "configured"
            except Exception as e:
                st.session_state.gateway_status = f"error: {str(e)}"
    
    def render_main_page(self):
        """Render the main TidyLLM web interface."""
        st.set_page_config(
            page_title="TidyLLM Enterprise Platform",
            page_icon="⭐",
            layout="wide"
        )
        
        st.title("TidyLLM Enterprise Platform")
        st.markdown("**Unified Gateway Management & AI Processing**")
        
        # Sidebar navigation
        page = st.sidebar.selectbox("Navigate", [
            "🏠 Dashboard",
            "🚪 Gateway Management", 
            "🤖 AI Processing",
            "🔧 System Configuration",
            "📊 Status & Monitoring"
        ])
        
        if page == "🏠 Dashboard":
            self._render_dashboard()
        elif page == "🚪 Gateway Management":
            self._render_gateway_management()
        elif page == "🤖 AI Processing":
            self._render_ai_processing()
        elif page == "🔧 System Configuration":
            self._render_system_config()
        elif page == "📊 Status & Monitoring":
            self._render_status_monitoring()
    
    def _render_dashboard(self):
        """Render main dashboard."""
        st.markdown("## 📊 System Overview")
        
        # Gateway status cards
        col1, col2, col3, col4 = st.columns(4)
        
        services = self.registry.list_services()
        service_status = {s['service_type']: s['initialized'] for s in services}
        
        with col1:
            status = "🟢 Active" if service_status.get('corporate_llm', False) else "🔴 Inactive"
            st.metric("Corporate LLM", status)
        
        with col2:
            status = "🟢 Active" if service_status.get('ai_processing', False) else "🔴 Inactive"
            st.metric("AI Processing", status)
        
        with col3:
            status = "🟢 Active" if service_status.get('workflow_optimizer', False) else "🔴 Inactive"
            st.metric("Workflow Optimizer", status)
        
        with col4:
            status = "🟢 Active" if service_status.get('knowledge_resources', False) else "🔴 Inactive"
            st.metric("Knowledge Resources", status)
        
        # Recent activity (mock data for demo)
        st.markdown("## 📈 Recent Activity")
        st.info("Gateway status monitoring, request logs, and system metrics would appear here.")
        
        # Quick actions
        st.markdown("## ⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart All Gateways", use_container_width=True):
                with st.spinner("Restarting gateways..."):
                    self.registry.auto_configure()
                st.success("Gateways restarted successfully!")
        
        with col2:
            if st.button("🔍 Health Check", use_container_width=True):
                health = self.registry.health_check()
                st.json(health)
        
        with col3:
            if st.button("📋 Export Configuration", use_container_width=True):
                config = self.registry.get_registry_stats()
                st.json(config)
    
    def _render_gateway_management(self):
        """Render gateway management interface."""
        st.markdown("## 🚪 Gateway Management")
        
        # List all gateways
        services = self.registry.list_services()
        
        for service in services:
            with st.expander(f"**{service['service_class']}** - {service['service_type']}"):
                st.markdown(f"**Description:** {service['description']}")
                st.markdown(f"**Status:** {'✅ Initialized' if service['initialized'] else '❌ Not Initialized'}")
                st.markdown(f"**Dependencies:** {', '.join(service['dependencies']) if service['dependencies'] else 'None'}")
                
                if service['capabilities']:
                    st.markdown("**Capabilities:**")
                    st.json(service['capabilities'])
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔄 Restart {service['service_type']}", key=f"restart_{service['service_type']}"):
                        st.info(f"Restarting {service['service_type']}...")
                
                with col2:
                    if st.button(f"ℹ️ Get Info {service['service_type']}", key=f"info_{service['service_type']}"):
                        info = self.registry.get_service_info(service['service_type'])
                        st.json(info)
    
    def _render_ai_processing(self):
        """Render AI processing interface."""
        st.markdown("## 🤖 AI Processing")
        
        # Corporate LLM Chat
        st.markdown("### 💼 Corporate LLM Chat")
        corporate_llm = self.registry.get("corporate_llm")
        
        if corporate_llm:
            user_message = st.text_area("Enter your message:", placeholder="Hello, how can I help you?")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                model = st.selectbox("Model:", ["claude", "gpt-4", "local"])
            with col2:
                compliance_level = st.selectbox("Compliance:", ["standard", "high", "maximum"])
            
            if st.button("💬 Send Message", type="primary"):
                if user_message:
                    with st.spinner("Processing..."):
                        try:
                            # Mock processing - would use actual gateway
                            response = f"Corporate LLM Response to: {user_message}\nModel: {model}\nCompliance: {compliance_level}"
                            st.success("Response received!")
                            st.markdown(f"**Response:**\n{response}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a message.")
        else:
            st.error("Corporate LLM Gateway not available. Check gateway configuration.")
        
        # AI Processing Tasks
        st.markdown("### 🔧 AI Processing Tasks")
        ai_processing = self.registry.get("ai_processing")
        
        if ai_processing:
            task = st.selectbox("Task Type:", [
                "text_analysis", "summarization", "translation", 
                "classification", "extraction", "generation"
            ])
            content = st.text_area("Content to process:", placeholder="Enter content here...")
            
            if st.button("Process", type="primary"):
                if content:
                    with st.spinner(f"Processing {task}..."):
                        try:
                            # Mock processing - would use actual gateway
                            result = f"AI Processing Result for {task}:\n\nProcessed: {content[:100]}..."
                            st.success("Processing complete!")
                            st.markdown(f"**Result:**\n{result}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter content to process.")
        else:
            st.error("AI Processing Gateway not available.")
    
    def _render_system_config(self):
        """Render system configuration interface."""
        st.markdown("## 🔧 System Configuration")
        
        # AWS Configuration
        st.markdown("### ☁️ AWS Configuration")
        with st.expander("AWS Settings"):
            aws_region = st.selectbox("AWS Region:", ["us-east-1", "us-west-2", "eu-west-1"])
            s3_bucket = st.text_input("S3 Bucket:", placeholder="tidyllm-data")
            bedrock_enabled = st.checkbox("Enable AWS Bedrock", value=True)
            
            if st.button("💾 Save AWS Config"):
                st.success("AWS configuration saved!")
        
        # Gateway Configuration
        st.markdown("### 🚪 Gateway Configuration")
        with st.expander("Gateway Settings"):
            max_requests = st.number_input("Max Concurrent Requests:", value=10, min_value=1, max_value=100)
            timeout_seconds = st.number_input("Request Timeout (seconds):", value=30, min_value=1, max_value=300)
            enable_caching = st.checkbox("Enable Response Caching", value=True)
            
            if st.button("💾 Save Gateway Config"):
                st.success("Gateway configuration saved!")
        
        # Export/Import Configuration
        st.markdown("### 📤 Export/Import Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Current Config"):
                config = {
                    "timestamp": datetime.now().isoformat(),
                    "registry_stats": self.registry.get_registry_stats(),
                    "services": self.registry.list_services()
                }
                st.download_button(
                    "Download Config JSON",
                    json.dumps(config, indent=2),
                    "tidyllm_config.json",
                    "application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader("📥 Import Config", type=['json'])
            if uploaded_file and st.button("📥 Import"):
                config_data = json.load(uploaded_file)
                st.json(config_data)
                st.info("Configuration imported (would apply settings)")
    
    def _render_status_monitoring(self):
        """Render status and monitoring interface."""
        st.markdown("## 📊 Status & Monitoring")
        
        # System Health
        st.markdown("### 🏥 System Health")
        if st.button("🔍 Run Health Check"):
            health_results = self.registry.health_check()
            
            if health_results.get("overall_healthy", False):
                st.success("🟢 All systems healthy!")
            else:
                st.error("🔴 System issues detected!")
            
            # Display detailed health
            for service_name, health_data in health_results.get("services", {}).items():
                with st.expander(f"{service_name} Health"):
                    if health_data.get("healthy", False):
                        st.success(f"✅ {service_name} is healthy")
                    else:
                        st.error(f"❌ {service_name} has issues")
                    
                    st.json(health_data)
        
        # Registry Statistics
        st.markdown("### 📈 Registry Statistics")
        stats = self.registry.get_registry_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Services", stats.get("total_services", 0))
        with col2:
            st.metric("Initialized Services", stats.get("initialized_services", 0))
        with col3:
            st.metric("Registry Status", "✅ Active" if stats.get("registry_initialized", False) else "❌ Inactive")
        
        # Detailed Statistics
        with st.expander("Detailed Statistics"):
            st.json(stats)
        
        # Live Monitoring (Mock)
        st.markdown("### 📡 Live Monitoring")
        st.info("Real-time metrics, request logs, and performance monitoring would appear here.")


def main():
    """Main entry point for web interface."""
    interface = TidyLLMWebInterface()
    interface.render_main_page()


if __name__ == "__main__":
    main()