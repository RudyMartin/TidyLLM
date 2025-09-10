#!/usr/bin/env python3
"""
Visual Demo Interface for TidyLLM Enhanced Demo Systems

Provides a beautiful, interactive web interface for demo team interactions.
Built with Streamlit for easy deployment and vetting.

This is the visual layer that makes the demo systems accessible and engaging.
"""

import streamlit as st
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import polars as pl

# Import our standalone demo systems
from sparse_agreements import SparseAgreementManager, execute_sparse_command
from error_tracker import PromptPipelineErrorTracker, track_dspy_error, ErrorSeverity
from demo_protection import DemoProtectionSystem, TransparentModeIndicator, protect_demo_input

# Page configuration
st.set_page_config(
    page_title="TidyLLM Enhanced Demo Systems",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mode-indicator {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .full-power {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .hybrid {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    .demo-mode {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .sparse-command {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        cursor: pointer;
    }
    .sparse-command:hover {
        background-color: #bbdefb;
    }
</style>
""", unsafe_allow_html=True)

class VisualDemoInterface:
    """Visual interface for the enhanced demo systems"""
    
    def __init__(self):
        # Initialize demo systems
        self.sparse_manager = SparseAgreementManager()
        self.error_tracker = PromptPipelineErrorTracker()
        self.protection_system = DemoProtectionSystem()
        self.mode_indicator = TransparentModeIndicator(self.protection_system)
        
        # Session state for tracking
        if 'execution_history' not in st.session_state:
            st.session_state.execution_history = []
        if 'error_history' not in st.session_state:
            st.session_state.error_history = []
        if 'protection_events' not in st.session_state:
            st.session_state.protection_events = []
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🎯 TidyLLM Enhanced Demo Systems</h1>', unsafe_allow_html=True)
        st.markdown("### Sophisticated demo systems inspired by tidy-mvr - Ready for demo team testing!")
    
    def render_mode_indicator(self):
        """Render the current mode indicator"""
        mode_info = self.mode_indicator.get_mode_indicator()
        
        # Determine CSS class based on mode
        if mode_info['mode'] == 'FULL_POWER':
            css_class = 'full-power'
            icon = '🚀'
        elif mode_info['mode'] == 'HYBRID':
            css_class = 'hybrid'
            icon = '⚡'
        else:
            css_class = 'demo-mode'
            icon = '🎭'
        
        st.markdown(f"""
        <div class="mode-indicator {css_class}">
            {icon} {mode_info['user_message']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show system status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CPU Usage", f"{mode_info['system_status']['cpu_percent']:.1f}%")
        with col2:
            st.metric("Memory Usage", f"{mode_info['system_status']['memory_percent']:.1f}%")
        with col3:
            st.metric("Confidence", f"{mode_info['confidence']:.1%}")
        with col4:
            st.metric("Active Rules", mode_info['protection_active'])
    
    def render_sparse_agreements_interface(self):
        """Render the SPARSE agreements interface"""
        st.header("🎭 SPARSE Agreements System")
        st.markdown("Intelligent shortcuts for demo team interactions")
        
        # Available commands
        st.subheader("Available Commands")
        available_commands = self.sparse_manager.get_available_agreements()
        
        # Create a grid of command buttons
        cols = st.columns(3)
        for i, command in enumerate(available_commands):
            with cols[i % 3]:
                if st.button(f"🔍 {command}", key=f"sparse_{i}"):
                    self.execute_sparse_command(command)
        
        # Manual command input
        st.subheader("Custom Command")
        custom_command = st.text_input("Enter a SPARSE command:", placeholder="[Performance Test]")
        if st.button("Execute Custom Command"):
            if custom_command:
                self.execute_sparse_command(custom_command)
    
    def execute_sparse_command(self, command: str):
        """Execute a SPARSE command and display results"""
        with st.spinner(f"Executing {command}..."):
            result = execute_sparse_command(command)
            
            # Store in session state
            st.session_state.execution_history.append({
                'timestamp': datetime.now(),
                'command': command,
                'result': result
            })
            
            # Display result
            st.success(f"✅ {command} executed successfully!")
            
            # Show execution details
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Execution Mode", result.get('execution_mode', 'unknown').title())
                st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
            
            with col2:
                if result.get('result'):
                    st.json(result['result'])
            
            # Show authenticity indicator
            authenticity = self.mode_indicator.show_analysis_authenticity(result)
            if authenticity['authenticity'] == 'REAL':
                st.success(authenticity['message'])
            elif authenticity['authenticity'] == 'SIMULATED':
                st.warning(authenticity['message'])
            else:
                st.error(authenticity['message'])
    
    def render_error_tracking_interface(self):
        """Render the error tracking interface"""
        st.header("📊 Intelligent Error Tracking & Alerting")
        st.markdown("Smart monitoring and alerting for demo operations")
        
        # Error summary
        st.subheader("Error Summary (Last 24 Hours)")
        summary = self.error_tracker.get_error_summary(hours=24)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Errors", summary['total_errors'])
        with col2:
            st.metric("Critical", summary['severity_breakdown'].get('critical', 0))
        with col3:
            st.metric("Warnings", summary['severity_breakdown'].get('warning', 0))
        with col4:
            st.metric("Info", summary['severity_breakdown'].get('info', 0))
        
        # Error type breakdown chart
        if summary['error_type_breakdown']:
            error_df = pl.DataFrame(list(summary['error_type_breakdown'].items()), 
                                  columns=['Error Type', 'Count'])
            fig = px.bar(error_df, x='Error Type', y='Count', 
                        title="Error Types (Last 24 Hours)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Test error tracking
        st.subheader("Test Error Tracking")
        test_error_type = st.selectbox("Error Type", [
            'prompt_timeout', 'llm_api_failure', 'cost_exceeded', 
            'dspy_wrapper_failure', 'high_latency', 'demo_test_error'
        ])
        test_error_message = st.text_input("Error Message", 
                                         f"Test error: {test_error_type}")
        
        if st.button("Track Test Error"):
            track_dspy_error(
                error_type=test_error_type,
                error_message=test_error_message,
                agent_name='demo_team',
                task_type='testing',
                user_facing=False
            )
            st.success("✅ Test error tracked successfully!")
    
    def render_protection_interface(self):
        """Render the demo protection interface"""
        st.header("🛡️ Anti-Sabotage Demo Protection")
        st.markdown("Multi-layer protection for demo stability")
        
        # Protection status
        st.subheader("Protection Status")
        status = self.protection_system.get_protection_status()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Requests", status['protection_stats']['total_requests'])
        with col2:
            st.metric("Blocked", status['protection_stats']['blocked_requests'])
        with col3:
            st.metric("Warned", status['protection_stats']['warned_requests'])
        with col4:
            st.metric("Sabotage Attempts", status['protection_stats']['sabotage_attempts'])
        
        # Test protection
        st.subheader("Test Input Protection")
        test_input = st.text_area("Enter test input:", 
                                 placeholder="Enter text to test protection...")
        input_type = st.selectbox("Input Type", ["text", "file", "dict"])
        
        if st.button("Test Protection"):
            if test_input:
                safe, result = protect_demo_input(test_input, input_type)
                
                # Store in session state
                st.session_state.protection_events.append({
                    'timestamp': datetime.now(),
                    'input': test_input,
                    'safe': safe,
                    'result': result
                })
                
                # Display result
                if safe:
                    st.success("✅ Input passed protection checks")
                else:
                    st.error("❌ Input blocked by protection system")
                
                st.json(result)
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.header("📈 Demo Dashboard")
        
        # Real-time metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SPARSE Executions", len(st.session_state.execution_history))
        with col2:
            st.metric("Errors Tracked", len(st.session_state.error_history))
        with col3:
            st.metric("Protection Events", len(st.session_state.protection_events))
        
        # Recent activity
        st.subheader("Recent Activity")
        
        # SPARSE executions
        if st.session_state.execution_history:
            st.write("**Recent SPARSE Executions:**")
            for execution in st.session_state.execution_history[-5:]:
                st.write(f"- {execution['command']} ({execution['result']['execution_mode']}) - {execution['timestamp'].strftime('%H:%M:%S')}")
        
        # Protection events
        if st.session_state.protection_events:
            st.write("**Recent Protection Events:**")
            for event in st.session_state.protection_events[-5:]:
                status = "✅" if event['safe'] else "❌"
                st.write(f"- {status} {event['input'][:50]}... - {event['timestamp'].strftime('%H:%M:%S')}")
    
    def render_sidebar(self):
        """Render the sidebar with navigation and controls"""
        st.sidebar.title("🎯 Demo Controls")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Choose Demo Section",
            ["Dashboard", "SPARSE Agreements", "Error Tracking", "Demo Protection", "System Status"]
        )
        
        # System controls
        st.sidebar.subheader("System Controls")
        
        if st.sidebar.button("🔄 Refresh All Systems"):
            st.rerun()
        
        if st.sidebar.button("🧹 Clear History"):
            st.session_state.execution_history = []
            st.session_state.error_history = []
            st.session_state.protection_events = []
            st.success("History cleared!")
        
        # Export data
        if st.sidebar.button("📊 Export Demo Data"):
            export_data = {
                'execution_history': st.session_state.execution_history,
                'error_history': st.session_state.error_history,
                'protection_events': st.session_state.protection_events,
                'timestamp': datetime.now().isoformat()
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"demo_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        return page
    
    def run(self):
        """Run the visual demo interface"""
        self.render_header()
        self.render_mode_indicator()
        
        # Get current page from sidebar
        page = self.render_sidebar()
        
        # Render appropriate page
        if page == "Dashboard":
            self.render_dashboard()
        elif page == "SPARSE Agreements":
            self.render_sparse_agreements_interface()
        elif page == "Error Tracking":
            self.render_error_tracking_interface()
        elif page == "Demo Protection":
            self.render_protection_interface()
        elif page == "System Status":
            self.render_system_status()
    
    def render_system_status(self):
        """Render detailed system status"""
        st.header("🔧 System Status")
        
        # SPARSE system status
        st.subheader("SPARSE Agreements System")
        agreements = self.sparse_manager.get_available_agreements()
        st.write(f"Available agreements: {len(agreements)}")
        for agreement in agreements:
            st.write(f"- {agreement}")
        
        # Error tracking status
        st.subheader("Error Tracking System")
        summary = self.error_tracker.get_error_summary(hours=1)
        st.write(f"Errors in last hour: {summary['total_errors']}")
        
        # Protection system status
        st.subheader("Demo Protection System")
        status = self.protection_system.get_protection_status()
        st.write(f"Active rules: {status['active_rules']}/{status['total_rules']}")
        st.write(f"Recent events: {status['recent_events']}")

def main():
    """Main function to run the visual demo"""
    # Create and run the visual interface
    interface = VisualDemoInterface()
    interface.run()

if __name__ == "__main__":
    main()
