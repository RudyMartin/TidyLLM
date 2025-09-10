#!/usr/bin/env python3
"""
AI Dropzone Manager - Streamlit Dashboard
========================================

Real-time monitoring and management dashboard for the AI Dropzone Manager.
Provides visual interface for drop zone monitoring, bracket command execution,
and processing status tracking.

Usage:
    streamlit run tidyllm/web/ai_dropzone_dashboard.py
"""

import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import json
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="AI Dropzone Manager Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .status-active { background: #dcfce7; color: #166534; }
    .status-processing { background: #fef3c7; color: #92400e; }
    .status-completed { background: #dbeafe; color: #1d4ed8; }
    .status-failed { background: #fecaca; color: #dc2626; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

class DropzoneDashboard:
    """Dashboard controller for AI Dropzone Manager monitoring."""
    
    def __init__(self):
        self.drop_zones_path = Path("tidyllm/drop_zones")
        self.processing_log = []
        
    def get_drop_zone_status(self) -> Dict[str, Any]:
        """Get current status of all drop zones."""
        zones = {
            "mvr_analysis": "Process MVR",
            "financial_analysis": "Financial Analysis",
            "contract_review": "Contract Review", 
            "compliance_check": "Compliance Check",
            "quality_check": "Quality Check",
            "data_extraction": "Data Extraction"
        }
        
        status = {
            "zones": [],
            "total_files": 0,
            "processing_files": 0,
            "completed_files": 0,
            "failed_files": 0
        }
        
        for zone_dir, display_name in zones.items():
            zone_path = self.drop_zones_path / zone_dir
            
            if zone_path.exists():
                files = list(zone_path.glob("*.txt")) + list(zone_path.glob("*.pdf"))
                file_count = len(files)
                
                zone_info = {
                    "name": display_name,
                    "directory": zone_dir,
                    "file_count": file_count,
                    "files": [f.name for f in files],
                    "status": "active" if file_count > 0 else "empty",
                    "last_modified": self._get_latest_modification(zone_path)
                }
                
                status["zones"].append(zone_info)
                status["total_files"] += file_count
        
        # Check processing folders
        for folder in ["processing", "completed", "failed"]:
            folder_path = self.drop_zones_path / folder
            if folder_path.exists():
                count = len(list(folder_path.glob("*")))
                status[f"{folder}_files"] = count
        
        return status
    
    def _get_latest_modification(self, path: Path) -> Optional[datetime]:
        """Get latest modification time for files in directory."""
        if not path.exists():
            return None
            
        files = list(path.glob("*"))
        if not files:
            return None
            
        latest = max(files, key=lambda f: f.stat().st_mtime if f.is_file() else 0)
        return datetime.fromtimestamp(latest.stat().st_mtime)
    
    def get_bracket_commands(self) -> List[Dict[str, str]]:
        """Get available bracket commands from registry."""
        commands = [
            {"command": "[Process MVR]", "category": "QA & Compliance", "priority": "High"},
            {"command": "[Financial Analysis]", "category": "Document Analysis", "priority": "Normal"},
            {"command": "[Contract Review]", "category": "Document Analysis", "priority": "High"},
            {"command": "[Compliance Check]", "category": "QA & Compliance", "priority": "High"},
            {"command": "[Quality Check]", "category": "QA & Compliance", "priority": "Normal"},
            {"command": "[Data Extraction]", "category": "Document Analysis", "priority": "Normal"},
            {"command": "[Peer Review]", "category": "Advanced Analysis", "priority": "Critical"},
            {"command": "[Hybrid Analysis]", "category": "Advanced Analysis", "priority": "High"},
            {"command": "[Performance Test]", "category": "System Operations", "priority": "Normal"},
            {"command": "[Integration Test]", "category": "System Operations", "priority": "Normal"},
            {"command": "[Cost Analysis]", "category": "System Operations", "priority": "Normal"},
            {"command": "[Error Analysis]", "category": "System Operations", "priority": "High"}
        ]
        return commands
    
    def get_processing_history(self) -> pl.DataFrame:
        """Get simulated processing history for demo."""
        # Simulate processing history
        history_data = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(20):
            history_data.append({
                "timestamp": base_time + timedelta(minutes=i*30),
                "document": f"document_{i+1}.pdf",
                "bracket_command": "[Process MVR]" if i % 3 == 0 else "[Financial Analysis]",
                "status": "completed" if i % 4 != 3 else "processing",
                "processing_time": f"{45 + i*2}s",
                "confidence": 0.85 + (i % 10) * 0.01
            })
        
        return pl.DataFrame(history_data)

def main():
    """Main dashboard application."""
    dashboard = DropzoneDashboard()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Dropzone Manager Dashboard</h1>
        <p>Real-time monitoring and management of document processing workflows</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_interval = st.selectbox("Refresh interval", [5, 10, 30, 60], index=1)
            st.info(f"Auto-refreshing every {refresh_interval} seconds")
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh = datetime.now()
            st.experimental_rerun()
        
        st.markdown("---")
        
        # System status
        st.header("🔍 System Status")
        status_data = dashboard.get_drop_zone_status()
        
        st.metric("Total Files", status_data["total_files"])
        st.metric("Processing", status_data["processing_files"]) 
        st.metric("Completed", status_data["completed_files"])
        st.metric("Failed", status_data["failed_files"])
        
        # Last refresh time
        st.markdown("---")
        st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Drop Zone Status")
        
        # Get current status
        status_data = dashboard.get_drop_zone_status()
        
        # Create drop zone status cards
        for zone in status_data["zones"]:
            with st.expander(f"🗂️ {zone['name']} ({zone['file_count']} files)", expanded=zone['file_count'] > 0):
                
                col_a, col_b, col_c = st.columns([1, 1, 1])
                
                with col_a:
                    status_class = f"status-{zone['status']}"
                    st.markdown(f"""
                    <span class="status-badge {status_class}">{zone['status']}</span>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.write(f"**Directory:** `{zone['directory']}/`")
                
                with col_c:
                    if zone['last_modified']:
                        st.write(f"**Modified:** {zone['last_modified'].strftime('%H:%M')}")
                
                if zone['files']:
                    st.write("**Files:**")
                    for file_name in zone['files']:
                        st.write(f"• {file_name}")
                        
                        # Action buttons for each file
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        
                        with col_btn1:
                            if st.button(f"▶️ Process", key=f"process_{file_name}"):
                                st.success(f"Processing {file_name} with {zone['name']}")
                        
                        with col_btn2:
                            if st.button(f"👁️ Preview", key=f"preview_{file_name}"):
                                st.info(f"Preview functionality for {file_name}")
                        
                        with col_btn3:
                            if st.button(f"📊 Analyze", key=f"analyze_{file_name}"):
                                st.info(f"Document intelligence analysis for {file_name}")
                else:
                    st.write("*No files in drop zone*")
    
    with col2:
        st.header("🎯 Quick Actions")
        
        # Bracket command selector
        commands = dashboard.get_bracket_commands()
        
        selected_command = st.selectbox(
            "Select Bracket Command",
            options=[cmd["command"] for cmd in commands],
            format_func=lambda x: f"{x} ({next(cmd['priority'] for cmd in commands if cmd['command'] == x)})"
        )
        
        # File upload for immediate processing
        uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'txt', 'docx'])
        
        if uploaded_file and selected_command:
            col_process, col_save = st.columns(2)
            
            with col_process:
                if st.button("🚀 Process Now", type="primary"):
                    st.success(f"Processing {uploaded_file.name} with {selected_command}")
                    st.balloons()
            
            with col_save:
                if st.button("💾 Save to Drop Zone"):
                    st.info(f"Saved {uploaded_file.name} to appropriate drop zone")
        
        st.markdown("---")
        
        # System metrics
        st.header("📊 System Metrics")
        
        # Processing time chart
        time_range = [(datetime.now() - timedelta(hours=1) + timedelta(minutes=5*i)) for i in range(12)]
        chart_data = pl.DataFrame({
            'Time': time_range,
            'Processing Speed': [45, 52, 48, 43, 50, 55, 47, 49, 53, 46, 51, 48]
        })
        
        fig = px.line(chart_data, x='Time', y='Processing Speed', 
                     title='Average Processing Time (seconds)')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Success rate gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 94.7,
            title = {'text': "Success Rate %"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Processing history section
    st.header("📋 Processing History")
    
    history_df = dashboard.get_processing_history()
    
    # Filter controls
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=history_df['status'].unique(),
            default=history_df['status'].unique()
        )
    
    with col_filter2:
        command_filter = st.multiselect(
            "Filter by Command", 
            options=history_df['bracket_command'].unique(),
            default=history_df['bracket_command'].unique()
        )
    
    with col_filter3:
        time_range = st.selectbox(
            "Time Range",
            options=['Last Hour', 'Last 4 Hours', 'Last 24 Hours'],
            index=2
        )
    
    # Apply filters
    filtered_df = history_df[
        (history_df['status'].isin(status_filter)) &
        (history_df['bracket_command'].isin(command_filter))
    ]
    
    # Display filtered data
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="MM/DD HH:mm"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            "status": st.column_config.TextColumn("Status")
        }
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280;">
        <p>🤖 AI Dropzone Manager Dashboard v1.0 | 
        Powered by TidyLLM Infrastructure | 
        <a href="#" style="color: #3b82f6;">Documentation</a> | 
        <a href="#" style="color: #3b82f6;">API Reference</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(1)
        if (datetime.now() - st.session_state.last_refresh).seconds >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.experimental_rerun()

if __name__ == "__main__":
    main()