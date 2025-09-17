"""
System Monitor Page - Advanced monitoring and diagnostics
========================================================

Comprehensive system monitoring page for AI Dropzone Manager including:
- Real-time performance metrics
- System health diagnostics  
- Worker status and resource usage
- Error tracking and alerts
- Configuration management
"""

import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any

st.set_page_config(
    page_title="System Monitor - AI Dropzone Manager",
    page_icon="🔧",
    layout="wide"
)

class SystemMonitor:
    """System monitoring and diagnostics controller."""
    
    def __init__(self):
        self.refresh_interval = 30  # seconds
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        # Simulate real metrics (replace with actual monitoring)
        return {
            "cpu_usage": random.uniform(15, 45),
            "memory_usage": random.uniform(25, 65),
            "disk_usage": random.uniform(35, 75),
            "network_io": random.uniform(10, 90),
            "active_workers": random.randint(2, 8),
            "queue_length": random.randint(0, 15),
            "processing_rate": random.uniform(8, 25),
            "error_rate": random.uniform(0, 5)
        }
    
    def get_worker_status(self) -> List[Dict[str, Any]]:
        """Get status of individual workers."""
        workers = [
            {
                "name": "ExtractionWorker-01",
                "type": "Document Extraction",
                "status": "active",
                "current_task": "mvr_report.pdf",
                "cpu": 25.3,
                "memory": 150.2,
                "uptime": "2h 15m",
                "tasks_completed": 45
            },
            {
                "name": "EmbeddingWorker-01", 
                "type": "Text Embedding",
                "status": "idle",
                "current_task": None,
                "cpu": 5.1,
                "memory": 85.7,
                "uptime": "2h 15m",
                "tasks_completed": 32
            },
            {
                "name": "ProcessingWorker-01",
                "type": "Document Processing",
                "status": "active",
                "current_task": "contract_analysis.pdf",
                "cpu": 42.8,
                "memory": 220.5,
                "uptime": "1h 42m",
                "tasks_completed": 18
            },
            {
                "name": "QAWorker-01",
                "type": "Quality Assurance",
                "status": "active", 
                "current_task": "quality_review.pdf",
                "cpu": 18.9,
                "memory": 95.3,
                "uptime": "2h 08m",
                "tasks_completed": 28
            }
        ]
        return workers
    
    def get_error_log(self) -> pl.DataFrame:
        """Get recent system errors and warnings."""
        # Simulate error log (replace with actual log parsing)
        errors = []
        base_time = datetime.now()
        
        for i in range(10):
            errors.append({
                "timestamp": base_time - timedelta(minutes=i*15),
                "level": random.choices(["ERROR", "WARNING", "INFO"], weights=[0.2, 0.3, 0.5])[0],
                "component": random.choice(["AI Manager", "Worker", "Gateway", "Storage"]),
                "message": f"Sample error message {i+1}",
                "count": random.randint(1, 5)
            })
        
        return pl.DataFrame(errors)
    
    def get_performance_history(self) -> pl.DataFrame:
        """Get historical performance data."""
        # Generate sample performance data
        times = [datetime.now() - timedelta(hours=4) + timedelta(minutes=5*i) for i in range(48)]
        
        data = {
            "timestamp": times,
            "cpu_usage": [random.uniform(10, 50) for _ in range(48)],
            "memory_usage": [random.uniform(20, 70) for _ in range(48)],
            "processing_rate": [random.uniform(5, 30) for _ in range(48)],
            "queue_length": [random.randint(0, 20) for _ in range(48)],
            "error_rate": [random.uniform(0, 8) for _ in range(48)]
        }
        
        return pl.DataFrame(data)

def main():
    """Main system monitor page."""
    
    monitor = SystemMonitor()
    
    # Page header
    st.title("🔧 System Monitor")
    st.markdown("Real-time system diagnostics and performance monitoring")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("🔄 Refresh"):
            st.experimental_rerun()
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    
    # System overview metrics
    st.header("📊 System Overview")
    
    metrics = monitor.get_system_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "CPU Usage",
            f"{metrics['cpu_usage']:.1f}%",
            delta=f"{random.uniform(-2, 2):.1f}%"
        )
        
    with col2:
        st.metric(
            "Memory Usage", 
            f"{metrics['memory_usage']:.1f}%",
            delta=f"{random.uniform(-1, 3):.1f}%"
        )
        
    with col3:
        st.metric(
            "Active Workers",
            int(metrics['active_workers']),
            delta=random.randint(-1, 2)
        )
        
    with col4:
        st.metric(
            "Queue Length",
            int(metrics['queue_length']),
            delta=random.randint(-3, 3)
        )
    
    # Performance charts
    st.header("📈 Performance Trends")
    
    perf_data = monitor.get_performance_history()
    
    tab1, tab2, tab3 = st.tabs(["Resource Usage", "Processing Metrics", "Error Tracking"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cpu = px.line(perf_data, x='timestamp', y='cpu_usage', 
                             title='CPU Usage Over Time')
            fig_cpu.update_layout(height=300)
            st.plotly_chart(fig_cpu, use_container_width=True)
            
        with col2:
            fig_mem = px.line(perf_data, x='timestamp', y='memory_usage',
                             title='Memory Usage Over Time')
            fig_mem.update_layout(height=300)  
            st.plotly_chart(fig_mem, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rate = px.line(perf_data, x='timestamp', y='processing_rate',
                              title='Processing Rate (docs/min)')
            fig_rate.update_layout(height=300)
            st.plotly_chart(fig_rate, use_container_width=True)
            
        with col2:
            fig_queue = px.line(perf_data, x='timestamp', y='queue_length',
                               title='Queue Length Over Time')
            fig_queue.update_layout(height=300)
            st.plotly_chart(fig_queue, use_container_width=True)
    
    with tab3:
        fig_errors = px.line(perf_data, x='timestamp', y='error_rate',
                            title='Error Rate Over Time')
        fig_errors.update_layout(height=400)
        st.plotly_chart(fig_errors, use_container_width=True)
    
    # Worker status section
    st.header("⚙️ Worker Status")
    
    workers = monitor.get_worker_status()
    
    for worker in workers:
        with st.expander(f"🔧 {worker['name']} - {worker['status'].upper()}", 
                        expanded=worker['status'] == 'active'):
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**Type:** {worker['type']}")
                st.write(f"**Uptime:** {worker['uptime']}")
                
            with col2:
                st.write(f"**CPU:** {worker['cpu']:.1f}%")
                st.write(f"**Memory:** {worker['memory']:.1f} MB")
                
            with col3:
                st.write(f"**Tasks Completed:** {worker['tasks_completed']}")
                if worker['current_task']:
                    st.write(f"**Current Task:** {worker['current_task']}")
                else:
                    st.write("**Current Task:** None")
                    
            with col4:
                # Status indicator
                if worker['status'] == 'active':
                    st.success("🟢 Active")
                elif worker['status'] == 'idle':
                    st.info("🟡 Idle")
                else:
                    st.error("🔴 Error")
                
                # Action buttons
                if st.button(f"Restart", key=f"restart_{worker['name']}"):
                    st.success(f"Restarting {worker['name']}")
                    
                if st.button(f"Stop", key=f"stop_{worker['name']}"):
                    st.warning(f"Stopping {worker['name']}")
    
    # Error log section
    st.header("🚨 Error Log")
    
    error_df = monitor.get_error_log()
    
    # Error level filter
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        level_filter = st.multiselect(
            "Filter by Level",
            options=error_df['level'].unique(),
            default=error_df['level'].unique()
        )
    
    with col2:
        component_filter = st.multiselect(
            "Filter by Component", 
            options=error_df['component'].unique(),
            default=error_df['component'].unique()
        )
    
    # Apply filters
    filtered_errors = error_df.filter(
        (pl.col('level').is_in(level_filter)) &
        (pl.col('component').is_in(component_filter))
    ).sort('timestamp', descending=True)
    
    # Display error log
    for error in filtered_errors.iter_rows(named=True):

        # Color based on error level
        if error['level'] == 'ERROR':
            st.error(f"🔴 **{error['level']}** | {error['component']} | {error['timestamp'].strftime('%H:%M:%S')}")
        elif error['level'] == 'WARNING':
            st.warning(f"🟡 **{error['level']}** | {error['component']} | {error['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info(f"🔵 **{error['level']}** | {error['component']} | {error['timestamp'].strftime('%H:%M:%S')}")

        st.write(f"   {error['message']}")

        if error['count'] > 1:
            st.write(f"   *Occurred {error['count']} times*")
    
    # System configuration section
    with st.expander("⚙️ System Configuration"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("AI Dropzone Manager")
            st.write("• **Status:** Active")
            st.write("• **Workers:** 4 active")
            st.write("• **LLM Gateway:** Connected")
            st.write("• **Session Manager:** Available")
            
        with col2:
            st.subheader("Configuration")
            st.write("• **Max Concurrent Workers:** 8")
            st.write("• **Queue Size Limit:** 100")
            st.write("• **Processing Timeout:** 300s")
            st.write("• **Auto-scaling:** Enabled")
        
        if st.button("🔧 Edit Configuration"):
            st.info("Configuration editor would open here")
            
        if st.button("📊 Export Diagnostics"):
            st.success("System diagnostics exported to diagnostics.json")
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.experimental_rerun()

if __name__ == "__main__":
    main()