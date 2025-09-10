"""
TidyLLM Onboarding Dashboard Page
=================================

Real-time monitoring and analytics dashboard page.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

def render_dashboard_page():
    """Render the dashboard page."""
    
    st.markdown('<div class="section-header">📊 Dashboard - System Monitoring</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Real-time monitoring and analytics:
    - **System Health**: Connection status and performance metrics
    - **Usage Analytics**: AI model usage and cost tracking
    - **Performance Metrics**: Response times and throughput
    - **Resource Utilization**: System resource monitoring
    """)
    
    # System overview
    render_system_overview()
    
    # Performance metrics
    render_performance_metrics()
    
    # Usage analytics
    render_usage_analytics()
    
    # Resource monitoring
    render_resource_monitoring()

def render_system_overview():
    """Render system overview section."""
    st.subheader("System Overview")
    
    # System status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="System Status",
            value="🟢 Online",
            delta="Stable"
        )
    
    with col2:
        st.metric(
            label="Active Users",
            value="12",
            delta="3"
        )
    
    with col3:
        st.metric(
            label="Requests Today",
            value="1,234",
            delta="156"
        )
    
    with col4:
        st.metric(
            label="Uptime",
            value="99.9%",
            delta="0.1%"
        )
    
    # Service status
    st.subheader("Service Status")
    
    services = [
        {"name": "AWS S3", "status": "🟢 Online", "latency": "45ms"},
        {"name": "AWS Bedrock", "status": "🟢 Online", "latency": "1.2s"},
        {"name": "PostgreSQL", "status": "🟢 Online", "latency": "23ms"},
        {"name": "CorporateLLMGateway", "status": "🟢 Online", "latency": "67ms"},
        {"name": "AIProcessingGateway", "status": "🟢 Online", "latency": "1.5s"},
        {"name": "DatabaseGateway", "status": "🟢 Online", "latency": "34ms"},
        {"name": "WorkflowOptimizerGateway", "status": "🟢 Online", "latency": "89ms"}
    ]
    
    for service in services:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**{service['name']}**")
        
        with col2:
            st.write(service['status'])
        
        with col3:
            st.write(f"Latency: {service['latency']}")

def render_performance_metrics():
    """Render performance metrics section."""
    st.subheader("Performance Metrics")
    
    # Generate sample data
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
    
    # Response time chart
    response_times = [1.2 + 0.3 * (i % 24) / 24 + 0.1 * (i % 7) for i in range(len(dates))]
    
    fig_response = px.line(
        x=dates,
        y=response_times,
        title="Response Time Over Time",
        labels={'x': 'Time', 'y': 'Response Time (seconds)'}
    )
    
    st.plotly_chart(fig_response, use_container_width=True)
    
    # Throughput chart
    throughput = [50 + 20 * (i % 24) / 24 + 10 * (i % 7) for i in range(len(dates))]
    
    fig_throughput = px.bar(
        x=dates,
        y=throughput,
        title="Requests Per Hour",
        labels={'x': 'Time', 'y': 'Requests'}
    )
    
    st.plotly_chart(fig_throughput, use_container_width=True)
    
    # Performance summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Response Time", "1.2s", "0.1s")
    
    with col2:
        st.metric("Peak Throughput", "89 req/min", "12")
    
    with col3:
        st.metric("Error Rate", "0.1%", "0.05%")
    
    with col4:
        st.metric("Success Rate", "99.9%", "0.1%")

def render_usage_analytics():
    """Render usage analytics section."""
    st.subheader("Usage Analytics")
    
    # AI model usage
    model_usage = {
        'Model': ['Claude-3-Sonnet', 'Claude-3-Haiku', 'GPT-4', 'GPT-3.5-Turbo'],
        'Requests': [450, 320, 180, 95],
        'Cost': [12.50, 8.20, 15.30, 4.10]
    }
    
    df_usage = pd.DataFrame(model_usage)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_requests = px.pie(
            df_usage,
            values='Requests',
            names='Model',
            title="Request Distribution by Model"
        )
        st.plotly_chart(fig_requests, use_container_width=True)
    
    with col2:
        fig_cost = px.bar(
            df_usage,
            x='Model',
            y='Cost',
            title="Cost by Model"
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Usage summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", "1,045", "156")
    
    with col2:
        st.metric("Total Cost", "$40.10", "$5.20")
    
    with col3:
        st.metric("Avg Cost/Request", "$0.038", "$0.005")
    
    with col4:
        st.metric("Most Used Model", "Claude-3-Sonnet", "43%")

def render_resource_monitoring():
    """Render resource monitoring section."""
    st.subheader("Resource Monitoring")
    
    # CPU and Memory usage
    time_points = pd.date_range(start=datetime.now() - timedelta(hours=1), end=datetime.now(), freq='5min')
    
    cpu_usage = [45 + 10 * (i % 12) / 12 + 5 * (i % 6) for i in range(len(time_points))]
    memory_usage = [60 + 15 * (i % 12) / 12 + 8 * (i % 6) for i in range(len(time_points))]
    
    fig_resources = go.Figure()
    
    fig_resources.add_trace(go.Scatter(
        x=time_points,
        y=cpu_usage,
        mode='lines',
        name='CPU Usage (%)',
        line=dict(color='blue')
    ))
    
    fig_resources.add_trace(go.Scatter(
        x=time_points,
        y=memory_usage,
        mode='lines',
        name='Memory Usage (%)',
        line=dict(color='red')
    ))
    
    fig_resources.update_layout(
        title="Resource Utilization",
        xaxis_title="Time",
        yaxis_title="Usage (%)",
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig_resources, use_container_width=True)
    
    # Storage usage
    storage_data = {
        'Component': ['S3 Buckets', 'PostgreSQL', 'Vector DB', 'Logs', 'Cache'],
        'Usage (GB)': [125.5, 45.2, 78.9, 12.3, 8.7],
        'Capacity (GB)': [500, 100, 200, 50, 20]
    }
    
    df_storage = pd.DataFrame(storage_data)
    df_storage['Usage %'] = (df_storage['Usage (GB)'] / df_storage['Capacity (GB)'] * 100).round(1)
    
    st.subheader("Storage Usage")
    
    for _, row in df_storage.iterrows():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.write(f"**{row['Component']}**")
        
        with col2:
            st.write(f"{row['Usage (GB)']:.1f} GB")
        
        with col3:
            st.write(f"{row['Capacity (GB)']:.0f} GB")
        
        with col4:
            usage_pct = row['Usage %']
            if usage_pct > 80:
                st.error(f"{usage_pct:.1f}%")
            elif usage_pct > 60:
                st.warning(f"{usage_pct:.1f}%")
            else:
                st.success(f"{usage_pct:.1f}%")
    
    # Resource summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", "45%", "5%")
    
    with col2:
        st.metric("Memory Usage", "60%", "8%")
    
    with col3:
        st.metric("Storage Used", "270.6 GB", "15.2 GB")
    
    with col4:
        st.metric("Network I/O", "125 MB/s", "12 MB/s")
