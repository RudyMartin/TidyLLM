#!/usr/bin/env python3
"""
MCP Hierarchical LLM Dashboard

A comprehensive Streamlit dashboard for monitoring and interacting with the
MCP (Model Context Protocol) hierarchical LLM system in VectorQA Sage.
"""

import streamlit as st
import sys
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Add the backend directory to the path (corrected for src structure)
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from backend.services.mcp_service import MCPService
except ImportError as e:
    st.error(f"Failed to import MCP service: {e}")
    st.info("Please ensure the backend services are properly configured.")
    st.stop()

def init_mcp_service():
    """Initialize the MCP service"""
    try:
        return MCPService()
    except Exception as e:
        st.error(f"Failed to initialize MCP service: {e}")
        return None

def display_system_status(mcp_service):
    """Display comprehensive system status"""
    st.subheader("🔧 System Status")
    
    try:
        status = mcp_service.get_system_status()
        
        # Overall status
        overall_status = status.get("overall_status", "unknown")
        status_color = {
            "healthy": "🟢",
            "degraded": "🟡", 
            "error": "🔴",
            "unknown": "⚪"
        }.get(overall_status, "⚪")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Overall Status",
                value=f"{status_color} {overall_status.title()}",
                delta=None
            )
        
        with col2:
            mcp_system = status.get("mcp_system", {})
            system_overview = mcp_system.get("system_overview", {})
            total_executions = system_overview.get("total_executions", 0)
            st.metric(
                label="Total Executions",
                value=total_executions,
                delta=None
            )
        
        with col3:
            success_rate = system_overview.get("success_rate", 0)
            st.metric(
                label="Success Rate",
                value=f"{success_rate:.1%}",
                delta=None
            )
        
        # Detailed status
        with st.expander("📊 Detailed System Status"):
            tab1, tab2, tab3 = st.tabs(["MCP System", "VectorQA Integration", "Configuration"])
            
            with tab1:
                st.json(status.get("mcp_system", {}))
            
            with tab2:
                st.json(status.get("vectorqa_integration", {}))
            
            with tab3:
                st.json(status.get("mcp_system", {}).get("configuration", {}))
                
    except Exception as e:
        st.error(f"Failed to get system status: {e}")

def display_analytics(mcp_service):
    """Display system analytics"""
    st.subheader("📈 Analytics Dashboard")
    
    try:
        analytics = mcp_service.get_analytics()
        
        # System analytics
        system_analytics = analytics.get("mcp_analytics", {}).get("system_analytics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_requests = system_analytics.get("total_requests", 0)
            st.metric("Total Requests", total_requests)
        
        with col2:
            success_rate = system_analytics.get("success_rate", 0)
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        with col3:
            avg_execution_time = system_analytics.get("average_execution_time", 0)
            st.metric("Avg Execution Time", f"{avg_execution_time:.2f}s")
        
        with col4:
            vectorqa_analytics = analytics.get("vectorqa_analytics", {})
            vector_integration_usage = analytics.get("combined_metrics", {}).get("vector_integration_usage", 0)
            st.metric("Vector Integration Usage", f"{vector_integration_usage:.1%}")
        
        # Request types chart
        request_types = system_analytics.get("request_types", {})
        if request_types:
            fig = px.pie(
                values=list(request_types.values()),
                names=list(request_types.keys()),
                title="Request Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance trends
        with st.expander("📊 Performance Trends"):
            # Create sample trend data (in real implementation, this would come from historical data)
            dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=7, freq='D')
            trend_data = pd.DataFrame({
                'Date': dates,
                'Success Rate': [0.85, 0.88, 0.92, 0.89, 0.91, 0.87, 0.90],
                'Execution Time (s)': [2.1, 1.9, 2.3, 2.0, 1.8, 2.2, 1.9],
                'Requests': [15, 18, 22, 19, 25, 21, 23]
            })
            
            fig = px.line(trend_data, x='Date', y=['Success Rate', 'Execution Time (s)', 'Requests'],
                         title="Performance Trends (Last 7 Days)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Worker analytics
        worker_analytics = analytics.get("mcp_analytics", {}).get("worker_analytics", {})
        if worker_analytics:
            with st.expander("🔧 Worker Performance"):
                worker_data = []
                for worker_name, worker_stats in worker_analytics.items():
                    if isinstance(worker_stats, dict):
                        worker_data.append({
                            'Worker': worker_name,
                            'Total Tasks': worker_stats.get('total_tasks', 0),
                            'Success Rate': worker_stats.get('success_rate', 0),
                            'Avg Duration': worker_stats.get('average_duration', 0)
                        })
                
                if worker_data:
                    df = pd.DataFrame(worker_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Worker performance chart
                    fig = px.bar(df, x='Worker', y=['Total Tasks', 'Success Rate'],
                                title="Worker Performance Overview",
                                barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                    
    except Exception as e:
        st.error(f"Failed to get analytics: {e}")

def qa_processing_interface(mcp_service):
    """QA Processing Interface"""
    st.subheader("❓ Question-Answering Interface")
    
    with st.form("qa_form"):
        question = st.text_area(
            "Enter your question:",
            placeholder="What is artificial intelligence?",
            height=100
        )
        
        context_documents = st.text_area(
            "Context documents (optional):",
            placeholder="Paste relevant documents or context here...",
            height=150
        )
        
        use_vector_search = st.checkbox("Use vector search", value=True)
        
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("🚀 Process with MCP")
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear")
        
        if clear_button:
            st.rerun()
    
    if submit_button and question:
        with st.spinner("Processing through MCP hierarchy..."):
            try:
                # Prepare context documents
                context_docs = None
                if context_documents.strip():
                    context_docs = [doc.strip() for doc in context_documents.split('\n\n') if doc.strip()]
                
                # Process request
                result = mcp_service.process_qa_request(
                    question, 
                    context_docs,
                    use_vector_search
                )
                
                if result["success"]:
                    st.success("✅ Processing completed!")
                    
                    # Display final response
                    st.subheader("🤖 Final Response")
                    st.write(result["final_response"])
                    
                    # Display execution details
                    with st.expander("📋 Execution Details"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Execution Plan:**")
                            st.json(result.get("execution_plan", {}))
                        
                        with col2:
                            st.write("**Execution Metadata:**")
                            st.json(result.get("execution_metadata", {}))
                    
                    # Display intermediate results
                    with st.expander("🔍 Intermediate Results"):
                        intermediate_results = result.get("intermediate_results", {})
                        for task_id, task_result in intermediate_results.items():
                            st.write(f"**Task {task_id}:**")
                            st.json(task_result)
                    
                    # Display VectorQA metadata
                    vectorqa_metadata = result.get("vectorqa_metadata", {})
                    if vectorqa_metadata:
                        with st.expander("🔗 VectorQA Integration Details"):
                            st.json(vectorqa_metadata)
                
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

def report_generation_interface(mcp_service):
    """Report Generation Interface"""
    st.subheader("📊 Report Generation Interface")
    
    with st.form("report_form"):
        topic = st.text_input(
            "Report topic:",
            placeholder="AI Trends in 2024"
        )
        
        data_sources = st.text_area(
            "Data sources (one per line):",
            placeholder="Enter data sources or documents...",
            height=150
        )
        
        report_type = st.selectbox(
            "Report type:",
            ["comprehensive", "executive_summary", "technical", "business"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("📝 Generate Report")
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear")
        
        if clear_button:
            st.rerun()
    
    if submit_button and topic:
        with st.spinner("Generating report through MCP hierarchy..."):
            try:
                # Prepare data sources
                sources = None
                if data_sources.strip():
                    sources = [source.strip() for source in data_sources.split('\n') if source.strip()]
                
                # Generate report
                result = mcp_service.generate_report(topic, sources, report_type)
                
                if result["success"]:
                    st.success("✅ Report generated successfully!")
                    
                    # Display report
                    st.subheader("📄 Generated Report")
                    st.write(result["final_response"])
                    
                    # Display report metadata
                    report_metadata = result.get("report_metadata", {})
                    if report_metadata:
                        with st.expander("📊 Report Metadata"):
                            st.json(report_metadata)
                
                else:
                    st.error(f"❌ Report generation failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error during report generation: {e}")

def document_analysis_interface(mcp_service):
    """Document Analysis Interface"""
    st.subheader("🔍 Document Analysis Interface")
    
    with st.form("analysis_form"):
        documents = st.text_area(
            "Documents to analyze:",
            placeholder="Paste documents to analyze...",
            height=200
        )
        
        analysis_type = st.selectbox(
            "Analysis type:",
            ["sentiment", "key_points", "compliance", "general", "technical", "financial"]
        )
        
        include_vector_analysis = st.checkbox("Include vector analysis", value=True)
        
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("🔍 Analyze Documents")
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear")
        
        if clear_button:
            st.rerun()
    
    if submit_button and documents:
        with st.spinner("Analyzing documents through MCP hierarchy..."):
            try:
                # Prepare documents
                docs = [doc.strip() for doc in documents.split('\n\n') if doc.strip()]
                
                # Analyze documents
                result = mcp_service.analyze_documents(docs, analysis_type, include_vector_analysis)
                
                if result["success"]:
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display analysis results
                    st.subheader("📊 Analysis Results")
                    st.write(result["final_response"])
                    
                    # Display analysis metadata
                    analysis_metadata = result.get("analysis_metadata", {})
                    if analysis_metadata:
                        with st.expander("📊 Analysis Metadata"):
                            st.json(analysis_metadata)
                
                else:
                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")

def system_management_interface(mcp_service):
    """System Management Interface"""
    st.subheader("⚙️ System Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset System"):
            try:
                mcp_service.reset_system()
                st.success("✅ System reset completed!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to reset system: {e}")
    
    with col3:
        if st.button("📊 Refresh Analytics"):
            st.rerun()
    
    # Configuration management
    with st.expander("⚙️ Configuration Management"):
        st.write("**Current Configuration:**")
        try:
            config = mcp_service.config
            st.json(config)
        except Exception as e:
            st.error(f"Failed to get configuration: {e}")
        
        # Configuration update interface
        st.write("**Update Configuration:**")
        new_config = st.text_area(
            "New configuration (JSON):",
            placeholder='{"planner_config": {"model": "gpt-4o"}}',
            height=150
        )
        
        if st.button("💾 Update Configuration"):
            try:
                if new_config.strip():
                    config_update = json.loads(new_config)
                    mcp_service.update_config(config_update)
                    st.success("✅ Configuration updated successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Please enter a valid JSON configuration.")
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format")
            except Exception as e:
                st.error(f"❌ Failed to update configuration: {e}")

def execution_history_interface(mcp_service):
    """Execution History Interface"""
    st.subheader("📜 Execution History")
    
    try:
        history = mcp_service.orchestrator.get_execution_history(limit=20)
        
        if history:
            # Create a DataFrame for better display
            history_data = []
            for record in history:
                history_data.append({
                    'Timestamp': record.get('timestamp', ''),
                    'Request': record.get('request', '')[:100] + '...' if len(record.get('request', '')) > 100 else record.get('request', ''),
                    'Success': record.get('success', False),
                    'Duration (s)': f"{record.get('execution_duration', 0):.2f}",
                    'Error': record.get('error', '')[:50] + '...' if record.get('error') and len(record.get('error', '')) > 50 else record.get('error', '')
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
            
            # Success rate over time
            if len(history) > 1:
                success_rates = []
                timestamps = []
                
                for i in range(1, min(len(history) + 1, 11)):  # Last 10 executions
                    recent_history = history[-i:]
                    success_count = sum(1 for record in recent_history if record.get('success', False))
                    success_rate = success_count / len(recent_history)
                    success_rates.append(success_rate)
                    timestamps.append(f"Last {i}")
                
                fig = px.line(
                    x=timestamps,
                    y=success_rates,
                    title="Success Rate Trend (Last 10 Executions)",
                    labels={'x': 'Execution Window', 'y': 'Success Rate'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No execution history available.")
            
    except Exception as e:
        st.error(f"Failed to get execution history: {e}")

def main():
    """Main dashboard function"""
    st.set_page_config(
        page_title="MCP Hierarchical LLM Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 MCP Hierarchical LLM Dashboard")
    st.markdown("**Model Context Protocol - Hierarchical LLM System for VectorQA Sage**")
    
    # Initialize MCP service
    mcp_service = init_mcp_service()
    if not mcp_service:
        st.error("Failed to initialize MCP service. Please check your configuration.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🎛️ MCP Controls")
    
    page = st.sidebar.selectbox(
        "Choose Operation",
        [
            "System Status",
            "Analytics Dashboard", 
            "QA Processing",
            "Report Generation",
            "Document Analysis",
            "System Management",
            "Execution History"
        ]
    )
    
    # Display selected page
    if page == "System Status":
        display_system_status(mcp_service)
        
    elif page == "Analytics Dashboard":
        display_analytics(mcp_service)
        
    elif page == "QA Processing":
        qa_processing_interface(mcp_service)
        
    elif page == "Report Generation":
        report_generation_interface(mcp_service)
        
    elif page == "Document Analysis":
        document_analysis_interface(mcp_service)
        
    elif page == "System Management":
        system_management_interface(mcp_service)
        
    elif page == "Execution History":
        execution_history_interface(mcp_service)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**MCP System Info:**")
    st.sidebar.markdown("- **Version:** 1.0.0")
    st.sidebar.markdown("- **Architecture:** Hierarchical LLM")
    st.sidebar.markdown("- **Framework:** Model Context Protocol")
    st.sidebar.markdown("- **Integration:** VectorQA Sage")
    
    # Quick status in sidebar
    try:
        status = mcp_service.get_system_status()
        overall_status = status.get("overall_status", "unknown")
        st.sidebar.markdown(f"**Status:** {overall_status.title()}")
    except:
        st.sidebar.markdown("**Status:** Unknown")

if __name__ == "__main__":
    main()
