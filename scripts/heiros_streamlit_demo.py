#!/usr/bin/env python3
"""
TidyLLM-HeirOS Configuration & Management Dashboard
Streamlit web interface for workflow management, SPARSE agreements, and system monitoring
"""

import streamlit as st
import pandas as pd
import json
import yaml
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import TidyLLM database integration
try:
    from tidyllm.tidyllm.database_integration import DatabaseManager, DatabaseConfig
    TIDYLLM_DB_AVAILABLE = True
except ImportError:
    # Fallback to direct psycopg2 if TidyLLM not available
    import psycopg2
    from psycopg2.extras import RealDictCursor
    TIDYLLM_DB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="HeirOS Control Center",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection using TidyLLM
@st.cache_resource
def get_database_manager():
    """Get TidyLLM DatabaseManager with caching"""
    try:
        if TIDYLLM_DB_AVAILABLE:
            # Use TidyLLM DatabaseManager
            config = DatabaseConfig(
                host='vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com',
                port=5432,
                database='vectorqa',
                username='vectorqa_user',
                password='REMOVED_PASSWORD',
                ssl_mode='require'
            )
            return DatabaseManager(config)
        else:
            # Fallback to direct psycopg2
            conn = psycopg2.connect(
                host='vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com',
                port=5432,
                database='vectorqa',
                user='vectorqa_user', 
                password='REMOVED_PASSWORD',
                sslmode='require'
            )
            return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# Load database queries
def load_query(query_name: str) -> str:
    """Load SQL query from file"""
    query_files = {
        'workflow_performance': """
            SELECT 
                w.workflow_id,
                w.name,
                w.status as workflow_status,
                w.compliance_level,
                COUNT(e.execution_id) as total_executions,
                COUNT(CASE WHEN e.status = 'success' THEN 1 END) as successful_executions,
                COUNT(CASE WHEN e.status = 'failure' THEN 1 END) as failed_executions,
                ROUND(AVG(e.duration_ms), 2) as avg_execution_time_ms,
                ROUND(AVG(e.nodes_executed), 1) as avg_nodes_per_execution,
                MAX(e.execution_date) as last_execution,
                ROUND(
                    (COUNT(CASE WHEN e.status = 'success' THEN 1 END)::decimal / 
                     NULLIF(COUNT(e.execution_id), 0)) * 100, 2
                ) as success_rate_percent
            FROM heiros_workflows w
            LEFT JOIN heiros_executions e ON w.workflow_id = e.workflow_id
            WHERE w.status = 'active'
            GROUP BY w.workflow_id, w.name, w.status, w.compliance_level
            ORDER BY total_executions DESC;
        """,
        'system_health': """
            SELECT 
                COUNT(*) as total_executions,
                COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
                COUNT(CASE WHEN status = 'failure' THEN 1 END) as failed,
                COUNT(CASE WHEN status = 'running' THEN 1 END) as currently_running,
                AVG(duration_ms) as avg_duration,
                ROUND((COUNT(CASE WHEN status = 'success' THEN 1 END)::decimal / NULLIF(COUNT(*), 0)) * 100, 2) as success_rate
            FROM heiros_executions 
            WHERE execution_date >= NOW() - INTERVAL '24 hours';
        """,
        'recent_executions': """
            SELECT 
                e.execution_id,
                w.name as workflow_name,
                e.execution_date,
                e.status,
                e.duration_ms,
                e.executed_by,
                e.nodes_executed
            FROM heiros_executions e
            JOIN heiros_workflows w ON e.workflow_id = w.workflow_id
            ORDER BY e.execution_date DESC
            LIMIT 10;
        """,
        'pending_agreements': """
            SELECT 
                agreement_id,
                title,
                business_purpose,
                business_owner,
                technical_owner,
                risk_level,
                created_date,
                EXTRACT(DAYS FROM (NOW() - created_date)) as days_pending
            FROM heiros_sparse_agreements 
            WHERE status = 'pending'
            ORDER BY created_date;
        """
    }
    return query_files.get(query_name, "")

def execute_query(query: str, params: tuple = None) -> pd.DataFrame:
    """Execute SQL query and return DataFrame using TidyLLM DatabaseManager"""
    db_manager = get_database_manager()
    if not db_manager:
        return pd.DataFrame()
    
    try:
        if TIDYLLM_DB_AVAILABLE:
            # Use TidyLLM DatabaseManager
            with db_manager._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                return df
        else:
            # Fallback to direct psycopg2
            df = pd.read_sql_query(query, db_manager, params=params)
            db_manager.close()
            return df
    except Exception as e:
        st.error(f"Query execution failed: {e}")
        return pd.DataFrame()

# Main Dashboard
def main_dashboard():
    """Main dashboard with system overview"""
    st.title("🌲 HeirOS Control Center")
    st.markdown("**TidyLLM Hierarchical Workflow Management System**")
    
    # System health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    health_df = execute_query(load_query('system_health'))
    if not health_df.empty:
        row = health_df.iloc[0]
        
        with col1:
            st.metric("24h Executions", int(row['total_executions']))
        with col2:
            st.metric("Success Rate", f"{row['success_rate']:.1f}%")
        with col3:
            st.metric("Currently Running", int(row['currently_running']))
        with col4:
            st.metric("Avg Duration", f"{row['avg_duration']:.0f}ms")
    
    # Workflow performance chart
    st.subheader("📊 Workflow Performance")
    
    perf_df = execute_query(load_query('workflow_performance'))
    if not perf_df.empty:
        fig = px.bar(
            perf_df.head(10), 
            x='name', 
            y='total_executions',
            color='success_rate_percent',
            title="Top Workflows by Execution Count",
            color_continuous_scale='RdYlGn'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("🔄 Recent Activity")
    
    recent_df = execute_query(load_query('recent_executions'))
    if not recent_df.empty:
        # Format the dataframe for display
        display_df = recent_df.copy()
        display_df['execution_date'] = pd.to_datetime(display_df['execution_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['duration_ms'] = display_df['duration_ms'].fillna(0).astype(int)
        
        st.dataframe(
            display_df,
            column_config={
                "execution_id": st.column_config.TextColumn("Execution ID", width="small"),
                "workflow_name": st.column_config.TextColumn("Workflow"),
                "execution_date": st.column_config.TextColumn("Date"),
                "status": st.column_config.TextColumn("Status"),
                "duration_ms": st.column_config.NumberColumn("Duration (ms)"),
                "executed_by": st.column_config.TextColumn("User"),
                "nodes_executed": st.column_config.NumberColumn("Nodes")
            },
            use_container_width=True
        )

def workflow_manager():
    """Workflow management interface"""
    st.title("🔧 Workflow Manager")
    
    tab1, tab2, tab3 = st.tabs(["📋 Active Workflows", "➕ Create Workflow", "📊 Analytics"])
    
    with tab1:
        # List all workflows
        workflows_query = """
            SELECT 
                workflow_id,
                name,
                description,
                compliance_level,
                version,
                status,
                created_date,
                created_by,
                tags
            FROM heiros_workflows 
            ORDER BY created_date DESC;
        """
        
        workflows_df = execute_query(workflows_query)
        
        if not workflows_df.empty:
            for _, workflow in workflows_df.iterrows():
                with st.expander(f"🌟 {workflow['name']} (v{workflow['version']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Status:** {workflow['status']}")
                        st.write(f"**Compliance:** {workflow['compliance_level']}")
                        st.write(f"**Created:** {workflow['created_date']}")
                    
                    with col2:
                        st.write(f"**Created by:** {workflow['created_by']}")
                        st.write(f"**Description:** {workflow['description']}")
                        
                        if workflow['tags']:
                            try:
                                tags = json.loads(workflow['tags'])
                                st.write(f"**Tags:** {', '.join(tags)}")
                            except:
                                pass
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"▶️ Execute", key=f"exec_{workflow['workflow_id']}"):
                            st.info("Workflow execution would be triggered here")
                    with col2:
                        if st.button(f"📝 Edit", key=f"edit_{workflow['workflow_id']}"):
                            st.info("Workflow editor would open here")
                    with col3:
                        if workflow['status'] == 'active':
                            if st.button(f"⏸️ Pause", key=f"pause_{workflow['workflow_id']}"):
                                st.info("Workflow would be paused")
        else:
            st.info("No workflows found. Create your first workflow!")
    
    with tab2:
        # Workflow creation form
        st.subheader("Create New Workflow")
        
        with st.form("create_workflow"):
            name = st.text_input("Workflow Name", placeholder="Document Processing Pipeline")
            description = st.text_area("Description", placeholder="Automated document validation and analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                compliance_level = st.selectbox(
                    "Compliance Level",
                    ["full_transparency", "summary_only", "minimal", "regulatory"]
                )
            with col2:
                created_by = st.text_input("Created By", value="system_admin")
            
            tags_input = st.text_input("Tags (comma-separated)", placeholder="document, automation, compliance")
            
            # Workflow JSON structure
            st.subheader("Workflow Structure")
            workflow_json = st.text_area(
                "Workflow JSON",
                value="""{
    "root": {
        "node_id": "main_sequence",
        "type": "sequence",
        "name": "Main Processing Flow",
        "children": [
            {
                "node_id": "validate_input",
                "type": "action",
                "name": "Input Validation",
                "action": "validate_document"
            },
            {
                "node_id": "process_document",
                "type": "action", 
                "name": "Process Document",
                "action": "extract_metadata"
            }
        ]
    }
}""",
                height=200
            )
            
            submitted = st.form_submit_button("🚀 Create Workflow")
            
            if submitted:
                if name and workflow_json:
                    try:
                        # Validate JSON
                        json.loads(workflow_json)
                        
                        # Prepare tags
                        tags = [tag.strip() for tag in tags_input.split(',')] if tags_input else []
                        
                        # Insert workflow (mock - would use actual database insert)
                        st.success(f"✅ Workflow '{name}' created successfully!")
                        st.info("In production, this would insert into heiros_workflows table")
                        
                        # Show what would be inserted
                        st.json({
                            "name": name,
                            "description": description,
                            "compliance_level": compliance_level,
                            "workflow_json": json.loads(workflow_json),
                            "created_by": created_by,
                            "tags": tags
                        })
                        
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON in workflow structure")
                else:
                    st.error("❌ Please provide workflow name and JSON structure")
    
    with tab3:
        # Workflow analytics
        st.subheader("📊 Workflow Analytics")
        
        perf_df = execute_query(load_query('workflow_performance'))
        
        if not perf_df.empty:
            # Success rate distribution
            fig1 = px.histogram(
                perf_df, 
                x='success_rate_percent',
                nbins=20,
                title="Workflow Success Rate Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Execution time vs Success rate scatter
            fig2 = px.scatter(
                perf_df,
                x='avg_execution_time_ms',
                y='success_rate_percent',
                size='total_executions',
                hover_name='name',
                title="Execution Time vs Success Rate",
                labels={
                    'avg_execution_time_ms': 'Average Execution Time (ms)',
                    'success_rate_percent': 'Success Rate (%)'
                }
            )
            st.plotly_chart(fig2, use_container_width=True)

def sparse_agreements():
    """SPARSE agreements management"""
    st.title("📜 SPARSE Agreements")
    st.markdown("**Structured Pre-Approved Reasoning for Systematic Execution**")
    
    tab1, tab2, tab3 = st.tabs(["⏳ Pending Approvals", "✅ Active Agreements", "➕ Create Agreement"])
    
    with tab1:
        pending_df = execute_query(load_query('pending_agreements'))
        
        if not pending_df.empty:
            st.subheader(f"📋 {len(pending_df)} Pending Approvals")
            
            for _, agreement in pending_df.iterrows():
                with st.expander(f"🔍 {agreement['title']} ({agreement['risk_level']} risk)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Business Owner:** {agreement['business_owner']}")
                        st.write(f"**Technical Owner:** {agreement['technical_owner']}")
                        st.write(f"**Risk Level:** {agreement['risk_level']}")
                    
                    with col2:
                        st.write(f"**Created:** {agreement['created_date']}")
                        st.write(f"**Pending for:** {int(agreement['days_pending'])} days")
                    
                    st.write(f"**Purpose:** {agreement['business_purpose']}")
                    
                    # Approval actions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"✅ Approve", key=f"approve_{agreement['agreement_id']}"):
                            st.success("Agreement approved!")
                    with col2:
                        if st.button(f"❌ Reject", key=f"reject_{agreement['agreement_id']}"):
                            st.error("Agreement rejected!")
                    with col3:
                        if st.button(f"📝 Review", key=f"review_{agreement['agreement_id']}"):
                            st.info("Opening detailed review...")
        else:
            st.success("✨ No pending approvals!")
    
    with tab2:
        # Active agreements
        active_agreements_query = """
            SELECT 
                agreement_id,
                title,
                business_owner,
                risk_level,
                approved_date,
                expiry_date,
                execution_count,
                last_execution_date,
                CASE 
                    WHEN expiry_date IS NULL THEN 'No Expiry'
                    WHEN expiry_date < NOW() THEN 'EXPIRED'
                    WHEN expiry_date < NOW() + INTERVAL '30 days' THEN 'EXPIRING SOON'
                    ELSE 'ACTIVE'
                END as expiry_status
            FROM heiros_sparse_agreements 
            WHERE status = 'approved'
            ORDER BY execution_count DESC;
        """
        
        active_df = execute_query(active_agreements_query)
        
        if not active_df.empty:
            st.subheader(f"🟢 {len(active_df)} Active Agreements")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Executions", active_df['execution_count'].sum())
            with col2:
                expiring_count = (active_df['expiry_status'] == 'EXPIRING SOON').sum()
                st.metric("Expiring Soon", expiring_count)
            with col3:
                high_risk_count = (active_df['risk_level'] == 'high').sum()
                st.metric("High Risk", high_risk_count)
            with col4:
                avg_executions = active_df['execution_count'].mean()
                st.metric("Avg Executions", f"{avg_executions:.1f}")
            
            # Agreements table
            display_df = active_df.copy()
            display_df['approved_date'] = pd.to_datetime(display_df['approved_date']).dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                display_df,
                column_config={
                    "agreement_id": st.column_config.TextColumn("ID", width="small"),
                    "title": st.column_config.TextColumn("Title"),
                    "business_owner": st.column_config.TextColumn("Owner"),
                    "risk_level": st.column_config.TextColumn("Risk"),
                    "execution_count": st.column_config.NumberColumn("Executions"),
                    "expiry_status": st.column_config.TextColumn("Status")
                },
                use_container_width=True
            )
        else:
            st.info("No active agreements found")
    
    with tab3:
        # Create new agreement
        st.subheader("Create New SPARSE Agreement")
        
        with st.form("create_agreement"):
            title = st.text_input("Agreement Title", placeholder="Automated Document Classification")
            description = st.text_area("Description", placeholder="ML-based document type classification")
            business_purpose = st.text_area("Business Purpose", placeholder="Streamline document intake process")
            
            col1, col2 = st.columns(2)
            with col1:
                business_owner = st.text_input("Business Owner", placeholder="Risk Management Team")
                risk_level = st.selectbox("Risk Level", ["minimal", "low", "medium", "high", "critical"])
            with col2:
                technical_owner = st.text_input("Technical Owner", placeholder="AI Systems Team")
                expiry_days = st.number_input("Expiry (days)", min_value=0, value=365)
            
            # Conditions
            st.subheader("Conditions")
            conditions = st.text_area(
                "Agreement Conditions (JSON)",
                value="""[
    {
        "description": "Document size must be under 50MB",
        "condition_type": "context_check",
        "parameters": {"max_size": 52428800}
    }
]""",
                height=150
            )
            
            # Actions
            st.subheader("Approved Actions")
            actions = st.text_area(
                "Approved Actions (JSON)",
                value="""[
    {
        "name": "Classify Document Type",
        "action_type": "ml_classification",
        "parameters": {"model": "document_classifier_v2"}
    }
]""",
                height=150
            )
            
            submitted = st.form_submit_button("📝 Create Agreement")
            
            if submitted:
                if title and description:
                    try:
                        # Validate JSON
                        json.loads(conditions)
                        json.loads(actions)
                        
                        st.success(f"✅ SPARSE Agreement '{title}' created successfully!")
                        st.info("Agreement is now pending approval")
                        
                        # Show what would be created
                        st.json({
                            "title": title,
                            "description": description,
                            "business_purpose": business_purpose,
                            "business_owner": business_owner,
                            "technical_owner": technical_owner,
                            "risk_level": risk_level,
                            "conditions": json.loads(conditions),
                            "approved_actions": json.loads(actions)
                        })
                        
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON in conditions or actions")
                else:
                    st.error("❌ Please provide title and description")

def system_analytics():
    """System analytics and monitoring"""
    st.title("📊 System Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance Trends", "🔍 Error Analysis", "💾 Database Health"])
    
    with tab1:
        # Execution trends over time
        trends_query = """
            SELECT 
                DATE(execution_date) as execution_date,
                COUNT(*) as total_executions,
                COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
                COUNT(CASE WHEN status = 'failure' THEN 1 END) as failed,
                AVG(duration_ms) as avg_duration_ms
            FROM heiros_executions 
            WHERE execution_date >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(execution_date)
            ORDER BY execution_date;
        """
        
        trends_df = execute_query(trends_query)
        
        if not trends_df.empty:
            # Execution volume over time
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=trends_df['execution_date'],
                y=trends_df['total_executions'],
                mode='lines+markers',
                name='Total Executions',
                line=dict(color='blue')
            ))
            fig1.add_trace(go.Scatter(
                x=trends_df['execution_date'],
                y=trends_df['successful'],
                mode='lines+markers',
                name='Successful',
                line=dict(color='green')
            ))
            fig1.add_trace(go.Scatter(
                x=trends_df['execution_date'],
                y=trends_df['failed'],
                mode='lines+markers',
                name='Failed',
                line=dict(color='red')
            ))
            fig1.update_layout(title="Execution Trends (30 days)")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Average duration trend
            fig2 = px.line(
                trends_df,
                x='execution_date',
                y='avg_duration_ms',
                title="Average Execution Duration Trend"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Error analysis
        errors_query = """
            SELECT 
                error_message,
                COUNT(*) as occurrence_count,
                COUNT(DISTINCT workflow_id) as workflows_affected,
                MAX(execution_date) as last_occurrence
            FROM heiros_executions 
            WHERE status = 'failure'
              AND execution_date >= NOW() - INTERVAL '7 days'
              AND error_message IS NOT NULL
            GROUP BY error_message
            ORDER BY occurrence_count DESC
            LIMIT 10;
        """
        
        errors_df = execute_query(errors_query)
        
        if not errors_df.empty:
            st.subheader("🚨 Common Errors (Last 7 days)")
            
            for _, error in errors_df.iterrows():
                with st.expander(f"❌ {error['error_message'][:60]}... ({error['occurrence_count']} occurrences)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Occurrences:** {error['occurrence_count']}")
                        st.write(f"**Workflows Affected:** {error['workflows_affected']}")
                    with col2:
                        st.write(f"**Last Occurrence:** {error['last_occurrence']}")
                    
                    st.write(f"**Full Error:** {error['error_message']}")
        else:
            st.success("✨ No errors in the last 7 days!")
    
    with tab3:
        # Database health
        health_query = """
            SELECT 
                'heiros_workflows' as table_name,
                COUNT(*) as row_count,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_records
            FROM heiros_workflows
            UNION ALL
            SELECT 
                'heiros_executions' as table_name,
                COUNT(*) as row_count,
                COUNT(CASE WHEN status = 'success' THEN 1 END) as active_records
            FROM heiros_executions
            UNION ALL
            SELECT 
                'heiros_sparse_agreements' as table_name,
                COUNT(*) as row_count,
                COUNT(CASE WHEN status = 'approved' THEN 1 END) as active_records
            FROM heiros_sparse_agreements
            ORDER BY table_name;
        """
        
        health_df = execute_query(health_query)
        
        if not health_df.empty:
            st.subheader("🗄️ Database Tables Health")
            
            col1, col2, col3 = st.columns(3)
            for i, (_, row) in enumerate(health_df.iterrows()):
                with [col1, col2, col3][i]:
                    st.metric(
                        row['table_name'].replace('heiros_', '').title(),
                        f"{row['row_count']:,}",
                        f"{row['active_records']} active"
                    )
            
            # Table size visualization
            fig = px.bar(
                health_df,
                x='table_name',
                y='row_count',
                title="Table Row Counts"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def node_templates():
    """Node template library"""
    st.title("🧩 Node Template Library")
    
    templates_query = """
        SELECT 
            template_id,
            name,
            description,
            node_type,
            category,
            version,
            usage_count,
            created_date,
            is_public,
            created_by
        FROM heiros_node_templates 
        WHERE is_public = true
        ORDER BY usage_count DESC;
    """
    
    templates_df = execute_query(templates_query)
    
    if not templates_df.empty:
        # Template categories
        categories = templates_df['category'].unique()
        selected_category = st.selectbox("Filter by Category", ['All'] + list(categories))
        
        if selected_category != 'All':
            filtered_df = templates_df[templates_df['category'] == selected_category]
        else:
            filtered_df = templates_df
        
        # Template grid
        cols = st.columns(3)
        
        for i, (_, template) in enumerate(filtered_df.iterrows()):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"**🔧 {template['name']}**")
                    st.markdown(f"*{template['node_type']} | {template['category']}*")
                    st.markdown(f"{template['description']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Usage:** {template['usage_count']}")
                    with col2:
                        st.markdown(f"**v{template['version']}**")
                    
                    if st.button(f"📋 Use Template", key=f"use_{template['template_id']}"):
                        st.info(f"Template '{template['name']}' added to workflow")
                    
                    st.markdown("---")
    else:
        st.info("No templates found. Create your first template!")

# Sidebar navigation
def sidebar():
    """Application sidebar navigation"""
    st.sidebar.title("🌲 HeirOS")
    st.sidebar.markdown("**Control Center**")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "🏠 Dashboard",
            "🔧 Workflow Manager", 
            "📜 SPARSE Agreements",
            "📊 System Analytics",
            "🧩 Node Templates"
        ]
    )
    
    # Connection status
    db_manager = get_database_manager()
    if db_manager:
        if TIDYLLM_DB_AVAILABLE:
            st.sidebar.success("🟢 TidyLLM Database Connected")
        else:
            st.sidebar.success("🟢 Database Connected (psycopg2)")
    else:
        st.sidebar.error("🔴 Database Disconnected")
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Info**")
    st.sidebar.markdown("Version: 1.0.0")
    st.sidebar.markdown("Environment: Development")
    
    return page

# Main application
def main():
    """Main application entry point"""
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    
    # Sidebar navigation
    page = sidebar()
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        main_dashboard()
    elif page == "🔧 Workflow Manager":
        workflow_manager()
    elif page == "📜 SPARSE Agreements":
        sparse_agreements()
    elif page == "📊 System Analytics":
        system_analytics()
    elif page == "🧩 Node Templates":
        node_templates()

if __name__ == "__main__":
    main()