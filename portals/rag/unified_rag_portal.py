#!/usr/bin/env python3
"""
Unified RAG Portal - Management Interface for All 5 RAG Systems
===============================================================

Streamlit portal for managing all 5 RAG systems through the UnifiedRAGManager:
1. AIPoweredRAGAdapter - AI-enhanced responses via CorporateLLMGateway
2. PostgresRAGAdapter - Authority-based precedence with SME infrastructure
3. JudgeRAGAdapter - External system integration with fallback
4. IntelligentRAGAdapter - Real content extraction with direct database
5. SMERAGSystem - Full document lifecycle with multi-model support

Features:
- System health monitoring and status dashboard
- Collection management across all RAG systems
- Unified query interface with system routing
- Performance metrics and optimization tracking
- CRUD operations for collections and documents
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import UnifiedRAGManager
try:
    from tidyllm.services.unified_rag_manager import (
        UnifiedRAGManager,
        RAGSystemType,
        UnifiedRAGQuery,
        RAGSystemInfo
    )
    MANAGER_AVAILABLE = True
except ImportError:
    MANAGER_AVAILABLE = False
    st.error("UnifiedRAGManager not available. Please check imports.")

# Configure Streamlit page
st.set_page_config(
    page_title="Unified RAG Portal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .system-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .system-card.healthy {
        border-color: #10b981;
        background: #f0fdf4;
    }
    .system-card.degraded {
        border-color: #f59e0b;
        background: #fffbeb;
    }
    .system-card.error {
        border-color: #ef4444;
        background: #fef2f2;
    }
    .system-card.unavailable {
        border-color: #6b7280;
        background: #f9fafb;
    }
    .capability-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 0.125rem;
    }
    .metric-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-message {
        background: #dcfce7;
        border: 1px solid #16a34a;
        color: #15803d;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-message {
        background: #fef2f2;
        border: 1px solid #dc2626;
        color: #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-healthy { color: #10b981; font-weight: bold; }
    .status-degraded { color: #f59e0b; font-weight: bold; }
    .status-error { color: #ef4444; font-weight: bold; }
    .status-unavailable { color: #6b7280; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_manager():
    """Get or create UnifiedRAGManager instance."""
    if MANAGER_AVAILABLE:
        return UnifiedRAGManager()
    return None

def main():
    """Main Unified RAG Portal application."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Unified RAG Portal</h1>
        <p>Comprehensive management interface for all 5 RAG systems</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if manager is available
    rag_manager = get_rag_manager()
    if not rag_manager:
        st.error("⚠️ UnifiedRAGManager not available. Please check your installation.")
        return

    # Sidebar navigation
    with st.sidebar:
        st.header("🚀 Navigation")

        page = st.selectbox(
            "Choose Section",
            [
                "System Dashboard",
                "Collection Manager",
                "Query Interface",
                "RAG2DAG Accelerator",
                "Health Monitor",
                "Performance Analytics"
            ]
        )

        st.markdown("---")

        # Quick stats
        if st.button("🔄 Refresh Stats"):
            st.cache_resource.clear()
            st.rerun()

        # System availability overview
        st.subheader("📊 Quick Stats")
        try:
            metrics = rag_manager.get_performance_metrics()
            st.metric("Total Systems", metrics["total_systems"])
            st.metric("Available", metrics["available_systems"])
            st.metric("Collections", metrics["total_collections"])
        except Exception as e:
            st.error(f"Error loading stats: {e}")

    # Main content based on selected page
    if page == "System Dashboard":
        render_system_dashboard(rag_manager)
    elif page == "Collection Manager":
        render_collection_manager(rag_manager)
    elif page == "Query Interface":
        render_query_interface(rag_manager)
    elif page == "RAG2DAG Accelerator":
        render_rag2dag_accelerator(rag_manager)
    elif page == "Health Monitor":
        render_health_monitor(rag_manager)
    elif page == "Performance Analytics":
        render_performance_analytics(rag_manager)

def render_system_dashboard(rag_manager):
    """Render the system dashboard page."""
    st.header("🎯 System Dashboard")

    st.markdown("""
    Overview of all 5 RAG systems with status, capabilities, and collection counts.
    """)

    # Get system status
    try:
        system_infos = rag_manager.get_system_status()
    except Exception as e:
        st.error(f"Error loading system status: {e}")
        return

    # System overview metrics
    col1, col2, col3, col4 = st.columns(4)

    healthy_count = sum(1 for info in system_infos if info.status == "healthy")
    total_collections = sum(info.collections_count for info in system_infos)
    avg_health_score = sum(info.health_score for info in system_infos) / len(system_infos)

    with col1:
        st.metric("Healthy Systems", f"{healthy_count}/{len(system_infos)}")
    with col2:
        st.metric("Total Collections", total_collections)
    with col3:
        st.metric("Avg Health Score", f"{avg_health_score:.2f}")
    with col4:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

    # System details
    st.subheader("📋 System Details")

    for info in system_infos:
        status_class = f"system-card {info.status}"

        st.markdown(f"""
        <div class="{status_class}">
            <h4>{info.name}</h4>
            <p><strong>Status:</strong> <span class="status-{info.status}">{info.status.upper()}</span></p>
            <p><strong>Collections:</strong> {info.collections_count}</p>
            <p><strong>Health Score:</strong> {info.health_score:.2f}/1.0</p>
            <p><strong>Description:</strong> {info.description}</p>
            <div>
                <strong>Capabilities:</strong><br>
                {' '.join([f'<span class="capability-badge">{cap}</span>' for cap in info.capabilities])}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action buttons for each system
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(f"📊 View Details", key=f"details_{info.system_type.value}"):
                show_system_details(rag_manager, info)

        with col2:
            if st.button(f"🔍 Test Query", key=f"test_{info.system_type.value}"):
                test_system_query(rag_manager, info.system_type)

        with col3:
            if st.button(f"📁 Manage Collections", key=f"collections_{info.system_type.value}"):
                st.session_state.selected_system = info.system_type
                st.session_state.page = "Collection Manager"
                st.rerun()

def render_collection_manager(rag_manager):
    """Render the collection manager page."""
    st.header("📁 Collection Manager")

    st.markdown("""
    Manage collections across all RAG systems. Create, view, and delete collections.
    """)

    # System selection
    system_type_names = {
        RAGSystemType.AI_POWERED: "AI-Powered RAG",
        RAGSystemType.POSTGRES: "Postgres RAG",
        RAGSystemType.JUDGE: "Judge RAG",
        RAGSystemType.INTELLIGENT: "Intelligent RAG",
        RAGSystemType.SME: "SME RAG System"
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 All Collections")

        # Get all collections
        try:
            all_collections = rag_manager.list_all_collections()
        except Exception as e:
            st.error(f"Error loading collections: {e}")
            return

        # Display collections by system
        for system_type, collections in all_collections.items():
            with st.expander(f"{system_type_names[system_type]} ({len(collections)} collections)", expanded=False):

                if collections:
                    # Create DataFrame for display
                    df_data = []
                    for collection in collections:
                        df_data.append({
                            "ID": collection.get("collection_id", "N/A"),
                            "Name": collection.get("collection_name", collection.get("name", "Unknown")),
                            "Description": collection.get("description", "No description")[:50] + "...",
                            "Documents": collection.get("document_count", 0),
                            "Created": collection.get("created_date", "Unknown")
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No collections found in this system.")

    with col2:
        st.subheader("➕ Create Collection")

        with st.form("create_collection_form"):
            selected_system = st.selectbox(
                "RAG System",
                options=list(system_type_names.keys()),
                format_func=lambda x: system_type_names[x]
            )

            collection_name = st.text_input("Collection Name *")
            collection_description = st.text_area("Description")

            # System-specific options
            if selected_system == RAGSystemType.POSTGRES:
                domain = st.text_input("Domain", value="general")
                authority_tier = st.selectbox("Authority Tier", [1, 2, 3], index=1)
                config = {"domain": domain, "authority_tier": authority_tier}

            elif selected_system == RAGSystemType.SME:
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-ada-002", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]
                )
                s3_bucket = st.text_input("S3 Bucket", value="dsai-2025-asu")
                tags = st.text_input("Tags (comma-separated)")
                config = {
                    "embedding_model": embedding_model,
                    "s3_bucket": s3_bucket,
                    "tags": tags.split(",") if tags else []
                }

            else:
                domain = st.text_input("Domain", value="general")
                config = {"domain": domain}

            submitted = st.form_submit_button("Create Collection")

            if submitted:
                if not collection_name:
                    st.error("Collection name is required!")
                else:
                    try:
                        result = rag_manager.create_collection(
                            system_type=selected_system,
                            name=collection_name,
                            description=collection_description,
                            config=config
                        )

                        if result.get("success"):
                            st.markdown(f"""
                            <div class="success-message">
                                <strong>SUCCESS!</strong> {result['message']}<br>
                                <strong>Collection ID:</strong> {result['collection_id']}
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown(f"""
                            <div class="error-message">
                                <strong>ERROR:</strong> {result['error']}
                            </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Failed to create collection: {e}")

def render_query_interface(rag_manager):
    """Render the unified query interface."""
    st.header("🔍 Unified Query Interface")

    st.markdown("""
    Query across all RAG systems with intelligent routing and response optimization.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Query Configuration")

        with st.form("unified_query_form"):
            query_text = st.text_area(
                "Query *",
                placeholder="Enter your question here...",
                height=100
            )

            col_config1, col_config2 = st.columns(2)

            with col_config1:
                domain = st.text_input("Domain", value="general")

                system_preference = st.selectbox(
                    "System Preference",
                    ["Auto (Best Match)", "AI-Powered", "Postgres", "Judge", "Intelligent", "SME"],
                    help="Choose specific system or let auto-routing find the best match"
                )

            with col_config2:
                confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
                max_results = st.number_input("Max Results", 1, 20, 5)

            # Advanced options
            with st.expander("Advanced Options"):
                authority_tier = st.selectbox("Authority Tier (for compliance queries)", [None, 1, 2, 3])
                collection_id = st.text_input("Specific Collection ID (optional)")

            submitted = st.form_submit_button("🚀 Execute Query", type="primary")

            if submitted:
                if not query_text:
                    st.error("Please enter a query!")
                else:
                    # Convert system preference to enum
                    system_type = None
                    if system_preference != "Auto (Best Match)":
                        system_mapping = {
                            "AI-Powered": RAGSystemType.AI_POWERED,
                            "Postgres": RAGSystemType.POSTGRES,
                            "Judge": RAGSystemType.JUDGE,
                            "Intelligent": RAGSystemType.INTELLIGENT,
                            "SME": RAGSystemType.SME
                        }
                        system_type = system_mapping[system_preference]

                    # Create unified query
                    unified_query = UnifiedRAGQuery(
                        query=query_text,
                        domain=domain,
                        system_type=system_type,
                        collection_id=collection_id if collection_id else None,
                        authority_tier=authority_tier,
                        confidence_threshold=confidence_threshold,
                        max_results=max_results
                    )

                    # Execute query
                    with st.spinner("Querying RAG systems..."):
                        try:
                            response = asyncio.run(rag_manager.query_unified(unified_query))
                            display_query_response(response)
                        except Exception as e:
                            st.error(f"Query failed: {e}")

    with col2:
        st.subheader("💡 Query Tips")

        st.markdown("""
        **System Strengths:**
        - **AI-Powered**: Best for analysis and synthesis
        - **Postgres**: Great for compliance and authority-based queries
        - **Judge**: External expert knowledge integration
        - **Intelligent**: Excellent for document content extraction
        - **SME**: Comprehensive document lifecycle management

        **Query Examples:**
        - "What are the compliance requirements for..."
        - "Analyze the financial risk in..."
        - "Extract key information from..."
        - "What best practices apply to..."

        **Tips:**
        - Use specific domains for better routing
        - Higher confidence thresholds = more precise results
        - Authority tier helps with regulatory queries
        """)

def render_health_monitor(rag_manager):
    """Render the health monitoring page."""
    st.header("🏥 Health Monitor")

    st.markdown("""
    Real-time health monitoring and diagnostics for all RAG systems.
    """)

    # Health check button
    if st.button("🔄 Run Health Check", type="primary"):
        with st.spinner("Running health checks..."):
            try:
                health_results = asyncio.run(rag_manager.health_check_all())
                st.session_state.health_results = health_results
                st.session_state.health_timestamp = datetime.now()
            except Exception as e:
                st.error(f"Health check failed: {e}")
                return

    # Display health results
    if hasattr(st.session_state, 'health_results'):
        st.subheader(f"📊 Health Results (Last updated: {st.session_state.health_timestamp.strftime('%H:%M:%S')})")

        # Overall health summary
        healthy_systems = sum(1 for result in st.session_state.health_results.values()
                            if result.get('status') == 'healthy')
        total_systems = len(st.session_state.health_results)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Healthy Systems", f"{healthy_systems}/{total_systems}")

        with col2:
            overall_health = healthy_systems / total_systems * 100
            st.metric("Overall Health", f"{overall_health:.1f}%")

        with col3:
            avg_response_time = sum(result.get('response_time_ms', 0)
                                  for result in st.session_state.health_results.values()) / total_systems
            st.metric("Avg Response Time", f"{avg_response_time:.0f}ms")

        # Individual system health
        for system_type, result in st.session_state.health_results.items():
            status = result.get('status', 'unknown')
            health_score = result.get('health_score', 0.0)
            response_time = result.get('response_time_ms', 0)

            with st.expander(f"🔍 {system_type.value.upper()} - {status.upper()}", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Status", status.upper())
                    st.metric("Health Score", f"{health_score:.2f}")

                with col2:
                    st.metric("Response Time", f"{response_time:.0f}ms")
                    collections = result.get('collections', 0)
                    st.metric("Collections", collections)

                with col3:
                    capabilities = result.get('capabilities', [])
                    st.write("**Capabilities:**")
                    for cap in capabilities:
                        st.write(f"• {cap}")

                # Show errors if any
                if 'error' in result:
                    st.error(f"Error: {result['error']}")

                # Show external status for Judge system
                if system_type == RAGSystemType.JUDGE and 'external_status' in result:
                    st.json(result['external_status'])

def render_performance_analytics(rag_manager):
    """Render performance analytics page."""
    st.header("📈 Performance Analytics")

    st.markdown("""
    Performance metrics, trends, and optimization insights across all RAG systems.
    """)

    # Get performance metrics
    try:
        metrics = rag_manager.get_performance_metrics()
    except Exception as e:
        st.error(f"Error loading performance metrics: {e}")
        return

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Systems", metrics["total_systems"])
    with col2:
        st.metric("Available Systems", metrics["available_systems"])
    with col3:
        st.metric("Total Collections", metrics["total_collections"])
    with col4:
        availability_pct = (metrics["available_systems"] / metrics["total_systems"]) * 100
        st.metric("Availability", f"{availability_pct:.1f}%")

    # System breakdown
    st.subheader("📊 System Breakdown")

    breakdown_data = []
    for system_name, system_data in metrics["system_breakdown"].items():
        breakdown_data.append({
            "System": system_name.replace("_", " ").title(),
            "Collections": system_data["collections"],
            "Status": system_data["status"]
        })

    df = pd.DataFrame(breakdown_data)

    # Collections chart
    fig_collections = px.bar(
        df,
        x="System",
        y="Collections",
        color="Status",
        title="Collections by RAG System"
    )
    st.plotly_chart(fig_collections, use_container_width=True)

    # Status distribution
    status_counts = df["Status"].value_counts()
    fig_status = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="System Status Distribution"
    )
    st.plotly_chart(fig_status, use_container_width=True)

    # Performance trends (mock data for demonstration)
    st.subheader("📈 Performance Trends")

    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
    trend_data = {
        "Date": dates,
        "Query Response Time": [200, 195, 210, 185, 175, 190, 180],
        "Success Rate": [99.5, 99.2, 99.8, 99.1, 99.6, 99.3, 99.7],
        "Active Collections": [15, 16, 16, 17, 18, 18, 19]
    }

    trend_df = pd.DataFrame(trend_data)

    # Response time trend
    fig_response = px.line(
        trend_df,
        x="Date",
        y="Query Response Time",
        title="Average Query Response Time (ms)"
    )
    st.plotly_chart(fig_response, use_container_width=True)

def display_query_response(response):
    """Display unified query response."""
    st.subheader("📋 Query Response")

    # Response metadata
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("System Used", response.system_type.value.replace("_", " ").title())
        st.metric("Confidence", f"{response.confidence:.2f}")

    with col2:
        st.metric("Authority Tier", response.authority_tier)
        st.metric("Processing Time", f"{response.processing_time_ms:.0f}ms")

    with col3:
        st.metric("Sources Found", len(response.sources))
        st.metric("Precedence Level", f"{response.precedence_level:.2f}")

    # Response content
    st.markdown("**Response:**")
    st.write(response.response)

    # Sources
    if response.sources:
        with st.expander(f"📚 Sources ({len(response.sources)})", expanded=False):
            for i, source in enumerate(response.sources):
                st.markdown(f"**Source {i+1}:**")
                st.json(source)

def show_system_details(rag_manager, info: RAGSystemInfo):
    """Show detailed information for a specific system."""
    st.modal(f"System Details: {info.name}")
    st.write(f"**System Type:** {info.system_type.value}")
    st.write(f"**Status:** {info.status}")
    st.write(f"**Collections:** {info.collections_count}")
    st.write(f"**Health Score:** {info.health_score:.2f}")
    st.write(f"**Description:** {info.description}")
    st.write(f"**Capabilities:** {', '.join(info.capabilities)}")

def test_system_query(rag_manager, system_type: RAGSystemType):
    """Test query for a specific system."""
    st.info(f"Testing query for {system_type.value} system...")

    test_query = UnifiedRAGQuery(
        query="What are the main capabilities of this system?",
        domain="test",
        system_type=system_type,
        confidence_threshold=0.5
    )

    try:
        response = asyncio.run(rag_manager.query_unified(test_query))
        st.success(f"Test successful! Confidence: {response.confidence:.2f}")
        st.write(response.response[:200] + "...")
    except Exception as e:
        st.error(f"Test failed: {e}")

def render_rag2dag_accelerator(rag_manager):
    """Render the RAG2DAG acceleration page."""
    st.header("⚡ RAG2DAG Accelerator")

    st.markdown("""
    RAG2DAG optimization converts linear RAG workflows into parallel DAG (Directed Acyclic Graph)
    workflows for improved performance and cost-effectiveness.
    """)

    # RAG2DAG availability check
    rag2dag_stats = rag_manager.get_rag2dag_stats()

    if not rag2dag_stats.get("available", False):
        st.error(f"⚠️ RAG2DAG service not available: {rag2dag_stats.get('reason', 'Unknown error')}")
        return

    # RAG2DAG statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Analyzed", rag2dag_stats.get("total_analyzed", 0))
    with col2:
        st.metric("Optimizations Applied", rag2dag_stats.get("optimizations_applied", 0))
    with col3:
        st.metric("Avg Speedup", f"{rag2dag_stats.get('avg_speedup', 0):.1f}x")
    with col4:
        st.metric("Optimization Rate", f"{rag2dag_stats.get('optimization_rate', 0):.1f}%")

    # Query optimization analysis
    st.subheader("🔍 Query Optimization Analysis")

    with st.form("rag2dag_analysis_form"):
        query_text = st.text_area(
            "Query to Analyze",
            placeholder="Enter your query to analyze RAG2DAG optimization potential...",
            height=100
        )

        col_config1, col_config2 = st.columns(2)

        with col_config1:
            domain = st.text_input("Domain", value="general")
            source_files = st.text_input("Source Files (comma-separated)", placeholder="file1.pdf, file2.docx")

        with col_config2:
            system_preference = st.selectbox(
                "Target System",
                ["Auto (Best Match)", "AI-Powered", "SME", "Postgres", "Judge", "Intelligent"]
            )

        analyze_button = st.form_submit_button("🔬 Analyze Optimization Potential")

        if analyze_button and query_text:
            # Convert system preference
            system_type = None
            if system_preference != "Auto (Best Match)":
                system_mapping = {
                    "AI-Powered": "ai_powered",
                    "SME": "sme",
                    "Postgres": "postgres",
                    "Judge": "judge",
                    "Intelligent": "intelligent"
                }
                from tidyllm.services.unified_rag_manager import RAGSystemType
                system_type = RAGSystemType(system_mapping[system_preference])

            # Create query for analysis
            from tidyllm.services.unified_rag_manager import UnifiedRAGQuery
            analysis_query = UnifiedRAGQuery(
                query=query_text,
                domain=domain,
                system_type=system_type,
                source_files=source_files.split(",") if source_files else None
            )

            # Analyze optimization potential
            optimization_analysis = rag_manager.should_use_rag2dag_for_query(analysis_query)

            # Display results
            st.subheader("📊 Optimization Analysis Results")

            # Overall recommendation
            if optimization_analysis["should_use"]:
                st.success(f"✅ {optimization_analysis['recommendation']}")
            else:
                st.info(f"ℹ️ {optimization_analysis['recommendation']}")

            # Optimization potential
            potential = optimization_analysis["optimization_potential"]
            st.metric("Optimization Potential", f"{potential:.1%}")
            st.progress(potential)

            # Factors contributing to optimization
            if optimization_analysis.get("factors"):
                st.write("**Optimization Factors:**")
                for factor, score in optimization_analysis["factors"]:
                    st.write(f"• {factor}: {score:.1%}")

if __name__ == "__main__":
    if MANAGER_AVAILABLE:
        main()
    else:
        st.error("⚠️ UnifiedRAGManager not available. Please check your installation and imports.")