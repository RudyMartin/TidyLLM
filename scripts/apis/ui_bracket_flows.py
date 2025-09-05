"""
TidyLLM UI Chain Interface Solution
===================================

Extends existing Streamlit UI to provide direct access to the 7 core document chain operations.
Builds on existing heiros_streamlit_demo.py but adds chain contract interfaces.
"""

import streamlit as st
import pandas as pd
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import requests
import asyncio

# Import existing components
try:
    from tidyllm.document_chains import BackendDocumentPipeline, FrontendDocumentAPI
    from tidyllm.gateways import get_global_registry
    TIDYLLM_CHAINS_AVAILABLE = True
except ImportError:
    TIDYLLM_CHAINS_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="TidyLLM Chain Operations",
    page_icon="⛓️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TidyLLMChainUI:
    """Streamlit UI for TidyLLM chain operations."""
    
    def __init__(self):
        if TIDYLLM_CHAINS_AVAILABLE:
            self.backend_pipeline = BackendDocumentPipeline()
            self.frontend_api = FrontendDocumentAPI()
        else:
            self.backend_pipeline = None
            self.frontend_api = None
        
        # Initialize session state
        if 'operations_history' not in st.session_state:
            st.session_state.operations_history = []
        if 'active_domain' not in st.session_state:
            st.session_state.active_domain = None
    
    def render_sidebar(self) -> str:
        """Render sidebar navigation."""
        st.sidebar.title("⛓️ TidyLLM Chains")
        st.sidebar.markdown("**Document Operations**")
        
        # Domain selection
        domains = self._get_available_domains()
        selected_domain = st.sidebar.selectbox(
            "Active Domain",
            options=["None"] + domains,
            index=0 if not st.session_state.active_domain else domains.index(st.session_state.active_domain) + 1
        )
        
        if selected_domain != "None":
            st.session_state.active_domain = selected_domain
        else:
            st.session_state.active_domain = None
        
        # Navigation
        page = st.sidebar.radio(
            "Navigate to:",
            [
                "🏠 Chain Dashboard",
                "📥 Ingest Documents", 
                "🧠 Generate Embeddings",
                "🗂️ Create Indices",
                "📊 Track Operations",
                "📋 Generate Reports",
                "🔍 Query Knowledge",
                "🔎 Search Documents",
                "⛓️ Chain Operations",
                "📈 Analytics"
            ]
        )
        
        # System status
        st.sidebar.markdown("---")
        self._render_sidebar_status()
        
        return page
    
    def _get_available_domains(self) -> List[str]:
        """Get list of available knowledge domains."""
        # Mock domains for demo - would fetch from actual system
        return ["legal", "technical", "financial", "compliance", "research"]
    
    def _render_sidebar_status(self):
        """Render system status in sidebar."""
        st.sidebar.markdown("**System Status**")
        
        if TIDYLLM_CHAINS_AVAILABLE:
            st.sidebar.success("🟢 Chain System Ready")
        else:
            st.sidebar.error("🔴 Chain System Unavailable")
        
        # Quick stats
        if st.session_state.active_domain:
            st.sidebar.info(f"Active: {st.session_state.active_domain}")
        
        recent_ops = len(st.session_state.operations_history)
        st.sidebar.metric("Recent Operations", recent_ops)
    
    def render_chain_dashboard(self):
        """Main chain dashboard."""
        st.title("⛓️ TidyLLM Chain Operations Dashboard")
        st.markdown("**Unified Document Processing with S3-First Architecture**")
        
        # Quick action buttons
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📥 Quick Ingest", type="primary"):
                st.session_state.quick_action = "ingest"
                st.rerun()
        
        with col2:
            if st.button("🔍 Quick Query", type="primary"):
                st.session_state.quick_action = "query"
                st.rerun()
        
        with col3:
            if st.button("⛓️ Run Chain", type="primary"):
                st.session_state.quick_action = "chain"
                st.rerun()
        
        with col4:
            if st.button("📊 View Status"):
                st.session_state.quick_action = "status"
                st.rerun()
        
        # Handle quick actions
        if hasattr(st.session_state, 'quick_action'):
            if st.session_state.quick_action == "query" and st.session_state.active_domain:
                self._render_quick_query()
            elif st.session_state.quick_action == "ingest":
                self._render_quick_ingest()
            elif st.session_state.quick_action == "chain":
                self._render_quick_chain()
            elif st.session_state.quick_action == "status":
                self._render_quick_status()
        
        # Operations overview
        st.subheader("📋 Document Operations Overview")
        
        # Two-layer architecture explanation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Backend Layer (Complex)")
            backend_ops = [
                "📥 **Ingest** - S3-first document upload",
                "🧠 **Embed** - tidyllm-sentence embeddings", 
                "🗂️ **Index** - tlm-based indexing",
                "📊 **Track** - PostgreSQL MLflow tracking",
                "📋 **Report** - Processing reports"
            ]
            for op in backend_ops:
                st.markdown(f"- {op}")
        
        with col2:
            st.markdown("### 🎯 Frontend Layer (Simple)")
            frontend_ops = [
                "🔍 **Query** - Natural language questions",
                "🔎 **Search** - Keyword search"
            ]
            for op in frontend_ops:
                st.markdown(f"- {op}")
            
            st.markdown("### ⛓️ Chain Execution")
            st.markdown("- **Sequential** - One after another")
            st.markdown("- **Pipeline** - Stream between stages") 
            st.markdown("- **Parallel** - Where possible")
        
        # Recent operations
        self._render_recent_operations()
    
    def _render_quick_query(self):
        """Quick query interface."""
        st.markdown("---")
        st.subheader(f"🔍 Quick Query - {st.session_state.active_domain}")
        
        with st.form("quick_query"):
            question = st.text_input(
                "Ask a question:",
                placeholder=f"What information do you need from {st.session_state.active_domain}?"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                limit = st.slider("Number of results", 1, 20, 5)
            with col2:
                threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7)
            
            if st.form_submit_button("🔍 Query", type="primary"):
                if question:
                    self._execute_query(st.session_state.active_domain, question, limit, threshold)
                else:
                    st.error("Please enter a question")
    
    def _render_quick_ingest(self):
        """Quick ingest interface."""
        st.markdown("---")
        st.subheader("📥 Quick Document Ingest")
        
        with st.form("quick_ingest"):
            domain = st.text_input("Domain", value=st.session_state.active_domain or "")
            
            # File upload options
            upload_method = st.radio(
                "Upload Method",
                ["File Upload", "S3 Path", "Local Path"]
            )
            
            if upload_method == "File Upload":
                uploaded_files = st.file_uploader(
                    "Choose files",
                    accept_multiple_files=True,
                    type=['pdf', 'docx', 'txt']
                )
                source = uploaded_files
            elif upload_method == "S3 Path":
                source = st.text_input("S3 URI", placeholder="s3://bucket/path/")
            else:
                source = st.text_input("Local Path", placeholder="./documents/")
            
            col1, col2 = st.columns(2)
            with col1:
                bucket = st.text_input("Target S3 Bucket", placeholder="processed-docs-bucket")
            with col2:
                batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
            
            if st.form_submit_button("📥 Ingest", type="primary"):
                if domain and source:
                    self._execute_ingest(domain, source, bucket, batch_size)
                else:
                    st.error("Please provide domain and source")
    
    def _render_quick_chain(self):
        """Quick chain execution interface.""" 
        st.markdown("---")
        st.subheader("⛓️ Quick Chain Execution")
        
        with st.form("quick_chain"):
            domain = st.text_input("Domain", value=st.session_state.active_domain or "")
            
            # Operation selection
            st.markdown("**Select Operations to Chain:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("*Backend Operations:*")
                ingest = st.checkbox("📥 Ingest", value=True)
                embed = st.checkbox("🧠 Embed", value=True) 
                index = st.checkbox("🗂️ Index", value=True)
                track = st.checkbox("📊 Track", value=False)
                report = st.checkbox("📋 Report", value=False)
            
            with col2:
                st.markdown("*Frontend Operations:*")
                query = st.checkbox("🔍 Query", value=False)
                search = st.checkbox("🔎 Search", value=False)
            
            # Chain configuration
            if ingest:
                source = st.text_input("Source", placeholder="./documents/ or s3://bucket/")
            if query:
                question = st.text_input("Question", placeholder="What should I query?")
            
            execution_mode = st.selectbox(
                "Execution Mode",
                ["auto", "sequential", "pipeline", "parallel"]
            )
            
            if st.form_submit_button("⛓️ Execute Chain", type="primary"):
                operations = []
                if ingest: operations.append("ingest")
                if embed: operations.append("embed")
                if index: operations.append("index") 
                if track: operations.append("track")
                if report: operations.append("report")
                if query: operations.append("query")
                if search: operations.append("search")
                
                if domain and operations:
                    self._execute_chain(domain, operations, execution_mode, 
                                      source=source if ingest else None,
                                      question=question if query else None)
                else:
                    st.error("Please provide domain and select operations")
    
    def _render_quick_status(self):
        """Quick status interface."""
        st.markdown("---")
        st.subheader("📊 System Status")
        
        # Mock status data - would fetch from actual system
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Domains", len(self._get_available_domains()))
        with col2:
            st.metric("Documents Processed", "1,234")
        with col3:
            st.metric("Queries Served", "5,678") 
        with col4:
            st.metric("Avg Response Time", "1.2s")
        
        # Domain-specific status
        if st.session_state.active_domain:
            st.markdown(f"### Status for '{st.session_state.active_domain}' domain:")
            
            # Mock domain status
            status_data = {
                "documents": 150,
                "embeddings": 145,
                "indices": 3,
                "last_query": "2 minutes ago",
                "health": "healthy"
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", status_data["documents"])
            with col2:
                st.metric("Embeddings", status_data["embeddings"])
            with col3:
                st.metric("Indices", status_data["indices"])
            
            st.info(f"Last query: {status_data['last_query']}")
            st.success(f"Health: {status_data['health']}")
    
    def _execute_query(self, domain: str, question: str, limit: int, threshold: float):
        """Execute query operation."""
        with st.spinner(f"Querying {domain} domain..."):
            try:
                # Mock query execution - would use actual frontend API
                time.sleep(2)  # Simulate processing
                
                # Mock results
                results = [
                    {
                        "title": f"Document 1 - {domain}",
                        "content": f"This document contains information relevant to: {question}",
                        "score": 0.92,
                        "source": "s3://docs/doc1.pdf"
                    },
                    {
                        "title": f"Document 2 - {domain}",
                        "content": f"Additional context about: {question}",
                        "score": 0.87,
                        "source": "s3://docs/doc2.pdf"
                    }
                ]
                
                # Display results
                st.success(f"Found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result['title']} (score: {result['score']:.3f})"):
                        st.write(result['content'])
                        st.caption(f"Source: {result['source']}")
                
                # Add to history
                operation_record = {
                    "timestamp": datetime.now(),
                    "operation": "query",
                    "domain": domain,
                    "details": f"Question: {question}",
                    "status": "completed",
                    "results_count": len(results)
                }
                st.session_state.operations_history.append(operation_record)
                
            except Exception as e:
                st.error(f"Query failed: {e}")
    
    def _execute_ingest(self, domain: str, source: Any, bucket: str, batch_size: int):
        """Execute ingest operation."""
        with st.spinner(f"Ingesting documents to {domain} domain..."):
            try:
                # Mock ingest execution - would use actual backend pipeline
                time.sleep(3)  # Simulate processing
                
                # Mock results
                documents_processed = 25 if hasattr(source, '__len__') and len(source) > 0 else 10
                
                st.success(f"Successfully ingested {documents_processed} documents")
                st.info(f"Documents stored in S3 bucket: {bucket}")
                
                # Add to history
                operation_record = {
                    "timestamp": datetime.now(),
                    "operation": "ingest",
                    "domain": domain,
                    "details": f"Source: {source}, Batch: {batch_size}",
                    "status": "completed",
                    "results_count": documents_processed
                }
                st.session_state.operations_history.append(operation_record)
                
            except Exception as e:
                st.error(f"Ingest failed: {e}")
    
    def _execute_chain(self, domain: str, operations: List[str], mode: str, source=None, question=None):
        """Execute chain operations."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_ops = len(operations)
            
            for i, operation in enumerate(operations):
                status_text.text(f"Executing: {operation.upper()}")
                progress_bar.progress((i + 1) / total_ops)
                time.sleep(1)  # Simulate processing
            
            st.success(f"Chain completed: {' → '.join(operations)}")
            
            # Add to history
            operation_record = {
                "timestamp": datetime.now(),
                "operation": "chain",
                "domain": domain,
                "details": f"Chain: {' → '.join(operations)} ({mode})",
                "status": "completed",
                "results_count": len(operations)
            }
            st.session_state.operations_history.append(operation_record)
            
        except Exception as e:
            st.error(f"Chain execution failed: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def _render_recent_operations(self):
        """Render recent operations history.""" 
        st.subheader("🕒 Recent Operations")
        
        if not st.session_state.operations_history:
            st.info("No recent operations")
            return
        
        # Show last 5 operations
        recent_ops = st.session_state.operations_history[-5:]
        
        for op in reversed(recent_ops):
            with st.expander(f"{op['operation'].upper()} - {op['domain']} ({op['timestamp'].strftime('%H:%M:%S')})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {op['status']}")
                    st.write(f"**Domain:** {op['domain']}")
                
                with col2:
                    st.write(f"**Time:** {op['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    if 'results_count' in op:
                        st.write(f"**Results:** {op['results_count']}")
                
                st.write(f"**Details:** {op['details']}")
    
    def render_ingest_interface(self):
        """Full ingest interface."""
        st.title("📥 Document Ingest")
        st.markdown("**S3-First document ingestion with batch processing**")
        
        with st.form("full_ingest"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Source Configuration")
                domain = st.text_input("Knowledge Domain", value=st.session_state.active_domain or "")
                source_type = st.selectbox("Source Type", ["Local Files", "S3 Bucket", "File Upload"])
                
                if source_type == "File Upload":
                    uploaded_files = st.file_uploader(
                        "Upload Documents",
                        accept_multiple_files=True,
                        type=['pdf', 'docx', 'txt', 'md']
                    )
                elif source_type == "S3 Bucket":
                    source_uri = st.text_input("S3 URI", placeholder="s3://source-bucket/path/")
                else:
                    source_path = st.text_input("Local Path", placeholder="./documents/")
                
                document_format = st.selectbox("Document Format", ["auto", "pdf", "docx", "txt", "md"])
            
            with col2:
                st.subheader("Processing Configuration")
                target_bucket = st.text_input("Target S3 Bucket", placeholder="processed-docs")
                batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
                parallel_workers = st.number_input("Parallel Workers", min_value=1, max_value=10, value=3)
                
                st.subheader("Metadata")
                metadata = st.text_area("Additional Metadata (JSON)", placeholder='{"owner": "team", "project": "research"}')
            
            # Advanced options
            with st.expander("Advanced Options"):
                enable_ocr = st.checkbox("Enable OCR for scanned documents")
                skip_duplicates = st.checkbox("Skip duplicate documents", value=True)
                validate_format = st.checkbox("Validate document format", value=True)
            
            submitted = st.form_submit_button("📥 Start Ingest", type="primary")
            
            if submitted:
                if domain:
                    # Would execute actual ingest here
                    st.success("Ingest operation started!")
                    st.info("Check the Chain Dashboard for progress updates")
                else:
                    st.error("Domain is required")
    
    def render_query_interface(self):
        """Full query interface."""
        st.title("🔍 Knowledge Query")
        st.markdown("**Natural language queries with semantic search**")
        
        # Query form
        with st.form("knowledge_query"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                domain = st.selectbox("Knowledge Domain", self._get_available_domains())
                question = st.text_area(
                    "Your Question",
                    placeholder="What information are you looking for?",
                    height=100
                )
            
            with col2:
                st.subheader("Query Parameters")
                limit = st.slider("Max Results", 1, 50, 10)
                similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)
                response_format = st.selectbox("Response Format", ["Detailed", "Summary", "Raw"])
            
            submitted = st.form_submit_button("🔍 Query", type="primary")
            
            if submitted and question:
                self._execute_query(domain, question, limit, similarity_threshold)


# Main application
def main():
    """Main UI application entry point."""
    ui = TidyLLMChainUI()
    
    # Render sidebar and get selected page
    page = ui.render_sidebar()
    
    # Route to appropriate interface
    if page == "🏠 Chain Dashboard":
        ui.render_chain_dashboard()
    elif page == "📥 Ingest Documents":
        ui.render_ingest_interface()
    elif page == "🔍 Query Knowledge":
        ui.render_query_interface()
    elif page == "🧠 Generate Embeddings":
        st.title("🧠 Generate Embeddings")
        st.info("Embedding interface - to be implemented")
    elif page == "🗂️ Create Indices":
        st.title("🗂️ Create Indices") 
        st.info("Indexing interface - to be implemented")
    elif page == "📊 Track Operations":
        st.title("📊 Track Operations")
        st.info("Tracking interface - to be implemented")
    elif page == "📋 Generate Reports":
        st.title("📋 Generate Reports")
        st.info("Reporting interface - to be implemented")
    elif page == "🔎 Search Documents":
        st.title("🔎 Search Documents")
        st.info("Search interface - to be implemented")
    elif page == "⛓️ Chain Operations":
        st.title("⛓️ Chain Operations")
        st.info("Advanced chaining interface - to be implemented")
    elif page == "📈 Analytics":
        st.title("📈 Analytics")
        st.info("Analytics interface - to be implemented")


if __name__ == "__main__":
    main()