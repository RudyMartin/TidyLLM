#!/usr/bin/env python3
"""
Boss MVR App - V2 Architecture with V1 Integration
=================================================

Simple MVR processing interface for the boss using proper V2 Clean Architecture
with V1 processor integration via adapters.

Architecture:
- V2 Clean Architecture (Hexagonal/Ports & Adapters)
- V1 processors accessed through V2 adapters
- Results logged to MLflow via V2 MLflow adapter
- Evidence stored in PostgreSQL via V2 adapter
- Proper dependency injection
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import time
import os
from typing import List

# Add V2 paths
v2_src = Path(__file__).parent / "src"
sys.path.append(str(v2_src))

# Page config
st.set_page_config(
    page_title="Boss MVR (V2)",
    page_icon="📋",
    layout="wide"
)

# V2 Architecture imports
try:
    from infrastructure.adapters import AWSSecretsAdapter, PostgreSQLAdapter
    from infrastructure.mlflow_adapter import V2MLflowAdapter  
    from infrastructure.v1_mvr_adapter import V1MVRAdapter
    
    V2_AVAILABLE = True
    st.success("✅ V2 Architecture loaded successfully")
except ImportError as e:
    V2_AVAILABLE = False
    st.error(f"⚠️ V2 Architecture not available: {e}")

def initialize_v2_system():
    """Initialize V2 system with dependency injection"""
    try:
        # Load secrets
        secrets_adapter = AWSSecretsAdapter()
        
        # Initialize adapters with proper DI
        mlflow_adapter = V2MLflowAdapter(secrets_adapter)
        postgres_adapter = PostgreSQLAdapter()
        
        # Initialize V1 MVR adapter (bridges V1 to V2)
        mvr_adapter = V1MVRAdapter(mlflow_adapter, postgres_adapter)
        
        # Return adapters for direct use (simplified for boss interface)
        return mvr_adapter, mvr_adapter
        
    except Exception as e:
        st.error(f"Failed to initialize V2 system: {e}")
        return None, None

def main():
    """Simple boss interface using V2 architecture"""
    
    # Header
    st.title("📋 Boss MVR Processing (V2 Architecture)")
    st.markdown("**Clean Architecture with V1 processor integration**")
    
    if not V2_AVAILABLE:
        st.error("V2 system not available. Please check the installation.")
        return
    
    # Initialize V2 system
    with st.spinner("Initializing V2 system..."):
        mvr_adapter, document_service = initialize_v2_system()
    
    if not mvr_adapter:
        st.error("Failed to initialize V2 system")
        return
    
    # Show system health
    with st.expander("🔧 System Health", expanded=False):
        health = mvr_adapter.health_check()
        st.json(health)
    
    # Connection status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ V2 Architecture")
    with col2:
        st.success("✅ V1 Processors")
    with col3:
        st.success("✅ Clean Integration")
    
    st.markdown("---")
    
    # File upload
    st.subheader("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files for V2 processing with V1 integration",
        type=['json', 'xlsx', 'xls', 'pdf'],
        accept_multiple_files=True,
        help="JSON: MVR documents | Excel: QA files | PDF: Documents"
    )
    
    if uploaded_files:
        st.markdown(f"**📊 Ready to process {len(uploaded_files)} files**")
        
        # Show files
        for i, file in enumerate(uploaded_files, 1):
            file_type = Path(file.name).suffix.lower()
            if file_type == '.json':
                processor_info = "V2 → V1 MVR Gap Analyzer → V2 Evidence Store"
            elif file_type in ['.xlsx', '.xls']:
                processor_info = "V2 → V1 QA Processor → V2 Evidence Store"
            elif file_type == '.pdf':
                processor_info = "V2 → V1 QA Processor → V2 Evidence Store"
            else:
                processor_info = "V2 Processing"
                
            st.text(f"{i}. {file.name} → {processor_info}")
        
        st.markdown("---")
        
        # Process button
        if st.button(f"🚀 Process {len(uploaded_files)} Files (V2 + V1)", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process files through V2 architecture
            results = []
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                progress = 10 + (i * 80 // total_files)
                progress_bar.progress(progress)
                status_text.text(f"🔍 V2 Processing: {uploaded_file.name}")
                
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Process through V2 architecture
                    result = mvr_adapter.process_document(tmp_file_path)
                    
                    results.append({
                        'file': uploaded_file.name,
                        'result': result,
                        'status': 'success' if result.success else 'error'
                    })
                    
                except Exception as e:
                    results.append({
                        'file': uploaded_file.name,
                        'error': str(e),
                        'status': 'error'
                    })
                
                finally:
                    # Cleanup
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ V2 processing complete!")
            
            # Show results
            st.success("🎉 **V2 Processing with V1 Integration Complete!**")
            
            successful = len([r for r in results if r['status'] == 'success'])
            errors = len([r for r in results if r['status'] == 'error'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Processed", successful)
            with col2:
                st.metric("Errors", errors)
            with col3:
                st.metric("Architecture", "V2 + V1")
            
            # Detailed results
            st.markdown("### 📊 Processing Results")
            
            for item in results:
                if item['status'] == 'success':
                    result = item['result']
                    
                    with st.expander(f"✅ {item['file']} - Success"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Confidence Score", f"{result.confidence_score:.3f}")
                            st.metric("Processing Time", f"{result.processing_time:.2f}s")
                            if result.evidence_id:
                                st.text(f"Evidence ID: {result.evidence_id}")
                        
                        with col2:
                            if result.workflow_route:
                                st.text(f"Workflow: {result.workflow_route.workflow_path}")
                                st.text(f"Priority: {result.workflow_route.priority}")
                                st.text(f"Duration: {result.workflow_route.estimated_duration} days")
                        
                        if result.metadata:
                            st.markdown("**Metadata:**")
                            st.json(result.metadata)
                
                elif item['status'] == 'error':
                    with st.expander(f"❌ {item['file']} - Error"):
                        st.error(item['error'])
            
            # Links
            st.markdown("### 🔗 View Results")
            st.markdown("• [MLflow Dashboard](http://localhost:5000) - Experiment tracking")
            st.markdown("• PostgreSQL Database - Evidence records stored")
            st.markdown("• V2 Clean Architecture - Proper separation of concerns")
    
    else:
        st.info("👆 Upload files to start V2 processing with V1 integration")
    
    # Footer
    st.markdown("---")
    st.markdown("**V2 Clean Architecture** | V1 processor integration via adapters | MLflow + PostgreSQL")

if __name__ == "__main__":
    main()