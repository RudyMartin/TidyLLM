#!/usr/bin/env python3
"""
Welcome Workflow Demo
====================

Simple welcome page with three workflow options:
1. Analyst Report
2. Section View  
3. Peer Review

Clean interface for workflow selection and document upload.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

# Import components with error handling
HUMAN_LOOP_AVAILABLE = False
try:
    from human_loop_mvr_interface import HumanLoopMVRInterface, DocumentState
    HUMAN_LOOP_AVAILABLE = True
except ImportError:
    pass

# Page configuration
st.set_page_config(
    page_title="DROP ZONES Workflow Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_workflow' not in st.session_state:
    st.session_state.selected_workflow = None
if 'human_interface' not in st.session_state:
    st.session_state.human_interface = None

def init_interface():
    """Initialize the MVR interface"""
    if not HUMAN_LOOP_AVAILABLE:
        return None
    
    if st.session_state.human_interface is None:
        try:
            workspace_path = Path.cwd() / "streamlit_mvr_workspace"
            st.session_state.human_interface = HumanLoopMVRInterface(str(workspace_path))
        except Exception as e:
            st.error(f"Failed to initialize interface: {e}")
            return None
    
    return st.session_state.human_interface

def execute_workflow_chain(interface, document_id: str, workflow_type: str) -> Dict[str, Any]:
    """Execute workflow chain based on selection"""
    
    workflow_stages = {
        "Analyst Report": ['mvr_tag', 'mvr_qa', 'mvr_report'],
        "Section View": ['mvr_tag', 'mvr_qa'],
        "Peer Review": ['mvr_tag', 'mvr_qa', 'mvr_peer', 'mvr_report']
    }
    
    required_stages = workflow_stages.get(workflow_type, [])
    if not required_stages:
        return {"success": False, "error": f"Unknown workflow type: {workflow_type}"}
    
    try:
        # Get current document state
        doc_state = interface.get_document_state(document_id)
        current_stage = doc_state.current_stage if doc_state else 'mvr_tag'
        
        result = {
            "success": True,
            "workflow_type": workflow_type,
            "stages_executed": [],
            "stages_skipped": [],
            "final_stage": current_stage,
            "outputs": []
        }
        
        # Execute required stages
        for target_stage in required_stages:
            if current_stage == target_stage:
                result["stages_skipped"].append(target_stage)
                continue
            
            # Advance to target stage
            if interface.mvr_workflow_stages.index(current_stage) < interface.mvr_workflow_stages.index(target_stage):
                advance_result = interface.advance_workflow_stage(document_id, force=True)
                if advance_result['success']:
                    current_stage = advance_result.get('current_stage', current_stage)
                    result["stages_executed"].append(target_stage)
                    
                    # Add stage outputs
                    if target_stage == 'mvr_qa':
                        result["outputs"].append({
                            "stage": "mvr_qa",
                            "type": "comparison_analysis"
                        })
                    elif target_stage == 'mvr_peer':
                        result["outputs"].append({
                            "stage": "mvr_peer",
                            "type": "peer_review"
                        })
                    elif target_stage == 'mvr_report':
                        result["outputs"].append({
                            "stage": "mvr_report", 
                            "type": "final_report"
                        })
                else:
                    result["success"] = False
                    result["error"] = f"Failed to advance to {target_stage}"
                    break
        
        result["final_stage"] = current_stage
        
        # Workflow-specific actions
        if workflow_type == "Section View" and result["success"]:
            result["action"] = "open_section_browser"
        elif workflow_type == "Peer Review" and result["success"]:
            result["action"] = "generate_peer_pdf"
            result["peer_review_pdf"] = f"peer_review_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "workflow_type": workflow_type,
            "error": str(e),
            "stages_executed": [],
            "final_stage": current_stage if 'current_stage' in locals() else 'unknown'
        }

def show_file_sidebar():
    """Show the file upload and classification sidebar"""
    
    st.sidebar.title("📁 Document Management")
    
    # Workflow status first
    with st.sidebar.expander("🎯 Workflow Status", expanded=True):
        if st.session_state.selected_workflow:
            st.success(f"Selected: {st.session_state.selected_workflow}")
            if hasattr(st.session_state, 'uploaded_files') and st.session_state.uploaded_files:
                st.info(f"Ready: {len(st.session_state.uploaded_files)} files")
            else:
                st.warning("Upload files to proceed")
        else:
            st.info("👈 Select workflow from main page first")
    
    # File upload section - only show if workflow is selected
    if st.session_state.selected_workflow:
        with st.sidebar.expander("➕ Upload Documents", expanded=True):
            uploaded_files = st.file_uploader(
                "Select documents",
                type=['txt', 'pdf', 'docx', 'md', 'json'],
                accept_multiple_files=True,
                help="Upload MVR, VST, or research documents",
                key="sidebar_uploader"
            )
            
            if uploaded_files:
                st.markdown(f"**{len(uploaded_files)} files uploaded:**")
                
                # Store files in session state with classifications
                if 'uploaded_files' not in st.session_state:
                    st.session_state.uploaded_files = []
                
                file_configs = []
                for i, uploaded_file in enumerate(uploaded_files):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"📄 `{uploaded_file.name}`")
                    with col2:
                        # Auto-detect document type
                        auto_type = 'mvr'
                        if 'vst' in uploaded_file.name.lower():
                            auto_type = 'vst'
                        elif 'research' in uploaded_file.name.lower() or uploaded_file.name.endswith('.pdf'):
                            auto_type = 'research'
                        
                        doc_type = st.selectbox(
                            "Type",
                            ['mvr', 'vst', 'research'],
                            index=['mvr', 'vst', 'research'].index(auto_type),
                            key=f"sidebar_type_{i}"
                        )
                    
                    file_configs.append({
                        'file': uploaded_file,
                        'name': uploaded_file.name,
                        'type': doc_type,
                        'size': len(uploaded_file.getvalue())
                    })
                
                # Update session state
                st.session_state.uploaded_files = file_configs
                
                # Batch options
                st.markdown("**⚙️ Options**")
                st.session_state.auto_pair = st.checkbox("Auto-pair MVR/VST documents", value=True)
    
    # Current files section
    if hasattr(st.session_state, 'uploaded_files') and st.session_state.uploaded_files:
        with st.sidebar.expander("📋 Current Files", expanded=False):
            for file_config in st.session_state.uploaded_files:
                st.markdown(f"📄 **{file_config['name']}**")
                st.markdown(f"   Type: {file_config['type'].upper()}")
                st.markdown(f"   Size: {file_config['size']:,} bytes")
    
    # Workflow status
    with st.sidebar.expander("🎯 Workflow Status"):
        if st.session_state.selected_workflow:
            st.info(f"Selected: {st.session_state.selected_workflow}")
            if hasattr(st.session_state, 'uploaded_files') and st.session_state.uploaded_files:
                st.success(f"Ready: {len(st.session_state.uploaded_files)} files")
            else:
                st.warning("Upload files to proceed")
        else:
            st.info("Select workflow from main page")

def show_welcome_page():
    """Show the main welcome page with workflow selection"""
    
    st.title("🎯 DROP ZONES Workflow Demo")
    st.markdown("### Choose your analysis workflow")
    
    # Sidebar for file management
    show_file_sidebar()
    
    # Three workflow options in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Analyst Report", type="primary"):
            st.session_state.selected_workflow = "Analyst Report"
    
    with col2:
        if st.button("📖 Section View", type="primary"):
            st.session_state.selected_workflow = "Section View"
    
    with col3:
        if st.button("🔍 Peer Review", type="primary"):
            st.session_state.selected_workflow = "Peer Review"
    
    # Workflow descriptions
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Analyst Report**
        - Complete MVR vs VST comparison
        - Automated analysis pipeline  
        - Final PDF/JSON report output
        - Path: mvr_tag → mvr_qa → mvr_report
        """)
    
    with col2:
        st.markdown("""
        **📖 Section View**
        - Interactive document browser
        - Section-by-section navigation
        - No final report generated
        - Path: mvr_tag → mvr_qa → browser
        """)
    
    with col3:
        st.markdown("""
        **🔍 Peer Review**
        - Full peer review analysis
        - Triangulation validation
        - PDF peer review report
        - Path: mvr_tag → mvr_qa → mvr_peer → report
        """)

def show_workflow_page(workflow_type):
    """Show the selected workflow execution page"""
    
    st.title(f"🎯 {workflow_type} Workflow")
    
    # Sidebar for file management
    show_file_sidebar()
    
    # Back button
    if st.button("← Back to Welcome"):
        st.session_state.selected_workflow = None
        st.rerun()
    
    # Workflow info
    workflow_info = {
        "Analyst Report": {
            "icon": "📊",
            "description": "Complete MVR vs VST comparison analysis with final report",
            "stages": "mvr_tag → mvr_qa → mvr_report"
        },
        "Section View": {
            "icon": "📖", 
            "description": "Interactive section-by-section document browser",
            "stages": "mvr_tag → mvr_qa → section_browser"
        },
        "Peer Review": {
            "icon": "🔍",
            "description": "Full peer review with triangulation analysis and PDF report",
            "stages": "mvr_tag → mvr_qa → mvr_peer → peer_report"
        }
    }
    
    info = workflow_info[workflow_type]
    st.info(f"{info['icon']} {info['description']}")
    st.code(f"Pipeline: {info['stages']}")
    
    # Check if files are uploaded in sidebar
    if hasattr(st.session_state, 'uploaded_files') and st.session_state.uploaded_files:
        st.markdown("### 📁 Ready Files")
        st.markdown(f"**{len(st.session_state.uploaded_files)} files ready for {workflow_type}:**")
        
        # Show file list from sidebar
        for file_config in st.session_state.uploaded_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 {file_config['name']}")
            with col2:
                st.write(f"**{file_config['type'].upper()}**")
            with col3:
                st.write(f"{file_config['size']:,} bytes")
        
        # Execute workflow button
        if st.button(f"🚀 Execute {workflow_type} Workflow", type="primary"):
            
            # Initialize interface
            interface = init_interface()
            if not interface:
                st.error("❌ MVR interface not available - running in demo mode")
                st.info("✅ Demo mode: Workflow would execute successfully")
                st.json({
                    "workflow": workflow_type,
                    "files": len(uploaded_files),
                    "status": "demo_success",
                    "message": "In production, files would be processed through the selected workflow pipeline"
                })
                return
            
            # Process files from sidebar session state
            results = []
            progress_bar = st.progress(0)
            
            for i, file_config in enumerate(st.session_state.uploaded_files):
                try:
                    # Save file
                    upload_path = interface.documents_dir / file_config['name']
                    with open(upload_path, 'wb') as f:
                        f.write(file_config['file'].getvalue())
                    
                    # Register document
                    doc_id = interface.register_document(str(upload_path), file_config['type'])
                    
                    # Execute workflow
                    workflow_result = execute_workflow_chain(interface, doc_id, workflow_type)
                    
                    results.append({
                        'filename': file_config['name'],
                        'doc_id': doc_id,
                        'success': workflow_result['success'],
                        **workflow_result
                    })
                    
                except Exception as e:
                    results.append({
                        'filename': file_config['name'],
                        'success': False,
                        'error': str(e)
                    })
                
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_files))
            
            # Show results
            successful = len([r for r in results if r['success']])
            st.success(f"✅ {workflow_type} completed for {successful}/{len(results)} documents")
            
            # Workflow-specific results
            if workflow_type == "Analyst Report":
                reports = len([r for r in results if r.get('success') and 'final_report' in str(r.get('outputs', []))])
                if reports > 0:
                    st.info(f"📊 {reports} analyst reports generated")
            
            elif workflow_type == "Section View":
                browser_ready = len([r for r in results if r.get('action') == 'open_section_browser'])
                if browser_ready > 0:
                    st.info(f"📖 {browser_ready} documents ready for section browsing")
            
            elif workflow_type == "Peer Review":
                pdfs = len([r for r in results if 'peer_review_pdf' in r])
                if pdfs > 0:
                    st.info(f"🔍 {pdfs} peer review PDFs generated")
            
            # Detailed results
            with st.expander("📋 Execution Details"):
                for result in results:
                    if result['success']:
                        st.write(f"✅ **{result['filename']}** - {result.get('workflow_type', 'Unknown')} workflow completed")
                        if 'stages_executed' in result:
                            st.write(f"   Stages executed: {', '.join(result['stages_executed'])}")
                        if 'final_stage' in result:
                            st.write(f"   Final stage: {result['final_stage']}")
                    else:
                        st.write(f"❌ **{result['filename']}** - Error: {result.get('error', 'Unknown error')}")
    
    else:
        # No files uploaded
        st.markdown("### 📁 No Files Uploaded")
        st.info("👈 Please upload documents using the sidebar to proceed with the workflow.")
        st.markdown("**How to use:**")
        st.markdown("1. Upload files in the sidebar")
        st.markdown("2. Classify each file (MVR, VST, or Research)")
        st.markdown("3. Return here to execute the workflow")

def main():
    """Main application"""
    
    # Show welcome page or workflow page based on selection
    if st.session_state.selected_workflow is None:
        show_welcome_page()
    else:
        show_workflow_page(st.session_state.selected_workflow)

if __name__ == "__main__":
    main()