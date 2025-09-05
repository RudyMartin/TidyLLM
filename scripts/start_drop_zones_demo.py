#!/usr/bin/env python3
"""
DROP ZONES Streamlit Demo
========================

Interactive Streamlit demo showcasing:
- Watchdog-based file monitoring
- Human-in-the-Loop MVR workflow management  
- SOP compliance integration
- Real-time document processing

Launch with: streamlit run start_drop_zones_demo.py --server.runOnSave=true
"""

import streamlit as st
import pandas as pd
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

# Import DROP ZONES components with error handling
HumanLoopMVRInterface = None
DocumentState = None
DropZoneTestFramework = None
BasicListener = None
ConfigManager = None

HUMAN_LOOP_AVAILABLE = False
WATCHDOG_TEST_AVAILABLE = False  
BASIC_LISTENER_AVAILABLE = False

try:
    from human_loop_mvr_interface import HumanLoopMVRInterface, DocumentState
    HUMAN_LOOP_AVAILABLE = True
    print("[SUCCESS] Human Loop MVR Interface loaded")
except ImportError as e:
    print(f"[ERROR] Human Loop MVR Interface not available: {e}")

try:
    from test_drop_zones_watchdog import DropZoneTestFramework
    WATCHDOG_TEST_AVAILABLE = True
    print("[SUCCESS] Watchdog Test Framework loaded")
except ImportError as e:
    print(f"[ERROR] Watchdog Test Framework not available: {e}")

try:
    sys.path.append(str(Path(__file__).parent.parent / "drop_zones"))
    from basic.listener import BasicListener
    from basic.config import ConfigManager
    BASIC_LISTENER_AVAILABLE = True
    print("[SUCCESS] Basic Listener loaded")
except ImportError as e:
    print(f"[ERROR] Basic Listener not available: {e}")

# Page configuration
st.set_page_config(
    page_title="DROP ZONES Demo",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'human_interface' not in st.session_state:
    st.session_state.human_interface = None
if 'watchdog_results' not in st.session_state:
    st.session_state.watchdog_results = []
if 'demo_documents' not in st.session_state:
    st.session_state.demo_documents = []

def init_human_interface():
    """Initialize Human-in-the-Loop interface"""
    if not HUMAN_LOOP_AVAILABLE:
        return None
    
    if st.session_state.human_interface is None:
        try:
            workspace_path = Path.cwd() / "streamlit_mvr_workspace"
            st.session_state.human_interface = HumanLoopMVRInterface(str(workspace_path))
            
            # Create some demo documents if none exist
            if not st.session_state.human_interface.active_documents:
                create_demo_documents(st.session_state.human_interface)
        except Exception as e:
            st.error(f"Failed to initialize interface: {e}")
            return None
    
    return st.session_state.human_interface

def create_document_pairs(interface, registration_results):
    """Auto-pair MVR/VST documents with matching REV IDs"""
    if not interface:
        return 0
    
    # Extract REV IDs from successful registrations
    mvr_docs = {}
    vst_docs = {}
    
    for result in registration_results:
        if not result['success']:
            continue
            
        doc_id = result['doc_id']
        doc_type = result['doc_type']
        
        # Extract REV ID from document ID
        if doc_id.startswith('REV') and len(doc_id) >= 8:
            rev_id = doc_id[:8]  # REV12345 format
            
            if doc_type == 'mvr':
                mvr_docs[rev_id] = doc_id
            elif doc_type == 'vst':
                vst_docs[rev_id] = doc_id
    
    # Create pairs for matching REV IDs
    pairs_created = 0
    for rev_id in mvr_docs:
        if rev_id in vst_docs:
            mvr_doc = mvr_docs[rev_id]
            vst_doc = vst_docs[rev_id]
            
            try:
                # Add pairing metadata to both documents
                mvr_state = interface.active_documents.get(mvr_doc)
                vst_state = interface.active_documents.get(vst_doc)
                
                if mvr_state and vst_state:
                    mvr_state.metadata['paired_document'] = vst_doc
                    mvr_state.metadata['pair_type'] = 'MVR-primary'
                    
                    vst_state.metadata['paired_document'] = mvr_doc
                    vst_state.metadata['pair_type'] = 'VST-validation'
                    
                    interface.save_document_state(mvr_state)
                    interface.save_document_state(vst_state)
                    
                    pairs_created += 1
            
            except Exception as e:
                print(f"Failed to pair {mvr_doc} with {vst_doc}: {e}")
    
    return pairs_created

def create_demo_documents(interface):
    """Create demo documents for testing"""
    if not interface:
        raise ValueError("Interface not available")
    
    print(f"[DEMO] Creating demo documents in: {interface.documents_dir}")
    
    demo_docs = [
        {
            'filename': 'REV12345_MVR_Analysis.txt',
            'content': '''REV12345 Motor Vehicle Record Analysis
=====================================

Document Type: MVR (Motor Vehicle Record)
REV ID: REV12345
Analysis Date: 2024-01-15

Driver Information:
- Name: John Doe
- License: ABC123456
- State: California

Violations:
- Speeding: 2023-05-10 (65 in 55 zone)
- Parking: 2023-08-22 (Expired meter)

YNSR Noise Factor: 0.15 (Low noise - high quality data)
Classification: Standard MVR Document
Business Purpose: Employment Verification''',
            'doc_type': 'mvr'
        },
        {
            'filename': 'REV12345_VST_Template.txt',  # Fixed: Same REV ID for pairing
            'content': '''REV12345 Validation Scoping Template
===================================

Document Type: VST (Validation Scoping Template)
REV ID: REV12345
Scope Date: 2024-01-10

Validation Requirements:
- Identity verification: Required
- Address verification: Required
- Employment verification: Optional

Expected Discrepancies:
- Minor address variations acceptable
- Nickname variations acceptable

Quality Threshold: 95%''',
            'doc_type': 'vst'
        },
        {
            'filename': 'REV12345_Peer_Review.txt',
            'content': '''REV12345 Peer Review Analysis
================================

Document Type: Peer Review
REV ID: REV12345
Review Date: 2024-01-16
Reviewer: Senior Analyst

Review Summary:
- MVR document structure: Compliant
- Data quality assessment: High
- VST alignment: Strong match
- Recommended actions: Proceed with standard processing

Triangulation Notes:
- Cross-referenced with state databases
- Verified employment history alignment
- Confirmed identity validation markers

Confidence Level: 95%''',
            'doc_type': 'peer_review'
        }
    ]
    
    created_docs = []
    
    for i, doc in enumerate(demo_docs):
        try:
            doc_path = interface.documents_dir / doc['filename']
            print(f"[DEMO] Creating document {i+1}/{len(demo_docs)}: {doc['filename']}")
            
            # Create the file
            with open(doc_path, 'w') as f:
                f.write(doc['content'])
            print(f"[DEMO] File written: {doc_path}")
            
            # Register with interface
            doc_id = interface.register_document(str(doc_path), doc['doc_type'])
            print(f"[DEMO] Document registered: {doc_id}")
            
            created_docs.append({
                'filename': doc['filename'],
                'doc_id': doc_id,
                'doc_type': doc['doc_type'],
                'success': True
            })
            
            # Add to session state
            if 'demo_documents' not in st.session_state:
                st.session_state.demo_documents = []
            st.session_state.demo_documents.append(doc['filename'])
            
        except Exception as e:
            print(f"[ERROR] Failed to create {doc['filename']}: {e}")
            created_docs.append({
                'filename': doc['filename'],
                'success': False,
                'error': str(e)
            })
            raise e  # Re-raise to show in UI
    
    # Create a demo collection if we have multiple documents
    if len([d for d in created_docs if d['success']]) >= 2:
        try:
            successful_docs = [d['doc_id'] for d in created_docs if d['success']]
            collection_id = interface.create_collection(
                name="Demo Analysis Collection",
                description="Demonstration collection with MVR, VST, and peer review",
                document_ids=successful_docs,
                collection_type="peer_review",
                primary_document=successful_docs[0] if successful_docs else None
            )
            print(f"[DEMO] Created demo collection: {collection_id}")
        except Exception as e:
            print(f"[WARNING] Failed to create demo collection: {e}")
    
    print(f"[DEMO] Created {len([d for d in created_docs if d['success']])}/{len(demo_docs)} demo documents")
    return created_docs

def execute_workflow_chain(interface, document_id: str, workflow_type: str) -> Dict[str, Any]:
    """Execute the complete workflow chain based on selection with dependency management"""
    
    workflow_stages = {
        "Analyst Report": ['mvr_tag', 'mvr_qa', 'mvr_report'],
        "Section View": ['mvr_tag', 'mvr_qa'],  # No final report - opens browser
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
        
        # Execute each required stage in sequence
        for target_stage in required_stages:
            if current_stage == target_stage:
                # Already at target stage, skip
                result["stages_skipped"].append(target_stage)
                continue
            
            # Check if we need to advance to this stage
            if interface.mvr_workflow_stages.index(current_stage) < interface.mvr_workflow_stages.index(target_stage):
                # Advance workflow to target stage
                advance_result = interface.advance_workflow_stage(document_id, force=True)
                if advance_result['success']:
                    current_stage = advance_result.get('current_stage', current_stage)
                    result["stages_executed"].append(target_stage)
                    
                    # Check for stage-specific outputs
                    if target_stage == 'mvr_qa':
                        # MVR QA comparison results available
                        try:
                            qa_results = interface.get_document_analysis(document_id)
                            result["outputs"].append({
                                "stage": "mvr_qa",
                                "type": "comparison_analysis",
                                "data": qa_results
                            })
                        except:
                            pass
                    
                    elif target_stage == 'mvr_peer':
                        # Peer review results available
                        try:
                            peer_results = interface.get_peer_review_analysis(document_id)
                            result["outputs"].append({
                                "stage": "mvr_peer", 
                                "type": "peer_review",
                                "data": peer_results
                            })
                        except:
                            pass
                    
                    elif target_stage == 'mvr_report':
                        # Final report generated
                        try:
                            report_path = interface.generate_document_report(document_id)
                            result["outputs"].append({
                                "stage": "mvr_report",
                                "type": "final_report",
                                "path": report_path
                            })
                        except Exception as e:
                            result["outputs"].append({
                                "stage": "mvr_report",
                                "type": "error",
                                "error": str(e)
                            })
                else:
                    result["success"] = False
                    result["error"] = f"Failed to advance to {target_stage}: {advance_result.get('message', 'Unknown error')}"
                    break
        
        result["final_stage"] = current_stage
        
        # Workflow-specific post-processing
        if workflow_type == "Section View" and result["success"]:
            # For Section View, we don't generate a report but prepare browser data
            result["action"] = "open_section_browser"
            result["browser_ready"] = True
        
        elif workflow_type == "Peer Review" and result["success"]:
            # For Peer Review, generate PDF report
            result["action"] = "generate_peer_pdf"
            try:
                pdf_path = generate_peer_review_pdf(interface, document_id)
                result["peer_review_pdf"] = pdf_path
            except Exception as e:
                result["peer_pdf_error"] = str(e)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "workflow_type": workflow_type,
            "error": str(e),
            "stages_executed": [],
            "final_stage": current_stage if 'current_stage' in locals() else 'unknown'
        }

def generate_peer_review_pdf(interface, document_id: str) -> str:
    """Generate PDF report for peer review workflow"""
    # This would integrate with actual PDF generation
    # For now, return a mock path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = f"peer_review_{document_id}_{timestamp}.pdf"
    return pdf_path

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📁 DROP ZONES Demo")
    st.markdown("""
    **Interactive demonstration of the DROP ZONES system for MVR workflow automation**
    
    Choose between automated Watchdog monitoring or manual Human-in-the-Loop processing.
    """)
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Controls")
    
    demo_mode = st.sidebar.selectbox(
        "Select Demo Mode:",
        ["Human-in-the-Loop MVR Interface", "Watchdog File Monitoring", "System Overview"],
        index=0
    )
    
    if demo_mode == "System Overview":
        show_system_overview()
    elif demo_mode == "Human-in-the-Loop MVR Interface":
        show_human_loop_interface()
    elif demo_mode == "Watchdog File Monitoring":
        show_watchdog_monitoring()

def show_system_overview():
    """Show system architecture overview"""
    st.header("🏗️ DROP ZONES System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Core Concept")
        st.markdown("""
        **DROP ZONES** bridges file-based document intake with MVR analysis workflows:
        
        - **File Monitoring**: Watchdog detects document drops
        - **Workflow Triggering**: Universal Bracket Flows activation  
        - **SOP Compliance**: Real-time guidance during processing
        - **Human Oversight**: Manual controls when needed
        """)
        
        st.subheader("🔄 Processing Approaches")
        st.markdown("""
        **1. Automated (Watchdog)**
        ```
        Drop → Detect → Process → Archive
        ```
        - High-volume processing
        - Built-in error recovery
        - Unattended operation
        
        **2. Human-in-the-Loop**
        ```
        Register → Guide → Validate → Advance
        ```
        - Complex document handling
        - SOP compliance checking
        - Interactive decision making
        """)
    
    with col2:
        st.subheader("🎯 MVR Workflow Integration")
        mvr_stages = [
            ("mvr_tag", "Document Classification & Tagging"),
            ("mvr_qa", "MVR vs VST Comparison Analysis"),
            ("mvr_peer", "Peer Review & Triangulation"), 
            ("mvr_report", "Final Report Generation")
        ]
        
        for stage, description in mvr_stages:
            st.markdown(f"**{stage}**: {description}")
        
        st.subheader("🔧 Available Components")
        components = {
            "Human Loop Interface": HUMAN_LOOP_AVAILABLE,
            "Watchdog Testing": WATCHDOG_TEST_AVAILABLE,
            "Basic Listener": BASIC_LISTENER_AVAILABLE
        }
        
        for component, available in components.items():
            status = "✅" if available else "❌"
            st.markdown(f"{status} {component}")
    
    # Architecture diagram placeholder
    st.subheader("📊 System Flow")
    
    # Create a simple flow diagram using columns
    flow_cols = st.columns(5)
    flow_steps = [
        ("📁", "Document\nDropped"),
        ("👁️", "Zone\nDetection"), 
        ("⚡", "Workflow\nTrigger"),
        ("🔍", "MVR\nProcessing"),
        ("📋", "Results\nArchive")
    ]
    
    for i, (icon, step) in enumerate(flow_steps):
        with flow_cols[i]:
            st.markdown(f"<div style='text-align: center'><h1>{icon}</h1><p>{step}</p></div>", 
                       unsafe_allow_html=True)
            if i < len(flow_steps) - 1:
                st.markdown("→")

def show_human_loop_interface():
    """Show Human-in-the-Loop MVR interface"""
    st.header("👤 Human-in-the-Loop MVR Interface")
    
    if not HUMAN_LOOP_AVAILABLE:
        st.error("❌ Human Loop MVR Interface not available.")
        st.markdown("""
        **Missing Components:**
        - `human_loop_mvr_interface.py` not found
        
        **To enable this feature:**
        1. Ensure the script files are in the same directory
        2. Check that all dependencies are installed
        3. Restart the Streamlit app
        """)
        
        # Show mock interface for demonstration
        show_mock_human_interface()
        return
    
    # Initialize interface
    interface = init_human_interface()
    if not interface:
        st.error("Failed to initialize Human Loop Interface")
        return
    
    # Sidebar controls
    st.sidebar.subheader("📄 Document Management")
    
    # Multi-file document registration
    with st.sidebar.expander("➕ Register Documents", expanded=True):
        st.markdown("**📁 Multi-File Upload**")
        
        # Multi-file uploader
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=['txt', 'pdf', 'docx', 'md', 'json'],
            accept_multiple_files=True,
            help="Upload multiple documents for batch processing"
        )
        
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} files selected:**")
            
            # Show file list with individual type selection
            file_configs = []
            for i, uploaded_file in enumerate(uploaded_files):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"📄 `{uploaded_file.name}`")
                with col2:
                    # Auto-detect document type based on filename
                    auto_type = 'mvr'
                    if 'vst' in uploaded_file.name.lower():
                        auto_type = 'vst'
                    elif 'research' in uploaded_file.name.lower() or uploaded_file.name.endswith('.pdf'):
                        auto_type = 'research'
                    
                    doc_type = st.selectbox(
                        "Type", 
                        ['mvr', 'vst', 'research'], 
                        index=['mvr', 'vst', 'research'].index(auto_type),
                        key=f"type_{i}"
                    )
                
                file_configs.append((uploaded_file, doc_type))
            
            # Workflow selection options
            st.markdown("**🎯 Analysis Workflow**")
            workflow_choice = st.radio(
                "Choose workflow type:",
                ["Analyst Report", "Section View", "Peer Review"],
                help="Select the type of analysis to perform on uploaded documents",
                horizontal=True
            )
            
            # Workflow descriptions
            workflow_descriptions = {
                "Analyst Report": "📊 Complete MVR vs VST comparison analysis with final report",
                "Section View": "📖 Interactive section-by-section document browser",
                "Peer Review": "🔍 Full peer review with triangulation analysis and PDF report"
            }
            st.info(workflow_descriptions[workflow_choice])
            
            create_pairs = st.checkbox("Auto-pair MVR/VST documents with same REV ID", value=True)
            
            # Bulk registration button
            col1, col2 = st.columns(2)
            with col1:
                button_text = f"🚀 Execute {workflow_choice}"
                if st.button(button_text, type="primary"):
                    registration_results = []
                    
                    with st.spinner(f"Executing {workflow_choice} workflow for {len(file_configs)} documents..."):
                        progress_bar = st.progress(0)
                        
                        for i, (uploaded_file, doc_type) in enumerate(file_configs):
                            try:
                                # Save uploaded file
                                upload_path = interface.documents_dir / uploaded_file.name
                                with open(upload_path, 'wb') as f:
                                    f.write(uploaded_file.getvalue())
                                
                                # Register with interface
                                doc_id = interface.register_document(str(upload_path), doc_type)
                                
                                result = {
                                    'filename': uploaded_file.name,
                                    'doc_id': doc_id,
                                    'doc_type': doc_type,
                                    'success': True,
                                    'size': len(uploaded_file.getvalue()),
                                    'workflow': workflow_choice
                                }
                                
                                # Execute workflow chain based on selection
                                workflow_result = execute_workflow_chain(interface, doc_id, workflow_choice)
                                result.update(workflow_result)
                                
                                registration_results.append(result)
                                
                            except Exception as e:
                                registration_results.append({
                                    'filename': uploaded_file.name,
                                    'doc_type': doc_type,
                                    'success': False,
                                    'error': str(e)
                                })
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(file_configs))
                    
                    # Show results
                    successful = len([r for r in registration_results if r['success']])
                    st.success(f"✅ {workflow_choice} workflow completed for {successful}/{len(registration_results)} documents")
                    
                    # Workflow-specific results summary
                    workflow_outputs = []
                    for result in registration_results:
                        if result.get('success') and 'outputs' in result:
                            workflow_outputs.extend(result['outputs'])
                    
                    if workflow_outputs:
                        if workflow_choice == "Analyst Report":
                            report_count = len([o for o in workflow_outputs if o.get('type') == 'final_report'])
                            if report_count > 0:
                                st.info(f"📊 {report_count} analyst reports generated")
                        
                        elif workflow_choice == "Section View":
                            browser_ready = len([r for r in registration_results if r.get('browser_ready')])
                            if browser_ready > 0:
                                st.info(f"📖 {browser_ready} documents ready for section browsing")
                        
                        elif workflow_choice == "Peer Review":
                            pdf_count = len([r for r in registration_results if 'peer_review_pdf' in r])
                            if pdf_count > 0:
                                st.info(f"🔍 {pdf_count} peer review PDFs generated")
                    
                    # Detailed results
                    with st.expander("📋 Workflow Execution Details"):
                        for result in registration_results:
                            if result['success']:
                                status = "✅"
                                details = f"ID: {result['doc_id']}, Type: {result['doc_type']}"
                                
                                # Show workflow execution details
                                if 'workflow_type' in result:
                                    details += f", Workflow: {result['workflow_type']}"
                                if 'final_stage' in result:
                                    details += f", Final Stage: {result['final_stage']}"
                                if 'stages_executed' in result and result['stages_executed']:
                                    details += f", Executed: {', '.join(result['stages_executed'])}"
                                if 'stages_skipped' in result and result['stages_skipped']:
                                    details += f", Skipped: {', '.join(result['stages_skipped'])}"
                                
                                # Show workflow-specific outputs
                                if 'action' in result:
                                    if result['action'] == 'open_section_browser':
                                        details += ", Action: Ready for Section Browser"
                                    elif result['action'] == 'generate_peer_pdf':
                                        if 'peer_review_pdf' in result:
                                            details += f", PDF: {result['peer_review_pdf']}"
                                
                            else:
                                status = "❌"
                                details = f"Error: {result.get('error', 'Unknown error')}"
                                if 'workflow_type' in result:
                                    details += f", Workflow: {result['workflow_type']}"
                            
                            st.markdown(f"{status} **{result['filename']}** - {details}")
                    
                    # Auto-pair MVR/VST documents
                    if create_pairs:
                        try:
                            pairs_created = create_document_pairs(interface, registration_results)
                            if pairs_created:
                                st.info(f"🔗 Created {pairs_created} MVR/VST document pairs")
                        except Exception as e:
                            st.warning(f"⚠️ Pairing failed: {e}")
                    
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Selection"):
                    st.rerun()
    
    # Manual document pairing section
    with st.sidebar.expander("🔗 Manual Document Pairing"):
        st.markdown("**Link MVR ↔ VST Documents**")
        
        # Get unpaired documents
        all_docs = interface.list_active_documents() if interface else []
        mvr_docs = [doc for doc in all_docs if doc.get('document_type') == 'mvr']
        vst_docs = [doc for doc in all_docs if doc.get('document_type') == 'vst']
        
        if mvr_docs and vst_docs:
            mvr_options = [f"{doc['document_id']} - {doc['file_name']}" for doc in mvr_docs]
            vst_options = [f"{doc['document_id']} - {doc['file_name']}" for doc in vst_docs]
            
            selected_mvr = st.selectbox("Select MVR Document:", mvr_options)
            selected_vst = st.selectbox("Select VST Document:", vst_options)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔗 Create Pair"):
                    try:
                        mvr_id = selected_mvr.split(' - ')[0]
                        vst_id = selected_vst.split(' - ')[0]
                        
                        result = interface.pair_documents(mvr_id, vst_id)
                        if result['success']:
                            st.success(f"✅ Paired {mvr_id} ↔ {vst_id}")
                            st.rerun()
                        else:
                            st.error("❌ Pairing failed")
                    except Exception as e:
                        st.error(f"❌ Pairing error: {e}")
            
            with col2:
                if st.button("📋 Show Pairs"):
                    try:
                        pairs = interface.list_document_pairs()
                        if pairs:
                            st.markdown("**Active Pairs:**")
                            for pair in pairs:
                                st.markdown(f"• {pair['mvr_document']} ↔ {pair['vst_document']}")
                        else:
                            st.info("No paired documents")
                    except Exception as e:
                        st.error(f"❌ Error listing pairs: {e}")
        else:
            st.info("Need both MVR and VST documents to create pairs")
    
    # Document Collections Management
    with st.sidebar.expander("📚 Document Collections", expanded=False):
        st.markdown("**Create & Manage Collections**")
        
        # Create new collection
        with st.form("create_collection"):
            collection_name = st.text_input("Collection Name", placeholder="MVR Analysis Set 1")
            collection_desc = st.text_area("Description", placeholder="Collection for peer review analysis...")
            
            # Collection type with readable labels
            collection_type_options = {
                '📋 MVR Analysis': 'mvr_analysis',
                '👥 Peer Review': 'peer_review', 
                '📊 Comparison Set': 'comparison_set',
                '🔧 Custom': 'custom'
            }
            
            collection_type_display = st.selectbox("Collection Type", 
                options=list(collection_type_options.keys()))
            collection_type = collection_type_options[collection_type_display]
            
            # Document selection for collection
            all_docs = interface.list_active_documents() if interface else []
            if all_docs:
                doc_options = [f"{doc['document_id']} ({doc['document_type']})" for doc in all_docs]
                selected_for_collection = st.multiselect(
                    "Select Documents:", 
                    options=doc_options,
                    help="Choose documents to include in this collection"
                )
                
                primary_doc = st.selectbox(
                    "Primary Document (optional):", 
                    options=["None"] + doc_options
                )
            
            if st.form_submit_button("📚 Create Collection"):
                if collection_name and selected_for_collection:
                    try:
                        doc_ids = [opt.split(' (')[0] for opt in selected_for_collection]
                        primary_id = None if primary_doc == "None" else primary_doc.split(' (')[0]
                        
                        collection_id = interface.create_collection(
                            name=collection_name,
                            description=collection_desc,
                            document_ids=doc_ids,
                            collection_type=collection_type,
                            primary_document=primary_id
                        )
                        
                        st.success(f"✅ Created collection: {collection_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Collection creation failed: {e}")
                else:
                    st.warning("Please provide name and select documents")
        
        # Manage existing collections
        try:
            collections = interface.list_collections() if interface else []
            if collections:
                st.markdown("**Existing Collections:**")
                for collection in collections[:3]:  # Show first 3
                    with st.expander(f"{collection['name']} ({collection['document_count']} docs)"):
                        st.markdown(f"**Type**: {collection['collection_type']}")
                        st.markdown(f"**Documents**: {collection['document_count']}")
                        st.markdown(f"**Types**: {', '.join(collection['document_types'])}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"ⓘ Info", key=f"info_{collection['collection_id']}"):
                                collection_info = interface.get_collection_info(collection['collection_id'])
                                st.json(collection_info)
                        
                        with col2:
                            if st.button(f"📊 Report", key=f"report_{collection['collection_id']}"):
                                try:
                                    report_result = interface.generate_collection_report(collection['collection_id'])
                                    st.success("✅ Collection report generated!")
                                except Exception as e:
                                    st.error(f"❌ Report failed: {e}")
        except Exception as e:
            st.warning(f"⚠️ Collections unavailable: {e}")
    
    # Main interface
    if not interface.active_documents:
        st.info("📝 No active documents. Upload a document or run demo setup.")
        if st.button("🎭 Create Demo Documents"):
            with st.spinner("Creating demo documents..."):
                try:
                    # Create documents and get results
                    created_docs = create_demo_documents(interface)
                    
                    # Show detailed results
                    successful = [d for d in created_docs if d['success']]
                    failed = [d for d in created_docs if not d['success']]
                    
                    if successful:
                        st.success(f"✅ Created {len(successful)} demo documents successfully!")
                        
                        # Show what was created
                        with st.expander("📋 Documents Created", expanded=True):
                            for doc in successful:
                                st.markdown(f"• **{doc['filename']}** ({doc['doc_type']}) - ID: `{doc['doc_id']}`")
                        
                        if failed:
                            st.warning(f"⚠️ {len(failed)} documents failed to create")
                            for doc in failed:
                                st.error(f"❌ {doc['filename']}: {doc.get('error', 'Unknown error')}")
                        
                        # Brief pause to show results
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ All demo documents failed to create")
                        for doc in failed:
                            st.error(f"❌ {doc['filename']}: {doc.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Demo document creation failed: {e}")
                    st.exception(e)  # Show full error details for debugging
        return
    
    # Document selection
    doc_options = list(interface.active_documents.keys())
    selected_doc = st.selectbox("Select Document:", doc_options)
    
    if not selected_doc:
        return
    
    # Document status and controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📄 Document: {selected_doc}")
        
        # Get current status
        try:
            status = interface.get_document_status(selected_doc)
            
            # Progress bar
            progress = status['progress_percentage']
            st.progress(progress / 100, f"Workflow Progress: {progress:.1f}%")
            
            # Current stage info
            st.markdown(f"""
            **Current Stage**: {status['current_stage']}  
            **Description**: {status['stage_description']}  
            **Last Updated**: {status['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}
            """)
            
            # Checklist management
            st.subheader("📋 Stage Checklist")
            
            # Get SOP guidance for checklist items
            sop_guidance = interface.get_sop_guidance(selected_doc)
            checklist_items = sop_guidance.get('checklist_items', [])
            
            if checklist_items:
                checklist_updates = {}
                current_checklist = status.get('checklist_status', {})
                
                for item in checklist_items[:5]:  # Show first 5 items
                    current_status = current_checklist.get(item, False)
                    new_status = st.checkbox(item, value=current_status, key=f"check_{item}")
                    if new_status != current_status:
                        checklist_updates[item] = new_status
                
                if checklist_updates and st.button("💾 Update Checklist"):
                    result = interface.update_checklist(selected_doc, checklist_updates)
                    if result['stage_complete']:
                        st.success("✅ Stage appears complete!")
                    st.rerun()
            else:
                st.info("No checklist items available for current stage")
            
            # Processing history
            if status['processing_history']:
                with st.expander("📜 Processing History"):
                    history_df = pd.DataFrame(status['processing_history'])
                    if not history_df.empty:
                        st.dataframe(history_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error getting document status: {e}")
    
    with col2:
        st.subheader("🎛️ Controls")
        
        # SOP Guidance
        st.markdown("**💡 SOP Guidance**")
        if st.button("📖 Get Stage Guidance"):
            try:
                guidance = interface.get_sop_guidance(selected_doc)
                with st.expander("SOP Guidance", expanded=True):
                    st.markdown(f"**Confidence**: {guidance['confidence']:.1%}")
                    st.markdown(f"**Guidance**: {guidance['guidance']}")
                    if guidance.get('recommendations'):
                        st.markdown("**Recommendations**:")
                        for rec in guidance['recommendations']:
                            st.markdown(f"- {rec}")
            except Exception as e:
                st.error(f"❌ Error getting SOP guidance: {e}")
        
        # SOP Chat
        st.markdown("**💬 Chat with SOP**")
        chat_question = st.text_input("Ask SOP a question:")
        if chat_question and st.button("💬 Ask SOP"):
            try:
                chat_response = interface.chat_with_sop(selected_doc, chat_question)
                with st.expander("SOP Response", expanded=True):
                    st.markdown(f"**Q**: {chat_question}")
                    st.markdown(f"**A**: {chat_response['guidance']}")
                    st.markdown(f"**Confidence**: {chat_response['confidence']:.1%}")
            except Exception as e:
                st.error(f"❌ Chat failed: {e}")
        
        # Document pairing status
        try:
            paired_doc = interface.get_paired_document(selected_doc)
            if paired_doc:
                st.markdown("**🔗 Document Pairing**")
                st.info(f"Paired with: {paired_doc}")
                
                if st.button("🔓 Unpair Document"):
                    try:
                        result = interface.unpair_document(selected_doc)
                        if result['success']:
                            st.success("✅ Document unpaired successfully")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                    except Exception as e:
                        st.error(f"❌ Unpair failed: {e}")
            else:
                st.markdown("**🔗 Document Pairing**")
                st.warning("Not paired with any document")
        except Exception as e:
            st.warning(f"⚠️ Pairing status unavailable: {e}")
        
        # Workflow controls
        st.markdown("**📊 Individual Document Actions**")
        
        if st.button("📊 Generate Report for This Document"):
            try:
                report_result = interface.generate_stage_report(selected_doc)
                if report_result['report_generated']:
                    st.success("✅ Report generated!")
                    with st.expander("Report Preview"):
                        report_data = report_result['report_data']
                        st.json(report_data)
            except Exception as e:
                st.error(f"❌ Report failed: {e}")
    
    # Use tabs to separate documents and collections
    tab1, tab2 = st.tabs(["📄 Individual Documents", "📚 Document Collections"])
    
    with tab1:
        st.subheader("📄 Individual Documents")
        try:
            active_docs = interface.list_active_documents()
            if active_docs:
                # Individual document operations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("⚡ Advance All Documents"):
                        with st.spinner("Advancing all documents individually..."):
                            advanced_count = 0
                            for doc in active_docs:
                                try:
                                    result = interface.advance_workflow_stage(doc['document_id'], force=True)
                                    if result['success']:
                                        advanced_count += 1
                                except:
                                    continue
                            st.success(f"✅ Advanced {advanced_count} documents")
                            st.rerun()
                
                with col2:
                    if st.button("📊 Generate All Document Reports"):
                        with st.spinner("Generating individual reports..."):
                            report_count = 0
                            for doc in active_docs:
                                try:
                                    interface.generate_stage_report(doc['document_id'])
                                    report_count += 1
                                except:
                                    continue
                            st.success(f"✅ Generated {report_count} individual reports")
                
                with col3:
                    # Show stage distribution
                    stages = [doc['current_stage'] for doc in active_docs]
                    most_common_stage = max(set(stages), key=stages.count) if stages else "None"
                    st.metric("📈 Most Common Stage", most_common_stage.replace('mvr_', '').title())
                
                # Documents table
                docs_df = pd.DataFrame(active_docs)
                if 'last_updated' in docs_df.columns:
                    docs_df['last_updated'] = pd.to_datetime(docs_df['last_updated']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(docs_df, use_container_width=True, hide_index=True)
                
            else:
                st.info("No active documents")
        except Exception as e:
            st.error(f"❌ Error listing documents: {e}")
    
    with tab2:
        st.subheader("📚 Document Collections")
        try:
            collections = interface.list_collections() if hasattr(interface, 'list_collections') else []
            if collections:
                # Collection operations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("⚡ Advance All Collections"):
                        with st.spinner("Advancing all collections..."):
                            try:
                                result = interface.advance_all_collections(force=True)
                                if result['success']:
                                    st.success(f"✅ Advanced {result['successful_advances']}/{result['total_documents']} documents across {result['total_collections']} collections")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result.get('message', 'No collections to advance')}")
                            except Exception as e:
                                st.error(f"❌ Collection advancement failed: {e}")
                
                with col2:
                    if st.button("📊 Generate All Collection Reports"):
                        with st.spinner("Generating collection reports..."):
                            report_count = 0
                            for collection in collections:
                                try:
                                    interface.generate_collection_report(collection['collection_id'])
                                    report_count += 1
                                except:
                                    continue
                            st.success(f"✅ Generated {report_count} collection reports")
                
                with col3:
                    st.metric("📚 Total Collections", len(collections))
                
                # Collections table
                collections_df = pd.DataFrame(collections)
                if 'last_updated' in collections_df.columns:
                    collections_df['last_updated'] = pd.to_datetime(collections_df['last_updated']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(collections_df, use_container_width=True, hide_index=True)
                
            else:
                st.info("No active collections. Create collections in the sidebar!")
        except Exception as e:
            st.error(f"❌ Error listing collections: {e}")

def show_mock_human_interface():
    """Show mock interface when real components aren't available"""
    st.info("🎭 Mock Demo Mode - Showing interface preview")
    
    # Mock multi-file upload
    st.sidebar.subheader("📄 Document Management (Mock)")
    with st.sidebar.expander("➕ Multi-File Upload (Mock)", expanded=True):
        st.markdown("**📁 Drag & Drop Multiple Files**")
        st.markdown("```\n• REV12345_MVR_Analysis.pdf\n• REV12345_VST_Template.docx\n• REV12346_MVR_Analysis.txt\n• Research_Paper_001.pdf\n```")
        
        # Mock batch options
        st.markdown("**⚙️ Batch Options**")
        st.checkbox("Auto-advance to next stage after registration (Mock)", value=False, disabled=True)
        st.checkbox("Auto-pair MVR/VST documents with same REV ID (Mock)", value=True, disabled=True)
        
        if st.button("📥 Register All Files (Mock)"):
            with st.spinner("Mock registration in progress..."):
                time.sleep(2)
            st.success("✅ Mock: Registered 4/4 documents successfully!")
            st.info("🔗 Mock: Created 2 MVR/VST document pairs")
    
    # Mock document selection with more options
    mock_docs = ["REV12345_MVR_Demo", "REV12345_VST_Demo", "REV12346_MVR_Demo", "Research_Paper_001"]
    selected_doc = st.selectbox("Select Document (Mock):", mock_docs)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📄 Document: {selected_doc}")
        
        # Mock progress
        st.progress(0.6, "Workflow Progress: 60%")
        
        st.markdown("""
        **Current Stage**: mvr_qa  
        **Description**: MVR vs VST Comparison Analysis  
        **Status**: In Progress (Mock Mode)
        """)
        
        # Mock checklist
        st.subheader("📋 Stage Checklist (Mock)")
        st.checkbox("REV00000 format ID extracted", value=True, disabled=True)
        st.checkbox("Document type classified (MVR/VST)", value=True, disabled=True)  
        st.checkbox("Section-by-section comparison completed", value=False, disabled=True)
        st.checkbox("Domain RAG validation performed", value=False, disabled=True)
    
    with col2:
        st.subheader("🎛️ Controls (Mock)")
        
        if st.button("📖 Get SOP Guidance"):
            st.info("Mock SOP Guidance: For MVR vs VST comparison, ensure REV numbers match and perform section-by-section analysis...")
        
        chat_question = st.text_input("Ask SOP (Mock):")
        if chat_question:
            st.info(f"Mock SOP Response: {chat_question} - This would be answered by the SOP system in live mode.")
        
        if st.button("▶️ Advance Stage"):
            st.success("Mock: Advanced to next stage!")
        
        if st.button("📊 Generate Report"):
            st.success("Mock: Report generated!")
    
    # Mock bulk operations
    st.subheader("📚 All Active Documents (Mock)")
    
    # Mock metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⚡ Total Docs", 4)
    with col2:
        st.metric("🔗 Paired Docs", 2)
    with col3:
        st.metric("📈 In Progress", 3)
    with col4:
        st.metric("✅ Completed", 1)
    
    # Mock document selection for batch ops
    mock_all_docs = ["REV12345_MVR_Demo", "REV12345_VST_Demo", "REV12346_MVR_Demo", "Research_Paper_001"]
    selected_for_batch = st.multiselect(
        "Select documents for batch operations (Mock):",
        options=mock_all_docs,
        default=["REV12345_MVR_Demo", "REV12346_MVR_Demo"],
        help="Mock multi-select for batch processing"
    )
    
    if selected_for_batch:
        st.markdown(f"**Selected {len(selected_for_batch)} documents for batch operations:**")
        
        batch_col1, batch_col2, batch_col3 = st.columns(3)
        
        with batch_col1:
            if st.button(f"⚡ Advance Selected ({len(selected_for_batch)}) - Mock"):
                st.success(f"✅ Mock: Advanced {len(selected_for_batch)}/{len(selected_for_batch)} documents")
        
        with batch_col2:
            if st.button(f"📊 Report Selected ({len(selected_for_batch)}) - Mock"):
                st.success(f"✅ Mock: Generated {len(selected_for_batch)} reports")
        
        with batch_col3:
            if st.button(f"💬 Batch SOP Check ({len(selected_for_batch)}) - Mock"):
                st.info("📋 Mock SOP Compliance Summary: All documents compliant")
    
    # Mock documents table
    mock_data = {
        'document_id': ['REV12345_MVR_Demo', 'REV12345_VST_Demo', 'REV12346_MVR_Demo', 'Research_Paper_001'],
        'document_type': ['mvr', 'vst', 'mvr', 'research'],
        'current_stage': ['mvr_qa', 'mvr_tag', 'mvr_peer', 'completed'],
        'progress_percentage': [50, 25, 75, 100],
        'last_updated': ['2024-01-15 14:30', '2024-01-15 14:25', '2024-01-15 14:20', '2024-01-15 14:15']
    }
    
    mock_df = pd.DataFrame(mock_data)
    st.dataframe(mock_df, use_container_width=True, hide_index=True)

def show_watchdog_monitoring():
    """Show Watchdog file monitoring interface"""
    st.header("👁️ Watchdog File Monitoring")
    
    if not WATCHDOG_TEST_AVAILABLE:
        st.error("❌ Watchdog Test Framework not available.")
        st.markdown("""
        **Missing Components:**
        - `test_drop_zones_watchdog.py` not found
        
        **To enable this feature:**
        1. Ensure the test script is available
        2. Check Watchdog dependencies are installed
        3. Restart the Streamlit app
        """)
        
        # Show mock monitoring interface
        show_mock_watchdog_interface()
        return
    
    st.markdown("""
    **Automated file monitoring and processing using Watchdog**
    
    This demonstrates how documents are automatically detected and processed when dropped in monitored folders.
    """)
    
    # Test controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Test Controls")
        test_duration = st.slider("Test Duration (seconds)", 5, 30, 10)
        
        if st.button("🚀 Run Watchdog Test"):
            with st.spinner("Running Watchdog monitoring test..."):
                try:
                    # Initialize test framework
                    test_framework = DropZoneTestFramework()
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run test with progress updates
                    for i in range(test_duration):
                        progress_bar.progress((i + 1) / test_duration)
                        status_text.text(f"Running test... {i + 1}/{test_duration} seconds")
                        time.sleep(1)
                    
                    # Run the actual test
                    result = test_framework.test_watchdog_monitoring(duration=test_duration)
                    
                    # Store results
                    st.session_state.watchdog_results.append(result)
                    
                    # Cleanup
                    test_framework.cleanup()
                    
                    if result.get('success', False):
                        st.success("✅ Watchdog test completed successfully!")
                    else:
                        st.error("❌ Watchdog test failed")
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Test failed: {e}")
    
    with col2:
        st.subheader("📊 Zone Configuration")
        
        # Mock zone configuration display
        zones_config = {
            'mvr_documents': {
                'patterns': ['*.pdf', '*.docx', '*.txt'],
                'events': ['created', 'modified'],
                'max_size': '50MB'
            },
            'vst_documents': {
                'patterns': ['*.pdf', '*.docx'],
                'events': ['created'],
                'max_size': '10MB'
            },
            'research_papers': {
                'patterns': ['*.pdf'],
                'events': ['created'],
                'max_size': '100MB'
            }
        }
        
        for zone_name, config in zones_config.items():
            with st.expander(f"📁 {zone_name}"):
                st.json(config)
    
    # Test results
    if st.session_state.watchdog_results:
        st.subheader("📈 Test Results")
        
        # Latest result
        latest_result = st.session_state.watchdog_results[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Files Detected", latest_result.get('statistics', {}).get('files_detected', 0))
        
        with col2:
            st.metric("Files Processed", latest_result.get('statistics', {}).get('files_processed', 0))
        
        with col3:
            processing_rate = latest_result.get('statistics', {}).get('files_per_minute', 0)
            st.metric("Processing Rate", f"{processing_rate:.1f}/min")
        
        # Detailed results
        with st.expander("📋 Detailed Test Results"):
            st.json(latest_result)
        
        # Results history
        if len(st.session_state.watchdog_results) > 1:
            with st.expander("📜 Test History"):
                history_df = pd.DataFrame([
                    {
                        'timestamp': r.get('timestamp', 'Unknown'),
                        'success': r.get('success', False),
                        'files_processed': r.get('statistics', {}).get('files_processed', 0),
                        'duration': r.get('duration_seconds', 0)
                    }
                    for r in st.session_state.watchdog_results
                ])
                st.dataframe(history_df, use_container_width=True)

def show_mock_watchdog_interface():
    """Show mock watchdog interface for demonstration"""
    st.info("🎭 Mock Demo Mode - Showing monitoring preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Test Controls (Mock)")
        test_duration = st.slider("Test Duration (seconds)", 5, 30, 10)
        
        if st.button("🚀 Run Mock Watchdog Test"):
            # Mock test execution
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(10):
                progress_bar.progress((i + 1) / 10)
                status_text.text(f"Mock test running... {i + 1}/10")
                time.sleep(0.5)
            
            # Mock results
            st.success("✅ Mock test completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Detected", 5)
            with col2:
                st.metric("Files Processed", 4)
            with col3:
                st.metric("Processing Rate", "24.0/min")
    
    with col2:
        st.subheader("📊 Zone Configuration (Mock)")
        
        zones_config = {
            'mvr_documents': {
                'patterns': ['*.pdf', '*.docx', '*.txt'],
                'events': ['created', 'modified'],
                'max_size': '50MB'
            },
            'vst_documents': {
                'patterns': ['*.pdf', '*.docx'],
                'events': ['created'],
                'max_size': '10MB'
            },
            'research_papers': {
                'patterns': ['*.pdf'],
                'events': ['created'],
                'max_size': '100MB'
            }
        }
        
        for zone_name, config in zones_config.items():
            with st.expander(f"📁 {zone_name}"):
                st.json(config)

if __name__ == "__main__":
    main()