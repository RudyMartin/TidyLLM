#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced QA Document Processing Demo

Streamlit interface for smart QA document processing with intelligent
orchestrator selection and user preference management.
"""

# Environment setup
from config.setup import setup_env
setup_env()

import streamlit as st
import zipfile
import tempfile
from typing import List, Dict, Any
from datetime import datetime
import json

from backend.mcp.orchestrators.smart_orchestrator_router import SmartOrchestratorRouter
import re


def extract_zip_safely(zip_file, max_size_mb=50, max_files=100):
    """Safely extract ZIP file with security checks"""
    # Simplified version for demo - same as original qa_demo.py
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_files = []
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir():
                        continue
                    
                    file_path = file_info.filename
                    content = zip_ref.read(file_path)
                    
                    extracted_files.append({
                        'name': file_path,
                        'content': content,
                        'type': 'unknown',
                        'size': len(content)
                    })
            
            return extracted_files
    except Exception as e:
        return f"Error extracting ZIP file: {str(e)}"


def validate_review_id(review_id):
    """Validate Review ID format"""
    if not review_id:
        return False, "Review ID is required"
    
    # Check format REVXXXXX
    pattern = r'^REV\d{5}$'
    if not re.match(pattern, review_id):
        return False, "Review ID must be in format REVXXXXX (e.g., REV00001)"
    
    return True, "Valid Review ID format"


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Enhanced QA Document Processing",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Enhanced QA Document Processing")
    st.markdown("### Smart Orchestrator Selection with Intelligent Routing")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Main App", use_container_width=True):
            st.switch_page("main.py")
    
    with col2:
        if st.button("📋 Basic QA Demo", use_container_width=True):
            st.switch_page("qa_demo.py")
    
    with col3:
        if st.button("📊 MCP Dashboard", use_container_width=True):
            st.switch_page("mcp_dashboard.py")
    
    st.markdown("---")
    
    # Initialize smart router
    if 'smart_router' not in st.session_state:
        st.session_state.smart_router = SmartOrchestratorRouter()
    
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Smart Configuration")
        
        # Review ID
        review_id = st.text_input(
            "Review ID", 
            placeholder="REV00001",
            help="Enter review number in format REVXXXXX"
        )
        
        if review_id:
            is_valid, message = validate_review_id(review_id)
            if not is_valid:
                st.error(message)
            else:
                st.success("✅ Valid Review ID format")
        
        st.markdown("---")
        
        # Orchestrator Selection
        st.subheader("🎯 Orchestrator Selection")
        
        selection_mode = st.radio(
            "Selection Mode",
            ["🤖 Smart Auto-Selection", "👤 Manual Selection"],
            help="Choose between automatic intelligent routing or manual orchestrator selection"
        )
        
        if selection_mode == "👤 Manual Selection":
            st.markdown("#### Manual Orchestrator Choice")
            
            orchestrator_choice = st.selectbox(
                "Select Orchestrator",
                [
                    "basic_qa",
                    "expert_qa", 
                    "llm_enhanced"
                ],
                format_func=lambda x: {
                    "basic_qa": "📄 Basic QA (Simple Processing)",
                    "expert_qa": "🔍 Expert QA (Professional Reviews)",
                    "llm_enhanced": "🧠 LLM Enhanced (AI-Powered Analysis)"
                }[x],
                help="Manually select which orchestrator to use"
            )
            
            # Show orchestrator capabilities
            capabilities = st.session_state.smart_router.orchestrator_capabilities[orchestrator_choice]
            st.info(f"**Capabilities**: {', '.join(capabilities['capabilities'])}")
            st.info(f"**Complexity**: {capabilities['complexity']}")
            st.info(f"**Expertise Level**: {capabilities['expertise_level']}")
        
        st.markdown("---")
        
        # User Preferences
        st.subheader("👤 User Preferences")
        
        expertise_level = st.selectbox(
            "Expertise Level",
            ["basic", "enhanced", "expert"],
            help="Your expertise level for processing"
        )
        
        llm_enhancement = st.checkbox(
            "Enable LLM Enhancement",
            value=False,
            help="Use LLM-powered analysis and enhancement"
        )
        
        compliance_focus = st.checkbox(
            "Compliance Focus",
            value=False,
            help="Focus on regulatory compliance and standards"
        )
        
        budget_limit = st.number_input(
            "Budget Limit ($)",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=5.0,
            help="Maximum budget for LLM processing"
        )
        
        # Workflow Context
        st.markdown("---")
        st.subheader("📋 Workflow Context")
        
        team_num = st.text_input("Team Number", value="QA Team 1")
        process_name = st.text_input("Process Name", value="QA Validation Review")
        reviewer_name = st.text_input("Reviewer Name", value="Alex")
        model_type = st.selectbox(
            "Model Type",
            ["Research Document", "Machine Learning Model", "Statistical Model", "Other"]
        )
        risk_tier = st.selectbox(
            "Risk Tier",
            ["Low", "Medium", "High", "Critical"]
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Document Upload")
        
        # Custom prompt
        st.subheader("💬 Custom Prompt (Optional)")
        custom_prompt = st.text_area(
            "Enter a custom prompt for QA processing",
            placeholder="Enter your custom QA instructions here...",
            height=100,
            help="Provide specific instructions for how to process and analyze the documents"
        )
        
        st.markdown("---")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload QA Documents",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'md', 'zip', 'gif', 'jpg', 'jpeg', 'png', 'svg', 'bmp', 'tiff', 'webp', 'py', 'ipynb', 'r', 'sql', 'sas', 'mat', 'm', 'sav', 'dta', 'rds', 'rdata', 'parquet', 'feather', 'h5', 'hdf5', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf', 'tex', 'bib', 'ris', 'enw'],
            accept_multiple_files=True,
            help="Upload one or more documents for QA processing"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            # Process uploaded files
            all_files = []
            for file in uploaded_files:
                if file.name.endswith('.zip'):
                    extracted = extract_zip_safely(file)
                    if isinstance(extracted, list):
                        all_files.extend(extracted)
                    else:
                        st.error(f"Error extracting {file.name}: {extracted}")
                else:
                    all_files.append({
                        'name': file.name,
                        'content': file.read(),
                        'type': file.type,
                        'size': file.size
                    })
            
            # Display file summary
            if all_files:
                st.subheader("📊 File Summary")
                file_types = {}
                total_size = 0
                
                for file in all_files:
                    file_type = file.get('type', 'unknown')
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    total_size += file.get('size', 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", len(all_files))
                with col2:
                    st.metric("Total Size", f"{total_size / 1024:.1f} KB")
                with col3:
                    st.metric("File Types", len(file_types))
                
                # Show file type breakdown
                if file_types:
                    st.write("**File Type Breakdown:**")
                    for file_type, count in file_types.items():
                        st.write(f"  - {file_type}: {count} files")
        
        st.markdown("---")
        
        # Smart Analysis Preview
        if uploaded_files and all_files:
            st.subheader("🔍 Smart Analysis Preview")
            
            # Prepare user preferences
            user_preferences = {
                'expertise_level': expertise_level,
                'llm_enhancement': llm_enhancement,
                'compliance_focus': compliance_focus,
                'budget_limit': budget_limit
            }
            
            if selection_mode == "👤 Manual Selection":
                user_preferences['preferred_orchestrator'] = orchestrator_choice
            
            # Prepare workflow context
            workflow_context = {
                'team_num': team_num,
                'process_name': process_name,
                'reviewer_name': reviewer_name,
                'review_id': review_id,
                'model_type': model_type,
                'risk_tier': risk_tier,
                'custom_prompt': custom_prompt
            }
            
            # Analyze documents
            analysis = st.session_state.smart_router.analyze_document_requirements(
                all_files, user_preferences, workflow_context
            )
            
            # Display analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Document Analysis:**")
                st.write(f"  - Document Count: {analysis['document_count']}")
                st.write(f"  - Total Size: {analysis['total_size'] / 1024:.1f} KB")
                st.write(f"  - Complexity Score: {analysis['complexity_score']}")
                st.write(f"  - File Types: {', '.join(analysis['file_types'])}")
            
            with col2:
                st.write("**Requirements Detected:**")
                if analysis['qa_requirements']:
                    st.write(f"  - QA: {', '.join(analysis['qa_requirements'])}")
                if analysis['llm_requirements']:
                    st.write(f"  - LLM: {', '.join(analysis['llm_requirements'])}")
                if analysis['compliance_requirements']:
                    st.write(f"  - Compliance: {', '.join(analysis['compliance_requirements'])}")
            
            # Predict orchestrator selection
            if selection_mode == "🤖 Smart Auto-Selection":
                orchestrator_type, orchestrator, selection_reason = st.session_state.smart_router.select_orchestrator(
                    analysis, user_preferences
                )
                
                st.info(f"**🤖 Predicted Orchestrator**: {orchestrator_type}")
                st.info(f"**Reasoning**: {selection_reason['reasoning']}")
                
                # Show scores
                scores = selection_reason['scores']
                st.write("**Orchestrator Scores:**")
                for orchestrator_name, score in scores.items():
                    display_name = {
                        'basic_qa': '📄 Basic QA',
                        'expert_qa': '🔍 Expert QA',
                        'llm_enhanced': '🧠 LLM Enhanced'
                    }[orchestrator_name]
                    st.write(f"  - {display_name}: {score}")
        
        # Process button
        if uploaded_files and all_files and review_id and validate_review_id(review_id)[0]:
            if st.button("🚀 Process Documents Smart", type="primary", use_container_width=True):
                with st.spinner("Processing documents with smart routing..."):
                    # Prepare files for processing
                    files_data = []
                    for file in all_files:
                        files_data.append({
                            "filename": file['name'],
                            "content": file['content'],
                            "type": file['type'],
                            "size": file['size']
                        })
                    
                    # Process with smart router
                    result = st.session_state.smart_router.process_documents_smart(
                        files=files_data,
                        user_preferences=user_preferences,
                        workflow_context=workflow_context
                    )
                    
                    st.session_state.workflow_result = result
                    st.rerun()
    
    with col2:
        st.header("📈 Processing Status")
        
        if st.session_state.workflow_result:
            result = st.session_state.workflow_result
            
            if result["status"] == "completed":
                st.success("✅ Processing Completed!")
                
                # Show smart routing info
                if 'smart_routing' in result:
                    routing = result['smart_routing']
                    st.subheader("🤖 Smart Routing Results")
                    
                    orchestrator_name = routing['selected_orchestrator']
                    display_name = {
                        'basic_qa': '📄 Basic QA',
                        'expert_qa': '🔍 Expert QA',
                        'llm_enhanced': '🧠 LLM Enhanced'
                    }.get(orchestrator_name, orchestrator_name)
                    
                    st.info(f"**Selected**: {display_name}")
                    
                    if 'reasoning' in routing['selection_reasoning']:
                        st.write(f"**Reasoning**: {routing['selection_reasoning']['reasoning']}")
                
                # Show processing results
                st.subheader("📊 Processing Results")
                
                if "document_result" in result:
                    st.write("✅ Document processing completed")
                
                if "extraction_result" in result:
                    st.write("✅ Field extraction completed")
                
                if "report_result" in result:
                    st.write("✅ Report generation completed")
                
                # Download results
                if st.button("📥 Download Results"):
                    # Create downloadable result
                    result_json = json.dumps(result, indent=2, default=str)
                    st.download_button(
                        "Download JSON Results",
                        result_json,
                        file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            elif result["status"] == "failed":
                st.error("❌ Processing Failed")
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        
        else:
            st.info("📋 Ready to process documents")
            st.write("Upload files and configure settings to begin processing.")


if __name__ == "__main__":
    main()
