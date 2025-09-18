#!/usr/bin/env python3
"""
Boss MVR - Simple Working Version
=================================

Simplified MVR processing focused on JSON files only.
Uses V1 processors without problematic dependencies.
"""

import streamlit as st
import sys
import tempfile
import time
import json
import os
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Boss MVR - Simple",
    page_icon="[BOARD]", 
    layout="wide"
)

# Import V1 processors
processors_loaded = {}

try:
    from v1_workflows.mvr_json.mvr_qa_gap_analyzer import MVRQAGapAnalyzer
    processors_loaded['MVR Gap Analyzer'] = "[OK] LOADED"
except ImportError as e:
    processors_loaded['MVR Gap Analyzer'] = f"[FAIL] {e}"

try:
    from v1_workflows.mvr_json.mvr_workflow_router import MVRWorkflowRouter
    processors_loaded['MVR Router'] = "[OK] LOADED"  
except ImportError as e:
    processors_loaded['MVR Router'] = f"[FAIL] {e}"

# Skip QA processor for now to avoid TidyLLM issues
processors_loaded['QA Processor'] = "[SKIP] SKIPPED (Focus on JSON only)"

def main():
    """Simple MVR interface focusing on JSON processing"""
    
    # Header
    st.title("[BOARD] Boss MVR - Simple & Fast")
    st.markdown("**JSON MVR Processing Only - No Dependencies Issues**")
    
    # Show processor status
    st.subheader("[FIX] Processor Status")
    for name, status in processors_loaded.items():
        if "[OK]" in status:
            st.success(f"**{name}**: {status}")
        elif "[SKIP]" in status:
            st.warning(f"**{name}**: {status}")
        else:
            st.error(f"**{name}**: {status}")
    
    # Connection status
    col1, col2, col3 = st.columns(3)
    with col1:
        if "[OK]" in processors_loaded.get('MVR Gap Analyzer', ''):
            st.success("[OK] Gap Analysis")
        else:
            st.error("[FAIL] Gap Analysis")
    with col2:
        if "[OK]" in processors_loaded.get('MVR Router', ''):
            st.success("[OK] Workflow Router")
        else:
            st.error("[FAIL] Workflow Router")
    with col3:
        st.info("[BOARD] JSON Focus")
    
    st.markdown("---")
    
    # File upload - JSON only
    st.subheader("[FILES] Upload JSON MVR Files")
    
    uploaded_files = st.file_uploader(
        "Drop your JSON MVR files here",
        type=['json'],
        accept_multiple_files=True,
        help="JSON MVR documents for gap analysis and workflow routing"
    )
    
    if uploaded_files:
        st.markdown(f"**[DATA] {len(uploaded_files)} JSON files ready for processing**")
        
        # Show files
        for i, file in enumerate(uploaded_files, 1):
            st.text(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")
        
        st.markdown("---")
        
        # Process button - only if both processors loaded
        can_process = all("[OK]" in processors_loaded.get(key, '') for key in ['MVR Gap Analyzer', 'MVR Router'])
        
        if can_process:
            if st.button(f"[RUN] PROCESS {len(uploaded_files)} JSON FILES", type="primary", use_container_width=True):
                
                # Initialize processors
                gap_analyzer = MVRQAGapAnalyzer()
                workflow_router = MVRWorkflowRouter()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("[SEARCH] Processing JSON files...")
                progress_bar.progress(10)
                
                results = []
                total_files = len(uploaded_files)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    file_progress = 20 + (i * 70 // total_files)
                    progress_bar.progress(file_progress)
                    status_text.text(f"[BOARD] Processing: {uploaded_file.name}")
                    
                    # Save file temporarily  
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Run gap analysis
                        gap_result = gap_analyzer.analyze_rev_file(tmp_file_path)
                        judge_report = gap_analyzer.generate_judge_report(gap_result)
                        
                        # Run workflow routing  
                        routing_result = workflow_router.route_json_document(tmp_file_path)
                        
                        result = {
                            'file': uploaded_file.name,
                            'status': 'SUCCESS',
                            'model_id': gap_result.model_id,
                            'overall_score': judge_report['overall']['weighted_score'],
                            'missing_sections': len(gap_result.missing_sections),
                            'incomplete_sections': len(gap_result.incomplete_sections), 
                            'findings': len(gap_result.findings),
                            'pending_items': len(gap_result.pending_items),
                            'workflow_route': routing_result['workflow_route']['workflow_path'],
                            'priority': routing_result['workflow_route']['priority']
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        results.append({
                            'file': uploaded_file.name,
                            'status': 'ERROR', 
                            'error': str(e)
                        })
                    
                    finally:
                        # Cleanup temp file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                
                # Complete
                progress_bar.progress(100)
                status_text.text("[OK] Processing complete!")
                
                # Show results
                st.success("[DONE] **JSON MVR Processing Complete!**")
                
                successful = len([r for r in results if r['status'] == 'SUCCESS'])
                errors = len([r for r in results if r['status'] == 'ERROR'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processed", successful)
                with col2:
                    st.metric("Errors", errors)
                with col3:
                    st.metric("Success Rate", f"{(successful/len(results)*100):.0f}%")
                
                # Detailed results
                st.markdown("### [DATA] Results")
                
                for result in results:
                    if result['status'] == 'SUCCESS':
                        with st.expander(f"[OK] {result['file']} - Score: {result['overall_score']}/100"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Score", f"{result['overall_score']}/100")
                                st.metric("Missing", result['missing_sections'])
                                st.metric("Findings", result['findings'])
                                
                            with col2:
                                st.metric("Incomplete", result['incomplete_sections'])  
                                st.metric("Pending", result['pending_items'])
                                st.text(f"Model: {result['model_id']}")
                            
                            st.markdown(f"**Route**: {result['workflow_route']}")
                            st.markdown(f"**Priority**: {result['priority']}")
                    
                    else:
                        with st.expander(f"[FAIL] {result['file']} - Error"):
                            st.error(result['error'])
                
                # Download results
                if results:
                    report_data = {
                        'timestamp': datetime.now().isoformat(),
                        'processor': 'Simple_MVR_V1',
                        'total_files': len(results),
                        'successful': successful,
                        'results': results
                    }
                    
                    st.download_button(
                        label="[FILES] Download Results",
                        data=json.dumps(report_data, indent=2),
                        file_name=f"mvr_results_{int(time.time())}.json",
                        mime="application/json"
                    )
        else:
            st.error("[FAIL] Cannot process - Required processors not loaded")
    
    else:
        st.info("Upload JSON MVR files to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("**Simple MVR System** | JSON Focus | V1 Processors | Fast & Reliable")

if __name__ == "__main__":
    main()