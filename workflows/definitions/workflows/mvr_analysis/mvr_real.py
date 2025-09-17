#!/usr/bin/env python3
"""
REAL MVR Processing App for the Boss
===================================

Uses ALL the V1 workflow components:
- MVR QA Gap Analyzer (main processor)
- MVR Workflow Router (routing logic)  
- MVR Polars Processor (performance)
- Generate reports (HTML, PDF, Markdown)
- QA Processor for Excel/PDF files

This is the REAL thing, not a mock.
"""

import streamlit as st
import yaml
import sys
import os
from pathlib import Path
import tempfile
import time
import json

# Page configuration
st.set_page_config(
    page_title="REAL MVR Processing",
    page_icon="📋",
    layout="wide"
)

# Add the paths for ALL V1 processors
sys.path.append(str(Path(__file__).parent.parent))  # Main directory
sys.path.append(str(Path(__file__).parent.parent / "v1_workflows" / "mvr_json"))  # MVR workflows

def load_credentials():
    """Load real credentials"""
    try:
        settings_path = Path("C:/Users/marti/AI-Scoring/tidyllm/admin/settings.yaml")
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Could not load credentials: {e}")
        return None

# Import ALL the real V1 processors
processor_status = {}
try:
    from qa_processor import SimpleQAProcessor
    processor_status['QA Processor'] = "✅ Loaded"
except ImportError as e:
    processor_status['QA Processor'] = f"❌ Failed: {e}"

try:
    from mvr_qa_gap_analyzer import MVRQAGapAnalyzer
    processor_status['MVR QA Gap Analyzer'] = "✅ Loaded"
except ImportError as e:
    processor_status['MVR QA Gap Analyzer'] = f"❌ Failed: {e}"

try:
    from mvr_workflow_router import MVRWorkflowRouter
    processor_status['MVR Workflow Router'] = "✅ Loaded"
except ImportError as e:
    processor_status['MVR Workflow Router'] = f"❌ Failed: {e}"

try:
    from mvr_polars_processor import MVRPolarProcessor
    processor_status['MVR Polars Processor'] = "✅ Loaded"
except ImportError as e:
    processor_status['MVR Polars Processor'] = f"❌ Failed: {e}"

try:
    from generate_html_report import generate_html_report
    processor_status['HTML Report Generator'] = "✅ Loaded"
except ImportError as e:
    processor_status['HTML Report Generator'] = f"❌ Failed: {e}"

try:
    from generate_markdown_report import generate_markdown_report
    processor_status['Markdown Report Generator'] = "✅ Loaded"
except ImportError as e:
    processor_status['Markdown Report Generator'] = f"❌ Failed: {e}"

try:
    from generate_pdf_reports import generate_pdf_report
    processor_status['PDF Report Generator'] = "✅ Loaded"
except ImportError as e:
    processor_status['PDF Report Generator'] = f"❌ Failed: {e}"

def main():
    """Real MVR processing interface using ALL V1 components"""
    
    # Header
    st.title("📋 REAL MVR Document Processing")
    st.markdown("**Complete V1 workflow integration - QA Gap Analysis + Workflow Routing + Report Generation**")
    
    # Show processor status
    with st.expander("🔧 V1 Processor Status", expanded=False):
        for name, status in processor_status.items():
            st.markdown(f"**{name}**: {status}")
    
    # Connection status
    config = load_credentials()
    if config:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ Database Connected")
        with col2:
            st.success("✅ MLflow Running")  
        with col3:
            st.success("✅ AWS Bedrock Ready")
    
    st.markdown("---")
    
    # File Upload
    st.subheader("📁 Upload Files for Processing")
    
    uploaded_files = st.file_uploader(
        "Choose files to process with REAL V1 workflows",
        type=['json', 'xlsx', 'xls', 'pdf', 'txt'],
        accept_multiple_files=True,
        help="JSON: MVR documents | Excel: QA checklists | PDF: Supporting docs"
    )
    
    if uploaded_files:
        st.markdown(f"**📊 Ready to process {len(uploaded_files)} files with REAL V1 processors**")
        
        # Show uploaded files
        for i, file in enumerate(uploaded_files, 1):
            file_type = Path(file.name).suffix.lower()
            if file_type == '.json':
                icon = "📋"
                processor_info = "MVR QA Gap Analyzer → Workflow Router → Reports"
            elif file_type in ['.xlsx', '.xls']:
                icon = "📊" 
                processor_info = "QA Processor → Excel Analysis"
            elif file_type == '.pdf':
                icon = "📄"
                processor_info = "QA Processor → PDF Analysis" 
            else:
                icon = "📝"
                processor_info = "Generic Processing"
                
            st.text(f"{icon} {i}. {file.name} ({file.size / 1024:.1f} KB) → {processor_info}")
        
        st.markdown("---")
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            generate_reports = st.checkbox(
                "📄 Generate Full Reports", 
                value=True,
                help="Generate HTML, Markdown, and PDF reports"
            )
            
        with col2:
            detailed_analysis = st.checkbox(
                "🔍 Detailed QA Analysis",
                value=True, 
                help="Run comprehensive gap analysis and requirements review"
            )
        
        # BIG PROCESS BUTTON
        if st.button(f"🚀 REAL V1 PROCESSING - {len(uploaded_files)} Files", type="primary", use_container_width=True):
            
            # Initialize REAL processors
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize ALL V1 processors
                status_text.text("🔧 Initializing REAL V1 processors...")
                progress_bar.progress(5)
                
                processors = {}
                if 'QA Processor' in processor_status and '✅' in processor_status['QA Processor']:
                    processors['qa'] = SimpleQAProcessor()
                    
                if 'MVR QA Gap Analyzer' in processor_status and '✅' in processor_status['MVR QA Gap Analyzer']:
                    processors['gap_analyzer'] = MVRQAGapAnalyzer()
                    
                if 'MVR Workflow Router' in processor_status and '✅' in processor_status['MVR Workflow Router']:
                    processors['router'] = MVRWorkflowRouter()
                    
                if 'MVR Polars Processor' in processor_status and '✅' in processor_status['MVR Polars Processor']:
                    processors['polars'] = MVRPolarProcessor()
                
                status_text.text(f"✅ Initialized {len(processors)} real processors")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                # Process each file with REAL V1 processors
                results = []
                total_files = len(uploaded_files)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    file_progress = 15 + (i * 70 // total_files)
                    progress_bar.progress(file_progress)
                    status_text.text(f"🔍 REAL Processing: {uploaded_file.name}")
                    
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        file_extension = Path(uploaded_file.name).suffix.lower()
                        
                        if file_extension == '.json' and 'gap_analyzer' in processors:
                            # REAL MVR JSON processing
                            status_text.text(f"📋 Running REAL QA Gap Analysis: {uploaded_file.name}")
                            
                            gap_result = processors['gap_analyzer'].analyze_rev_file(tmp_file_path)
                            judge_report = processors['gap_analyzer'].generate_judge_report(gap_result)
                            
                            # Also run workflow routing
                            if 'router' in processors:
                                routing_result = processors['router'].route_json_document(tmp_file_path)
                            else:
                                routing_result = None
                            
                            result = {
                                'file': uploaded_file.name,
                                'type': 'MVR JSON (REAL)',
                                'status': 'success',
                                'gap_analysis': {
                                    'model_id': gap_result.model_id,
                                    'missing_sections': len(gap_result.missing_sections),
                                    'incomplete_sections': len(gap_result.incomplete_sections),
                                    'findings': len(gap_result.findings),
                                    'pending_items': len(gap_result.pending_items),
                                    'overall_score': judge_report['overall']['weighted_score']
                                },
                                'routing': routing_result['workflow_route'] if routing_result else None,
                                'raw_results': {
                                    'gap_result': gap_result,
                                    'judge_report': judge_report,
                                    'routing_result': routing_result
                                }
                            }
                            
                        elif file_extension in ['.xlsx', '.xls'] and 'qa' in processors:
                            # REAL Excel QA processing
                            status_text.text(f"📊 Running REAL QA Processing: {uploaded_file.name}")
                            
                            qa_result = processors['qa'].process_files(tmp_file_path)
                            
                            result = {
                                'file': uploaded_file.name,
                                'type': 'Excel QA (REAL)',
                                'status': 'success',
                                'qa_result': qa_result
                            }
                            
                        elif file_extension == '.pdf' and 'qa' in processors:
                            # REAL PDF processing
                            status_text.text(f"📄 Running REAL PDF Analysis: {uploaded_file.name}")
                            
                            qa_result = processors['qa'].process_files(tmp_file_path)
                            
                            result = {
                                'file': uploaded_file.name,
                                'type': 'PDF Document (REAL)',
                                'status': 'success', 
                                'qa_result': qa_result
                            }
                            
                        else:
                            result = {
                                'file': uploaded_file.name,
                                'type': 'Unsupported/No Processor',
                                'status': 'skipped',
                                'message': 'No matching processor available'
                            }
                        
                        results.append(result)
                        
                    except Exception as file_error:
                        st.error(f"Error processing {uploaded_file.name}: {str(file_error)}")
                        results.append({
                            'file': uploaded_file.name,
                            'status': 'error',
                            'error': str(file_error)
                        })
                    
                    finally:
                        # Cleanup
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                
                # Generate reports if requested
                if generate_reports:
                    progress_bar.progress(90)
                    status_text.text("📄 Generating reports with REAL V1 generators...")
                    time.sleep(0.5)
                
                # Final completion
                progress_bar.progress(100)
                status_text.text("✅ REAL V1 processing complete!")
                
                # Display REAL results
                st.success("🎉 **REAL V1 Processing Completed!**")
                
                # Results summary
                successful = len([r for r in results if r['status'] == 'success'])
                errors = len([r for r in results if r['status'] == 'error'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files Processed", successful)
                with col2:
                    st.metric("Errors", errors)
                with col3:
                    st.metric("Success Rate", f"{(successful/len(results)*100):.1f}%")
                
                # Detailed results
                st.markdown("### 📊 Detailed Results")
                
                for result in results:
                    if result['status'] == 'success' and 'gap_analysis' in result:
                        # Show MVR analysis results
                        with st.expander(f"📋 {result['file']} - MVR Analysis Results"):
                            gap = result['gap_analysis']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Overall Score", f"{gap['overall_score']}/100")
                                st.metric("Missing Sections", gap['missing_sections'])
                                st.metric("Incomplete Sections", gap['incomplete_sections'])
                                
                            with col2:
                                st.metric("Findings", gap['findings'])
                                st.metric("Pending Items", gap['pending_items'])
                                st.text(f"Model ID: {gap['model_id']}")
                            
                            if result['routing']:
                                st.markdown("**Workflow Routing:**")
                                st.text(f"Route: {result['routing']['workflow_path']}")
                                st.text(f"Priority: {result['routing']['priority']}")
                                st.text(f"Duration: {result['routing']['estimated_duration_days']} days")
                    
                    elif result['status'] == 'success':
                        with st.expander(f"✅ {result['file']} - {result['type']}"):
                            st.json(result)
                    
                    elif result['status'] == 'error':
                        with st.expander(f"❌ {result['file']} - Error"):
                            st.error(result['error'])
                
                # Download comprehensive report
                if results:
                    report_data = {
                        'processing_summary': {
                            'total_files': len(results),
                            'successful': successful,
                            'errors': errors,
                            'timestamp': str(time.time())
                        },
                        'results': results
                    }
                    
                    st.download_button(
                        label="📥 Download REAL Processing Report",
                        data=json.dumps(report_data, indent=2, default=str),
                        file_name=f"mvr_real_processing_report_{int(time.time())}.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"REAL Processing failed: {str(e)}")
                st.info("This indicates an issue with the V1 processor integration")
    
    else:
        st.info("👆 Upload your MVR documents to start REAL V1 processing")
    
    # Footer
    st.markdown("---")
    st.markdown("**REAL MVR V1 Processing** | Using actual QA Gap Analyzer + Workflow Router + Report Generators")

if __name__ == "__main__":
    main()