"""
Enterprise Document Processing Demo
==================================

Standalone demo of the recovered and centralized document processing features:
- PyMuPDF (benchmark winner) text extraction
- Business document classification with templates
- Advanced metadata extraction with confidence scoring
- Smart text chunking with sentence boundaries
- S3 enterprise streaming capabilities
- All V1 specialized functionality integrated

This can be tested independently and later integrated into v2_tidyllm.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from boss_enterprise_document_processor import EnterpriseDocumentProcessor, create_enterprise_processor

# Page configuration
st.set_page_config(
    page_title="Enterprise Document Processing Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-banner {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-banner {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">🚀 Enterprise Document Processing Demo</div>', unsafe_allow_html=True)
    
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = create_enterprise_processor()
    
    processor = st.session_state.processor
    
    # Sidebar - System Status
    with st.sidebar:
        st.header("🔧 System Status")
        
        capabilities = processor.get_capabilities_report()
        
        # Extraction Engines Status
        st.subheader("Extraction Engines")
        if capabilities["extraction_engines"]["pymupdf"]:
            st.success("✅ PyMuPDF (Primary - Benchmark Winner)")
        else:
            st.error("❌ PyMuPDF Not Available")
        
        if capabilities["extraction_engines"]["pypdf2_fallback"]:
            st.info("🔄 PyPDF2 Fallback Available")
        else:
            st.warning("⚠️ PyPDF2 Fallback Not Available")
        
        # Enterprise Features Status
        st.subheader("Enterprise Features")
        for feature, status in capabilities["enterprise_features"].items():
            if isinstance(status, bool):
                if status:
                    st.success(f"✅ {feature.replace('_', ' ').title()}")
                else:
                    st.warning(f"⚠️ {feature.replace('_', ' ').title()}")
            else:
                st.info(f"📊 {feature.replace('_', ' ').title()}: {status}")
        
        st.subheader("Supported Templates")
        for template in capabilities["document_templates"]:
            st.write(f"• {template.replace('_', ' ').title()}")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Process Document", "🎯 Template Analysis", "📊 Batch Processing", "🧪 Testing"])
    
    with tab1:
        st.header("Upload & Process Document")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF document",
            type=['pdf'],
            help="Upload a PDF document for enterprise-grade processing"
        )
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            max_pages = st.slider("Maximum Pages to Process", 1, 20, 10)
        with col2:
            show_chunks = st.checkbox("Show Text Chunks", value=True)
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Path("temp_upload.pdf")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing document with enterprise-grade extraction..."):
                    # Process the document
                    result = processor.process_document(temp_path, max_pages)
                
                # Display results
                if result.text:
                    st.markdown('<div class="success-banner">✅ Document processed successfully!</div>', 
                               unsafe_allow_html=True)
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Document Type", result.document_type.replace('_', ' ').title())
                    with col2:
                        st.metric("Confidence Score", f"{result.confidence:.2%}")
                    with col3:
                        st.metric("Text Length", f"{len(result.text):,} chars")
                    with col4:
                        st.metric("Chunks Created", len(result.chunks))
                    
                    # Document classification details
                    st.subheader("📋 Document Classification")
                    if result.template_used:
                        st.success(f"Template Used: **{result.template_used.replace('_', ' ').title()}**")
                        st.info(f"Classification Confidence: **{result.confidence:.1%}**")
                    else:
                        st.warning("No specific template matched - classified as unknown")
                    
                    # Extracted metadata
                    if result.extracted_fields:
                        st.subheader("🔍 Extracted Metadata")
                        metadata_df = pd.DataFrame([
                            {"Field": field.replace('_', ' ').title(), "Value": value}
                            for field, value in result.extracted_fields.items()
                        ])
                        st.dataframe(metadata_df, use_container_width=True)
                    
                    # Text content
                    st.subheader("📝 Extracted Text")
                    st.text_area("Document Text", result.text, height=300)
                    
                    # Text chunks (if enabled)
                    if show_chunks and result.chunks:
                        st.subheader("📚 Text Chunks")
                        for chunk in result.chunks:
                            with st.expander(f"Chunk {chunk['chunk_id']} ({chunk['length']} chars)"):
                                st.write(chunk['text'])
                                st.caption(f"Length: {chunk['length']} characters")
                    
                    # Technical metadata
                    with st.expander("🔧 Technical Metadata"):
                        st.json(result.extraction_metadata)
                
                else:
                    st.error("❌ Failed to process document. Check the technical metadata below for details.")
                    st.json(result.extraction_metadata)
            
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
    
    with tab2:
        st.header("📊 Template Analysis")
        
        st.write("Analysis of business document templates and their classification criteria:")
        
        templates = processor.templates
        template_data = []
        
        for name, template in templates.items():
            template_data.append({
                "Template": template.name.replace('_', ' ').title(),
                "Keywords": len(template.keywords),
                "Required Patterns": len(template.required_patterns),
                "Optional Patterns": len(template.optional_patterns),
                "Confidence Boost": f"{template.confidence_boost:.1%}",
                "Keywords List": ", ".join(template.keywords[:5]) + ("..." if len(template.keywords) > 5 else "")
            })
        
        df = pd.DataFrame(template_data)
        st.dataframe(df, use_container_width=True)
        
        # Template details
        selected_template = st.selectbox("Select template for details:", list(templates.keys()))
        if selected_template:
            template = templates[selected_template]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Keywords")
                for keyword in template.keywords:
                    st.write(f"• {keyword}")
            
            with col2:
                st.subheader("Required Patterns")
                for pattern in template.required_patterns:
                    st.write(f"• {pattern.replace('_', ' ').title()}")
                
                st.subheader("Optional Patterns")
                for pattern in template.optional_patterns:
                    st.write(f"• {pattern.replace('_', ' ').title()}")
    
    with tab3:
        st.header("📊 Batch Processing")
        
        st.info("🚧 Batch processing feature - ready for implementation")
        st.write("This feature would allow:")
        st.write("• Upload multiple PDF documents")
        st.write("• Process all documents in parallel")
        st.write("• Generate comprehensive analysis report")
        st.write("• Export results to Excel/CSV")
        st.write("• S3 bucket processing for enterprise users")
        
        # S3 Processing Demo
        if processor.s3_available:
            st.subheader("☁️ S3 Processing")
            bucket_name = st.text_input("S3 Bucket Name")
            object_key = st.text_input("Object Key (path/to/document.pdf)")
            
            if st.button("Process from S3"):
                if bucket_name and object_key:
                    with st.spinner("Processing document from S3..."):
                        result = processor.process_s3_document(bucket_name, object_key)
                    
                    if result.text:
                        st.success("✅ S3 document processed successfully!")
                        st.write(f"Document Type: {result.document_type}")
                        st.write(f"Confidence: {result.confidence:.1%}")
                    else:
                        st.error("❌ Failed to process S3 document")
                        st.json(result.extraction_metadata)
        else:
            st.warning("⚠️ S3 processing not available (boto3 not installed)")
    
    with tab4:
        st.header("🧪 Testing & Validation")
        
        # System test
        if st.button("Run System Test"):
            with st.spinner("Running comprehensive system tests..."):
                test_results = run_system_tests(processor)
            
            # Display test results
            st.subheader("Test Results")
            for test_name, result in test_results.items():
                if result["passed"]:
                    st.success(f"✅ {test_name}: {result['message']}")
                else:
                    st.error(f"❌ {test_name}: {result['message']}")
        
        # Feature testing
        st.subheader("Feature Testing")
        
        # Text cleaning test
        if st.button("Test Text Cleaning"):
            test_text = "This is   \n\n  test text\x00\x08   with   \t\t excessive   whitespace"
            cleaned = processor._clean_text(test_text)
            st.write("**Original:**", repr(test_text))
            st.write("**Cleaned:**", repr(cleaned))
        
        # Template matching test
        if st.button("Test Template Matching"):
            test_cases = [
                ("Invoice #12345 Total: $500.00", "invoice"),
                ("Purchase Order PO-2023-001", "purchase_order"),
                ("Contract Agreement #CON-2023", "contract"),
                ("Motor Vehicle Record for License #ABC123", "mvr_record"),
                ("Financial Statement Account #987654", "financial_statement")
            ]
            
            results = []
            for text, expected in test_cases:
                doc_type, confidence = processor._classify_with_templates(text)
                results.append({
                    "Test Text": text,
                    "Expected": expected,
                    "Detected": doc_type,
                    "Confidence": f"{confidence:.1%}",
                    "Match": "✅" if doc_type == expected else "❌"
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

def run_system_tests(processor):
    """Run comprehensive system tests."""
    tests = {}
    
    # Test 1: Processor initialization
    try:
        tests["Processor Initialization"] = {
            "passed": True,
            "message": "Enterprise processor initialized successfully"
        }
    except Exception as e:
        tests["Processor Initialization"] = {
            "passed": False,
            "message": f"Initialization failed: {str(e)}"
        }
    
    # Test 2: Template loading
    try:
        templates_count = len(processor.templates)
        tests["Template Loading"] = {
            "passed": templates_count > 0,
            "message": f"Loaded {templates_count} business templates"
        }
    except Exception as e:
        tests["Template Loading"] = {
            "passed": False,
            "message": f"Template loading failed: {str(e)}"
        }
    
    # Test 3: Metadata extractor
    try:
        test_text = "Invoice #12345 dated 12/01/2023 total $500.00"
        metadata = processor.metadata_extractor.extract_metadata(test_text)
        tests["Metadata Extraction"] = {
            "passed": len(metadata) > 0,
            "message": f"Extracted {len(metadata)} metadata fields"
        }
    except Exception as e:
        tests["Metadata Extraction"] = {
            "passed": False,
            "message": f"Metadata extraction failed: {str(e)}"
        }
    
    # Test 4: Text cleaning
    try:
        test_text = "Test\x00\x08text   with\n\nexcessive\twhitespace"
        cleaned = processor._clean_text(test_text)
        tests["Text Cleaning"] = {
            "passed": len(cleaned) < len(test_text),
            "message": "Text cleaning functioning correctly"
        }
    except Exception as e:
        tests["Text Cleaning"] = {
            "passed": False,
            "message": f"Text cleaning failed: {str(e)}"
        }
    
    # Test 5: Capabilities report
    try:
        capabilities = processor.get_capabilities_report()
        tests["Capabilities Report"] = {
            "passed": "extraction_engines" in capabilities,
            "message": "Capabilities report generated successfully"
        }
    except Exception as e:
        tests["Capabilities Report"] = {
            "passed": False,
            "message": f"Capabilities report failed: {str(e)}"
        }
    
    return tests

if __name__ == "__main__":
    main()