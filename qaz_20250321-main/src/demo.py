#!/usr/bin/env python3
"""
QA Enhanced Demo - Simple Chatbox with Document Upload and Comparison

A single-page Streamlit application that demonstrates QA "Enhanced" capabilities
using the MCP architecture we created. Features include:
- Document upload and processing
- Chat interface for questions
- Document comparison capabilities
- Integration with our integrated workers (TOC, Bibliography, Tables)
"""

import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any
import time

# Add src to path for imports
try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    # Handle case where __file__ is not defined
    sys.path.insert(0, str(Path.cwd() / "src"))

# Environment setup
try:
    from config.setup import setup_env
    setup_env()
except ImportError:
    st.warning("⚠️ Environment setup module not found. Some features may not work.")

# Import our integrated workers (with error handling)
WORKERS_AVAILABLE = False
try:
    # Try to import individual workers
    from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker
    from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker
    from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
    WORKERS_AVAILABLE = True
    st.success("✅ All workers imported successfully!")
except ImportError as e:
    st.error(f"❌ Failed to import workers: {e}")
    WORKERS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="QA Enhanced Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px dashed #dee2e6;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .status-success {
        color: #4caf50;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
    .status-warning {
        color: #ff9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

def process_document_simple(file, filename: str) -> Dict[str, Any]:
    """Process an uploaded document using individual workers"""
    if not WORKERS_AVAILABLE:
        return {"success": False, "error": "Workers not available"}
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        start_time = time.time()
        results = {}
        
        # Process with Modern PDF Processor
        try:
            pdf_processor = ModernPDFProcessor()
            pdf_result = pdf_processor.process_pdf(tmp_path)
            results['pdf'] = pdf_result
        except Exception as e:
            results['pdf'] = {"error": str(e)}
        
        # Process with TOC Extractor
        try:
            toc_worker = TOCExtractorWorker()
            toc_result = toc_worker.extract_toc(tmp_path)
            results['toc'] = toc_result
        except Exception as e:
            results['toc'] = {"error": str(e)}
        
        # Process with Bibliography Builder
        try:
            bib_worker = BibliographyBuilderWorker()
            bib_result = bib_worker.extract_bibliography(file_path=tmp_path)
            results['bibliography'] = bib_result
        except Exception as e:
            results['bibliography'] = {"error": str(e)}
        
        processing_time = time.time() - start_time
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Extract metrics
        toc_sections = len(results.get('toc', {}).get('toc_structure', {}).get('entries', [])) if 'error' not in results.get('toc', {}) else 0
        citations = len(results.get('bibliography', {}).get('citations', [])) if 'error' not in results.get('bibliography', {}) else 0
        
        return {
            "success": True,
            "toc_sections": toc_sections,
            "citations": citations,
            "processing_time": processing_time,
            "metadata": results,
            "error": None
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def add_chat_message(role: str, content: str, metadata: Dict[str, Any] = None):
    """Add a message to the chat history"""
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {}
    }
    st.session_state.chat_history.append(message)

def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Assistant:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 QA Enhanced Demo</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Simple chatbox with document upload and comparison capabilities<br>
            Powered by our integrated MCP architecture
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for document management
    with st.sidebar:
        st.markdown('<h3 class="sub-header">📁 Document Management</h3>', unsafe_allow_html=True)
        
        # Document upload
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to process with our integrated workers"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process uploaded files
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    for file in uploaded_files:
                        if file.name not in st.session_state.uploaded_documents:
                            st.session_state.processing_status[file.name] = "Processing..."
                            
                            # Process the document
                            result = process_document_simple(file, file.name)
                            
                            if result["success"]:
                                st.session_state.uploaded_documents[file.name] = result
                                st.session_state.processing_status[file.name] = "✅ Success"
                                add_chat_message(
                                    "assistant", 
                                    f"Successfully processed '{file.name}': {result['toc_sections']} TOC sections, {result['citations']} citations found."
                                )
                            else:
                                st.session_state.processing_status[file.name] = f"❌ Error: {result['error']}"
                                add_chat_message(
                                    "assistant", 
                                    f"Failed to process '{file.name}': {result['error']}"
                                )
        
        # Display processing status
        if st.session_state.processing_status:
            st.markdown("### 📊 Processing Status")
            for filename, status in st.session_state.processing_status.items():
                if "✅" in status:
                    st.markdown(f"<span class='status-success'>{filename}: {status}</span>", unsafe_allow_html=True)
                elif "❌" in status:
                    st.markdown(f"<span class='status-error'>{filename}: {status}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='status-warning'>{filename}: {status}</span>", unsafe_allow_html=True)
        
        # Document list
        if st.session_state.uploaded_documents:
            st.markdown("### 📚 Processed Documents")
            for filename, data in st.session_state.uploaded_documents.items():
                with st.expander(f"📄 {filename}"):
                    st.write(f"**TOC Sections:** {data['toc_sections']}")
                    st.write(f"**Citations:** {data['citations']}")
                    st.write(f"**Processing Time:** {data['processing_time']:.2f}s")
                    
                    # Show metadata if available
                    if data.get('metadata'):
                        st.write("**Metadata:**")
                        st.json(data['metadata'])
        
        # Clear documents
        if st.session_state.uploaded_documents:
            if st.button("🗑️ Clear All Documents"):
                st.session_state.uploaded_documents = {}
                st.session_state.processing_status = {}
                st.session_state.chat_history = []
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">💬 Chat Interface</h3>', unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.chat_history:
            display_chat_history()
        else:
            st.info("👋 Welcome! Upload some documents and start asking questions.")
        
        # Chat input
        user_input = st.text_area(
            "Ask a question about your documents:",
            placeholder="e.g., 'What are the main sections in the documents?' or 'Compare the citations between documents'",
            height=100
        )
        
        if st.button("🚀 Send", type="primary") and user_input:
            # Add user message
            add_chat_message("user", user_input)
            
            # Generate response based on available documents
            if st.session_state.uploaded_documents:
                # Simple response generation based on document content
                response = generate_response(user_input, st.session_state.uploaded_documents)
                add_chat_message("assistant", response)
            else:
                add_chat_message("assistant", "Please upload some documents first to ask questions about them.")
            
            st.rerun()
    
    with col2:
        st.markdown('<h3 class="sub-header">📈 Quick Stats</h3>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_documents:
            total_docs = len(st.session_state.uploaded_documents)
            total_toc = sum(doc['toc_sections'] for doc in st.session_state.uploaded_documents.values())
            total_citations = sum(doc['citations'] for doc in st.session_state.uploaded_documents.values())
            total_time = sum(doc['processing_time'] for doc in st.session_state.uploaded_documents.values())
            
            st.metric("📄 Documents", total_docs)
            st.metric("📑 TOC Sections", total_toc)
            st.metric("📚 Citations", total_citations)
            st.metric("⏱️ Total Processing Time", f"{total_time:.1f}s")
        else:
            st.info("No documents processed yet.")
        
        # System status
        st.markdown("### 🔧 System Status")
        if WORKERS_AVAILABLE:
            st.success("✅ Workers Available")
        else:
            st.error("❌ Workers Not Available")

def generate_response(user_input: str, documents: Dict[str, Any]) -> str:
    """Generate a response based on user input and available documents"""
    input_lower = user_input.lower()
    
    # Simple keyword-based response generation
    if "section" in input_lower or "toc" in input_lower:
        sections_info = []
        for filename, data in documents.items():
            if data['toc_sections'] > 0:
                sections_info.append(f"'{filename}' has {data['toc_sections']} sections")
        
        if sections_info:
            return f"Here are the TOC sections found:\n" + "\n".join(f"• {info}" for info in sections_info)
        else:
            return "No table of contents sections were found in the processed documents."
    
    elif "citation" in input_lower or "reference" in input_lower:
        citations_info = []
        for filename, data in documents.items():
            if data['citations'] > 0:
                citations_info.append(f"'{filename}' has {data['citations']} citations")
        
        if citations_info:
            return f"Here are the citations found:\n" + "\n".join(f"• {info}" for info in citations_info)
        else:
            return "No citations were found in the processed documents."
    
    elif "compare" in input_lower:
        if len(documents) < 2:
            return "You need at least 2 documents to make comparisons."
        
        comparison = "Document Comparison:\n"
        for filename, data in documents.items():
            comparison += f"\n📄 {filename}:\n"
            comparison += f"   • TOC Sections: {data['toc_sections']}\n"
            comparison += f"   • Citations: {data['citations']}\n"
            comparison += f"   • Processing Time: {data['processing_time']:.2f}s\n"
        
        return comparison
    
    elif "table" in input_lower:
        tables_found = []
        for filename, data in documents.items():
            if 'metadata' in data and 'pdf' in data['metadata']:
                pdf_data = data['metadata']['pdf']
                if 'tables' in pdf_data and isinstance(pdf_data['tables'], list):
                    table_count = len(pdf_data['tables'])
                    if table_count > 0:
                        tables_found.append(f"'{filename}' has {table_count} tables")
        
        if tables_found:
            return f"Tables found in documents:\n" + "\n".join(f"• {info}" for info in tables_found)
        else:
            return "No tables were found in the processed documents."
    
    else:
        return f"I understand you're asking about: '{user_input}'. I can help you with:\n" + \
               "• TOC sections and document structure\n" + \
               "• Citations and references\n" + \
               "• Document comparisons\n" + \
               "• Tables and structured data\n" + \
               "Try asking about these topics!"

if __name__ == "__main__":
    main()
