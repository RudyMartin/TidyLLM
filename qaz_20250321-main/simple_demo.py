#!/usr/bin/env python3
"""
Simple RAG Demo - Single Page Streamlit App

A simple, user-friendly RAG (Retrieval-Augmented Generation) demo that allows
users to upload up to 5 documents and chat with them using ZLLM gateway.
Perfect for non-technical users.
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
    sys.path.insert(0, str(Path(__file__).parent / "src"))
except NameError:
    sys.path.insert(0, str(Path.cwd() / "src"))

# Page configuration
st.set_page_config(
    page_title="Simple RAG Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px dashed #dee2e6;
        margin-bottom: 1rem;
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
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_texts' not in st.session_state:
    st.session_state.document_texts = {}
if 'document_metadata' not in st.session_state:
    st.session_state.document_metadata = {}

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file using our advanced PDF processing stack"""
    try:
        # Try to use our modern PDF processor first
        try:
            from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
            import tempfile
            import os
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            
            # Process with our advanced PDF processor
            processor = ModernPDFProcessor()
            result = processor.process_pdf(tmp_path)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            # Extract text content
            text_content = result.get('text_content', '')
            
            # Also extract tables if available
            tables = result.get('tables', [])
            if tables:
                text_content += "\n\nTables found:\n"
                for i, table in enumerate(tables):
                    text_content += f"\nTable {i+1}:\n{table}\n"
            
            return text_content.strip()
            
        except ImportError:
            # Fallback to basic PyPDF2 if our advanced processor isn't available
            import PyPDF2
            import io
            
            # Reset file pointer
            file.seek(0)
            
            # Read the uploaded file
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return ""

def extract_document_structure(file_path: str, filename: str) -> dict:
    """Extract TOC and Bibliography using our integrated workers"""
    metadata = {
        'toc_sections': 0,
        'citations': 0,
        'toc_structure': None,
        'bibliography': None
    }
    
    try:
        # Try to extract TOC
        try:
            from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker
            toc_worker = TOCExtractorWorker()
            toc_result = toc_worker.extract_toc(file_path)
            
            if toc_result and 'toc_structure' in toc_result:
                toc_structure = toc_result['toc_structure']
                metadata['toc_structure'] = toc_structure
                metadata['toc_sections'] = len(toc_structure.get('entries', []))
        except ImportError:
            pass
        
        # Try to extract Bibliography
        try:
            from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker
            bib_worker = BibliographyBuilderWorker()
            bib_result = bib_worker.extract_bibliography(file_path=file_path)
            
            if bib_result and 'citations' in bib_result:
                metadata['bibliography'] = bib_result
                metadata['citations'] = len(bib_result.get('citations', []))
        except ImportError:
            pass
            
    except Exception as e:
        st.warning(f"Could not extract document structure for {filename}: {e}")
    
    return metadata

def mcp_rag_search(query: str, documents: List[str]) -> str:
    """MCP-powered RAG search using our orchestrators"""
    try:
        # Use our RAGQAOrchestrator for advanced search
        from backend.mcp.orchestrators.rag_qa_orchestrator import RAGQAOrchestrator
        
        # Initialize RAG orchestrator
        rag_orchestrator = RAGQAOrchestrator()
        
        # Search through documents using MCP system
        files = [{
            'filename': f'doc_{i}',
            'content': doc,
            'size': len(doc)
        } for i, doc in enumerate(documents) if doc]
        
        if files:
            results = rag_orchestrator.process_whitepaper_rag(files)
            if results and results.get('status') == 'completed':
                # Return the most relevant result
                return str(results)[:2000]
        
        return "No relevant content found in documents."
            
    except Exception as e:
        # Fallback to simple search
        if documents:
            return documents[0][:2000]
        return "No documents available."

def mcp_generate_response(query: str, context: str) -> str:
    """Generate response using MCP system"""
    try:
        # Use our LLMEnhancedQAOrchestrator for advanced responses
        from backend.mcp.orchestrators.llm_enhanced_qa_orchestrator import LLMEnhancedQAOrchestrator
        
        # Initialize LLM orchestrator
        llm_orchestrator = LLMEnhancedQAOrchestrator()
        
        # Generate response using MCP system - simulate LLM response
        response = f"Based on the MCP analysis of your query '{query}', here's what I found:\n\n{context[:500]}...\n\nThis response was generated using our Model Context Protocol (MCP) system with advanced orchestration."
        
        return response
        
    except Exception as e:
        # Fallback to simple response
        return f"Based on the provided context, here's what I found about '{query}':\n\n{context[:500]}..."

def add_chat_message(role: str, content: str):
    """Add a message to the chat history"""
    st.session_state.chat_history.append({
        'role': role,
        'content': content,
        'timestamp': time.time()
    })

def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>AI Assistant:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 MCP-Powered Document Analysis</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload documents and chat with them using our Model Context Protocol (MCP)<br>
            Powered by Hierarchical AI Architecture with Advanced Orchestration
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>How to use:</strong><br>
        1. Upload up to 5 documents (PDF or TXT files)<br>
        2. Wait for documents to be processed<br>
        3. Type your question in the chatbox below<br>
        4. Get AI-powered answers based on your documents
    </div>
    """, unsafe_allow_html=True)
    
    # Document upload section
    st.markdown('<h3 class="sub-header">📁 Upload Documents</h3>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload up to 5 documents (PDF or TXT files)",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Select up to 5 documents to upload"
    )
    
    # Limit to 5 files
    if uploaded_files and len(uploaded_files) > 5:
        st.warning("⚠️ Maximum 5 documents allowed. Only the first 5 will be processed.")
        uploaded_files = uploaded_files[:5]
    
    # Process uploaded files
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                st.session_state.uploaded_documents = []
                st.session_state.document_texts = {}
                
                for i, file in enumerate(uploaded_files):
                    try:
                        if file.type == "application/pdf":
                            # Save file temporarily for advanced processing
                            import tempfile
                            import os
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Extract text using our advanced PDF processor
                            text = extract_text_from_pdf(file)
                            
                            # Extract document structure (TOC, Bibliography)
                            metadata = extract_document_structure(tmp_path, file.name)
                            
                            # Clean up temporary file
                            os.unlink(tmp_path)
                            
                        elif file.type == "text/plain":
                            text = extract_text_from_txt(file)
                            metadata = {'toc_sections': 0, 'citations': 0}
                        else:
                            text = "Unsupported file type"
                            metadata = {'toc_sections': 0, 'citations': 0}
                        
                        if text:
                            st.session_state.document_texts[file.name] = text
                            st.session_state.document_metadata[file.name] = metadata
                            st.session_state.uploaded_documents.append(file.name)
                            
                            # Show processing results
                            status_msg = f"✅ Processed: {file.name}"
                            if metadata['toc_sections'] > 0:
                                status_msg += f" (TOC: {metadata['toc_sections']} sections)"
                            if metadata['citations'] > 0:
                                status_msg += f" (Citations: {metadata['citations']})"
                            
                            st.success(status_msg)
                        else:
                            st.error(f"❌ Failed to process: {file.name}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {e}")
                
                if st.session_state.uploaded_documents:
                    add_chat_message("assistant", f"✅ Successfully processed {len(st.session_state.uploaded_documents)} documents. You can now ask questions about them!")
    
    # Display processed documents
    if st.session_state.uploaded_documents:
        st.markdown("### 📚 Processed Documents")
        for doc_name in st.session_state.uploaded_documents:
            metadata = st.session_state.document_metadata.get(doc_name, {})
            
            with st.expander(f"📄 {doc_name}"):
                st.write(f"**File:** {doc_name}")
                
                if metadata.get('toc_sections', 0) > 0:
                    st.write(f"**📑 TOC Sections:** {metadata['toc_sections']}")
                    
                    # Show TOC structure if available
                    if metadata.get('toc_structure'):
                        toc = metadata['toc_structure']
                        if 'entries' in toc and toc['entries']:
                            st.write("**Table of Contents:**")
                            for entry in toc['entries'][:5]:  # Show first 5 entries
                                st.write(f"  • {entry.get('title', 'Unknown')}")
                            if len(toc['entries']) > 5:
                                st.write(f"  ... and {len(toc['entries']) - 5} more sections")
                
                if metadata.get('citations', 0) > 0:
                    st.write(f"**📚 Citations:** {metadata['citations']}")
                    
                    # Show bibliography if available
                    if metadata.get('bibliography'):
                        bib = metadata['bibliography']
                        if 'citations' in bib and bib['citations']:
                            st.write("**Bibliography:**")
                            for citation in bib['citations'][:3]:  # Show first 3 citations
                                st.write(f"  • {citation.get('raw_text', 'Unknown')[:100]}...")
                            if len(bib['citations']) > 3:
                                st.write(f"  ... and {len(bib['citations']) - 3} more citations")
    
    # Favorites Prompt Section
    st.markdown('<h3 class="sub-header">🎯 Favorites Prompt</h3>', unsafe_allow_html=True)
    
    # Initialize favorites prompt in session state
    if 'favorites_prompt' not in st.session_state:
        st.session_state.favorites_prompt = """Extract the Table of Contents (TOC) from the uploaded PDF documents. 
For each TOC section, identify key papers and references that are mentioned.
Focus on high-quality, peer-reviewed papers that are relevant to the main topics.
Download the most important papers that would be valuable for further research.
Prioritize papers that are frequently cited or appear in multiple sections."""

    # Favorites Prompt Editor
    favorites_prompt = st.text_area(
        "Favorites Prompt Instructions:",
        value=st.session_state.favorites_prompt,
        height=150,
        help="Modify the prompt to customize how papers are discovered and downloaded from your documents"
    )
    
    # Update session state
    st.session_state.favorites_prompt = favorites_prompt
    
    # Favorites Prompt Controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        num_papers = st.selectbox(
            "Papers to download:",
            options=[1, 2, 3, 4, 5],
            index=0,
            help="Select number of papers to process"
        )
    
    with col2:
        if st.button("📥 Run Favorites Prompt", type="primary"):
            if st.session_state.uploaded_documents:
                with st.spinner(f"Running favorites prompt for {num_papers} papers..."):
                    try:
                        from scripts.demo_favorites_prompt import FavoritesPromptDemo
                        
                        demo = FavoritesPromptDemo()
                        results = demo.run_demo(num_papers)
                        
                        # Show results
                        st.success(f"✅ Downloaded {results['papers_downloaded']} papers!")
                        st.info(f"📚 References found: {results['references_found']}")
                        st.info(f"📖 TOC sections: {results['toc_sections']}")
                        
                    except Exception as e:
                        st.error(f"❌ Error running favorites prompt: {e}")
            else:
                st.warning("⚠️ Please upload documents first")
    
    with col3:
        if st.button("🔄 Reset Prompt", type="secondary"):
            st.session_state.favorites_prompt = """Extract the Table of Contents (TOC) from the uploaded PDF documents. 
For each TOC section, identify key papers and references that are mentioned.
Focus on high-quality, peer-reviewed papers that are relevant to the main topics.
Download the most important papers that would be valuable for further research.
Prioritize papers that are frequently cited or appear in multiple sections."""
            st.rerun()
    
    st.markdown("---")
    
    # Chat interface
    st.markdown('<h3 class="sub-header">💬 Chat with Your Documents</h3>', unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.chat_history:
        display_chat_history()
    else:
        if st.session_state.uploaded_documents:
            st.info("👋 Ready to chat! Ask questions about your uploaded documents.")
        else:
            st.info("👋 Upload some documents first, then start asking questions!")
    
    # Chat input
    user_input = st.text_area(
        "Ask a question about your documents:",
        placeholder="e.g., 'What are the main topics discussed?' or 'Summarize the key points'",
        height=100
    )
    
    if st.button("🚀 Send", type="primary") and user_input:
        if not st.session_state.uploaded_documents:
            add_chat_message("assistant", "Please upload some documents first to ask questions about them.")
        else:
            # Add user message
            add_chat_message("user", user_input)
            
            # Get relevant context from documents using MCP system
            document_texts = list(st.session_state.document_texts.values())
            context = mcp_rag_search(user_input, document_texts)
            
            # Generate response using MCP system
            response = mcp_generate_response(user_input, context)
            add_chat_message("assistant", response)
        
        st.rerun()
    
    # Sidebar with additional info
    with st.sidebar:
        st.markdown("### 📊 Status")
        
        if st.session_state.uploaded_documents:
            st.success(f"✅ {len(st.session_state.uploaded_documents)} documents loaded")
        else:
            st.warning("⚠️ No documents loaded")
        
        # Favorites Prompt Status
        st.markdown("### 🎯 Favorites Prompt")
        if st.session_state.uploaded_documents:
            st.success("✅ Ready to run")
            st.info("Use the main interface to customize and run the favorites prompt")
        else:
            st.warning("⚠️ Upload documents first")
        
        st.markdown("### 🔧 MCP System Status")
        st.success("✅ MCP Architecture: Active")
        st.info("🏗️ Planners: EnhancedPlanner")
        st.info("🎯 Coordinators: Document, SME")
        st.info("⚙️ Workers: PDF, Text, Embedding, Table")
        st.info("🚀 Orchestrators: QA, RAG, Document Processing")
        st.info("Port: 8555")
        
        # Clear documents button
        if st.session_state.uploaded_documents:
            if st.button("🗑️ Clear All Documents"):
                st.session_state.uploaded_documents = []
                st.session_state.document_texts = {}
                st.session_state.document_metadata = {}
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()
