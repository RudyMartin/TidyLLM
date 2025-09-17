"""
SME RAG Portal - Streamlit Interface
===================================

Advanced SME (Subject Matter Expert) RAG Portal for v2 Boss Portal with:
- Collection Management
- Document Upload with S3 Integration
- Embedding Model Selection
- Manual Indexing/Reindexing
- Chat with SME functionality
- Document Status Monitoring
"""

import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any

# Import our SME RAG system
import sys
sys.path.append(str(Path(__file__).parent))

try:
    from sme_rag_system import SMERAGSystem, EmbeddingModel, DocumentStatus, Collection, DocumentMetadata
except ImportError:
    st.error("SME RAG System not available. Please ensure sme_rag_system.py is in the correct location.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="SME RAG Portal - v2 Boss Portal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .collection-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .doc-status-uploaded { color: #6c757d; }
    .doc-status-processing { color: #ffc107; }
    .doc-status-indexed { color: #28a745; }
    .doc-status-error { color: #dc3545; }
    .doc-status-reindexing { color: #17a2b8; }
    .metrics-row {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if 'sme_rag_system' not in st.session_state:
        try:
            st.session_state.sme_rag_system = SMERAGSystem()
        except Exception as e:
            st.error(f"Failed to initialize SME RAG System: {str(e)}")
            st.stop()
    
    if 'selected_collection' not in st.session_state:
        st.session_state.selected_collection = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processing_docs' not in st.session_state:
        st.session_state.processing_docs = set()

def render_header():
    """Render the main header."""
    st.markdown("""
        <div class="main-header">
            🧠 SME RAG Portal - Subject Matter Expert Knowledge Management
        </div>
    """, unsafe_allow_html=True)

def render_collection_management():
    """Render collection management interface."""
    st.header("📚 Collection Management")
    
    # Create new collection
    with st.expander("➕ Create New Collection", expanded=False):
        with st.form("create_collection"):
            col1, col2 = st.columns(2)
            
            with col1:
                collection_name = st.text_input("Collection Name *", 
                                              placeholder="e.g., Model Risk Management Standards")
                description = st.text_area("Description *", 
                                          placeholder="Detailed description of this collection's purpose...")
                
            with col2:
                s3_bucket = st.text_input("S3 Bucket *", 
                                        value="your-sme-documents",
                                        placeholder="S3 bucket for document storage")
                s3_prefix = st.text_input("S3 Prefix", 
                                        placeholder="Optional path prefix in S3")
                
                embedding_model = st.selectbox("Embedding Model *", 
                                             options=[model.value for model in EmbeddingModel],
                                             index=0)
            
            tags = st.text_input("Tags (comma-separated)", 
                                placeholder="model-risk, compliance, standards")
            
            submit_collection = st.form_submit_button("Create Collection", type="primary")
            
            if submit_collection:
                if collection_name and description and s3_bucket:
                    try:
                        tags_list = [tag.strip() for tag in tags.split(',')] if tags else []
                        
                        collection_id = st.session_state.sme_rag_system.create_collection(
                            name=collection_name,
                            description=description,
                            embedding_model=EmbeddingModel(embedding_model),
                            s3_bucket=s3_bucket,
                            s3_prefix=s3_prefix,
                            tags=tags_list
                        )
                        
                        st.success(f"✅ Collection '{collection_name}' created successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Failed to create collection: {str(e)}")
                else:
                    st.error("❌ Please fill in all required fields")
    
    # Display existing collections
    try:
        collections = st.session_state.sme_rag_system.get_collections()
        
        if collections:
            st.subheader("📊 Existing Collections")
            
            for collection in collections:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                            <div class="collection-card">
                                <h4>{collection.name}</h4>
                                <p>{collection.description}</p>
                                <small>
                                    📅 Created: {collection.created_date.strftime('%Y-%m-%d %H:%M')} | 
                                    🔧 Model: {collection.embedding_model} | 
                                    🏷️ Tags: {', '.join(collection.tags) if collection.tags else 'None'}
                                </small>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Documents", collection.document_count)
                    
                    with col3:
                        st.metric("Chunks", collection.total_chunks)
                    
                    with col4:
                        if st.button(f"Select", key=f"select_{collection.collection_id}"):
                            st.session_state.selected_collection = collection.collection_id
                            st.rerun()
                        
                        if st.session_state.selected_collection == collection.collection_id:
                            st.success("✓ Selected")
        else:
            st.info("📝 No collections found. Create your first collection above!")
            
    except Exception as e:
        st.error(f"❌ Error loading collections: {str(e)}")

def render_document_management():
    """Render document management interface."""
    if not st.session_state.selected_collection:
        st.warning("⚠️ Please select a collection first from the Collection Management section.")
        return
    
    st.header("📄 Document Management")
    
    # Get selected collection info
    try:
        collection = st.session_state.sme_rag_system.get_collection(st.session_state.selected_collection)
        if not collection:
            st.error("❌ Selected collection not found")
            return
        
        st.info(f"📂 Working with collection: **{collection.name}**")
        
        # Document upload section
        with st.expander("📤 Upload Documents", expanded=True):
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload PDF documents to add to this collection"
            )
            
            if uploaded_files:
                col1, col2 = st.columns(2)
                
                with col1:
                    tags_input = st.text_input("Tags for uploaded documents (comma-separated)",
                                             placeholder="regulation, standard, guidance")
                
                with col2:
                    metadata_input = st.text_area("Additional Metadata (JSON format)",
                                                placeholder='{"department": "Risk Management", "version": "2.1"}')
                
                if st.button("🚀 Upload and Process Documents", type="primary"):
                    tags_list = [tag.strip() for tag in tags_input.split(',')] if tags_input else []
                    
                    try:
                        metadata_dict = json.loads(metadata_input) if metadata_input else {}
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON in metadata field")
                        return
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            status_text.text(f"Processing {uploaded_file.name}...")
                            
                            # Save uploaded file to temp location
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.getbuffer())
                                tmp_path = tmp_file.name
                            
                            # Upload to S3 and database
                            doc_id = st.session_state.sme_rag_system.upload_document(
                                file_path=tmp_path,
                                collection_id=collection.collection_id,
                                original_filename=uploaded_file.name,
                                tags=tags_list,
                                metadata=metadata_dict
                            )
                            
                            # Process document (extract text, create embeddings)
                            st.session_state.processing_docs.add(doc_id)
                            processing_result = st.session_state.sme_rag_system.process_document(doc_id)
                            
                            if processing_result['status'] == 'success':
                                st.success(f"✅ {uploaded_file.name} processed successfully!")
                            else:
                                st.error(f"❌ Error processing {uploaded_file.name}: {processing_result.get('error', 'Unknown error')}")
                            
                            # Clean up temp file
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"❌ Error uploading {uploaded_file.name}: {str(e)}")
                        
                        finally:
                            progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text("✅ Upload complete!")
                    time.sleep(2)
                    st.rerun()
        
        # Document list and management
        st.subheader("📋 Collection Documents")
        
        # Collection actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Document List"):
                st.rerun()
        
        with col2:
            if st.button("🔧 Reindex Collection", help="Reprocess all documents in this collection"):
                with st.spinner("Reindexing collection..."):
                    try:
                        result = st.session_state.sme_rag_system.reindex_collection(collection.collection_id)
                        st.success(f"✅ Reindexing completed! Processed {result['total_documents']} documents.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Reindexing failed: {str(e)}")
        
        with col3:
            # Collection stats
            updated_collection = st.session_state.sme_rag_system.get_collection(collection.collection_id)
            st.metric("Total Documents", updated_collection.document_count)
        
        # Get documents in collection
        documents = st.session_state.sme_rag_system.get_collection_documents(collection.collection_id)
        
        if documents:
            # Create DataFrame for better display
            doc_data = []
            for doc in documents:
                status_class = f"doc-status-{doc.status.value}"
                doc_data.append({
                    'Filename': doc.original_filename,
                    'Status': doc.status.value.title(),
                    'Chunks': doc.chunk_count,
                    'Size (KB)': round(doc.file_size / 1024, 2) if doc.file_size else 0,
                    'Processing Time (s)': round(doc.processing_time, 2),
                    'Uploaded': doc.upload_date.strftime('%Y-%m-%d %H:%M'),
                    'Tags': ', '.join(doc.tags) if doc.tags else '',
                    'Error': doc.error_message if doc.error_message else ''
                })
            
            df = pd.DataFrame(doc_data)
            
            # Display documents table
            st.dataframe(
                df,
                use_container_width=True,
                height=400,
                column_config={
                    'Status': st.column_config.TextColumn(
                        help="Document processing status"
                    ),
                    'Chunks': st.column_config.NumberColumn(
                        help="Number of text chunks created"
                    ),
                    'Processing Time (s)': st.column_config.NumberColumn(
                        help="Time taken to process document"
                    )
                }
            )
            
            # Status legend
            st.markdown("""
                **Status Legend:**
                <span class="doc-status-uploaded">● Uploaded</span> |
                <span class="doc-status-processing">● Processing</span> |
                <span class="doc-status-indexed">● Indexed</span> |
                <span class="doc-status-error">● Error</span> |
                <span class="doc-status-reindexing">● Reindexing</span>
            """, unsafe_allow_html=True)
            
        else:
            st.info("📁 No documents in this collection yet. Upload some documents above!")
        
    except Exception as e:
        st.error(f"❌ Error in document management: {str(e)}")

def render_chat_interface():
    """Render Chat with SME interface."""
    if not st.session_state.selected_collection:
        st.warning("⚠️ Please select a collection first from the Collection Management section.")
        return
    
    st.header("💬 Chat with SME")
    
    try:
        collection = st.session_state.sme_rag_system.get_collection(st.session_state.selected_collection)
        if not collection:
            st.error("❌ Selected collection not found")
            return
        
        st.info(f"🧠 Chatting with SME knowledge from: **{collection.name}**")
        
        # Search settings
        with st.expander("🔧 Search Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                search_limit = st.slider("Number of relevant chunks to retrieve", 1, 20, 5)
                similarity_threshold = st.slider("Similarity threshold", 0.5, 0.95, 0.7, 0.05)
            
            with col2:
                st.info(f"""
                **Current Settings:**
                - Embedding Model: {collection.embedding_model}
                - Documents: {collection.document_count}
                - Total Chunks: {collection.total_chunks}
                """)
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>You:</strong> {message['content']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>SME Assistant:</strong> {message['content']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources if available
                    if 'sources' in message and message['sources']:
                        with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                            for j, source in enumerate(message['sources']):
                                st.markdown(f"""
                                    **Source {j+1}:** {source['filename']} (Chunk {source['chunk_index']}, Similarity: {source['similarity_score']:.3f})
                                    
                                    {source['content'][:300]}...
                                """)
        
        # Query input
        with st.form("chat_form", clear_on_submit=True):
            user_query = st.text_area(
                "Ask your question:",
                placeholder="e.g., What are the key requirements for model risk management according to the guidelines?",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_query = st.form_submit_button("🚀 Ask SME", type="primary")
            with col2:
                if st.form_submit_button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        if submit_query and user_query:
            with st.spinner("🔍 Searching knowledge base and generating response..."):
                try:
                    # Perform semantic search
                    search_results = st.session_state.sme_rag_system.semantic_search(
                        query=user_query,
                        collection_id=collection.collection_id,
                        limit=search_limit,
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Add user message to chat
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_query
                    })
                    
                    if search_results:
                        # Prepare context from search results
                        context_chunks = [result['content'] for result in search_results]
                        context = "\n\n".join(context_chunks)
                        
                        # Generate response using the context
                        # For now, we'll create a simple response
                        # In production, you'd use OpenAI or another LLM
                        
                        response = f"""Based on the documents in the {collection.name} collection, I found {len(search_results)} relevant sections that address your question.

**Key Information:**

{context[:1500]}...

**Summary:** The documents indicate that this topic is covered across {len(set([r['filename'] for r in search_results]))} different documents in your knowledge base. The most relevant information comes from {search_results[0]['filename']} with a similarity score of {search_results[0]['similarity_score']:.3f}.

*Note: This is a simplified response. In production, this would be enhanced with a full LLM integration for more sophisticated answer generation.*"""
                        
                        # Add assistant response to chat
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response,
                            'sources': search_results
                        })
                        
                    else:
                        # No relevant results found
                        response = f"""I couldn't find relevant information in the {collection.name} collection that matches your query. 

This could mean:
- The information isn't available in the current documents
- Try rephrasing your question
- Consider lowering the similarity threshold in search settings
- Check if the relevant documents have been successfully indexed

Current collection contains {collection.document_count} documents with {collection.total_chunks} searchable chunks."""
                        
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
        
        # Quick example queries
        st.markdown("### 💡 Example Questions")
        example_queries = [
            "What are the key model validation requirements?",
            "How should model risk be assessed and managed?",
            "What documentation is required for model approval?",
            "What are the regulatory expectations for model governance?",
            "How often should models be reviewed and validated?"
        ]
        
        cols = st.columns(len(example_queries))
        for i, query in enumerate(example_queries):
            with cols[i % len(cols)]:
                if st.button(f"💬 {query[:30]}...", key=f"example_{i}", help=query):
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': query
                    })
                    # Trigger processing...
                    st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error in chat interface: {str(e)}")

def main():
    """Main application function."""
    init_session_state()
    render_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧠 SME RAG Portal")
        st.markdown("---")
        
        nav_option = st.radio(
            "Navigate to:",
            ["📚 Collection Management", "📄 Document Management", "💬 Chat with SME"],
            help="Select a section to work with"
        )
        
        st.markdown("---")
        
        # Show current collection
        if st.session_state.selected_collection:
            try:
                collection = st.session_state.sme_rag_system.get_collection(st.session_state.selected_collection)
                if collection:
                    st.success(f"📂 **Selected Collection:**\n{collection.name}")
                    st.caption(f"Documents: {collection.document_count} | Chunks: {collection.total_chunks}")
            except:
                pass
        else:
            st.info("ℹ️ No collection selected")
        
        st.markdown("---")
        
        # System status
        try:
            collections = st.session_state.sme_rag_system.get_collections()
            total_docs = sum(c.document_count for c in collections)
            total_chunks = sum(c.total_chunks for c in collections)
            
            st.markdown("### 📊 System Status")
            st.metric("Collections", len(collections))
            st.metric("Total Documents", total_docs)
            st.metric("Total Chunks", total_chunks)
            
        except:
            st.error("❌ System Status Unavailable")
    
    # Main content area
    if nav_option == "📚 Collection Management":
        render_collection_management()
    elif nav_option == "📄 Document Management":
        render_document_management()
    elif nav_option == "💬 Chat with SME":
        render_chat_interface()

if __name__ == "__main__":
    main()