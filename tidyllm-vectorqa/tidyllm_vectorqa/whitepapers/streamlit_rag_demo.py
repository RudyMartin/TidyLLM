#!/usr/bin/env python3
"""
Streamlit wrapper for Simple RAG Demo
Interactive web interface for paper-based RAG system
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import the SimpleRAG class from the existing demo
from simple_rag_demo import SimpleRAG

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="VectorQA Simple RAG Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 VectorQA Simple RAG Demo")
    st.markdown("### Retrieval-Augmented Generation with Paper Repository")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
        st.session_state.papers_loaded = False
    
    # Sidebar for system info
    with st.sidebar:
        st.header("📊 System Status")
        
        if st.button("🚀 Initialize RAG System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                try:
                    # Force reload the module to get latest changes
                    import importlib
                    import simple_rag_demo
                    importlib.reload(simple_rag_demo)
                    from simple_rag_demo import SimpleRAG
                    
                    st.session_state.rag_system = SimpleRAG()
                    st.success("✅ RAG system initialized with latest fixes!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        if st.session_state.rag_system and st.button("📚 Load Paper Repository"):
            with st.spinner("Loading papers and creating RAG chunks..."):
                try:
                    if st.session_state.rag_system.load_papers_for_rag():
                        rag_ready = st.session_state.rag_system.build_rag_index()
                        st.session_state.papers_loaded = True
                        
                        # Display stats
                        st.success("✅ Papers loaded successfully!")
                        st.metric("Papers", len(st.session_state.rag_system.papers))
                        st.metric("Chunks", len(st.session_state.rag_system.paper_chunks))
                        
                        method = "Semantic Embeddings RAG" if rag_ready else "Text Matching RAG"
                        st.info(f"🔧 Method: {method}")
                    else:
                        st.error("❌ Failed to load papers")
                except Exception as e:
                    st.error(f"❌ Error loading papers: {e}")
    
    # Main content area
    if not st.session_state.rag_system:
        st.info("👈 Click 'Initialize RAG System' in the sidebar to start")
        
        # Show example queries while waiting
        st.subheader("📝 Example Queries")
        example_queries = [
            "How do heat engines work and what are measurement challenges?",
            "What role does AI play in creative processes and art generation?",
            "What are the main approaches to signal decomposition in mathematics?",
            "How effective is reservoir computing for separating signals from noise?"
        ]
        
        for i, query in enumerate(example_queries, 1):
            st.write(f"{i}. {query}")
        
        return
    
    if not st.session_state.papers_loaded:
        st.info("👈 Click 'Load Paper Repository' in the sidebar to continue")
        return
    
    # Query interface
    st.subheader("🤔 Ask a Question")
    
    # Predefined questions
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Quick Questions:**")
        if st.button("Heat Engines & Measurement"):
            st.session_state.user_query = "How do heat engines work and what are measurement challenges?"
        if st.button("AI in Creative Processes"):
            st.session_state.user_query = "What role does AI play in creative processes and art generation?"
    
    with col2:
        st.write("**More Questions:**")
        if st.button("Signal Decomposition"):
            st.session_state.user_query = "What are the main approaches to signal decomposition in mathematics?"
        if st.button("Reservoir Computing"):
            st.session_state.user_query = "How effective is reservoir computing for separating signals from noise?"
    
    # Custom query input
    user_input = st.text_input(
        "Or enter your own question:",
        value=st.session_state.get('user_query', ''),
        placeholder="What would you like to know about the research papers?"
    )
    
    if st.button("🔍 Search", type="primary", disabled=not user_input):
        if user_input:
            with st.spinner("Processing your query..."):
                try:
                    # Get RAG response
                    response = st.session_state.rag_system.rag_qa(user_input)
                    
                    # Display response
                    st.subheader("📄 RAG Response")
                    st.markdown(response)
                    
                    # Show retrieved chunks details
                    with st.expander("🔍 View Retrieved Chunks"):
                        retrieved_chunks = st.session_state.rag_system.retrieve(user_input, top_k=3)
                        
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            st.write(f"**Chunk {i}** (Relevance: {chunk.get('relevance_score', 0):.3f})")
                            st.write(f"**Paper:** {chunk['paper']['title']}")
                            st.write(f"**Authors:** {', '.join(chunk['paper']['authors'])}")
                            st.write(f"**Y-Score:** {chunk['paper']['y_score']}")
                            st.write(f"**Content:** {chunk['content'][:300]}...")
                            st.divider()
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
                    st.exception(e)
    
    # Show repository statistics
    if st.session_state.papers_loaded:
        st.subheader("📊 Repository Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Papers", 
                len(st.session_state.rag_system.papers)
            )
        
        with col2:
            st.metric(
                "Total Chunks", 
                len(st.session_state.rag_system.paper_chunks)
            )
        
        with col3:
            avg_y_score = sum(p['y_score'] for p in st.session_state.rag_system.papers) / len(st.session_state.rag_system.papers)
            st.metric(
                "Avg Y-Score", 
                f"{avg_y_score:.2f}"
            )
        
        # Show paper list
        with st.expander("📚 View All Papers"):
            for paper in st.session_state.rag_system.papers:
                st.write(f"**{paper['title']}** (Y-Score: {paper['y_score']})")
                st.write(f"Authors: {', '.join(paper['authors'])}")
                st.write(f"Collections: {', '.join(paper['collections'])}")
                st.divider()

if __name__ == "__main__":
    main()