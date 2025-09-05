#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Query Demo - Interactive Document Query Interface

Streamlit interface for querying processed documents using RAG (Retrieval-Augmented Generation)
with database integration and intelligent search capabilities.
"""

# Environment setup
from config.setup import setup_env
setup_env()

import streamlit as st
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.mcp.orchestrators.rag_qa_orchestrator import RAGQAOrchestrator
from dotenv import load_dotenv

# Load environment variables
load_dotenv('src/backend/config/credentials.env')

class RAGQueryInterface:
    """Interactive RAG query interface for processed documents"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.rag_orchestrator = None
        self.db_connection = None
        self._setup_connections()
    
    def _setup_connections(self):
        """Setup database and RAG orchestrator connections"""
        try:
            # Setup database connection
            if self.database_url:
                self.db_connection = psycopg2.connect(self.database_url)
                st.success("✅ Database connection established")
            else:
                st.error("❌ DATABASE_URL not found in environment")
                return
            
            # Setup RAG orchestrator
            self.rag_orchestrator = RAGQAOrchestrator()
            st.success("✅ RAG Orchestrator initialized")
            
        except Exception as e:
            st.error(f"❌ Error setting up connections: {e}")
    
    def get_available_documents(self) -> List[Dict[str, Any]]:
        """Get list of available documents from database"""
        if not self.db_connection:
            return []
        
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT DISTINCT doc_id, COUNT(*) as chunk_count
                    FROM document_chunks 
                    GROUP BY doc_id
                    ORDER BY doc_id
                """)
                documents = cursor.fetchall()
                return [dict(doc) for doc in documents]
        except Exception as e:
            st.error(f"❌ Error fetching documents: {e}")
            return []
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get overall document statistics"""
        if not self.db_connection:
            return {}
        
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Total documents
                cursor.execute("SELECT COUNT(DISTINCT doc_id) as total_docs FROM document_chunks")
                total_docs = cursor.fetchone()['total_docs']
                
                # Total chunks
                cursor.execute("SELECT COUNT(*) as total_chunks FROM document_chunks")
                total_chunks = cursor.fetchone()['total_chunks']
                
                # Average chunks per document
                cursor.execute("""
                    SELECT AVG(chunk_count) as avg_chunks_per_doc
                    FROM (
                        SELECT doc_id, COUNT(*) as chunk_count
                        FROM document_chunks 
                        GROUP BY doc_id
                    ) as doc_chunks
                """)
                avg_chunks = cursor.fetchone()['avg_chunks_per_doc']
                
                # Document types
                cursor.execute("""
                    SELECT doc_id, COUNT(*) as chunk_count
                    FROM document_chunks 
                    GROUP BY doc_id
                    ORDER BY chunk_count DESC
                    LIMIT 10
                """)
                top_docs = cursor.fetchall()
                
                return {
                    'total_documents': total_docs,
                    'total_chunks': total_chunks,
                    'avg_chunks_per_doc': round(avg_chunks, 2) if avg_chunks else 0,
                    'top_documents': [dict(doc) for doc in top_docs]
                }
        except Exception as e:
            st.error(f"❌ Error fetching statistics: {e}")
            return {}
    
    def search_documents(self, query: str, top_k: int = 5, selected_docs: List[str] = None) -> List[Dict[str, Any]]:
        """Search documents using RAG capabilities"""
        if not self.rag_orchestrator:
            st.error("❌ RAG Orchestrator not available")
            return []
        
        try:
            # Use the RAG orchestrator to search
            results = self.rag_orchestrator.search_documents(
                query=query,
                top_k=top_k,
                document_filter=selected_docs
            )
            return results
        except Exception as e:
            st.error(f"❌ Error searching documents: {e}")
            return []
    
    def answer_question(self, question: str, context_docs: List[str] = None) -> Dict[str, Any]:
        """Answer questions using RAG capabilities"""
        if not self.rag_orchestrator:
            st.error("❌ RAG Orchestrator not available")
            return {}
        
        try:
            # Use the RAG orchestrator to answer questions
            answer = self.rag_orchestrator.answer_question_with_rag(
                question=question,
                context_documents=context_docs
            )
            return answer
        except Exception as e:
            st.error(f"❌ Error answering question: {e}")
            return {}

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="RAG Document Query Interface",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG Document Query Interface")
    st.markdown("### Interactive Query Interface for Processed Documents")
    
    # Initialize RAG interface
    if 'rag_interface' not in st.session_state:
        st.session_state.rag_interface = RAGQueryInterface()
    
    rag_interface = st.session_state.rag_interface
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Main App", use_container_width=True):
            st.switch_page("main.py")
    
    with col2:
        if st.button("📋 QA Demo", use_container_width=True):
            st.switch_page("qa_demo.py")
    
    with col3:
        if st.button("🤖 Enhanced QA", use_container_width=True):
            st.switch_page("enhanced_qa_demo.py")
    
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Query Configuration")
        
        # Search parameters
        st.subheader("🔍 Search Parameters")
        
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of top results to return"
        )
        
        search_type = st.selectbox(
            "Search Type",
            ["semantic", "keyword", "hybrid"],
            help="Type of search to perform"
        )
        
        st.markdown("---")
        
        # Document selection
        st.subheader("📄 Document Selection")
        
        # Get available documents
        available_docs = rag_interface.get_available_documents()
        
        if available_docs:
            doc_options = [doc['doc_id'] for doc in available_docs]
            selected_docs = st.multiselect(
                "Select Documents to Search",
                options=doc_options,
                default=doc_options[:5] if len(doc_options) > 5 else doc_options,
                help="Select specific documents to search in (leave empty for all documents)"
            )
        else:
            st.warning("⚠️ No documents found in database")
            selected_docs = []
        
        st.markdown("---")
        
        # Document statistics
        st.subheader("📊 Document Statistics")
        
        stats = rag_interface.get_document_stats()
        if stats:
            st.metric("Total Documents", stats.get('total_documents', 0))
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Avg Chunks/Doc", stats.get('avg_chunks_per_doc', 0))
            
            if stats.get('top_documents'):
                st.write("**Top Documents by Chunk Count:**")
                for doc in stats['top_documents'][:5]:
                    st.write(f"  - {doc['doc_id']}: {doc['chunk_count']} chunks")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Document Search & Query")
        
        # Search interface
        st.subheader("📝 Search Documents")
        
        search_query = st.text_input(
            "Enter your search query",
            placeholder="e.g., model validation best practices, risk assessment methods, compliance requirements...",
            help="Enter keywords or phrases to search for in the documents"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Search Documents", type="primary", use_container_width=True):
                if search_query:
                    with st.spinner("Searching documents..."):
                        search_results = rag_interface.search_documents(
                            query=search_query,
                            top_k=top_k,
                            selected_docs=selected_docs if selected_docs else None
                        )
                        
                        if search_results:
                            st.session_state.search_results = search_results
                            st.success(f"✅ Found {len(search_results)} results")
                        else:
                            st.warning("⚠️ No results found")
                else:
                    st.error("❌ Please enter a search query")
        
        with col2:
            if st.button("🔄 Clear Results", use_container_width=True):
                if 'search_results' in st.session_state:
                    del st.session_state.search_results
                st.rerun()
        
        st.markdown("---")
        
        # Question answering interface
        st.subheader("❓ Ask Questions")
        
        question = st.text_area(
            "Ask a question about the documents",
            placeholder="e.g., What are the key steps in model validation? How do different papers approach risk assessment?",
            height=100,
            help="Ask specific questions about the content in the documents"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 Get Answer", type="primary", use_container_width=True):
                if question:
                    with st.spinner("Generating answer..."):
                        answer_result = rag_interface.answer_question(
                            question=question,
                            context_docs=selected_docs if selected_docs else None
                        )
                        
                        if answer_result:
                            st.session_state.answer_result = answer_result
                            st.success("✅ Answer generated")
                        else:
                            st.warning("⚠️ Could not generate answer")
                else:
                    st.error("❌ Please enter a question")
        
        with col2:
            if st.button("🔄 Clear Answer", use_container_width=True):
                if 'answer_result' in st.session_state:
                    del st.session_state.answer_result
                st.rerun()
        
        st.markdown("---")
        
        # Display search results
        if 'search_results' in st.session_state and st.session_state.search_results:
            st.subheader("📋 Search Results")
            
            for i, result in enumerate(st.session_state.search_results, 1):
                with st.expander(f"Result {i}: {result.get('document', 'Unknown Document')}"):
                    st.write(f"**Document:** {result.get('document', 'Unknown')}")
                    st.write(f"**Page:** {result.get('page', 'Unknown')}")
                    st.write(f"**Relevance Score:** {result.get('score', 0):.3f}")
                    
                    # Display chunk text
                    chunk_text = result.get('chunk_text', 'No text available')
                    st.write("**Content:**")
                    st.text_area(
                        f"Chunk {i}",
                        value=chunk_text,
                        height=150,
                        key=f"chunk_{i}",
                        disabled=True
                    )
                    
                    # Show metadata if available
                    if 'metadata' in result:
                        st.write("**Metadata:**")
                        st.json(result['metadata'])
        
        # Display answer results
        if 'answer_result' in st.session_state and st.session_state.answer_result:
            st.subheader("🤖 Generated Answer")
            
            answer = st.session_state.answer_result
            
            # Display the answer
            st.write("**Answer:**")
            st.write(answer.get('answer', 'No answer generated'))
            
            # Display confidence if available
            if 'confidence' in answer:
                st.write(f"**Confidence:** {answer['confidence']:.2f}")
            
            # Display sources if available
            if 'sources' in answer and answer['sources']:
                st.write("**Sources:**")
                for source in answer['sources']:
                    st.write(f"  - {source}")
            
            # Display context if available
            if 'context' in answer and answer['context']:
                with st.expander("View Context Used"):
                    st.write(answer['context'])
    
    with col2:
        st.header("📈 Query Analytics")
        
        # Query history
        st.subheader("📝 Recent Queries")
        
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        # Add current query to history
        if search_query and search_query not in [q['query'] for q in st.session_state.query_history]:
            st.session_state.query_history.append({
                'query': search_query,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'search'
            })
        
        if question and question not in [q['query'] for q in st.session_state.query_history]:
            st.session_state.query_history.append({
                'query': question,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'question'
            })
        
        # Display recent queries
        for query_entry in st.session_state.query_history[-5:]:
            query_type_icon = "🔍" if query_entry['type'] == 'search' else "❓"
            st.write(f"{query_type_icon} **{query_entry['timestamp']}**")
            st.write(f"*{query_entry['query'][:50]}{'...' if len(query_entry['query']) > 50 else ''}*")
            st.markdown("---")
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.query_history = []
            st.rerun()
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        # Predefined queries
        predefined_queries = [
            "model validation best practices",
            "risk assessment methodology",
            "compliance requirements",
            "data quality standards",
            "validation framework",
            "performance metrics",
            "documentation standards",
            "review process"
        ]
        
        for query in predefined_queries:
            if st.button(f"🔍 {query}", use_container_width=True):
                st.session_state.quick_query = query
                st.rerun()
        
        # Quick questions
        predefined_questions = [
            "What are the key steps in model validation?",
            "How do different approaches handle risk assessment?",
            "What compliance standards are mentioned?",
            "What are the common validation metrics?",
            "How is data quality assessed?"
        ]
        
        for question in predefined_questions:
            if st.button(f"❓ {question[:30]}...", use_container_width=True):
                st.session_state.quick_question = question
                st.rerun()
        
        # Handle quick actions
        if 'quick_query' in st.session_state:
            search_query = st.session_state.quick_query
            del st.session_state.quick_query
        
        if 'quick_question' in st.session_state:
            question = st.session_state.quick_question
            del st.session_state.quick_question

if __name__ == "__main__":
    main()
