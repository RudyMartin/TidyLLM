#!/usr/bin/env python3
"""
AI-Powered RAG Adapter - Real AI Analysis, Not Just Chunks
==========================================================

Uses existing LLM infrastructure to actually READ and ANALYZE documents
instead of just returning raw chunks like a search engine.
"""

import uuid
import psycopg2
import yaml
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import existing LLM infrastructure
sys.path.append('tidyllm')
try:
    from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway
    from tidyllm.infrastructure.standards import TidyLLMStandardRequest, TidyLLMStandardResponse, ResponseStatus
    from tidyllm.infrastructure.session.unified import UnifiedSessionManager
    LLM_GATEWAY_AVAILABLE = True
    SESSION_MANAGER_AVAILABLE = True
except ImportError as e:
    LLM_GATEWAY_AVAILABLE = False
    SESSION_MANAGER_AVAILABLE = False
    print(f"MLflow integration not available - using fallback: {e}")

@dataclass
class RAGQuery:
    """Query for AI-powered RAG system."""
    query: str
    domain: str
    authority_tier: Optional[int] = None
    confidence_threshold: float = 0.7

@dataclass
class RAGResponse:
    """AI-generated response from RAG system."""
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    authority_tier: int
    collection_name: str
    precedence_level: float
    ai_analysis: Optional[str] = None

class AIPoweredRAGAdapter:
    """
    RAG adapter that uses AI to actually READ and analyze documents.

    Instead of returning raw chunks, this adapter:
    1. Retrieves relevant document chunks
    2. Feeds them to Claude/GPT for analysis
    3. Returns intelligent AI-generated responses
    """

    def __init__(self):
        """Initialize with AI capabilities."""
        self.db_config = self._load_database_config()

        # Initialize LLM gateway with V2 architecture
        if LLM_GATEWAY_AVAILABLE and SESSION_MANAGER_AVAILABLE:
            try:
                # Initialize UnifiedSessionManager first (V2 architecture requirement)
                self.session_manager = UnifiedSessionManager()

                # Initialize Corporate LLM Gateway (V2 architecture)
                self.llm_gateway = CorporateLLMGateway()

                # Set session manager after initialization (V2 architecture pattern)
                self.llm_gateway.session_manager = self.session_manager

                print("AI-Powered RAG with Corporate LLM Gateway + UnifiedSessionManager initialized (V2 Architecture)")
            except Exception as e:
                print(f"WARNING: LLM Gateway V2 setup failed: {e}")
                self.llm_gateway = None
                self.session_manager = None
        else:
            self.llm_gateway = None
            self.session_manager = None
            print("WARNING: AI-Powered RAG initialized without LLM Gateway (missing V2 components)")

    def _load_database_config(self):
        """Load database configuration."""
        settings_path = Path("tidyllm/admin/settings.yaml")
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['credentials']['postgresql']

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['username'],
            password=self.db_config['password'],
            sslmode=self.db_config['ssl_mode']
        )

    def _create_analysis_prompt(self, query: str, context_chunks: List[str]) -> str:
        """Create prompt for AI to analyze document content."""
        context = "\n\n".join([f"Document Excerpt {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

        prompt = f"""You are an expert document analyzer. A user has asked a question about their document collection, and I've retrieved the most relevant excerpts. Your job is to READ and ANALYZE these excerpts to provide an intelligent, comprehensive answer.

USER QUESTION: {query}

RELEVANT DOCUMENT EXCERPTS:
{context}

INSTRUCTIONS:
1. READ through all the document excerpts carefully
2. ANALYZE the content to understand the key information
3. SYNTHESIZE an intelligent response that directly answers the user's question
4. CITE specific information from the documents when relevant
5. If the excerpts don't contain enough information, say so honestly
6. Provide a clear, professional response as if you're a knowledgeable assistant

RESPONSE FORMAT:
Provide a clear, direct answer to the user's question based on your analysis of the documents. Include specific references to the document content where appropriate."""

        return prompt

    def _fallback_response_generation(self, query: str, context_chunks: List[str]) -> str:
        """Fallback response generation when LLM not available."""
        if not context_chunks:
            return f"I couldn't find specific information about '{query}' in the document collection."

        # Extract key sentences that match query terms
        query_terms = query.lower().split()
        relevant_sentences = []

        for chunk in context_chunks[:3]:  # Top 3 chunks
            sentences = chunk.split('.')
            for sentence in sentences:
                if any(term in sentence.lower() for term in query_terms if len(term) > 2):
                    relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            content = '. '.join(relevant_sentences[:3])  # Top 3 sentences
            return f"""Based on analysis of the documents, here's what I found regarding '{query}':

{content}

This information comes from analyzing the most relevant sections of your document collection."""
        else:
            return f"The documents contain information related to '{query}', but I need better AI analysis capabilities to provide a detailed response. Current content preview: {context_chunks[0][:200]}..."

    def intelligent_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Create search query
            search_query = f"%{query.lower()}%"
            print(f"Searching for: {search_query}")

            cursor.execute("""
                SELECT
                    dc.chunk_id,
                    dc.chunk_text,
                    dc.doc_id,
                    dm.title
                FROM document_chunks dc
                JOIN document_metadata dm ON dc.doc_id = dm.doc_id
                WHERE dc.chunk_text ILIKE %s
                AND dc.chunk_text NOT LIKE '%%Real extraction requires file upload%%'
                AND dc.chunk_text NOT LIKE '%%Content would be extracted here%%'
                AND LENGTH(dc.chunk_text) > 20
                ORDER BY dc.created_at DESC
                LIMIT %s
            """, (search_query, limit))

            rows = cursor.fetchall()
            print(f"Found {len(rows)} matching chunks")

            results = []
            for row in rows:
                if len(row) >= 4:  # Ensure we have all expected columns
                    results.append({
                        'chunk_id': row[0],
                        'content': row[1],
                        'doc_id': row[2],
                        'filename': row[3],
                        'embedding_model': 'ai_powered_rag',
                        'similarity_score': 0.8
                    })
                else:
                    print(f"Skipping malformed row with {len(row)} columns")

            # If no results, try broader search
            if not results:
                print("No specific matches, trying broader search...")
                cursor.execute("""
                    SELECT
                        dc.chunk_id,
                        dc.chunk_text,
                        dc.doc_id,
                        dm.title
                    FROM document_chunks dc
                    JOIN document_metadata dm ON dc.doc_id = dm.doc_id
                    WHERE dc.chunk_text NOT LIKE '%%Real extraction requires file upload%%'
                    AND dc.chunk_text NOT LIKE '%%Content would be extracted here%%'
                    AND LENGTH(dc.chunk_text) > 20
                    ORDER BY dc.created_at DESC
                    LIMIT %s
                """, (limit,))

                rows = cursor.fetchall()
                print(f"Broader search found {len(rows)} chunks")

                for row in rows:
                    if len(row) >= 4:  # Ensure we have all expected columns
                        results.append({
                            'chunk_id': row[0],
                            'content': row[1],
                            'doc_id': row[2],
                            'filename': row[3],
                            'embedding_model': 'ai_powered_rag',
                            'similarity_score': 0.5
                        })
                    else:
                        print(f"Skipping malformed row with {len(row)} columns")

            conn.close()
            print(f"Returning {len(results)} results")
            return results

        except Exception as e:
            conn.close()
            print(f"Search error: {e}")
            raise e

    def query_unified_rag(self, query: RAGQuery) -> RAGResponse:
        """AI-powered RAG query that actually analyzes content."""
        print(f"AI-Powered RAG processing: {query.query}")

        # Get relevant chunks
        results = self.intelligent_search(query.query)

        if not results:
            return RAGResponse(
                response=f"I couldn't find any relevant documents for '{query.query}' in your collection. Please try different keywords or upload relevant documents.",
                confidence=0.0,
                sources=[],
                authority_tier=0,
                collection_name=query.domain,
                precedence_level=0.0,
                ai_analysis="No relevant content found"
            )

        # Extract content for AI analysis
        context_chunks = [r['content'] for r in results[:3] if len(r['content'].strip()) > 20]

        if not context_chunks:
            return RAGResponse(
                response=f"Found documents related to '{query.query}' but they contain insufficient content for analysis.",
                confidence=0.2,
                sources=results,
                authority_tier=query.authority_tier or 1,
                collection_name=query.domain,
                precedence_level=0.3,
                ai_analysis="Insufficient content"
            )

        # Generate AI-powered response
        if self.llm_gateway:
            try:
                prompt = self._create_analysis_prompt(query.query, context_chunks)

                # Create standard LLM request
                llm_request = TidyLLMStandardRequest(
                    model_id="claude-3-sonnet",
                    user_id="rag_system",
                    session_id=f"rag_{uuid.uuid4().hex[:8]}",
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1500
                )

                print("Sending to AI for analysis...")
                ai_response = self.llm_gateway.process_llm_request(llm_request)

                if ai_response.status == ResponseStatus.SUCCESS:
                    ai_analysis = ai_response.data
                    print("AI analysis complete")

                    return RAGResponse(
                        response=ai_analysis,
                        confidence=0.9,
                        sources=results,
                        authority_tier=query.authority_tier or 2,
                        collection_name=query.domain,
                        precedence_level=0.95,
                        ai_analysis="AI-generated analysis"
                    )
                else:
                    print(f"WARNING: AI analysis failed: {ai_response.error}")
                    # Fallback to basic response
                    fallback_response = self._fallback_response_generation(query.query, context_chunks)

                    return RAGResponse(
                        response=fallback_response,
                        confidence=0.7,
                        sources=results,
                        authority_tier=query.authority_tier or 1,
                        collection_name=query.domain,
                        precedence_level=0.8,
                        ai_analysis="Fallback analysis"
                    )

            except Exception as e:
                print(f"ERROR: AI processing error: {e}")
                # Fallback to basic response
                fallback_response = self._fallback_response_generation(query.query, context_chunks)

                return RAGResponse(
                    response=fallback_response,
                    confidence=0.6,
                    sources=results,
                    authority_tier=query.authority_tier or 1,
                    collection_name=query.domain,
                    precedence_level=0.7,
                    ai_analysis="Error fallback"
                )
        else:
            # No LLM available - use fallback
            fallback_response = self._fallback_response_generation(query.query, context_chunks)

            return RAGResponse(
                response=fallback_response,
                confidence=0.75,
                sources=results,
                authority_tier=query.authority_tier or 1,
                collection_name=query.domain,
                precedence_level=0.8,
                ai_analysis="Basic analysis (no LLM)"
            )

    async def create_document_from_file(self, file_content: str, filename: str, collection_id: str) -> str:
        """Create document with smart chunking."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # DEDUPLICATION CHECK: See if this file already exists
            cursor.execute("""
                SELECT doc_id, doc_type FROM document_metadata
                WHERE title = %s AND doc_type = 'ai_powered_rag'
            """, (filename,))

            existing_doc = cursor.fetchone()
            if existing_doc:
                print(f"Document {filename} already exists with doc_id {existing_doc[0]} - skipping duplicate")
                conn.close()
                return existing_doc[0]

            # Create document metadata
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"

            cursor.execute("""
                INSERT INTO document_metadata (
                    doc_id, title, doc_type, status, last_processed
                ) VALUES (%s, %s, %s, %s, %s)
            """, (
                doc_id,
                filename,
                "ai_powered_rag",
                "processed",
                datetime.now()
            ))

            # Smart chunking
            chunks = self._create_smart_chunks(file_content)
            print(f"Created {len(chunks)} chunks for AI-powered RAG")

            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i:03d}"

                cursor.execute("""
                    INSERT INTO document_chunks (
                        doc_id, chunk_id, page_num, chunk_text, embedding_model,
                        token_estimate, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    doc_id,
                    chunk_id,
                    i + 1,
                    chunk_text,
                    "ai_powered_rag_v1",
                    len(chunk_text.split()),
                    datetime.now()
                ))

            conn.commit()
            conn.close()

            print(f"AI-Powered RAG document created: {doc_id}")
            return doc_id

        except Exception as e:
            conn.rollback()
            conn.close()
            raise e

    def _create_smart_chunks(self, text: str, max_size: int = 1000) -> List[str]:
        """Create smart text chunks for better AI analysis."""
        if len(text) <= max_size:
            return [text]

        # Split on paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def get_or_create_authority_collection(self, domain: str, authority_tier: int, description: str) -> str:
        """Create collection for AI-powered RAG."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Check if collection exists
            cursor.execute("""
                SELECT id FROM yrsn_paper_collections
                WHERE collection_name LIKE %s
            """, (f"{domain}%",))

            existing = cursor.fetchone()
            if existing:
                conn.close()
                return str(existing[0])

            # Create new collection
            cursor.execute("""
                INSERT INTO yrsn_paper_collections (collection_name, description, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (
                f"{domain}_ai_powered_tier_{authority_tier}",
                f"AI-Powered RAG: {description}",
                datetime.now(),
                datetime.now()
            ))

            new_id = cursor.fetchone()[0]
            conn.commit()
            conn.close()

            print(f"Created AI-Powered collection: {domain}_ai_powered_tier_{authority_tier}")
            return str(new_id)

        except Exception as e:
            conn.rollback()
            conn.close()
            raise e

    def list_collections(self) -> List[Dict[str, Any]]:
        """List collections."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    c.id,
                    c.collection_name,
                    c.description,
                    c.created_at,
                    COUNT(dm.doc_id) as doc_count
                FROM yrsn_paper_collections c
                LEFT JOIN document_metadata dm ON TRUE
                GROUP BY c.id, c.collection_name, c.description, c.created_at
                ORDER BY c.created_at DESC
            """)

            collections = []
            for row in cursor.fetchall():
                collections.append({
                    'collection_id': str(row[0]),
                    'collection_name': row[1],
                    'description': row[2],
                    'created_at': row[3],
                    'doc_count': row[4]
                })

            conn.close()
            return collections

        except Exception as e:
            conn.close()
            raise e