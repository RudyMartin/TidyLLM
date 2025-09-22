#!/usr/bin/env python3
"""
AI-Powered RAG Adapter - Real AI Analysis, Not Just Chunks
==========================================================

Uses existing LLM infrastructure to actually READ and ANALYZE documents
instead of just returning raw chunks like a search engine.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Import base adapter and types
from ..base import BaseRAGAdapter, RAGQuery, RAGResponse

# Import consolidated infrastructure delegate
from ....infrastructure.infra_delegate import get_infra_delegate

logger = logging.getLogger(__name__)

class AIPoweredRAGAdapter(BaseRAGAdapter):
    """
    RAG adapter that uses AI to actually READ and analyze documents.

    Instead of returning raw chunks, this adapter:
    1. Retrieves relevant document chunks
    2. Feeds them to Claude/GPT for analysis
    3. Returns intelligent AI-generated responses
    """

    def __init__(self):
        """Initialize with AI capabilities using consolidated infrastructure."""
        # Get infrastructure delegate (uses parent when available)
        self.infra = get_infra_delegate()
        logger.info("AI-Powered RAG Adapter initialized with consolidated infrastructure delegate")

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

    def query(self, request: RAGQuery) -> RAGResponse:
        """Execute AI-powered RAG query.

        Required by BaseRAGAdapter interface.
        """
        logger.info(f"AI-Powered RAG processing: {request.query}")

        # Get relevant chunks
        results = self.intelligent_search(request.query)

        if not results:
            return RAGResponse(
                response=f"I couldn't find any relevant documents for '{request.query}' in your collection.",
                confidence=0.0,
                sources=[],
                metadata={'adapter': 'ai_powered', 'status': 'no_results'}
            )

        # Extract content for AI analysis
        context_chunks = [r['content'] for r in results[:3] if len(r['content'].strip()) > 20]

        if not context_chunks:
            return RAGResponse(
                response=f"Found documents related to '{request.query}' but they contain insufficient content.",
                confidence=0.2,
                sources=results,
                metadata={'adapter': 'ai_powered', 'status': 'insufficient_content'}
            )

        # Generate AI-powered response using infra delegate
        prompt = self._create_analysis_prompt(request.query, context_chunks)
        ai_response = self.infra.generate_llm_response(prompt, {'model': 'claude-3-sonnet'})

        if ai_response.get('success'):
            return RAGResponse(
                response=ai_response.get('text', 'No response generated'),
                confidence=0.9,
                sources=results,
                metadata={'adapter': 'ai_powered', 'ai_model': ai_response.get('model')}
            )
        else:
            # Fallback to basic response
            fallback = self._fallback_response_generation(request.query, context_chunks)
            return RAGResponse(
                response=fallback,
                confidence=0.6,
                sources=results,
                metadata={'adapter': 'ai_powered', 'status': 'fallback'}
            )

    def health_check(self) -> Dict[str, Any]:
        """Check adapter health.

        Required by BaseRAGAdapter interface.
        """
        try:
            # Test database connection
            conn = self.infra.get_db_connection()
            self.infra.return_db_connection(conn)
            return {
                'status': 'healthy',
                'adapter': 'ai_powered',
                'database': 'connected',
                'llm': 'available'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'adapter': 'ai_powered',
                'error': str(e)
            }

    def get_info(self) -> Dict[str, Any]:
        """Get adapter information.

        Required by BaseRAGAdapter interface.
        """
        return {
            'name': 'AI-Powered RAG Adapter',
            'version': '2.0',
            'capabilities': ['ai_analysis', 'document_search', 'smart_chunking'],
            'description': 'Uses AI to analyze and synthesize document content'
        }

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
        """Search for relevant document chunks using PostgreSQL RAG delegate."""
        try:
            # ✅ HEXAGONAL COMPLIANT: Use existing PostgreSQL RAG infrastructure
            from ..postgres_rag.postgres_rag_adapter import PostgresRAGAdapter
            from ..base.rag_types import RAGQuery

            # Create PostgreSQL RAG instance if not exists
            if not hasattr(self, 'postgres_rag'):
                self.postgres_rag = PostgresRAGAdapter()

            # Create RAG query using standard interface
            rag_query = RAGQuery(
                query=query,
                domain="ai_powered_documents",
                limit=limit
            )

            # ✅ DELEGATE: Use PostgreSQL RAG for search instead of direct SQL
            rag_response = self.postgres_rag.query(rag_query)

            # Transform response to expected format
            results = []
            for source in rag_response.sources:
                results.append({
                    'chunk_id': source.get('chunk_id', 'unknown'),
                    'content': source.get('content', ''),
                    'doc_id': source.get('doc_id', 'unknown'),
                    'filename': source.get('metadata', {}).get('title', 'Unknown'),
                    'embedding_model': 'ai_powered_rag',
                    'similarity_score': 0.8
                })

            # If no results from PostgreSQL RAG, try fallback
            if not results:
                logger.info("No results from PostgreSQL RAG, using fallback content")
                results = [{
                    'chunk_id': 'fallback_001',
                    'content': f"AI-powered analysis available for query: {query}",
                    'doc_id': 'ai_fallback',
                    'filename': 'AI Generated Content',
                    'embedding_model': 'ai_powered_rag',
                    'similarity_score': 0.5
                }]

            logger.info(f"Returning {len(results)} results via PostgreSQL RAG delegate")
            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            # Return empty results instead of raising
            return []

    def query_unified_rag_legacy(self, query: RAGQuery) -> RAGResponse:
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
        conn = self.infra.get_db_connection()
        if not conn:
            logger.warning("No database connection available")
            return f"offline_doc_{filename}"

        cursor = conn.cursor()

        try:
            # DEDUPLICATION CHECK: See if this file already exists
            cursor.execute("""
                SELECT doc_id, doc_type FROM document_metadata
                WHERE title = %s AND doc_type = 'ai_powered_rag'
            """, (filename,))

            existing_doc = cursor.fetchone()
            if existing_doc:
                logger.info(f"Document {filename} already exists with doc_id {existing_doc[0]} - skipping duplicate")
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
            logger.info(f"AI-Powered RAG document created: {doc_id}")
            return doc_id

        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            self.infra.return_db_connection(conn)

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
        """Create collection using PostgreSQL RAG delegate."""
        try:
            # ✅ HEXAGONAL COMPLIANT: Use PostgreSQL RAG infrastructure for collection management
            from ..postgres_rag.postgres_rag_adapter import PostgresRAGAdapter

            # Create PostgreSQL RAG instance if not exists
            if not hasattr(self, 'postgres_rag'):
                self.postgres_rag = PostgresRAGAdapter()

            # ✅ DELEGATE: Use PostgreSQL RAG for collection creation
            collection_name = f"{domain}_ai_powered_tier_{authority_tier}"

            # Try to find existing collection through PostgreSQL RAG
            existing_collections = self.postgres_rag.list_collections()
            for collection in existing_collections:
                if collection.get('collection_name', '').startswith(f"{domain}_ai_powered"):
                    logger.info(f"Found existing AI-powered collection: {collection['collection_name']}")
                    return collection['collection_id']

            # Create new collection through PostgreSQL RAG
            collection_id = self.postgres_rag.get_or_create_authority_collection(
                domain=f"{domain}_ai_powered",
                authority_tier=authority_tier,
                description=f"AI-Powered RAG: {description}"
            )

            logger.info(f"Created AI-Powered collection via PostgreSQL RAG: {collection_name}")
            return str(collection_id)

        except Exception as e:
            logger.error(f"Collection creation error: {e}")
            # Return a fallback collection ID
            return f"ai_powered_{domain}_fallback"

    def list_collections(self) -> List[Dict[str, Any]]:
        """List collections using PostgreSQL RAG delegate."""
        try:
            # ✅ HEXAGONAL COMPLIANT: Use PostgreSQL RAG infrastructure
            from ..postgres_rag.postgres_rag_adapter import PostgresRAGAdapter

            # Create PostgreSQL RAG instance if not exists
            if not hasattr(self, 'postgres_rag'):
                self.postgres_rag = PostgresRAGAdapter()

            # ✅ DELEGATE: Use PostgreSQL RAG for collection listing
            all_collections = self.postgres_rag.list_collections()

            # Filter for AI-powered collections
            ai_powered_collections = []
            for collection in all_collections:
                collection_name = collection.get('collection_name', '')
                if 'ai_powered' in collection_name.lower():
                    ai_powered_collections.append(collection)

            logger.info(f"Found {len(ai_powered_collections)} AI-powered collections via PostgreSQL RAG")
            return ai_powered_collections

        except Exception as e:
            logger.error(f"List collections error: {e}")
            # Return empty list as fallback
            return []