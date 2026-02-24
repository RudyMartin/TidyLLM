#!/usr/bin/env python3
"""
SME RAG Adapter - Subject Matter Expert RAG System
==================================================

A specialized RAG adapter for managing subject matter expert knowledge
with collection management, authority tiers, and rich metadata support.

Features:
- Collection-based document organization
- Authority tier management for expertise levels
- Smart document chunking
- Rich JSONB metadata storage
- PostgreSQL-backed persistence
"""

import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Import base adapter and types
from ..base import BaseRAGAdapter, RAGQuery, RAGResponse

# Import SME RAG System for internal operations
from .sme_rag_system import SMERAGSystem, Collection

# Import consolidated infrastructure delegate
from ....infrastructure.infra_delegate import get_infra_delegate

logger = logging.getLogger(__name__)


class SMERAGAdapter(BaseRAGAdapter):
    """
    Subject Matter Expert RAG Adapter.

    Provides specialized RAG functionality for managing expert knowledge
    with collections, authority tiers, and rich metadata support.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize SME RAG Adapter with configuration."""
        super().__init__()
        self.config = config or {}

        # Initialize infrastructure delegate
        self.infra = get_infra_delegate()

        # Initialize SME RAG System for internal operations
        self.sme_system = SMERAGSystem()

        # Configuration
        self.default_collection = self.config.get(
            "default_collection", "general_knowledge"
        )
        self.chunk_size = self.config.get("chunk_size", 800)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)

        logger.info("SME RAG Adapter initialized with collection management support")

    def query(self, request: RAGQuery) -> RAGResponse:
        """
        Execute SME-specific RAG query with collection and authority filtering.

        Args:
            request: RAG query request with domain and authority tier

        Returns:
            RAG response with expert knowledge
        """
        logger.info(f"SME RAG query: {request.query} in domain: {request.domain}")

        # Determine collection to search
        collection_name = request.domain or self.default_collection

        # Search within the specified collection
        try:
            results = self.sme_system.search_collection(
                collection_name=collection_name,
                query=request.query,
                limit=request.max_results or 5,
            )

            if results:
                # Build expert response from results
                response_text = self._build_expert_response(request.query, results)

                # Calculate confidence based on match quality
                confidence = self._calculate_confidence(results)

                return RAGResponse(
                    response=response_text,
                    confidence=confidence,
                    sources=results,
                    authority_tier=request.authority_tier or 2,
                    collection_name=collection_name,
                    precedence_level=confidence * 0.9,
                    adapter_type="sme_rag",
                    metadata={
                        "adapter": "sme_rag",
                        "collection": collection_name,
                        "result_count": len(results),
                    },
                )
            else:
                return RAGResponse(
                    response=f"No expert knowledge found for '{request.query}' in {collection_name} collection.",
                    confidence=0.0,
                    sources=[],
                    authority_tier=request.authority_tier or 0,
                    collection_name=collection_name,
                    precedence_level=0.0,
                    adapter_type="sme_rag",
                    metadata={
                        "adapter": "sme_rag",
                        "collection": collection_name,
                        "status": "no_results",
                    },
                )

        except Exception as e:
            logger.error(f"SME RAG query error: {e}")
            return RAGResponse(
                response=f"Error searching expert knowledge: {str(e)}",
                confidence=0.0,
                sources=[],
                authority_tier=request.authority_tier or 0,
                collection_name=request.domain or self.default_collection,
                precedence_level=0.0,
                adapter_type="sme_rag",
                metadata={"adapter": "sme_rag", "error": str(e)},
            )

    def health_check(self) -> Dict[str, Any]:
        """
        Check SME RAG adapter health.

        Returns:
            Health status dictionary
        """
        try:
            # Test database connection through SME system
            collections = self.sme_system.get_all_collections_including_legacy()

            # Test infrastructure delegate
            conn = self.infra.get_db_connection()
            connected = conn is not None
            if conn:
                self.infra.return_db_connection(conn)

            return {
                "status": "healthy",
                "adapter": "sme_rag",
                "database": "connected" if connected else "disconnected",
                "collections_available": len(collections),
                "infrastructure": "operational",
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "adapter": "sme_rag", "error": str(e)}

    def get_info(self) -> Dict[str, Any]:
        """
        Get SME RAG adapter information.

        Returns:
            Adapter information dictionary
        """
        return {
            "name": "SME RAG Adapter",
            "version": "2.0",
            "type": "subject_matter_expert",
            "capabilities": [
                "collection_management",
                "authority_tiers",
                "document_chunking",
                "metadata_storage",
                "expert_search",
            ],
            "description": "Specialized RAG for subject matter expert knowledge management",
            "configuration": {
                "default_collection": self.default_collection,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        }

    def create_collection(
        self, name: str, description: str = "", settings: Dict = None
    ) -> Dict[str, Any]:
        """
        Create a new SME knowledge collection.

        Args:
            name: Collection name
            description: Collection description
            settings: Additional collection settings

        Returns:
            Collection creation result
        """
        try:
            collection_id = self.sme_system.create_collection(
                name=name, description=description, settings=settings
            )

            logger.info(f"Created SME collection: {name} with ID: {collection_id}")

            return {
                "success": True,
                "collection_id": collection_id,
                "collection_name": name,
                "message": f"Collection '{name}' created successfully",
            }

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to create collection '{name}'",
            }

    def add_document(
        self, collection_name: str, content: str, metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Add a document to an SME collection.

        Args:
            collection_name: Target collection name
            content: Document content
            metadata: Document metadata

        Returns:
            Document addition result
        """
        try:
            # Find or create collection
            collections = self.sme_system.get_all_collections_including_legacy()
            collection = None

            for col in collections:
                if col.name == collection_name:
                    collection = col
                    break

            if not collection:
                # Create collection if it doesn't exist
                collection_id = self.sme_system.create_collection(
                    name=collection_name,
                    description=f"Auto-created collection for {collection_name}",
                )
            else:
                collection_id = collection.collection_id

            # Store document
            doc_id = self.sme_system.store_document(
                collection_id=collection_id, content=content, metadata=metadata
            )

            # Create chunks for the document
            chunks = self._create_document_chunks(content)
            chunk_count = self._store_chunks(doc_id, chunks, metadata)

            logger.info(
                f"Added document {doc_id} to collection {collection_name} with {chunk_count} chunks"
            )

            return {
                "success": True,
                "doc_id": doc_id,
                "collection": collection_name,
                "chunks_created": chunk_count,
                "message": f"Document added to '{collection_name}' successfully",
            }

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to add document to '{collection_name}'",
            }

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available SME collections.

        Returns:
            List of collection information
        """
        try:
            collections = self.sme_system.get_all_collections_including_legacy()

            return [
                {
                    "collection_id": col.collection_id,
                    "name": col.name,
                    "description": col.description,
                    "settings": col.settings,
                    "type": "sme_collection",
                }
                for col in collections
            ]

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def search_collection(
        self, collection_name: str, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific SME collection.

        Args:
            collection_name: Collection to search
            query: Search query
            limit: Maximum results

        Returns:
            List of search results
        """
        try:
            return self.sme_system.search_collection(
                collection_name=collection_name, query=query, limit=limit
            )
        except Exception as e:
            logger.error(f"Collection search error: {e}")
            return []

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific collection.

        Args:
            collection_name: Collection name

        Returns:
            Collection statistics
        """
        try:
            conn = self.infra.get_db_connection()
            if not conn:
                return {"error": "No database connection"}

            cursor = conn.cursor()

            # Get collection info and document count
            cursor.execute(
                """
                SELECT
                        c.collection_id,
                        c.collection_name,
                        c.description,
                        COUNT(DISTINCT d.doc_id) as doc_count,
                        COUNT(DISTINCT dc.chunk_id) as chunk_count
                    FROM sme_collections c
                    LEFT JOIN document_metadata d ON d.doc_type = 'sme_rag'
                    LEFT JOIN document_chunks dc ON dc.doc_id = d.doc_id
                    WHERE c.collection_name = %s
                    GROUP BY c.collection_id, c.collection_name, c.description
                """,
                (collection_name,),
            )

            result = cursor.fetchone()

            if result:
                return {
                    "collection_id": result[0],
                    "name": result[1],
                    "description": result[2],
                    "document_count": result[3],
                    "chunk_count": result[4],
                    "status": "active",
                }
            else:
                return {"name": collection_name, "status": "not_found"}

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                self.infra.return_db_connection(conn)

    def _build_expert_response(self, query: str, results: List[Dict]) -> str:
        """Build an expert response from search results."""
        if not results:
            return f"No expert knowledge available for '{query}'."

        # Extract top results
        top_chunks = []
        for result in results[:3]:  # Top 3 results
            content = result.get("content", "")
            if content:
                top_chunks.append(content)

        if not top_chunks:
            return f"Found references to '{query}' but no detailed content available."

        # Build response
        response = f"Based on expert knowledge regarding '{query}':\n\n"

        for i, chunk in enumerate(top_chunks, 1):
            # Clean and format chunk
            chunk = chunk.strip()
            if len(chunk) > 300:
                chunk = chunk[:300] + "..."
            response += f"{i}. {chunk}\n\n"

        response += f"\nFound {len(results)} relevant expert knowledge entries."

        return response

    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score based on results."""
        if not results:
            return 0.0

        # Simple confidence calculation based on result count
        # Could be enhanced with similarity scores
        base_confidence = min(len(results) / 5.0, 1.0)  # Max confidence at 5+ results

        # Adjust based on metadata quality
        has_metadata = sum(1 for r in results if r.get("metadata"))
        metadata_factor = has_metadata / len(results) if results else 0

        confidence = (base_confidence * 0.7) + (metadata_factor * 0.3)

        return min(confidence, 0.95)  # Cap at 0.95

    def _create_document_chunks(self, content: str) -> List[str]:
        """Create smart chunks from document content."""
        if len(content) <= self.chunk_size:
            return [content]

        chunks = []

        # Split on paragraphs first
        paragraphs = content.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph fits in current chunk
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)

                # Start new chunk with overlap
                if chunks and self.chunk_overlap > 0:
                    # Get last part of previous chunk for overlap
                    last_chunk = chunks[-1]
                    overlap_text = last_chunk[-(self.chunk_overlap) :]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _store_chunks(
        self, doc_id: str, chunks: List[str], metadata: Dict = None
    ) -> int:
        """Store document chunks in database."""
        conn = self.infra.get_db_connection()
        if not conn:
            logger.warning("No database connection for chunk storage")
            return 0

        try:
            cursor = conn.cursor()
            chunk_count = 0

            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i:03d}"

                cursor.execute(
                    """
                    INSERT INTO document_chunks (
                        doc_id, chunk_id, page_num, chunk_text,
                        embedding_model, token_estimate, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET chunk_text = EXCLUDED.chunk_text,
                        token_estimate = EXCLUDED.token_estimate
                """,
                    (
                        doc_id,
                        chunk_id,
                        i + 1,
                        chunk_text,
                        "sme_rag_v2",
                        len(chunk_text.split()),
                        datetime.now(),
                    ),
                )
                chunk_count += 1

            conn.commit()
            self.infra.return_db_connection(conn)
            return chunk_count

        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            if conn:
                conn.rollback()
                self.infra.return_db_connection(conn)
            return 0


def get_sme_rag_adapter(config: Dict[str, Any] = None) -> SMERAGAdapter:
    """Factory function to get SME RAG Adapter instance."""
    return SMERAGAdapter(config or {})
