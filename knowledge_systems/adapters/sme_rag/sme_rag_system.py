#!/usr/bin/env python3
"""
SME RAG System - Real Implementation using PostgreSQL
=====================================================

A real implementation that uses the existing PostgreSQL infrastructure
for document storage and retrieval.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Embedding model for document processing."""

    def __init__(self, model_name: str = "titan-embed-text-v1"):
        self.model_name = model_name


class Collection:
    """Collection representation."""

    def __init__(
        self,
        collection_id: str,
        name: str,
        description: str = "",
        settings: Dict = None,
    ):
        self.collection_id = collection_id
        self.collection_name = name
        self.name = name
        self.description = description
        self.settings = settings or {}

    def to_dict(self):
        return {
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "description": self.description,
            "settings": self.settings,
        }


class SMERAGSystem:
    """
    Real SME RAG System implementation using PostgreSQL.
    Uses actual database operations through infrastructure delegate.
    """

    def __init__(self):
        """Initialize SME RAG system with real database connection."""
        # Import here to avoid circular dependency
        from ....infrastructure.infra_delegate import get_infra_delegate

        self.infra = get_infra_delegate()
        logger.info("SME RAG System initialized with real PostgreSQL connection")

    def get_all_collections_including_legacy(self) -> List[Collection]:
        """Get all collections from database including legacy ones."""
        collections = []
        conn = self.infra.get_db_connection()
        if not conn:
            logger.warning("No database connection available")
            return []

        try:
            cursor = conn.cursor()

            # Try sme_collections table first
            try:
                cursor.execute(
                    """
                    SELECT id, collection_name, description, settings
                    FROM sme_collections
                    ORDER BY created_at DESC
                    """
                )

                rows = cursor.fetchall()
                for row in rows:
                    collections.append(
                        Collection(
                            collection_id=str(row[0]),
                            name=row[1],
                            description=row[2] or "",
                            settings=row[3] if isinstance(row[3], dict) else {},
                        )
                    )
            except Exception as e:
                logger.debug(f"sme_collections not available: {e}")

            # Try yrsn_paper_collections as fallback
            if not collections:
                try:
                    cursor.execute(
                        """
                        SELECT id, collection_name, description
                        FROM yrsn_paper_collections
                        ORDER BY created_at DESC
                        """
                    )

                    rows = cursor.fetchall()
                    for row in rows:
                        collections.append(
                            Collection(
                                collection_id=str(row[0]),
                                name=row[1],
                                description=row[2] or "",
                                settings={},
                            )
                        )
                except Exception as e:
                    logger.debug(f"yrsn_paper_collections not available: {e}")

        except Exception as e:
            logger.error(f"Error getting collections: {e}")
        finally:
            self.infra.return_db_connection(conn)

        logger.info(f"Found {len(collections)} collections")
        return collections

    def search_collection(
        self, collection_name: str, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific collection using real database queries."""
        results = []
        conn = self.infra.get_db_connection()
        if not conn:
            logger.warning("No database connection available")
            return []

        try:
            cursor = conn.cursor()

            # Search in sme_document_chunks
            cursor.execute(
                """
                    SELECT
                        dc.id,
                        dc.content,
                        dc.metadata,
                        d.id as doc_id
                    FROM sme_document_chunks dc
                    JOIN sme_documents d ON dc.document_id = d.id
                    WHERE dc.content ILIKE %s
                    LIMIT %s
                """,
                (f"%{query}%", limit),
            )

            rows = cursor.fetchall()
            for row in rows:
                results.append(
                    {
                        "chunk_id": str(row[0]),
                        "content": row[1],
                        "metadata": row[2] if isinstance(row[2], dict) else {},
                        "doc_id": str(row[3]),
                        "collection": collection_name,
                    }
                )

        except Exception as e:
            logger.debug(f"Search in sme_document_chunks failed: {e}")

            # Fallback to document_chunks table
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                        SELECT
                            chunk_id,
                            chunk_text,
                            doc_id
                        FROM document_chunks
                        WHERE chunk_text ILIKE %s
                        LIMIT %s
                    """,
                    (f"%{query}%", limit),
                )

                rows = cursor.fetchall()
                for row in rows:
                    results.append(
                        {
                            "chunk_id": row[0],
                            "content": row[1],
                            "metadata": {},
                            "doc_id": row[2],
                            "collection": collection_name,
                        }
                    )
            except Exception as e2:
                logger.error(f"Fallback search also failed: {e2}")
        finally:
            self.infra.return_db_connection(conn)

        logger.info(f"Search found {len(results)} results for query: {query}")
        return results

    def create_collection(
        self, name: str, description: str = "", settings: Dict = None
    ) -> str:
        """Create a new collection in the database."""
        conn = self.infra.get_db_connection()
        if not conn:
            logger.warning("No database connection available")
            return f"offline_collection_{name}"

        try:
            cursor = conn.cursor()

            # Try sme_collections first
            try:
                cursor.execute(
                    """
                    INSERT INTO sme_collections (collection_name, description, settings, created_at)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (name, description, json.dumps(settings or {}), datetime.now()),
                )

                collection_id = cursor.fetchone()[0]
                conn.commit()

            except Exception as e:
                # Fallback to yrsn_paper_collections
                cursor.execute(
                    """
                        INSERT INTO yrsn_paper_collections (collection_name, description, created_at, updated_at)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """,
                    (name, description, datetime.now(), datetime.now()),
                )

                collection_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"Created collection {name} with ID {collection_id}")
                return str(collection_id)

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return f"error_collection_{name}"
        finally:
            self.infra.return_db_connection(conn)

    def store_document(
        self, collection_id: str, content: str, metadata: Dict = None
    ) -> str:
        """Store a document in the specified collection."""
        conn = self.infra.get_db_connection()
        if not conn:
            logger.warning("No database connection available")
            return "offline_doc_id"

        try:
            cursor = conn.cursor()

            # Store in sme_documents
            cursor.execute(
                """
                INSERT INTO sme_documents (content, metadata, created_at)
                VALUES (%s, %s, %s)
                RETURNING id
            """,
                (content, json.dumps(metadata or {}), datetime.now()),
            )

            doc_id = cursor.fetchone()[0]
            conn.commit()

            logger.info(f"Stored document with ID {doc_id}")
            return str(doc_id)

        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return "error_doc_id"
        finally:
            self.infra.return_db_connection(conn)
