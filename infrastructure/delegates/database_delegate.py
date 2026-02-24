#!/usr/bin/env python3
"""
Database Delegate - Infrastructure Layer
========================================

Delegate for database operations following hexagonal architecture.
Provides clean interface for adapters without exposing infrastructure details.
"""

from typing import Dict, Any, Optional, List
import logging
import psycopg2
import yaml
from pathlib import Path
from datetime import datetime
import uuid
import sys

logger = logging.getLogger(__name__)

# Try to import parent infrastructure's ResilientPoolManager
try:
    # Add parent infrastructure to path
    qa_root = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(qa_root))

    from infrastructure.services.resilient_pool_manager import ResilientPoolManager
    from infrastructure.services.credential_carrier import get_credential_carrier
    RESILIENT_POOL_AVAILABLE = True
except ImportError:
    RESILIENT_POOL_AVAILABLE = False
    logger.info("ResilientPoolManager not available, using fallback connection pooling")


class DatabaseDelegate:
    """
    Delegate for database operations with resilient connection pooling.

    Uses parent infrastructure's ResilientPoolManager when available for:
    - Primary/backup pool failover
    - Health monitoring and automatic recovery
    - Transparent failover for applications

    Falls back to simple ThreadedConnectionPool if not available.
    """

    def __init__(self):
        """Initialize database delegate with resilient pooling."""
        self._pool_manager = None
        self._connection_pool = None  # Fallback pool
        self._config = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of database connection pool."""
        if self._initialized:
            return True

        try:
            # Try to use ResilientPoolManager first
            if RESILIENT_POOL_AVAILABLE:
                try:
                    # Get credential carrier for configuration
                    credential_carrier = get_credential_carrier() if 'get_credential_carrier' in globals() else None

                    # Initialize resilient pool manager
                    self._pool_manager = ResilientPoolManager(credential_carrier)

                    self._initialized = True
                    logger.info("Database delegate initialized with ResilientPoolManager (primary/backup/failover pools)")
                    return True

                except Exception as e:
                    logger.warning(f"ResilientPoolManager failed, falling back to simple pool: {e}")
                    # Fall through to simple pool

            # Fallback to simple connection pool
            self._config = self._load_database_config()

            # Import pool only when needed
            from psycopg2 import pool

            # Create simple connection pool (min 2, max 10 connections)
            self._connection_pool = pool.ThreadedConnectionPool(
                2,  # Min connections
                10,  # Max connections
                host=self._config.get('host', 'localhost'),
                port=self._config.get('port', 5432),
                database=self._config.get('database', 'rag_db'),
                user=self._config.get('username', 'rag_user'),
                password=self._config.get('password', 'rag_password'),
                sslmode=self._config.get('ssl_mode', 'prefer')
            )

            self._initialized = True
            logger.info("Database delegate initialized with simple ThreadedConnectionPool")
            return True

        except Exception as e:
            logger.warning(f"Database infrastructure not available: {e}")
            self._initialized = False
            return False

    def _load_database_config(self) -> Dict[str, Any]:
        """Load database configuration."""
        # Try multiple paths for settings
        settings_paths = [
            Path(__file__).parent.parent.parent / "admin" / "settings.yaml",
            Path("packages/tidyllm/admin/settings.yaml"),
            Path("tidyllm/admin/settings.yaml")
        ]

        for settings_path in settings_paths:
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('credentials', {}).get('postgresql', {})

        # Fallback configuration
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'rag_db',
            'username': 'rag_user',
            'password': 'rag_password',
            'ssl_mode': 'prefer'
        }

    def _get_connection(self):
        """Get database connection from pool."""
        if not self._initialized:
            if not self._initialize():
                raise Exception("Failed to initialize database connection pool")

        # Use ResilientPoolManager if available
        if self._pool_manager:
            return self._pool_manager.get_connection()

        # Fallback to simple pool
        return self._connection_pool.getconn()

    def _return_connection(self, conn):
        """Return connection to pool."""
        # Use ResilientPoolManager if available
        if self._pool_manager:
            self._pool_manager.return_connection(conn)
        elif self._connection_pool and conn:
            self._connection_pool.putconn(conn)

    def search_documents(self, query: str, domain: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents matching query.

        Args:
            query: Search query (SQL LIKE pattern)
            domain: Optional domain filter
            limit: Maximum results

        Returns:
            List of matching documents
        """
        if not self._initialize():
            return []

        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build query based on domain
            if domain:
                cursor.execute("""
                    SELECT
                        dc.chunk_id,
                        dc.chunk_text,
                        dc.doc_id,
                        dm.title,
                        dc.embedding_model,
                        dc.created_at
                    FROM document_chunks dc
                    LEFT JOIN document_metadata dm ON dc.doc_id = dm.doc_id
                    WHERE dc.chunk_text ILIKE %s
                    AND dm.doc_type = %s
                    AND LENGTH(dc.chunk_text) > 20
                    ORDER BY dc.created_at DESC
                    LIMIT %s
                """, (query, domain, limit))
            else:
                cursor.execute("""
                    SELECT
                        dc.chunk_id,
                        dc.chunk_text,
                        dc.doc_id,
                        dm.title,
                        dc.embedding_model,
                        dc.created_at
                    FROM document_chunks dc
                    LEFT JOIN document_metadata dm ON dc.doc_id = dm.doc_id
                    WHERE dc.chunk_text ILIKE %s
                    AND LENGTH(dc.chunk_text) > 20
                    ORDER BY dc.created_at DESC
                    LIMIT %s
                """, (query, limit))

            rows = cursor.fetchall()
            results = []

            for row in rows:
                results.append({
                    'chunk_id': row[0],
                    'content': row[1],
                    'doc_id': row[2],
                    'filename': row[3] or 'unknown',
                    'embedding_model': row[4] or 'unknown',
                    'created_at': row[5],
                    'similarity_score': 0.8  # Default score
                })

            return results

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)

    def create_collection(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new collection.

        Args:
            name: Collection name
            config: Collection configuration

        Returns:
            Result dictionary with success status
        """
        if not self._initialize():
            return {'success': False, 'error': 'Database not available'}

        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if collection exists
            cursor.execute("""
                SELECT id FROM yrsn_paper_collections
                WHERE collection_name = %s
            """, (name,))

            if cursor.fetchone():
                return {
                    'success': False,
                    'error': f'Collection {name} already exists'
                }

            # Create collection
            cursor.execute("""
                INSERT INTO yrsn_paper_collections
                (collection_name, description, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (
                name,
                config.get('description', f'Collection {name}'),
                datetime.now(),
                datetime.now()
            ))

            collection_id = cursor.fetchone()[0]
            conn.commit()

            return {
                'success': True,
                'collection_id': collection_id,
                'collection_name': name
            }

        except Exception as e:
            logger.error(f"Collection creation failed: {e}")
            if conn:
                conn.rollback()
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)

    def list_collections(self, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        List collections with optional filter.

        Args:
            filter: Optional filter criteria

        Returns:
            List of collections
        """
        if not self._initialize():
            return []

        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build query
            if filter and 'adapter_type' in filter:
                cursor.execute("""
                    SELECT
                        c.id,
                        c.collection_name,
                        c.description,
                        c.created_at,
                        COUNT(DISTINCT dm.doc_id) as doc_count
                    FROM yrsn_paper_collections c
                    LEFT JOIN document_metadata dm ON TRUE
                    WHERE c.description LIKE %s
                    GROUP BY c.id, c.collection_name, c.description, c.created_at
                    ORDER BY c.created_at DESC
                """, (f"%{filter['adapter_type']}%",))
            else:
                cursor.execute("""
                    SELECT
                        c.id,
                        c.collection_name,
                        c.description,
                        c.created_at,
                        COUNT(DISTINCT dm.doc_id) as doc_count
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

            return collections

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)

    def create_document(self, doc_id: str, title: str, doc_type: str,
                       content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Create a new document with chunks.

        Args:
            doc_id: Document ID
            title: Document title
            doc_type: Document type/domain
            content: Document content
            metadata: Optional metadata

        Returns:
            Success status
        """
        if not self._initialize():
            return False

        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check for existing document
            cursor.execute("""
                SELECT doc_id FROM document_metadata
                WHERE doc_id = %s
            """, (doc_id,))

            if cursor.fetchone():
                logger.info(f"Document {doc_id} already exists")
                return False

            # Create document metadata
            cursor.execute("""
                INSERT INTO document_metadata
                (doc_id, title, doc_type, status, last_processed)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                doc_id,
                title,
                doc_type,
                'processed',
                datetime.now()
            ))

            # Create chunks
            chunks = self._create_chunks(content)
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i:03d}"

                cursor.execute("""
                    INSERT INTO document_chunks
                    (doc_id, chunk_id, page_num, chunk_text,
                     embedding_model, token_estimate, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    doc_id,
                    chunk_id,
                    i + 1,
                    chunk_text,
                    metadata.get('embedding_model', 'default'),
                    len(chunk_text.split()),
                    datetime.now()
                ))

            conn.commit()
            logger.info(f"Created document {doc_id} with {len(chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Document creation failed: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)

    def _create_chunks(self, text: str, max_size: int = 1000) -> List[str]:
        """Create text chunks."""
        if len(text) <= max_size:
            return [text]

        # Split on paragraphs
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

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute arbitrary query (for advanced use).

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Query results as list of dicts
        """
        if not self._initialize():
            return []

        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Convert to list of dicts
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            return results

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)

    def is_available(self) -> bool:
        """Check if database is available."""
        return self._initialize()


class DatabaseDelegateFactory:
    """Factory for creating database delegates."""

    _instance = None

    @classmethod
    def get_delegate(cls) -> DatabaseDelegate:
        """Get singleton database delegate instance."""
        if cls._instance is None:
            cls._instance = DatabaseDelegate()
        return cls._instance


def get_database_delegate() -> DatabaseDelegate:
    """Get database delegate instance."""
    return DatabaseDelegateFactory.get_delegate()