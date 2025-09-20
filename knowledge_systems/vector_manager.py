"""
Unified Vector Manager
======================

Consolidates all vector database functionality from scattered implementations across:
- tests/vectorqa_setup.py
- src/tidyllm/vectorqa.py  
- tidyllm-cross-integration/document_to_vectorqa_pipeline.py
- Various RAG implementations

Provides single unified interface for all vector operations in TidyLLM.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import random
import math

# Import embedding standardization
from .embedding_config import EmbeddingStandardizer, standardize_embedding, get_target_dimension

# Use UnifiedSessionManager for database connections
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from tidyllm.infrastructure.session.unified import UnifiedSessionManager
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    UNIFIED_SESSION_AVAILABLE = False

try:
    # #future_fix: Convert to use enhanced service infrastructure
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger("vector_manager")

@dataclass
class VectorConfig:
    """Vector database configuration"""
    host: str = "vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com"
    port: int = 5432
    database: str = "vectorqa"
    user: str = "vectorqa_user"
    password: Optional[str] = None
    vector_dimension: int = 1024  # Standardized dimension for all embeddings
    embedding_model: str = "text-embedding-3-small"

@dataclass
class Document:
    """Document representation"""
    id: Optional[str] = None
    title: Optional[str] = None
    content: str = ""
    source: Optional[str] = None
    doc_type: str = "text"
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

@dataclass
class DocumentChunk:
    """Document chunk with embedding"""
    id: Optional[str] = None
    document_id: str = ""
    content: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SearchResult:
    """Vector search result"""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class VectorManager:
    """Unified vector database operations manager"""
    
    def __init__(self, config: VectorConfig = None, auto_connect: bool = True):
        self.config = config or VectorConfig()
        if not self.config.password:
            # Try to get password from environment, fail if not available
            self.config.password = os.environ.get('POSTGRES_PASSWORD')
            if not self.config.password:
                try:
                    from config.environment_manager import get_db_config
                    db_config = get_db_config()
                    self.config.password = db_config.password
                except ImportError:
                    raise ValueError("No database password provided. Set POSTGRES_PASSWORD environment variable.")
        
        # Initialize embedding standardizer
        self.embedding_standardizer = EmbeddingStandardizer(
            target_dimension=self.config.vector_dimension
        )
        
        # Use UnifiedSessionManager for database connections
        if UNIFIED_SESSION_AVAILABLE:
            self.session_mgr = UnifiedSessionManager()
            self.connection = None  # Legacy fallback connection
        else:
            self.session_mgr = None
            self.connection = None
            if not POSTGRES_AVAILABLE:
                logger.warning("Neither UnifiedSessionManager nor psycopg2 available")
        
        self._embedding_cache = {}
        
        if auto_connect:
            self.connect()
    
    def connect(self) -> Dict[str, Any]:
        """Connect to vector database"""
        # Use UnifiedSessionManager if available
        if self.session_mgr:
            try:
                # Test connection by trying to get PostgreSQL connection
                test_conn = self.session_mgr.get_postgres_connection()
                if test_conn:
                    test_conn.close()
                    return {
                        "success": True,
                        "message": "Connected via UnifiedSessionManager",
                        "connection_type": "UnifiedSessionManager"
                    }
                else:
                    return {
                        "success": False,
                        "error": "UnifiedSessionManager connection failed"
                    }
            except Exception as e:
                logger.error(f"UnifiedSessionManager connection failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        # Fallback to direct connection
        if not POSTGRES_AVAILABLE:
            return {
                "success": False,
                "error": "Neither UnifiedSessionManager nor PostgreSQL driver available"
            }
        
        try:
    # #future_fix: Convert to use enhanced service infrastructure
            self.connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            self.connection.autocommit = True
            
            return {
                "success": True,
                "message": "Connected to vector database (fallback)",
                "host": self.config.host,
                "database": self.config.database,
                "connection_type": "direct_fallback"
            }
        except Exception as e:
            logger.error(f"Failed to connect to vector database: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def check_extensions(self) -> Dict[str, Any]:
        """Check if required extensions are available"""
        try:
            # Use UnifiedSessionManager if available
            if self.session_mgr:
                # Check for pgvector extension
                pgvector_result = self.session_mgr.execute_postgres_query("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    ) as pgvector_available;
                """)
                
                # Check available extensions
                extensions_result = self.session_mgr.execute_postgres_query("""
                    SELECT name FROM pg_available_extensions 
                    WHERE name IN ('vector', 'pgvector') 
                    ORDER BY name;
                """)
                
                if pgvector_result and extensions_result:
                    return {
                        "success": True,
                        "pgvector_installed": pgvector_result[0]['pgvector_available'],
                        "available_extensions": [row['name'] for row in extensions_result]
                    }
                else:
                    return {"success": False, "error": "Query failed via UnifiedSessionManager"}
            
            # Fallback to direct connection
            elif self.connection:
                with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Check for pgvector extension
                    cursor.execute("""
                        SELECT EXISTS(
                            SELECT 1 FROM pg_extension WHERE extname = 'vector'
                        ) as pgvector_available;
                    """)
                    pgvector_result = cursor.fetchone()
                    
                    # Check available extensions
                    cursor.execute("""
                        SELECT name FROM pg_available_extensions 
                        WHERE name IN ('vector', 'pgvector') 
                        ORDER BY name;
                    """)
                    available_extensions = [row['name'] for row in cursor.fetchall()]
                    
                    return {
                        "success": True,
                        "pgvector_installed": pgvector_result['pgvector_available'],
                        "available_extensions": available_extensions
                    }
            else:
                return {"success": False, "error": "Not connected to database"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def setup_database(self) -> Dict[str, Any]:
        """Setup vector database tables and schema"""
        if not self.connection:
            return {"success": False, "error": "Not connected to database"}
        
        try:
            with self.connection.cursor() as cursor:
                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        title VARCHAR(500),
                        content TEXT,
                        source VARCHAR(500),
                        doc_type VARCHAR(100),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create chunks table with vector support
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        chunk_index INTEGER,
                        start_char INTEGER,
                        end_char INTEGER,
                        embedding vector({self.config.vector_dimension}),
                        model_used VARCHAR(100),
                        native_dimension INTEGER,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                    ON document_chunks(document_id);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine 
                    ON document_chunks USING ivfflat (embedding vector_cosine_ops);
                """)
                
                return {
                    "success": True,
                    "message": "Vector database schema created successfully"
                }
                
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            return {"success": False, "error": str(e)}
    
    def add_document(self, document: Document) -> Dict[str, Any]:
        """Add document to vector database"""
        if not self.connection:
            return {"success": False, "error": "Not connected to database"}
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    INSERT INTO documents (title, content, source, doc_type, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, created_at;
                """, (
                    document.title,
                    document.content,
                    document.source,
                    document.doc_type,
                    json.dumps(document.metadata or {})
                ))
                
                result = cursor.fetchone()
                
                return {
                    "success": True,
                    "document_id": str(result['id']),
                    "created_at": result['created_at'].isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return {"success": False, "error": str(e)}
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Split text into chunks for embedding"""
        if TIKTOKEN_AVAILABLE:
            # Use tiktoken for better tokenization
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            
            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                start = end - overlap
            
            return chunks
        else:
            # Fallback to character-based chunking
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size * 4, len(text))  # Rough char estimate
                chunk_text = text[start:end]
                chunks.append(chunk_text)
                start = end - overlap * 4
            
            return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        # Check cache first
        text_hash = hash(text)
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        # Mock embedding generation (replace with actual embedding service)
        # This would typically call OpenAI API or local embedding model
        embedding = [random.gauss(0, 1) for _ in range(self.config.vector_dimension)]
        
        # Normalize embedding using TLM
        try:
            import tlm
            embedding = tlm.l2_normalize([embedding])[0]  # l2_normalize expects 2D, returns 2D
        except ImportError:
            # Fallback to manual normalization if TLM not available
            norm = math.sqrt(sum(x*x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]
        
        # Cache result
        self._embedding_cache[text_hash] = embedding
        
        return embedding
    
    def add_document_chunks(self, document_id: str, content: str, chunk_size: int = 512) -> Dict[str, Any]:
        """Add document chunks with embeddings"""
        if not self.connection:
            return {"success": False, "error": "Not connected to database"}
        
        try:
            chunks = self.chunk_text(content, chunk_size)
            chunk_results = []
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                for i, chunk_text in enumerate(chunks):
                    # Generate embedding
                    embedding = self.generate_embedding(chunk_text)
                    
                    # Calculate character positions
                    start_char = i * chunk_size
                    end_char = min(start_char + len(chunk_text), len(content))
                    
                    # Insert chunk
                    cursor.execute("""
                        INSERT INTO document_chunks 
                        (document_id, content, chunk_index, start_char, end_char, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (
                        document_id,
                        chunk_text,
                        i,
                        start_char,
                        end_char,
                        json.dumps(embedding)
                    ))
                    
                    chunk_id = cursor.fetchone()['id']
                    chunk_results.append({
                        "chunk_id": str(chunk_id),
                        "chunk_index": i,
                        "content_length": len(chunk_text)
                    })
            
            return {
                "success": True,
                "chunks_added": len(chunks),
                "chunk_details": chunk_results
            }
            
        except Exception as e:
            logger.error(f"Failed to add document chunks: {e}")
            return {"success": False, "error": str(e)}
    
    def search_similar(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[SearchResult]:
        """Search for similar document chunks"""
        if not self.connection:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Perform vector similarity search
                cursor.execute("""
                    SELECT 
                        dc.id as chunk_id,
                        dc.document_id,
                        dc.content,
                        d.title,
                        d.source,
                        dc.metadata,
                        1 - (dc.embedding <=> %s::vector) as similarity_score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE 1 - (dc.embedding <=> %s::vector) > %s
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT %s;
                """, (
                    json.dumps(query_embedding),
                    json.dumps(query_embedding),
                    similarity_threshold,
                    json.dumps(query_embedding),
                    top_k
                ))
                
                results = []
                for row in cursor.fetchall():
                    result = SearchResult(
                        document_id=str(row['document_id']),
                        chunk_id=str(row['chunk_id']),
                        content=row['content'],
                        score=float(row['similarity_score']),
                        metadata={
                            "document_title": row['title'],
                            "document_source": row['source'],
                            "chunk_metadata": row['metadata']
                        }
                    )
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        if not self.connection:
            return None
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM documents WHERE id = %s;
                """, (document_id,))
                
                row = cursor.fetchone()
                if row:
                    return Document(
                        id=str(row['id']),
                        title=row['title'],
                        content=row['content'],
                        source=row['source'],
                        doc_type=row['doc_type'],
                        metadata=row['metadata'],
                        created_at=row['created_at']
                    )
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
        
        return None
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List documents in database"""
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        id, title, source, doc_type, 
                        LENGTH(content) as content_length,
                        created_at, updated_at
                    FROM documents 
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s;
                """, (limit, offset))
                
                documents = []
                for row in cursor.fetchall():
                    documents.append({
                        "id": str(row['id']),
                        "title": row['title'],
                        "source": row['source'],
                        "doc_type": row['doc_type'],
                        "content_length": row['content_length'],
                        "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                        "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks"""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM documents WHERE id = %s;
                """, (document_id,))
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.connection:
            return {"success": False, "error": "Not connected to database"}
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Count documents and chunks
                cursor.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM documents) as document_count,
                        (SELECT COUNT(*) FROM document_chunks) as chunk_count,
                        (SELECT AVG(LENGTH(content)) FROM documents) as avg_document_length,
                        (SELECT AVG(LENGTH(content)) FROM document_chunks) as avg_chunk_length;
                """)
                
                stats = cursor.fetchone()
                
                return {
                    "success": True,
                    "document_count": stats['document_count'],
                    "chunk_count": stats['chunk_count'],
                    "avg_document_length": float(stats['avg_document_length']) if stats['avg_document_length'] else 0,
                    "avg_chunk_length": float(stats['avg_chunk_length']) if stats['avg_chunk_length'] else 0,
                    "vector_dimension": self.config.vector_dimension
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def standardize_and_store_embedding(self, embedding: List[float], model_key: str, 
                                      chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize embedding to target dimension and store in database
        
        Args:
            embedding: Raw embedding from model
            model_key: Key identifying the model used (e.g., "titan_v2_256") 
            chunk_data: Data for the chunk (content, document_id, etc.)
            
        Returns:
            Dict with success status and chunk_id
        """
        if not self.connection:
            return {"success": False, "error": "Not connected to database"}
            
        try:
            # Get model info
            model_info = self.embedding_standardizer.get_model_info(model_key)
            if not model_info:
                return {
                    "success": False,
                    "error": f"Unknown model: {model_key}"
                }
            
            # Standardize embedding to 1024 dimensions
            standardized_embedding = self.embedding_standardizer.standardize(embedding, model_key)
            
            # Store in database
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO document_chunks (
                        document_id, content, chunk_index, start_char, end_char,
                        embedding, model_used, native_dimension, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    chunk_data.get("document_id"),
                    chunk_data.get("content"),
                    chunk_data.get("chunk_index", 0),
                    chunk_data.get("start_char", 0),
                    chunk_data.get("end_char", 0),
                    standardized_embedding,
                    model_key,
                    model_info["native_dimension"],
                    json.dumps(chunk_data.get("metadata", {}))
                ))
                
                result = cursor.fetchone()
                chunk_id = result[0] if isinstance(result, tuple) else result['id']
                
            return {
                "success": True,
                "chunk_id": str(chunk_id),
                "standardized_dimension": len(standardized_embedding),
                "model_info": model_info
            }
            
        except Exception as e:
            logger.error(f"Failed to standardize and store embedding: {e}")
            return {"success": False, "error": str(e)}

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None

# Global instance for easy access
_vector_manager_instance = None

def get_vector_manager(config: VectorConfig = None) -> VectorManager:
    """Get global vector manager instance"""
    global _vector_manager_instance
    
    if _vector_manager_instance is None or config is not None:
        _vector_manager_instance = VectorManager(config)
    
    return _vector_manager_instance

def reset_vector_manager():
    """Reset global vector manager instance"""
    global _vector_manager_instance
    if _vector_manager_instance:
        _vector_manager_instance.close()
    _vector_manager_instance = None