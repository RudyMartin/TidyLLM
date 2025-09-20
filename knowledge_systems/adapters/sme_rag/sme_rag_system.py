"""
SME RAG System for v2 Boss Portal
=================================

Advanced Subject Matter Expert Retrieval-Augmented Generation system with:
- Document upload and S3 storage management
- Multiple embedding model support
- PGVector integration for semantic search
- Collection management for different standards
- Manual indexing and reindexing capabilities
- Chat with SME functionality
"""

import os
import json
import uuid
import boto3
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Use centralized document service for all document operations
try:
    from tidyllm.services import CentralizedDocumentService
    CENTRALIZED_DOC_SERVICE_AVAILABLE = True
except ImportError:
    CENTRALIZED_DOC_SERVICE_AVAILABLE = False
    print("WARNING: Centralized document service not available - using fallback processing")
import numpy as np

class EmbeddingModel(Enum):
    """Supported embedding models."""
    OPENAI_ADA_002 = "text-embedding-ada-002"
    SENTENCE_BERT_BASE = "all-MiniLM-L6-v2"
    SENTENCE_BERT_LARGE = "all-mpnet-base-v2"
    INSTRUCTOR_XL = "hkunlp/instructor-xl"
    BGE_LARGE = "BAAI/bge-large-en-v1.5"

class DocumentStatus(Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    ERROR = "error"
    REINDEXING = "reindexing"

@dataclass
class DocumentMetadata:
    """Document metadata structure."""
    doc_id: str
    filename: str
    original_filename: str
    file_size: int
    upload_date: datetime
    s3_bucket: str
    s3_key: str
    s3_prefix: str
    collection_name: str
    embedding_model: str
    status: DocumentStatus
    chunk_count: int = 0
    processing_time: float = 0.0
    error_message: str = ""
    tags: List[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class Collection:
    """Document collection structure."""
    collection_id: str
    name: str
    description: str
    created_date: datetime
    document_count: int
    total_chunks: int
    embedding_model: str
    s3_bucket: str
    s3_prefix: str
    tags: List[str] = None
    settings: Dict[str, Any] = None

class SMERAGSystem:
    """SME RAG System for document management and semantic search."""
    
    def __init__(self, 
                 pg_connection_string: str = None,
                 aws_access_key: str = None,
                 aws_secret_key: str = None,
                 aws_region: str = "us-east-1",
                 openai_api_key: str = None):
        """Initialize SME RAG System."""
        
        # Database connection
        self.pg_conn_string = pg_connection_string or os.getenv('POSTGRESQL_CONNECTION_STRING')
        if not self.pg_conn_string:
            # Get database connection from environment manager
            try:
                from config.environment_manager import get_db_config
                db_config = get_db_config()
                self.pg_conn_string = db_config.connection_url
            except ImportError:
                raise ValueError("No database connection string provided and environment_manager not available")
        
        # #future_fix: Convert to use enhanced service infrastructure
        self.engine = create_engine(self.pg_conn_string)
        
        # AWS S3 setup
        self.aws_access_key = aws_access_key or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = aws_secret_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = aws_region
        
        if self.aws_access_key and self.aws_secret_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
        else:
            self.s3_client = boto3.client('s3')  # Use IAM role if available
        
        # OpenAI setup
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

        # Initialize centralized document service for all document operations
        if CENTRALIZED_DOC_SERVICE_AVAILABLE:
            try:
                self.document_service = CentralizedDocumentService(auto_load_credentials=True)
                print("SUCCESS: Centralized document service initialized with USM credentials")
            except Exception as e:
                print(f"WARNING: Failed to initialize centralized document service: {e}")
                self.document_service = None
        else:
            self.document_service = None
        
        # Initialize database tables
        self._init_database()
        
        # Cache for embedding models
        self._embedding_models_cache = {}
    
    def _init_database(self):
        """Initialize database tables for SME RAG system."""
        with self.engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Collections table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sme_collections (
                    collection_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    document_count INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    embedding_model VARCHAR(100) NOT NULL,
                    s3_bucket VARCHAR(255),
                    s3_prefix VARCHAR(255),
                    tags TEXT[], -- PostgreSQL array
                    settings JSONB,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Documents table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sme_documents (
                    doc_id VARCHAR(36) PRIMARY KEY,
                    collection_id VARCHAR(36) REFERENCES sme_collections(collection_id) ON DELETE CASCADE,
                    filename VARCHAR(255) NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    file_size BIGINT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    s3_bucket VARCHAR(255),
                    s3_key VARCHAR(500),
                    s3_prefix VARCHAR(255),
                    embedding_model VARCHAR(100),
                    status VARCHAR(20) DEFAULT 'uploaded',
                    chunk_count INTEGER DEFAULT 0,
                    processing_time FLOAT DEFAULT 0.0,
                    error_message TEXT,
                    tags TEXT[],
                    metadata JSONB,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Document chunks table with vector embeddings
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sme_document_chunks (
                    chunk_id VARCHAR(36) PRIMARY KEY,
                    doc_id VARCHAR(36) REFERENCES sme_documents(doc_id) ON DELETE CASCADE,
                    collection_id VARCHAR(36) REFERENCES sme_collections(collection_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(1536), -- Default for OpenAI ada-002, will adjust based on model
                    chunk_metadata JSONB,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Create indexes for better performance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sme_chunks_collection 
                ON sme_document_chunks(collection_id);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sme_chunks_doc 
                ON sme_document_chunks(doc_id);
            """))
            
            conn.commit()
    
    def create_collection(self, 
                         name: str, 
                         description: str,
                         embedding_model: EmbeddingModel,
                         s3_bucket: str,
                         s3_prefix: str = "",
                         tags: List[str] = None,
                         settings: Dict[str, Any] = None) -> str:
        """Create a new document collection."""
        
        collection_id = str(uuid.uuid4())
        
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO sme_collections 
                (collection_id, name, description, embedding_model, s3_bucket, s3_prefix, tags, settings)
                VALUES (:collection_id, :name, :description, :embedding_model, :s3_bucket, :s3_prefix, :tags, :settings)
            """), {
                'collection_id': collection_id,
                'name': name,
                'description': description,
                'embedding_model': embedding_model.value,
                's3_bucket': s3_bucket,
                's3_prefix': s3_prefix,
                'tags': tags or [],
                'settings': json.dumps(settings or {})
            })
            conn.commit()
        
        return collection_id
    
    def get_collections(self) -> List[Collection]:
        """Get all collections."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM sme_collections ORDER BY created_date DESC
            """))
            
            collections = []
            for row in result:
                collections.append(Collection(
                    collection_id=row.collection_id,
                    name=row.name,
                    description=row.description,
                    created_date=row.created_date,
                    document_count=row.document_count,
                    total_chunks=row.total_chunks,
                    embedding_model=row.embedding_model,
                    s3_bucket=row.s3_bucket,
                    s3_prefix=row.s3_prefix,
                    tags=row.tags or [],
                    settings=row.settings if isinstance(row.settings, dict) else (json.loads(row.settings) if row.settings else {})
                ))
            
            return collections

    def get_all_collections_including_legacy(self) -> List[Collection]:
        """Get ALL collections from both sme_collections AND langchain_pg_embedding tables."""
        collections = []

        with self.engine.connect() as conn:
            # First get V2 collections from sme_collections
            try:
                result = conn.execute(text("""
                    SELECT * FROM sme_collections ORDER BY created_date DESC
                """))

                for row in result:
                    collections.append(Collection(
                        collection_id=row.collection_id,
                        name=row.name,
                        description=row.description,
                        created_date=row.created_date,
                        document_count=row.document_count,
                        total_chunks=row.total_chunks,
                        embedding_model=row.embedding_model,
                        s3_bucket=row.s3_bucket,
                        s3_prefix=row.s3_prefix,
                        tags=row.tags or [],
                        settings=row.settings if isinstance(row.settings, dict) else (json.loads(row.settings) if row.settings else {})
                    ))
            except Exception as e:
                print(f"Warning: Could not fetch sme_collections: {e}")

            # Then get legacy collections from langchain_pg_embedding
            try:
                result = conn.execute(text("""
                    SELECT
                        collection_name,
                        COUNT(*) as document_count,
                        MIN(created_at) as created_date
                    FROM langchain_pg_embedding
                    GROUP BY collection_name
                    ORDER BY MIN(created_at) DESC
                """))

                for row in result:
                    # Create Collection object for legacy collections
                    collections.append(Collection(
                        collection_id=f"legacy_{row.collection_name}",
                        name=f"Live {row.collection_name}",
                        description=f"Legacy collection with {row.document_count} documents",
                        created_date=row.created_date,
                        document_count=row.document_count,
                        total_chunks=row.document_count,
                        embedding_model="legacy",
                        s3_bucket=None,
                        s3_prefix=None,
                        tags=["legacy", "historical"],
                        settings={"legacy": True, "source": "langchain_pg_embedding"}
                    ))
            except Exception as e:
                print(f"Warning: Could not fetch langchain_pg_embedding collections: {e}")

            return collections

    def upload_document(self,
                       file_path: str,
                       collection_id: str,
                       original_filename: str = None,
                       tags: List[str] = None,
                       metadata: Dict[str, Any] = None) -> str:
        """Upload a document to S3 and add to collection."""
        
        # Get collection info
        collection = self.get_collection(collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")
        
        # Generate document ID and S3 key
        doc_id = str(uuid.uuid4())
        file_path_obj = Path(file_path)
        filename = f"{doc_id}_{file_path_obj.name}"
        s3_key = f"{collection.s3_prefix}/{filename}" if collection.s3_prefix else filename
        
        # Upload to S3
        try:
            file_size = file_path_obj.stat().st_size
            self.s3_client.upload_file(
                file_path, 
                collection.s3_bucket, 
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'doc_id': doc_id,
                        'collection_id': collection_id,
                        'original_filename': original_filename or file_path_obj.name,
                        'upload_date': datetime.utcnow().isoformat()
                    }
                }
            )
        except Exception as e:
            raise Exception(f"Failed to upload to S3: {str(e)}")
        
        # Add document record to database
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO sme_documents 
                (doc_id, collection_id, filename, original_filename, file_size, 
                 s3_bucket, s3_key, s3_prefix, embedding_model, tags, metadata)
                VALUES (:doc_id, :collection_id, :filename, :original_filename, :file_size,
                        :s3_bucket, :s3_key, :s3_prefix, :embedding_model, :tags, :metadata)
            """), {
                'doc_id': doc_id,
                'collection_id': collection_id,
                'filename': filename,
                'original_filename': original_filename or file_path_obj.name,
                'file_size': file_size,
                's3_bucket': collection.s3_bucket,
                's3_key': s3_key,
                's3_prefix': collection.s3_prefix,
                'embedding_model': collection.embedding_model,
                'tags': tags or [],
                'metadata': json.dumps(metadata or {})
            })
            conn.commit()
        
        return doc_id
    
    def get_collection(self, collection_id: str) -> Optional[Collection]:
        """Get collection by ID."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM sme_collections WHERE collection_id = :collection_id
            """), {'collection_id': collection_id})
            
            row = result.fetchone()
            if not row:
                return None
            
            return Collection(
                collection_id=row.collection_id,
                name=row.name,
                description=row.description,
                created_date=row.created_date,
                document_count=row.document_count,
                total_chunks=row.total_chunks,
                embedding_model=row.embedding_model,
                s3_bucket=row.s3_bucket,
                s3_prefix=row.s3_prefix,
                tags=row.tags or [],
                settings=json.loads(row.settings) if row.settings else {}
            )
    
    def get_embedding_model(self, model_name: str):
        """Get or load embedding model."""
        if model_name in self._embedding_models_cache:
            return self._embedding_models_cache[model_name]
        
        if model_name == EmbeddingModel.OPENAI_ADA_002.value:
            # OpenAI embeddings are accessed via API
            return None  # Will use API calls
        else:
            # Load sentence transformer model
            try:
                model = SentenceTransformer(model_name)
                self._embedding_models_cache[model_name] = model
                return model
            except Exception as e:
                raise Exception(f"Failed to load model {model_name}: {str(e)}")
    
    def get_embedding(self, text: str, model_name: str) -> List[float]:
        """Get embedding for text using specified model."""
        if model_name == EmbeddingModel.OPENAI_ADA_002.value:
            # Use OpenAI API
            if not self.openai_api_key:
                raise Exception("OpenAI API key required for OpenAI embeddings")
            
            response = openai.Embedding.create(
                input=text,
                model=model_name
            )
            return response['data'][0]['embedding']
        else:
            # Use sentence transformer
            model = self.get_embedding_model(model_name)
            if model is None:
                raise Exception(f"Could not load model {model_name}")
            
            embedding = model.encode([text])[0]
            return embedding.tolist()
    
    def process_document(self, doc_id: str) -> Dict[str, Any]:
        """Process a document: extract text, create chunks, generate embeddings."""
        
        # Update status to processing
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE sme_documents SET status = 'processing', updated_date = CURRENT_TIMESTAMP
                WHERE doc_id = :doc_id
            """), {'doc_id': doc_id})
            conn.commit()
        
        start_time = datetime.utcnow()
        
        try:
            # Get document info
            doc = self.get_document(doc_id)
            if not doc:
                raise Exception(f"Document {doc_id} not found")
            
            # Download from S3 to temp file
            temp_file = f"/tmp/{doc.filename}"
            self.s3_client.download_file(doc.s3_bucket, doc.s3_key, temp_file)
            
            # Extract text using centralized document service (replaces all scattered imports)
            if self.document_service:
                try:
                    result = self.document_service.process_document(temp_file)
                    full_text = result.get('text', '')
                    processor_used = result.get('processor_used', 'unknown')
                    print(f"SUCCESS: Used centralized document service ({processor_used}) for {temp_file}")
                except Exception as e:
                    print(f"Centralized document service failed: {e}, using simple fallback")
                    with open(temp_file, 'r', encoding='utf-8', errors='ignore') as f:
                        full_text = f.read()
            else:
                # Simple fallback when centralized service unavailable
                print(f"WARNING: Using simple fallback processing for {temp_file}")
                with open(temp_file, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text = f.read()
            
            # Split into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            # Generate embeddings and store chunks
            processed_chunks = 0
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self.get_embedding(chunk, doc.embedding_model)
                    
                    # Store chunk with embedding
                    chunk_id = str(uuid.uuid4())
                    with self.engine.connect() as conn:
                        conn.execute(text("""
                            INSERT INTO sme_document_chunks 
                            (chunk_id, doc_id, collection_id, chunk_index, content, embedding, chunk_metadata)
                            VALUES (:chunk_id, :doc_id, :collection_id, :chunk_index, :content, :embedding, :metadata)
                        """), {
                            'chunk_id': chunk_id,
                            'doc_id': doc_id,
                            'collection_id': doc.collection_id,
                            'chunk_index': i,
                            'content': chunk,
                            'embedding': str(embedding),  # Convert to string for PostgreSQL
                            'metadata': json.dumps({
                                'chunk_length': len(chunk),
                                'page_numbers': [page.metadata.get('page', i) for page in pages if chunk in page.page_content][:3]
                            })
                        })
                        conn.commit()
                    
                    processed_chunks += 1
                except Exception as e:
                    print(f"Error processing chunk {i}: {str(e)}")
                    continue
            
            # Update document status
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE sme_documents 
                    SET status = 'indexed', chunk_count = :chunk_count, 
                        processing_time = :processing_time, updated_date = CURRENT_TIMESTAMP
                    WHERE doc_id = :doc_id
                """), {
                    'doc_id': doc_id,
                    'chunk_count': processed_chunks,
                    'processing_time': processing_time
                })
                
                # Update collection stats
                conn.execute(text("""
                    UPDATE sme_collections 
                    SET document_count = (SELECT COUNT(*) FROM sme_documents WHERE collection_id = :collection_id),
                        total_chunks = (SELECT COUNT(*) FROM sme_document_chunks WHERE collection_id = :collection_id),
                        updated_date = CURRENT_TIMESTAMP
                    WHERE collection_id = :collection_id
                """), {'collection_id': doc.collection_id})
                
                conn.commit()
            
            # Clean up temp file
            os.unlink(temp_file)
            
            return {
                'status': 'success',
                'chunks_processed': processed_chunks,
                'processing_time': processing_time,
                'total_chunks': len(chunks)
            }
            
        except Exception as e:
            # Update status to error
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE sme_documents 
                    SET status = 'error', error_message = :error_message, updated_date = CURRENT_TIMESTAMP
                    WHERE doc_id = :doc_id
                """), {
                    'doc_id': doc_id,
                    'error_message': str(e)
                })
                conn.commit()
            
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.utcnow() - start_time).total_seconds()
            }
    
    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get document by ID."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM sme_documents WHERE doc_id = :doc_id
            """), {'doc_id': doc_id})
            
            row = result.fetchone()
            if not row:
                return None
            
            return DocumentMetadata(
                doc_id=row.doc_id,
                filename=row.filename,
                original_filename=row.original_filename,
                file_size=row.file_size,
                upload_date=row.upload_date,
                s3_bucket=row.s3_bucket,
                s3_key=row.s3_key,
                s3_prefix=row.s3_prefix,
                collection_name="",  # Will be filled separately if needed
                embedding_model=row.embedding_model,
                status=DocumentStatus(row.status),
                chunk_count=row.chunk_count or 0,
                processing_time=row.processing_time or 0.0,
                error_message=row.error_message or "",
                tags=row.tags or [],
                metadata=json.loads(row.metadata) if row.metadata else {}
            )
    
    def semantic_search(self, 
                       query: str, 
                       collection_id: str,
                       limit: int = 5,
                       similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search within a collection."""
        
        # Get collection info
        collection = self.get_collection(collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")
        
        # Generate query embedding
        query_embedding = self.get_embedding(query, collection.embedding_model)
        
        # Perform vector similarity search
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    c.chunk_id,
                    c.content,
                    c.chunk_metadata,
                    c.chunk_index,
                    d.original_filename,
                    d.doc_id,
                    (c.embedding <-> :query_embedding::vector) as distance
                FROM sme_document_chunks c
                JOIN sme_documents d ON c.doc_id = d.doc_id
                WHERE c.collection_id = :collection_id
                AND (c.embedding <-> :query_embedding::vector) < :threshold
                ORDER BY distance
                LIMIT :limit
            """), {
                'collection_id': collection_id,
                'query_embedding': str(query_embedding),
                'threshold': 1.0 - similarity_threshold,  # Convert similarity to distance
                'limit': limit
            })
            
            results = []
            for row in result:
                results.append({
                    'chunk_id': row.chunk_id,
                    'content': row.content,
                    'metadata': json.loads(row.chunk_metadata) if row.chunk_metadata else {},
                    'chunk_index': row.chunk_index,
                    'filename': row.original_filename,
                    'doc_id': row.doc_id,
                    'similarity_score': 1.0 - row.distance,
                    'distance': row.distance
                })
            
            return results
    
    def reindex_collection(self, collection_id: str) -> Dict[str, Any]:
        """Reindex all documents in a collection."""
        
        # Get all documents in collection
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT doc_id FROM sme_documents 
                WHERE collection_id = :collection_id AND status IN ('indexed', 'error')
            """), {'collection_id': collection_id})
            
            doc_ids = [row.doc_id for row in result]
        
        # Clear existing chunks
        with self.engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM sme_document_chunks WHERE collection_id = :collection_id
            """), {'collection_id': collection_id})
            
            # Reset document status
            conn.execute(text("""
                UPDATE sme_documents 
                SET status = 'uploaded', chunk_count = 0, error_message = '', updated_date = CURRENT_TIMESTAMP
                WHERE collection_id = :collection_id
            """), {'collection_id': collection_id})
            
            conn.commit()
        
        # Reprocess all documents
        results = []
        for doc_id in doc_ids:
            result = self.process_document(doc_id)
            results.append({
                'doc_id': doc_id,
                'result': result
            })
        
        return {
            'status': 'completed',
            'total_documents': len(doc_ids),
            'results': results
        }
    
    def get_collection_documents(self, collection_id: str) -> List[DocumentMetadata]:
        """Get all documents in a collection."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT d.*, c.name as collection_name
                FROM sme_documents d
                JOIN sme_collections c ON d.collection_id = c.collection_id
                WHERE d.collection_id = :collection_id
                ORDER BY d.upload_date DESC
            """), {'collection_id': collection_id})
            
            documents = []
            for row in result:
                doc = DocumentMetadata(
                    doc_id=row.doc_id,
                    filename=row.filename,
                    original_filename=row.original_filename,
                    file_size=row.file_size,
                    upload_date=row.upload_date,
                    s3_bucket=row.s3_bucket,
                    s3_key=row.s3_key,
                    s3_prefix=row.s3_prefix,
                    collection_name=row.collection_name,
                    embedding_model=row.embedding_model,
                    status=DocumentStatus(row.status),
                    chunk_count=row.chunk_count or 0,
                    processing_time=row.processing_time or 0.0,
                    error_message=row.error_message or "",
                    tags=row.tags or [],
                    metadata=json.loads(row.metadata) if row.metadata else {}
                )
                documents.append(doc)
            
            return documents