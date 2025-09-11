"""
Knowledge Sources - Data Source Implementations
==============================================

Implements various knowledge sources for the Knowledge Resource Server.
"""

import json
import logging
import hashlib
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("knowledge_sources")


class KnowledgeSource(ABC):
    """Abstract base class for knowledge sources."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the knowledge source."""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get total number of documents in this source."""
        pass
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for documents matching query."""
        pass
    
    @abstractmethod
    def retrieve_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        pass
    
    @abstractmethod
    def retrieve_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents matching criteria."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this source."""
        return {
            "type": self.__class__.__name__,
            "initialized": hasattr(self, '_initialized') and self._initialized,
            "document_count": self.get_document_count() if hasattr(self, '_initialized') else 0
        }


class S3KnowledgeSource(KnowledgeSource):
    """Knowledge source backed by AWS S3 storage with real S3 integration."""
    
    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1"):
        """
        Initialize S3 knowledge source.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix (folder path)
            region: AWS region
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/' if prefix and not prefix.endswith('/') else prefix
        self.region = region
        self._initialized = False
        self._documents = {}
        self._s3_client = None
        self._session_manager = None
        
    def initialize(self) -> None:
        """Initialize real S3 connection and load document index."""
        try:
            # ALWAYS use global UnifiedSessionManager instance (universal architecture)
            try:
                from tidyllm.infrastructure.session.unified import get_global_session_manager
                self._session_manager = get_global_session_manager()
            except ImportError as ie:
                # Fallback for absolute import issues
                try:
                    from ...infrastructure.session.unified import get_global_session_manager
                    self._session_manager = get_global_session_manager()
                except ImportError:
                    raise ImportError(f"Cannot import get_global_session_manager: {ie}")
            self._s3_client = self._session_manager.get_s3_client()
            logger.info("[OK] S3 client initialized via UnifiedSessionManager")
            
            # List and load documents from S3
            self._load_documents_from_s3()
            
            self._initialized = True
            logger.info(f"[OK] Real S3 knowledge source initialized: s3://{self.bucket}/{self.prefix} ({len(self._documents)} documents)")
            
        except ImportError as e:
            logger.warning(f"UnifiedSessionManager not available: {e}")
            self._fallback_to_mock("UnifiedSessionManager not available")
            
        except Exception as e:
            # Don't try direct boto3 - if UnifiedSessionManager fails, use mock data
            logger.warning(f"S3 connection failed: {e}")
            self._fallback_to_mock(str(e))
    
    def _fallback_to_mock(self, reason: str) -> None:
        """Fallback to mock data with clear messaging."""
        logger.info(f"Using mock S3 data for development (reason: {reason})")
        try:
            self._load_mock_documents()
            self._initialized = True
            logger.info(f"[OK] S3 source initialized with mock data ({len(self._documents)} documents)")
        except Exception as mock_error:
            logger.error(f"Failed to load mock data: {mock_error}")
            raise
    
    def _load_documents_from_s3(self) -> None:
        """Load real documents from S3."""
        try:
            paginator = self._s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
            
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):  # Skip directories
                        continue
                        
                    # Generate document metadata
                    doc_id = f"s3_{hashlib.md5(key.encode()).hexdigest()[:8]}"
                    filename = key.split('/')[-1]
                    
                    # Read document content (limit to text files for now)
                    content = self._read_s3_content(key)
                    
                    self._documents[key] = {
                        "id": doc_id,
                        "title": filename,
                        "content": content,
                        "metadata": {
                            "bucket": self.bucket,
                            "key": key,
                            "size": obj['Size'],
                            "last_modified": obj['LastModified'].isoformat(),
                            "etag": obj['ETag'].strip('"')
                        },
                        "source_uri": f"s3://{self.bucket}/{key}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to load documents from S3: {e}")
            raise
    
    def _read_s3_content(self, key: str) -> str:
        """Read content from S3 object."""
        try:
            # Only read text-based files to avoid binary content
            if any(key.lower().endswith(ext) for ext in ['.txt', '.md', '.json', '.csv', '.log']):
                response = self._s3_client.get_object(Bucket=self.bucket, Key=key)
                content = response['Body'].read().decode('utf-8', errors='ignore')
                # Limit content size for performance
                return content[:10000] if len(content) > 10000 else content
            else:
                # For binary files, return metadata description
                return f"Binary file: {key.split('/')[-1]}"
        except Exception as e:
            logger.warning(f"Failed to read S3 object {key}: {e}")
            return f"Content unavailable for {key}"
    
    def _load_mock_documents(self) -> None:
        """Fallback mock documents for development."""
        self._documents = {
            f"s3://{self.bucket}/{self.prefix}legal_contract_001.pdf": {
                "id": "s3_legal1",
                "title": "Legal Contract 001 - Service Agreement",
                "content": "This service agreement contains termination clauses in section 8. The contract may be terminated by either party with 30 days written notice. Termination procedures include data return and confidentiality obligations.",
                "metadata": {
                    "bucket": self.bucket,
                    "key": f"{self.prefix}legal_contract_001.pdf",
                    "size": 15420,
                    "last_modified": "2024-09-01T10:00:00Z",
                    "document_type": "legal_contract"
                },
                "source_uri": f"s3://{self.bucket}/{self.prefix}legal_contract_001.pdf"
            },
            f"s3://{self.bucket}/{self.prefix}technical_spec_v2.md": {
                "id": "s3_tech1",
                "title": "Technical Specification v2.0",
                "content": "Technical specifications for the document processing system. Includes API endpoints, data formats, and integration requirements. The system supports contract analysis and automated review workflows.",
                "metadata": {
                    "bucket": self.bucket,
                    "key": f"{self.prefix}technical_spec_v2.md",
                    "size": 8950,
                    "last_modified": "2024-09-05T14:30:00Z",
                    "document_type": "technical_documentation"
                },
                "source_uri": f"s3://{self.bucket}/{self.prefix}technical_spec_v2.md"
            }
        }
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self._documents) if self._initialized else 0
    
    def search(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search S3 documents with enhanced semantic matching."""
        if not self._initialized:
            return []
        
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for doc in self._documents.values():
            score = self._calculate_similarity_score(query, query_words, doc)
            
            if score >= similarity_threshold:
                result = doc.copy()
                result["similarity_score"] = score
                results.append(result)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:max_results]
    
    def _calculate_similarity_score(self, query: str, query_words: set, doc: Dict[str, Any]) -> float:
        """Calculate enhanced similarity score using multiple factors."""
        query_lower = query.lower()
        content_lower = doc["content"].lower()
        title_lower = doc["title"].lower()
        
        score = 0.0
        
        # Exact phrase matching (highest weight)
        if query_lower in title_lower:
            score += 0.9
        if query_lower in content_lower:
            score += 0.8
            
        # Word overlap scoring
        title_words = set(title_lower.split())
        content_words = set(content_lower.split())
        
        # Title word matches (high weight)
        title_overlap = len(query_words.intersection(title_words))
        if title_overlap > 0:
            score += 0.7 * (title_overlap / len(query_words))
        
        # Content word matches (medium weight)
        content_overlap = len(query_words.intersection(content_words))
        if content_overlap > 0:
            score += 0.5 * (content_overlap / len(query_words))
            
        # Metadata matching (document type, tags, etc.)
        metadata_text = " ".join(str(v).lower() for v in doc.get("metadata", {}).values())
        if query_lower in metadata_text:
            score += 0.4
            
        # Boost for legal/contract terms if searching legal content
        legal_terms = {"contract", "agreement", "termination", "clause", "legal", "compliance"}
        if query_words.intersection(legal_terms) and any(term in content_lower for term in legal_terms):
            score += 0.3
            
        return min(score, 1.0)  # Cap at 1.0
    
    def retrieve_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        if not self._initialized:
            return None
        
        for doc in self._documents.values():
            if doc["id"] == document_id:
                return doc
        return None
    
    def retrieve_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents by criteria."""
        if not self._initialized:
            return []
        
        results = []
        for doc in self._documents.values():
            match = True
            
            # Check metadata criteria
            for key, value in criteria.items():
                if key in doc["metadata"]:
                    if doc["metadata"][key] != value:
                        match = False
                        break
                elif key in doc:
                    if doc[key] != value:
                        match = False
                        break
            
            if match:
                results.append(doc)
        
        return results


class LocalKnowledgeSource(KnowledgeSource):
    """Knowledge source backed by local file system."""
    
    def __init__(self, directory: str, file_patterns: List[str] = None):
        """
        Initialize local knowledge source.
        
        Args:
            directory: Local directory path
            file_patterns: File patterns to include (e.g., ['*.pdf', '*.txt'])
        """
        self.directory = Path(directory)
        self.file_patterns = file_patterns or ["*.txt", "*.md", "*.pdf"]
        self._initialized = False
        self._documents = {}
    
    def initialize(self) -> None:
        """Initialize local file system source."""
        try:
            if not self.directory.exists():
                raise ValueError(f"Directory does not exist: {self.directory}")
            
            # Scan directory for matching files
            doc_count = 0
            for pattern in self.file_patterns:
                for file_path in self.directory.glob(pattern):
                    if file_path.is_file():
                        doc_id = f"local_{file_path.stem}"
                        
                        # Read file content (simplified)
                        try:
                            content = file_path.read_text(encoding='utf-8')
                        except:
                            content = f"Content from {file_path.name}"
                        
                        self._documents[str(file_path)] = {
                            "id": doc_id,
                            "title": file_path.name,
                            "content": content,
                            "metadata": {
                                "file_path": str(file_path),
                                "file_size": file_path.stat().st_size,
                                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                            },
                            "source_uri": f"file://{file_path}"
                        }
                        doc_count += 1
            
            self._initialized = True
            logger.info(f"Initialized local knowledge source: {self.directory} ({doc_count} documents)")
            
        except Exception as e:
            logger.error(f"Failed to initialize local source: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self._documents) if self._initialized else 0
    
    def search(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search local documents."""
        if not self._initialized:
            return []
        
        results = []
        query_lower = query.lower()
        
        for doc in self._documents.values():
            content_lower = doc["content"].lower()
            title_lower = doc["title"].lower()
            
            score = 0.0
            if query_lower in title_lower:
                score += 0.8
            if query_lower in content_lower:
                score += 0.6
                
            if score >= similarity_threshold:
                result = doc.copy()
                result["similarity_score"] = score
                results.append(result)
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:max_results]
    
    def retrieve_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        if not self._initialized:
            return None
        
        for doc in self._documents.values():
            if doc["id"] == document_id:
                return doc
        return None
    
    def retrieve_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents by criteria."""
        if not self._initialized:
            return []
        
        results = []
        for doc in self._documents.values():
            match = True
            
            for key, value in criteria.items():
                if key in doc["metadata"]:
                    if doc["metadata"][key] != value:
                        match = False
                        break
                elif key in doc:
                    if doc[key] != value:
                        match = False
                        break
            
            if match:
                results.append(doc)
        
        return results


class DatabaseKnowledgeSource(KnowledgeSource):
    """Knowledge source backed by real database via UnifiedSessionManager."""
    
    def __init__(self, table_name: str = "documents", schema: str = "public"):
        """
        Initialize database knowledge source.
        
        Args:
            table_name: Table containing documents
            schema: Database schema
        """
        self.table_name = table_name
        self.schema = schema
        self._initialized = False
        self._connection = None
        self._session_manager = None
        self._documents_cache = {}
    
    def initialize(self) -> None:
        """Initialize real database connection."""
        try:
            # Import and use global UnifiedSessionManager instance (universal architecture)
            try:
                from tidyllm.infrastructure.session.unified import get_global_session_manager
                self._session_manager = get_global_session_manager()
            except ImportError as ie:
                # Try relative import fallback
                try:
                    from ...infrastructure.session.unified import get_global_session_manager
                    self._session_manager = get_global_session_manager()
                except ImportError:
                    raise ImportError(f"Cannot import get_global_session_manager: {ie}")
            self._connection = self._session_manager.get_postgres_connection()
            logger.info("[OK] Database connection established via UnifiedSessionManager")
            
            # Load documents from database
            self._load_documents_from_database()
            
        except ImportError as e:
            logger.warning(f"UnifiedSessionManager not available: {e}")
            self._load_mock_documents()
        except Exception as e:
            logger.warning(f"Database connection failed: {e}, using mock data")
            self._load_mock_documents()
        
        self._initialized = True
        logger.info(f"[OK] Database knowledge source initialized: {self.schema}.{self.table_name} ({len(self._documents_cache)} documents)")
    
    def _load_documents_from_database(self) -> None:
        """Load documents from real database."""
        try:
            with self._connection.cursor() as cursor:
                # Query assumes table has columns: id, title, content, metadata, created_at
                query = f"""
                SELECT id, title, content, metadata, created_at, updated_at
                FROM {self.schema}.{self.table_name}
                WHERE content IS NOT NULL AND content != ''
                ORDER BY created_at DESC
                LIMIT 1000
                """
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                for row in rows:
                    doc_id, title, content, metadata, created_at, updated_at = row
                    
                    # Handle metadata JSON
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {"raw_metadata": metadata}
                    elif metadata is None:
                        metadata = {}
                    
                    self._documents_cache[doc_id] = {
                        "id": doc_id,
                        "title": title or f"Document {doc_id}",
                        "content": content[:10000] if content and len(content) > 10000 else content or "",
                        "metadata": {
                            **metadata,
                            "table": self.table_name,
                            "schema": self.schema,
                            "created_at": created_at.isoformat() if created_at else None,
                            "updated_at": updated_at.isoformat() if updated_at else None
                        },
                        "source_uri": f"db://{self.schema}.{self.table_name}/{doc_id}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to load documents from database: {e}")
            # Don't raise, let it fall back to mock data
            self._load_mock_documents()
    
    def _load_mock_documents(self) -> None:
        """Load mock documents for development/fallback."""
        self._documents_cache = {
            "db_legal_contract_123": {
                "id": "db_legal_contract_123",
                "title": "Corporate Service Agreement - DB Source",
                "content": "This database-stored legal contract includes comprehensive termination clauses. Section 9 outlines termination procedures, including 60-day notice requirements and data handling obligations. The agreement covers service level commitments and compliance requirements.",
                "metadata": {
                    "table": self.table_name,
                    "schema": self.schema,
                    "document_type": "legal_contract",
                    "status": "active",
                    "created_at": "2024-08-15T09:30:00Z",
                    "updated_at": "2024-09-01T16:45:00Z"
                },
                "source_uri": f"db://{self.schema}.{self.table_name}/db_legal_contract_123"
            },
            "db_policy_doc_456": {
                "id": "db_policy_doc_456", 
                "title": "Data Governance Policy - DB Source",
                "content": "Database-stored policy document outlining data governance requirements. Covers data classification, retention policies, and compliance frameworks. Includes procedures for contract data handling and legal document processing workflows.",
                "metadata": {
                    "table": self.table_name,
                    "schema": self.schema,
                    "document_type": "policy",
                    "status": "approved",
                    "created_at": "2024-07-20T11:15:00Z",
                    "updated_at": "2024-08-30T14:20:00Z"
                },
                "source_uri": f"db://{self.schema}.{self.table_name}/db_policy_doc_456"
            }
        }
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self._documents_cache) if self._initialized else 0
    
    def search(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search database documents with enhanced matching."""
        if not self._initialized:
            return []
        
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for doc in self._documents_cache.values():
            score = self._calculate_db_similarity_score(query, query_words, doc)
            
            if score >= similarity_threshold:
                result = doc.copy()
                result["similarity_score"] = score
                results.append(result)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:max_results]
    
    def _calculate_db_similarity_score(self, query: str, query_words: set, doc: Dict[str, Any]) -> float:
        """Calculate similarity score for database documents."""
        query_lower = query.lower()
        content_lower = doc["content"].lower()
        title_lower = doc["title"].lower()
        
        score = 0.0
        
        # Exact phrase matching
        if query_lower in title_lower:
            score += 0.9
        if query_lower in content_lower:
            score += 0.8
            
        # Word overlap
        title_words = set(title_lower.split())
        content_words = set(content_lower.split())
        
        title_overlap = len(query_words.intersection(title_words))
        if title_overlap > 0:
            score += 0.7 * (title_overlap / len(query_words))
        
        content_overlap = len(query_words.intersection(content_words))
        if content_overlap > 0:
            score += 0.5 * (content_overlap / len(query_words))
            
        # Metadata matching
        metadata = doc.get("metadata", {})
        metadata_text = " ".join(str(v).lower() for v in metadata.values())
        if query_lower in metadata_text:
            score += 0.4
            
        return min(score, 1.0)
    
    def retrieve_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        if not self._initialized:
            return None
        
        return self._documents_cache.get(document_id)
    
    def retrieve_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents by criteria."""
        if not self._initialized:
            return []
        
        results = []
        for doc in self._documents_cache.values():
            match = True
            
            for key, value in criteria.items():
                if key in doc["metadata"]:
                    if doc["metadata"][key] != value:
                        match = False
                        break
                elif key in doc:
                    if doc[key] != value:
                        match = False
                        break
            
            if match:
                results.append(doc)
        
        return results