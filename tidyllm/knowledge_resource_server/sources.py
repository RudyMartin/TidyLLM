"""
Knowledge Sources - Data Source Implementations
==============================================

Implements various knowledge sources for the Knowledge Resource Server.
"""

import json
import logging
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
    """Knowledge source backed by AWS S3 storage."""
    
    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1"):
        """
        Initialize S3 knowledge source.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix (folder path)
            region: AWS region
        """
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._initialized = False
        self._documents = {}
        
    def initialize(self) -> None:
        """Initialize S3 connection and load document index."""
        try:
            # In production, would initialize boto3 S3 client
            # For now, simulate with mock data
            self._documents = {
                f"s3://{self.bucket}/{self.prefix}doc1.pdf": {
                    "id": "s3_doc1",
                    "title": "Sample S3 Document 1",
                    "content": "This is a sample document from S3 storage...",
                    "metadata": {
                        "bucket": self.bucket,
                        "key": f"{self.prefix}doc1.pdf",
                        "size": 1024,
                        "last_modified": "2024-01-01T00:00:00Z"
                    },
                    "source_uri": f"s3://{self.bucket}/{self.prefix}doc1.pdf"
                },
                f"s3://{self.bucket}/{self.prefix}doc2.pdf": {
                    "id": "s3_doc2", 
                    "title": "Sample S3 Document 2",
                    "content": "Another sample document with different content...",
                    "metadata": {
                        "bucket": self.bucket,
                        "key": f"{self.prefix}doc2.pdf",
                        "size": 2048,
                        "last_modified": "2024-01-02T00:00:00Z"
                    },
                    "source_uri": f"s3://{self.bucket}/{self.prefix}doc2.pdf"
                }
            }
            
            self._initialized = True
            logger.info(f"Initialized S3 knowledge source: s3://{self.bucket}/{self.prefix}")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 source: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self._documents) if self._initialized else 0
    
    def search(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search S3 documents."""
        if not self._initialized:
            return []
        
        results = []
        query_lower = query.lower()
        
        for doc in self._documents.values():
            # Simple text matching - in production would use vector similarity
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
        
        # Sort by score and limit results
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
    """Knowledge source backed by database."""
    
    def __init__(self, connection_string: str, table_name: str = "documents"):
        """
        Initialize database knowledge source.
        
        Args:
            connection_string: Database connection string
            table_name: Table containing documents
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self._initialized = False
        self._connection = None
    
    def initialize(self) -> None:
        """Initialize database connection."""
        try:
            # In production, would establish actual database connection
            # For now, simulate with mock data
            self._mock_documents = {
                "db_doc1": {
                    "id": "db_doc1",
                    "title": "Database Document 1",
                    "content": "Document content from database...",
                    "metadata": {
                        "table": self.table_name,
                        "created_at": "2024-01-01T00:00:00Z"
                    },
                    "source_uri": f"db://{self.table_name}/db_doc1"
                }
            }
            
            self._initialized = True
            logger.info(f"Initialized database knowledge source: table '{self.table_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize database source: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self._mock_documents) if self._initialized else 0
    
    def search(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search database documents."""
        if not self._initialized:
            return []
        
        # Mock search implementation
        results = []
        for doc in self._mock_documents.values():
            if query.lower() in doc["content"].lower():
                result = doc.copy()
                result["similarity_score"] = 0.8
                results.append(result)
        
        return results[:max_results]
    
    def retrieve_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        if not self._initialized:
            return None
        
        return self._mock_documents.get(document_id)
    
    def retrieve_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents by criteria."""
        if not self._initialized:
            return []
        
        # Mock criteria matching
        return list(self._mock_documents.values())