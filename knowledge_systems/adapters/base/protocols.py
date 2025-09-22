"""
Protocol Definitions for RAG Delegates
=======================================

Defines protocol interfaces that delegates must implement.
Follows hexagonal architecture - adapters depend on protocols, not implementations.
"""

from typing import Protocol, Dict, Any, List, Optional
from datetime import datetime


class RAGDelegateProtocol(Protocol):
    """Protocol for main RAG delegate."""

    def is_available(self) -> bool:
        """Check if RAG services are available."""
        ...

    def query(self, system_type: str, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a RAG query."""
        ...

    def create_system(self, system_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new RAG system instance."""
        ...

    def update_system(self, system_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update RAG system configuration."""
        ...

    def delete_system(self, system_id: str) -> Dict[str, Any]:
        """Delete a RAG system instance."""
        ...

    def list_systems(self) -> List[Dict[str, Any]]:
        """List all RAG system instances."""
        ...

    def get_system_health(self, system_type: str) -> Dict[str, Any]:
        """Get health status of a RAG system."""
        ...


class DatabaseDelegateProtocol(Protocol):
    """Protocol for database access delegate."""

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a database query."""
        ...

    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute a database update."""
        ...

    def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information."""
        ...

    def health_check(self) -> bool:
        """Check database health."""
        ...


class EmbeddingDelegateProtocol(Protocol):
    """Protocol for embedding service delegate."""

    def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Generate embedding for text."""
        ...

    def generate_embeddings_batch(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...

    def get_available_models(self) -> List[str]:
        """Get list of available embedding models."""
        ...

    def get_embedding_dimension(self, model: str = None) -> int:
        """Get dimension of embeddings for a model."""
        ...


class LLMDelegateProtocol(Protocol):
    """Protocol for LLM service delegate."""

    def generate_response(self, prompt: str, config: Dict[str, Any] = None) -> str:
        """Generate LLM response."""
        ...

    def generate_structured_response(self, prompt: str, schema: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate structured LLM response."""
        ...

    def get_available_models(self) -> List[str]:
        """Get list of available LLM models."""
        ...


class StorageDelegateProtocol(Protocol):
    """Protocol for storage service delegate (S3, etc)."""

    def upload_file(self, file_content: bytes, key: str, metadata: Dict[str, Any] = None) -> str:
        """Upload file to storage."""
        ...

    def download_file(self, key: str) -> bytes:
        """Download file from storage."""
        ...

    def list_files(self, prefix: str = None) -> List[Dict[str, Any]]:
        """List files in storage."""
        ...

    def delete_file(self, key: str) -> bool:
        """Delete file from storage."""
        ...

    def get_file_url(self, key: str, expiry_seconds: int = 3600) -> str:
        """Get presigned URL for file."""
        ...