"""
Base RAG Adapter
================

Abstract base class for all RAG adapters.
Enforces hexagonal architecture and standard interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import logging

from .rag_types import RAGQuery, RAGResponse, RAGHealthStatus, RAGSystemInfo, HealthStatus
from .protocols import RAGDelegateProtocol

logger = logging.getLogger(__name__)


class BaseRAGAdapter(ABC):
    """
    Abstract base class for all RAG adapters.

    All RAG adapters MUST:
    1. Extend this class
    2. Use delegate pattern (NO direct infrastructure imports)
    3. Return standard types (RAGQuery, RAGResponse, etc)
    4. Implement all abstract methods
    """

    def __init__(self, delegate: Optional[RAGDelegateProtocol] = None):
        """
        Initialize adapter with optional delegate.

        Args:
            delegate: RAG delegate for infrastructure access (follows hexagonal architecture)
        """
        self.delegate = delegate
        self._adapter_type = self.__class__.__name__.replace('Adapter', '')
        self._version = "1.0.0"
        logger.info(f"Initializing {self._adapter_type} adapter v{self._version}")

    @abstractmethod
    def query(self, request: RAGQuery) -> RAGResponse:
        """
        Execute a RAG query.

        Args:
            request: Standard RAG query request

        Returns:
            Standard RAG response
        """
        pass

    @abstractmethod
    def health_check(self) -> RAGHealthStatus:
        """
        Check adapter health status.

        Returns:
            Standard health status with dependencies
        """
        pass

    @abstractmethod
    def get_info(self) -> RAGSystemInfo:
        """
        Get adapter information and capabilities.

        Returns:
            Standard system information
        """
        pass

    def query_with_timing(self, request: RAGQuery) -> RAGResponse:
        """
        Execute query with timing information.

        Args:
            request: Standard RAG query request

        Returns:
            Standard RAG response with timing
        """
        start_time = time.time()
        try:
            response = self.query(request)
            response.processing_time_ms = (time.time() - start_time) * 1000
            response.adapter_type = self._adapter_type
            return response
        except Exception as e:
            logger.error(f"Query failed in {self._adapter_type}: {e}")
            # Return error response
            return RAGResponse(
                response=f"Query failed: {str(e)}",
                confidence=0.0,
                sources=[],
                authority_tier=0,
                collection_name=request.domain,
                precedence_level=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                adapter_type=self._adapter_type,
                metadata={'error': str(e)}
            )

    def validate_query(self, request: RAGQuery) -> bool:
        """
        Validate query request.

        Args:
            request: Query to validate

        Returns:
            True if valid, False otherwise
        """
        if not request.query:
            logger.warning("Empty query provided")
            return False

        if not request.domain:
            logger.warning("No domain specified")
            return False

        if request.confidence_threshold < 0 or request.confidence_threshold > 1:
            logger.warning(f"Invalid confidence threshold: {request.confidence_threshold}")
            return False

        return True

    def check_delegate_health(self) -> HealthStatus:
        """
        Check health of delegate connection.

        Returns:
            Health status of delegate
        """
        if not self.delegate:
            return HealthStatus.UNHEALTHY

        try:
            if hasattr(self.delegate, 'is_available') and self.delegate.is_available():
                return HealthStatus.HEALTHY
            return HealthStatus.DEGRADED
        except Exception as e:
            logger.error(f"Delegate health check failed: {e}")
            return HealthStatus.UNHEALTHY

    def get_base_health_status(self) -> RAGHealthStatus:
        """
        Get base health status with common checks.

        Returns:
            Base health status
        """
        return RAGHealthStatus(
            status=HealthStatus.UNKNOWN,
            adapter_type=self._adapter_type,
            last_checked=datetime.now(),
            dependencies={
                'delegate': self.check_delegate_health()
            }
        )

    def get_base_info(self) -> RAGSystemInfo:
        """
        Get base system information.

        Returns:
            Base system info
        """
        return RAGSystemInfo(
            adapter_type=self._adapter_type,
            version=self._version,
            description=f"{self._adapter_type} RAG Adapter",
            capabilities=[],
            supported_domains=[],
            configuration={
                'has_delegate': self.delegate is not None
            }
        )

    # Optional methods that adapters can override

    def create_collection(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new collection (optional).

        Args:
            name: Collection name
            config: Collection configuration

        Returns:
            Creation result
        """
        return {
            'success': False,
            'message': f'{self._adapter_type} does not support collection creation'
        }

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List available collections (optional).

        Returns:
            List of collections
        """
        return []

    def upload_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload a document (optional).

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            Upload result
        """
        return {
            'success': False,
            'message': f'{self._adapter_type} does not support document upload'
        }