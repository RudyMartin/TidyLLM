#!/usr/bin/env python3
"""
RAG Master Delegate - Infrastructure Layer
==========================================

Master delegate that combines all infrastructure delegates.
Provides unified access point for all RAG adapters.

IMPORTANT: This delegate uses the consolidated infra_delegate
which properly detects and uses parent infrastructure (ResilientPoolManager).
"""

from typing import Dict, Any, Optional, Protocol, List
import logging

# Import the consolidated infrastructure delegate
from ..infra_delegate import get_infra_delegate

# Still import delegate types for backward compatibility
from .llm_delegate import LLMDelegate
from .database_delegate import DatabaseDelegate
# from .aws_delegate import AWSDelegate  # Not implemented yet
from .embedding_delegate import EmbeddingDelegate
from .dspy_delegate import DSPyDelegate

logger = logging.getLogger(__name__)


class RAGDelegateProtocol(Protocol):
    """Protocol for RAG delegates."""

    def get_llm_delegate(self) -> Optional[LLMDelegate]:
        """Get LLM delegate."""
        ...

    def get_db_delegate(self) -> Optional[DatabaseDelegate]:
        """Get database delegate."""
        ...

    def get_aws_delegate(self) -> Optional[Any]:
        """Get AWS delegate."""
        ...

    def get_embedding_delegate(self) -> Optional[EmbeddingDelegate]:
        """Get embedding delegate."""
        ...

    def get_dspy_delegate(self) -> Optional[DSPyDelegate]:
        """Get DSPy delegate."""
        ...


class RAGMasterDelegate:
    """
    Master delegate that provides access to all infrastructure delegates.

    This is the main delegate that RAG adapters use.
    It provides lazy-loaded access to all infrastructure services.
    """

    def __init__(self):
        """Initialize master delegate using consolidated infrastructure."""
        # Use the consolidated infrastructure delegate
        # This properly detects and uses parent infrastructure (ResilientPoolManager)
        self._infra = get_infra_delegate()

        # Legacy delegate references (for backward compatibility)
        self._llm_delegate = None
        self._db_delegate = None
        self._aws_delegate = None
        self._embedding_delegate = None
        self._dspy_delegate = None
        self._initialized = False

        logger.info("RAGMasterDelegate initialized with consolidated infrastructure")

    def get_llm_delegate(self) -> Optional[LLMDelegate]:
        """Get LLM delegate for language model operations."""
        if self._llm_delegate is None:
            try:
                self._llm_delegate = get_llm_delegate()
                logger.info("LLM delegate initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM delegate: {e}")
        return self._llm_delegate

    def get_db_delegate(self) -> Optional[DatabaseDelegate]:
        """Get database delegate for PostgreSQL operations.

        Uses consolidated infrastructure which properly detects
        and uses parent ResilientPoolManager when available.
        """
        # Return the infra delegate which has database access
        # The infra delegate properly uses ResilientPoolManager from parent infrastructure
        if self._infra:
            return self._infra  # The infra delegate has get_db_connection() method
        return None

    def get_aws_delegate(self) -> Optional[Any]:
        """Get AWS delegate for Bedrock/S3 operations."""
        if self._aws_delegate is None:
            try:
                self._aws_delegate = get_aws_delegate()
                logger.info("AWS delegate initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AWS delegate: {e}")
        return self._aws_delegate

    def get_embedding_delegate(self) -> Optional[EmbeddingDelegate]:
        """Get embedding delegate for vector operations."""
        if self._embedding_delegate is None:
            try:
                self._embedding_delegate = get_embedding_delegate()
                logger.info("Embedding delegate initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding delegate: {e}")
        return self._embedding_delegate

    def get_dspy_delegate(self) -> Optional[DSPyDelegate]:
        """Get DSPy delegate for workflow operations."""
        if self._dspy_delegate is None:
            try:
                self._dspy_delegate = get_dspy_delegate()
                logger.info("DSPy delegate initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize DSPy delegate: {e}")
        return self._dspy_delegate

    def check_health(self) -> Dict[str, bool]:
        """
        Check health of all delegates.

        Uses consolidated infrastructure to check health.
        Returns:
            Dictionary of delegate health status
        """
        health = {}

        # Check infra delegate health (it has database via ResilientPoolManager)
        if self._infra:
            try:
                # Check database connection through infra delegate
                conn = self._infra.get_db_connection()
                if conn:
                    self._infra.return_db_connection(conn)
                    health['database'] = True
                else:
                    health['database'] = False
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                health['database'] = False

            # Check other services through infra
            health['llm'] = hasattr(self._infra, 'generate_llm_response')
            health['aws'] = hasattr(self._infra, 'invoke_bedrock')
            health['embedding'] = hasattr(self._infra, 'generate_embedding')
            health['dspy'] = False  # DSPy is separate

        else:
            # No infra delegate available
            health = {
                'llm': False,
                'database': False,
                'aws': False,
                'embedding': False,
                'dspy': False
            }

        return health

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of available delegates.

        Returns:
            Dictionary of available capabilities
        """
        capabilities = {
            'delegates': {},
            'services': []
        }

        health = self.check_health()

        # LLM capabilities
        if health.get('llm'):
            capabilities['delegates']['llm'] = True
            capabilities['services'].extend([
                'text_generation',
                'structured_generation',
                'model_selection'
            ])

        # Database capabilities
        if health.get('database'):
            capabilities['delegates']['database'] = True
            capabilities['services'].extend([
                'document_search',
                'collection_management',
                'document_storage'
            ])

        # AWS capabilities
        if health.get('aws'):
            capabilities['delegates']['aws'] = True
            capabilities['services'].extend([
                'bedrock_knowledge_base',
                'bedrock_generation',
                's3_storage'
            ])

        # Embedding capabilities
        if health.get('embedding'):
            capabilities['delegates']['embedding'] = True
            capabilities['services'].extend([
                'text_embedding',
                'similarity_search',
                'document_clustering'
            ])

        # DSPy capabilities
        if health.get('dspy'):
            capabilities['delegates']['dspy'] = True
            capabilities['services'].extend([
                'workflow_execution',
                'chain_of_thought',
                'program_compilation',
                'multi_hop_rag'
            ])

        return capabilities

    def get_available_systems(self) -> Dict[str, str]:
        """
        Get available RAG systems.

        Returns the 6 standardized RAG adapters.
        This method is used by portals for system selection.
        """
        return {
            "ai_powered": "AI-Powered RAG",
            "postgres": "PostgreSQL RAG",
            "judge": "Judge RAG",
            "intelligent": "Intelligent RAG",
            "sme": "SME RAG System",
            "dspy": "DSPy RAG"
        }

    def is_available(self) -> bool:
        """
        Check if delegate is available.

        Returns True if at least one sub-delegate is available.
        """
        health = self.check_health()
        return any(health.values())

    def get_infra(self):
        """
        Get the consolidated infrastructure delegate.

        This delegate properly uses:
        - Parent ResilientPoolManager for database connections
        - Parent credential_carrier for credentials
        - Parent aws_service for AWS operations

        Returns:
            The consolidated infrastructure delegate
        """
        return self._infra

    def list_systems(self) -> List[Dict[str, Any]]:
        """
        List active RAG system instances.

        Returns list of system instances with their status.
        """
        systems = []
        available_systems = self.get_available_systems()

        for system_id, system_name in available_systems.items():
            # Check if this system type is available
            try:
                # Try to check if adapter can be imported
                module_name = f"tidyllm.knowledge_systems.adapters.{system_id}"
                __import__(module_name)
                systems.append({
                    'id': system_id,
                    'name': system_name,
                    'status': 'available',
                    'type': 'adapter'
                })
            except ImportError:
                systems.append({
                    'id': system_id,
                    'name': system_name,
                    'status': 'unavailable',
                    'type': 'adapter'
                })

        return systems

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.

        Returns metrics about RAG system performance.
        """
        # Get health status
        health = self.check_health()
        available_count = sum(1 for v in health.values() if v)

        # Calculate success rate based on available delegates
        success_rate = available_count / len(health) if health else 0

        return {
            'overall_success_rate': success_rate,
            'available_delegates': available_count,
            'total_delegates': len(health),
            'health_status': health,
            'system_count': len(self.get_available_systems()),
            'active_instances': len([s for s in self.list_systems() if s['status'] == 'available'])
        }


class RAGMasterDelegateFactory:
    """Factory for creating master delegates."""

    _instance = None

    @classmethod
    def get_delegate(cls) -> RAGMasterDelegate:
        """Get singleton master delegate instance."""
        if cls._instance is None:
            cls._instance = RAGMasterDelegate()
        return cls._instance


def get_rag_delegate() -> RAGMasterDelegate:
    """
    Get RAG master delegate instance.

    This is the main entry point for getting a delegate
    that RAG adapters should use.
    """
    return RAGMasterDelegateFactory.get_delegate()


# Export the protocol for type hints
__all__ = ['RAGDelegateProtocol', 'RAGMasterDelegate', 'get_rag_delegate']