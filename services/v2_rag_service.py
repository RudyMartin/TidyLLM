"""
V2 RAG Service
==============

Simple centralized RAG service that eliminates all path dependencies.
No complex adapters, no knowledge_systems imports, just basic functionality.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Basic V2 services
try:
    from tidyllm.services import CentralizedDocumentService
    from tidyllm.infrastructure.session.unified import get_global_session_manager
    from tidyllm.gateways.gateway_registry import get_global_registry
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SimpleRAGQuery:
    """Basic RAG query."""
    query: str
    domain: str = "general"
    collection_name: Optional[str] = None

@dataclass
class SimpleRAGResponse:
    """Basic RAG response."""
    response: str
    confidence: float = 0.0
    sources: List[Dict] = None
    collection_name: str = ""

class V2RAGService:
    """
    Simple V2 RAG Service.

    Provides basic RAG functionality without complex adapter dependencies.
    Uses TidyLLM V2 centralized services only.
    """

    def __init__(self):
        """Initialize V2 RAG service."""
        self.collections = []
        self.session_manager = None
        self.document_service = None

        if V2_AVAILABLE:
            try:
                self.session_manager = get_global_session_manager()
                self.document_service = CentralizedDocumentService()
                logger.info("V2 RAG service initialized")
            except Exception as e:
                logger.warning(f"V2 service initialization failed: {e}")

    def list_collections(self) -> List[Dict[str, Any]]:
        """List available RAG collections."""
        # Mock collections for V2 demo
        return [
            {
                'collection_id': 'v2_demo_001',
                'collection_name': 'V2_Demo_Collection',
                'description': 'V2 Architecture Demo Collection',
                'domain': 'software_architecture',
                'document_count': 5
            },
            {
                'collection_id': 'v2_financial_002',
                'collection_name': 'V2_Financial_Risk_Analysis',
                'description': 'Financial Risk Management Collection',
                'domain': 'finance',
                'document_count': 8
            }
        ]

    def create_collection(self, name: str, description: str = "", domain: str = "general") -> str:
        """Create new RAG collection."""
        collection_id = f"v2_{name.lower().replace(' ', '_')}"

        # Mock creation for V2
        new_collection = {
            'collection_id': collection_id,
            'collection_name': name,
            'description': description,
            'domain': domain,
            'document_count': 0,
            'created': True
        }

        # Add to mock list
        self.collections.append(new_collection)

        logger.info(f"Created V2 collection: {collection_id}")
        return collection_id

    def add_documents(self, collection_id: str, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to collection."""
        if not self.document_service:
            return {'success': False, 'error': 'Document service not available'}

        processed_docs = []

        for file_path in file_paths:
            try:
                # Use centralized document service
                result = self.document_service.process_document(file_path)
                if result.get('success', False):
                    processed_docs.append({
                        'file_path': file_path,
                        'text_length': len(result.get('text', '')),
                        'processor': result.get('processor_used', 'unknown')
                    })
            except Exception as e:
                logger.warning(f"Document processing failed for {file_path}: {e}")

        return {
            'success': len(processed_docs) > 0,
            'documents_processed': len(processed_docs),
            'documents': processed_docs,
            'collection_id': collection_id
        }

    def query_collection(self, query: SimpleRAGQuery) -> SimpleRAGResponse:
        """Query RAG collection."""
        # Mock query response for V2 demo
        if query.collection_name:
            response_text = f"V2 RAG Response for query '{query.query}' in collection '{query.collection_name}'"
            confidence = 0.85
        else:
            response_text = f"V2 RAG Response for query '{query.query}' across all collections"
            confidence = 0.75

        return SimpleRAGResponse(
            response=response_text,
            confidence=confidence,
            sources=[{
                'source': 'V2_mock_document.pdf',
                'content': f"Mock content relevant to: {query.query}",
                'score': confidence
            }],
            collection_name=query.collection_name or "all_collections"
        )

    def get_service_status(self) -> Dict[str, Any]:
        """Get V2 RAG service status."""
        return {
            'v2_available': V2_AVAILABLE,
            'session_manager': self.session_manager is not None,
            'document_service': self.document_service is not None,
            'collections_count': len(self.list_collections()),
            'service_type': 'V2_Centralized_RAG_Service'
        }

# Global V2 RAG service instance
_v2_rag_service = None

def get_v2_rag_service() -> V2RAGService:
    """Get global V2 RAG service instance."""
    global _v2_rag_service
    if _v2_rag_service is None:
        _v2_rag_service = V2RAGService()
    return _v2_rag_service