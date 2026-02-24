"""
Standard RAG Types
==================

Dataclasses for standardized RAG request/response formats.
All adapters MUST use these types for consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class RAGSystemType(str, Enum):
    """RAG system types matching the 6 orchestrators."""
    AI_POWERED = "ai_powered"
    POSTGRES = "postgres"
    JUDGE = "judge"
    INTELLIGENT = "intelligent"
    SME = "sme"
    DSPY = "dspy"


class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class RAGQuery:
    """
    Standard RAG query format.
    All adapters MUST accept this format.
    """
    query: str
    domain: str
    authority_tier: Optional[int] = None  # 1=Regulatory, 2=SOP, 3=Technical
    collection_name: Optional[str] = None
    confidence_threshold: float = 0.7
    max_results: int = 5
    include_sources: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'domain': self.domain,
            'authority_tier': self.authority_tier,
            'collection_name': self.collection_name,
            'confidence_threshold': self.confidence_threshold,
            'max_results': self.max_results,
            'include_sources': self.include_sources,
            'metadata': self.metadata
        }


@dataclass
class RAGResponse:
    """
    Standard RAG response format.
    All adapters MUST return this format.
    """
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    authority_tier: int
    collection_name: str
    precedence_level: float
    processing_time_ms: float = 0.0
    adapter_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'response': self.response,
            'confidence': self.confidence,
            'sources': self.sources,
            'authority_tier': self.authority_tier,
            'collection_name': self.collection_name,
            'precedence_level': self.precedence_level,
            'processing_time_ms': self.processing_time_ms,
            'adapter_type': self.adapter_type,
            'metadata': self.metadata
        }


@dataclass
class RAGHealthStatus:
    """
    Standard health status format.
    All adapters MUST return this format from health_check().
    """
    status: HealthStatus
    adapter_type: str
    last_checked: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    dependencies: Dict[str, HealthStatus] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'status': self.status.value,
            'adapter_type': self.adapter_type,
            'last_checked': self.last_checked.isoformat(),
            'response_time_ms': self.response_time_ms,
            'error_message': self.error_message,
            'dependencies': {k: v.value for k, v in self.dependencies.items()},
            'metrics': self.metrics
        }


@dataclass
class RAGSystemInfo:
    """
    Standard system information format.
    All adapters MUST return this format from get_info().
    """
    adapter_type: str
    version: str
    description: str
    capabilities: List[str]
    supported_domains: List[str]
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'adapter_type': self.adapter_type,
            'version': self.version,
            'description': self.description,
            'capabilities': self.capabilities,
            'supported_domains': self.supported_domains,
            'configuration': self.configuration,
            'metadata': self.metadata
        }


@dataclass
class DocumentMetadata:
    """Standard document metadata for RAG systems."""
    doc_id: str
    filename: str
    collection_name: str
    upload_date: datetime
    chunk_count: int = 0
    embedding_model: Optional[str] = None
    status: str = "uploaded"
    metadata: Dict[str, Any] = field(default_factory=dict)