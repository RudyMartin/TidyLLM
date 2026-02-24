"""
Outbound Ports - Interfaces for external services that domain needs
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Domain entity for documents"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class ComplianceRule:
    """Domain entity for compliance rules"""
    id: str
    authority_tier: int  # 1=Regulatory, 2=SOP, 3=Technical
    rule_text: str
    precedence: float


class DocumentRepositoryPort(ABC):
    """Port for document storage operations"""

    @abstractmethod
    async def find_by_query(self, query: str, limit: int = 10) -> List[Document]:
        """Find documents matching query"""
        pass

    @abstractmethod
    async def save_document(self, document: Document) -> str:
        """Save document and return ID"""
        pass

    @abstractmethod
    async def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        pass


class ComplianceRepositoryPort(ABC):
    """Port for compliance data operations"""

    @abstractmethod
    async def find_rules_by_domain(self, domain: str) -> List[ComplianceRule]:
        """Find compliance rules for domain"""
        pass

    @abstractmethod
    async def find_by_authority_tier(self, tier: int) -> List[ComplianceRule]:
        """Find rules by authority tier"""
        pass

    @abstractmethod
    async def save_rule(self, rule: ComplianceRule) -> str:
        """Save compliance rule"""
        pass


class LLMServicePort(ABC):
    """Port for LLM operations"""

    @abstractmethod
    async def generate_completion(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text completion"""
        pass

    @abstractmethod
    async def analyze_document(self, document: str, query: str) -> Dict[str, Any]:
        """Analyze document with query"""
        pass


class EmbeddingServicePort(ABC):
    """Port for embedding operations"""

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass

    @abstractmethod
    async def find_similar(self, embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Find similar embeddings"""
        pass


class WorkflowRepositoryPort(ABC):
    """Port for workflow operations"""

    @abstractmethod
    async def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow definition"""
        pass

    @abstractmethod
    async def save_workflow_result(self, workflow_id: str, result: Dict) -> str:
        """Save workflow execution result"""
        pass