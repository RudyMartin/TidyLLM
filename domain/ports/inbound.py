"""
Inbound Ports - Use case interfaces that external world can call
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ComplianceQuery:
    """Request for compliance analysis"""
    domain: str
    question: str
    context: Dict[str, Any]


@dataclass
class ComplianceDecision:
    """Compliance decision result"""
    decision: str
    confidence: float
    authority_tier: int
    supporting_rules: List[str]
    recommendations: List[str]


class ComplianceQueryUseCase(ABC):
    """Use case for compliance queries"""

    @abstractmethod
    async def execute(self, query: ComplianceQuery) -> ComplianceDecision:
        """Execute compliance query and return decision"""
        pass


class DocumentAnalysisUseCase(ABC):
    """Use case for document analysis"""

    @abstractmethod
    async def analyze(self, document_path: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze document and return results"""
        pass


class WorkflowExecutionUseCase(ABC):
    """Use case for workflow execution"""

    @abstractmethod
    async def execute(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with inputs"""
        pass