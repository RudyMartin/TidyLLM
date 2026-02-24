#!/usr/bin/env python3
"""
DSPy RAG Adapter - Prompt Optimization and Reasoning Chains
===========================================================

Wraps DSPyAdvisor and DSPyService in standard adapter pattern.
Provides DSPy-powered RAG with ChainOfThought, ReAct, and signature optimization.

Features:
- Signature optimization for better prompts
- ChainOfThought reasoning
- Bootstrap learning from examples
- Query enhancement and rewriting
- Hexagonal architecture compliant
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Base adapter imports
from ..base import BaseRAGAdapter, RAGQuery, RAGResponse

# Import consolidated infrastructure delegate
from ....infrastructure.infra_delegate import get_infra_delegate

logger = logging.getLogger(__name__)


class DSPyRAGAdapter(BaseRAGAdapter):
    """
    DSPy-powered RAG adapter with prompt optimization.

    Uses DSPyAdvisor and DSPyService through consolidated delegate.
    Provides reasoning chains and signature optimization.
    """

    def __init__(self):
        """
        Initialize DSPy RAG adapter with consolidated infrastructure.
        """
        # Get infrastructure delegate (uses parent when available)
        self.infra = get_infra_delegate()
        self._version = "2.0"
        self._dspy_advisor = None
        self._dspy_service = None
        self._initialized = False

        logger.info("DSPy RAG Adapter initialized with consolidated infrastructure delegate")

    def _initialize_services(self):
        """Initialize DSPy services through delegate (lazy loading)."""
        if self._initialized:
            return True

        try:
            # Use delegate to access DSPy services
            if self.delegate and hasattr(self.delegate, 'get_dspy_services'):
                services = self.delegate.get_dspy_services()
                self._dspy_advisor = services.get('advisor')
                self._dspy_service = services.get('service')
                self._initialized = True
                logger.info("DSPy services initialized through delegate")
                return True
            else:
                # Fallback: try to import directly (for backwards compatibility)
                try:
                    from packages.tidyllm.services.dspy_advisor import DSPyAdvisor
                    from packages.tidyllm.services.dspy_service import DSPyService

                    self._dspy_advisor = DSPyAdvisor()
                    self._dspy_service = DSPyService()
                    self._initialized = True
                    logger.info("DSPy services initialized directly (fallback mode)")
                    return True
                except ImportError as e:
                    logger.error(f"Could not initialize DSPy services: {e}")
                    return False
        except Exception as e:
            logger.error(f"Failed to initialize DSPy services: {e}")
            return False

    def query(self, request: RAGQuery) -> RAGResponse:
        """
        Execute DSPy-powered RAG query with reasoning chains.

        Args:
            request: Standard RAG query

        Returns:
            DSPy-enhanced RAG response
        """
        # Initialize services if needed
        if not self._initialize_services():
            return self._create_error_response(
                request,
                "DSPy services not available",
                confidence=0.0
            )

        # Validate query
        if not self.validate_query(request):
            return self._create_error_response(
                request,
                "Invalid query parameters",
                confidence=0.0
            )

        try:
            # Prepare DSPy context
            dspy_context = self._prepare_dspy_context(request)

            # Get DSPy-enhanced response
            if self._dspy_advisor and hasattr(self._dspy_advisor, 'get_workflow_advice'):
                # Use DSPyAdvisor for workflow-style queries
                result = self._query_with_advisor(request, dspy_context)
            elif self._dspy_service and hasattr(self._dspy_service, 'enhance_rag_query'):
                # Use DSPyService for general RAG enhancement
                result = self._query_with_service(request, dspy_context)
            else:
                # Fallback to basic processing
                result = self._fallback_query(request)

            return result

        except Exception as e:
            logger.error(f"DSPy query failed: {e}")
            return self._create_error_response(request, str(e), confidence=0.1)

    def _prepare_dspy_context(self, request: RAGQuery) -> Dict[str, Any]:
        """Prepare context for DSPy processing."""
        return {
            'domain': request.domain,
            'authority_tier': request.authority_tier,
            'confidence_threshold': request.confidence_threshold,
            'metadata': request.metadata,
            'use_chain_of_thought': request.metadata.get('use_cot', True),
            'signature_type': request.metadata.get('signature_type', 'ChainOfThought')
        }

    def _query_with_advisor(self, request: RAGQuery, context: Dict[str, Any]) -> RAGResponse:
        """Query using DSPyAdvisor for workflow-style reasoning."""
        try:
            # Prepare advisor inputs
            advisor_result = self._dspy_advisor.get_workflow_advice(
                criteria={'domain': request.domain, 'authority': request.authority_tier},
                template_fields={},
                recent_activity=[],
                final_results={},
                user_question=request.query,
                use_cases=[request.domain]
            )

            if advisor_result.get('success'):
                # Extract reasoning and advice
                reasoning = advisor_result.get('reasoning', '')
                advice = advisor_result.get('advice', '')
                context_analyzed = advisor_result.get('context_analyzed', {})

                # Combine reasoning and advice for response
                response_text = f"{advice}\n\nReasoning: {reasoning}" if reasoning else advice

                return RAGResponse(
                    response=response_text,
                    confidence=0.9,  # DSPy responses are high confidence
                    sources=[{
                        'type': 'dspy_advisor',
                        'reasoning': reasoning,
                        'context': context_analyzed
                    }],
                    authority_tier=request.authority_tier or 2,
                    collection_name=f"dspy_{request.domain}",
                    precedence_level=0.95,
                    adapter_type='DSPyRAG',
                    metadata={
                        'dspy_method': 'advisor',
                        'signature_type': 'WorkflowAdvice',
                        'has_reasoning': bool(reasoning)
                    }
                )
            else:
                return self._fallback_query(request)

        except Exception as e:
            logger.error(f"DSPyAdvisor query failed: {e}")
            return self._fallback_query(request)

    def _query_with_service(self, request: RAGQuery, context: Dict[str, Any]) -> RAGResponse:
        """Query using DSPyService for general enhancement."""
        try:
            # Enhance query with DSPy
            enhanced_query = self._dspy_service.enhance_rag_query(request.query)

            # Get signature-based response
            signature_type = context.get('signature_type', 'ChainOfThought')

            if signature_type == 'ChainOfThought' and hasattr(self._dspy_service, 'chain_of_thought_rag'):
                result = self._dspy_service.chain_of_thought_rag(
                    query=enhanced_query,
                    context=request.domain
                )
            else:
                # Use basic enhancement
                result = {
                    'response': enhanced_query,
                    'confidence': 0.85
                }

            return RAGResponse(
                response=result.get('response', enhanced_query),
                confidence=result.get('confidence', 0.85),
                sources=[{
                    'type': 'dspy_service',
                    'enhanced_query': enhanced_query,
                    'signature_type': signature_type
                }],
                authority_tier=request.authority_tier or 2,
                collection_name=f"dspy_{request.domain}",
                precedence_level=0.9,
                adapter_type='DSPyRAG',
                metadata={
                    'dspy_method': 'service',
                    'signature_type': signature_type,
                    'query_enhanced': enhanced_query != request.query
                }
            )

        except Exception as e:
            logger.error(f"DSPyService query failed: {e}")
            return self._fallback_query(request)

    def _fallback_query(self, request: RAGQuery) -> RAGResponse:
        """Fallback query when DSPy services unavailable."""
        return RAGResponse(
            response=f"DSPy processing unavailable. Query: '{request.query}' in domain '{request.domain}'",
            confidence=0.5,
            sources=[],
            authority_tier=request.authority_tier or 3,
            collection_name=request.domain,
            precedence_level=0.5,
            adapter_type='DSPyRAG',
            metadata={'fallback': True}
        )

    def _create_error_response(self, request: RAGQuery, error: str, confidence: float = 0.0) -> RAGResponse:
        """Create error response."""
        return RAGResponse(
            response=f"DSPy RAG Error: {error}",
            confidence=confidence,
            sources=[],
            authority_tier=0,
            collection_name=request.domain,
            precedence_level=0.0,
            adapter_type='DSPyRAG',
            metadata={'error': error}
        )

    def health_check(self) -> Dict[str, Any]:
        """Check DSPy adapter health."""
        base_health = self.get_base_health_status()

        # Check DSPy services
        try:
            self._initialize_services()

            dependencies = base_health.dependencies
            dependencies['dspy_advisor'] = 'healthy' if self._dspy_advisor else 'unhealthy'
            dependencies['dspy_service'] = 'healthy' if self._dspy_service else 'unhealthy'

            # Overall status
            if self._dspy_advisor or self._dspy_service:
                base_health['status'] = 'healthy'
            elif self.delegate:
                base_health['status'] = 'degraded'
            else:
                base_health['status'] = 'unhealthy'

            base_health.dependencies = dependencies

        except Exception as e:
            base_health['status'] = 'unhealthy'
            base_health.error_message = str(e)

        return base_health

    def get_info(self) -> Dict[str, Any]:
        """Get DSPy adapter information."""
        base_info = self.get_base_info()

        base_info.description = "DSPy-powered RAG with prompt optimization and reasoning chains"
        base_info.capabilities = [
            "Signature optimization",
            "ChainOfThought reasoning",
            "Query enhancement",
            "Bootstrap learning",
            "Prompt engineering",
            "ReAct patterns",
            "ProgramOfThought"
        ]
        base_info.supported_domains = [
            "workflow_advice",
            "reasoning_chains",
            "prompt_optimization",
            "general_qa"
        ]

        # Add DSPy-specific configuration
        base_info.configuration.update({
            'has_advisor': self._dspy_advisor is not None,
            'has_service': self._dspy_service is not None,
            'signature_types': ['ChainOfThought', 'ReAct', 'ProgramOfThought'],
            'default_model': 'claude-3-sonnet'
        })

        return base_info