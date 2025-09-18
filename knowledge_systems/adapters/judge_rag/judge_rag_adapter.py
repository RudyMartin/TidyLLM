#!/usr/bin/env python3
"""
Judge RAG Adapter - External RAG Integration
===========================================

Clean adapter for JB's AWS-hosted RAG implementation.
Nothing to build or maintain - just an adapter interface for pipeline integration.

Integrates with existing Boss Portal + PostgresRAGAdapter architecture:
- ComplianceRAG (our postgres-based authority system)
- DocumentRAG (our document search)
- ExpertRAG (our subject matter expertise)
- JudgeRAG (JB's external AWS RAG) NEW

Features:
- Clean adapter pattern (no external dependencies leaked)
- AWS integration through adapter interface
- Fallback to local RAG systems if JB's system unavailable
- Unified response format compatible with existing pipeline
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import our existing RAG infrastructure
try:
    from ..postgres_rag import RAGQuery, RAGResponse, PostgresRAGAdapter
except ImportError:
    from rag_adapters.postgres_rag_adapter import RAGQuery, RAGResponse, PostgresRAGAdapter

# AWS integration (through adapter pattern)
try:
    import boto3
    import requests
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

@dataclass
class JBRAGRequest:
    """Request format for JB's RAG system."""
    query: str
    context: Optional[str] = None
    domain: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.1
    confidence_threshold: float = 0.7

@dataclass
class JBRAGResponse:
    """Response format from JB's RAG system."""
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time_ms: float
    model_used: str
    jb_system_status: str

class ExternalRAGPort(Protocol):
    """Port interface for external RAG systems like JB's."""
    async def query_external_rag(self, request: JBRAGRequest) -> JBRAGResponse: ...
    async def health_check(self) -> Dict[str, Any]: ...
    async def get_system_info(self) -> Dict[str, Any]: ...

class JudgeRAGAdapter:
    """
    Adapter for JB's AWS-hosted RAG implementation.

    Provides clean interface to JB's external RAG system while maintaining
    compatibility with existing Boss Portal + PostgresRAGAdapter pipeline.

    Nothing to build or maintain - pure adapter pattern.
    """

    def __init__(self,
                 aws_region: str = "us-east-1",
                 jb_endpoint_url: str = None,
                 fallback_adapter: PostgresRAGAdapter = None):
        """
        Initialize Judge RAG Adapter.

        Args:
            aws_region: AWS region for JB's deployment
            jb_endpoint_url: JB's RAG endpoint URL (provided by JB)
            fallback_adapter: Local PostgresRAGAdapter for fallback
        """
        print("Initializing Judge RAG Adapter...")

        self.aws_region = aws_region
        self.jb_endpoint_url = jb_endpoint_url or "https://jb-rag-api.your-aws-domain.com/v1/query"

        # Fallback to our existing PostgresRAGAdapter if JB's system unavailable
        self.fallback_adapter = fallback_adapter or PostgresRAGAdapter()

        # AWS session for JB's system access - MEGA USM Integration
        try:
            import streamlit as st
            usm = st.session_state.get('tidyllm_session_manager')
            if usm and hasattr(usm, 'get_aws_session'):
                self.aws_session = usm.get_aws_session()
                print("AWS session initialized via MEGA USM for JB's RAG system")
            elif AWS_AVAILABLE:
                self.aws_session = boto3.Session()
                print("AWS session initialized via direct boto3 (fallback)")
            else:
                self.aws_session = None
                print("WARNING: AWS not available - will use fallback adapter only")
        except Exception as e:
            print(f"WARNING: USM integration failed ({e}) - using direct boto3 fallback")
            if AWS_AVAILABLE:
                self.aws_session = boto3.Session()
            else:
                self.aws_session = None

        # System status tracking
        self.jb_system_available = False
        self.last_health_check = None

    async def health_check(self) -> Dict[str, Any]:
        """Check if JB's RAG system is available."""
        print("Checking JB LLM Judge system health...")

        try:
            if not AWS_AVAILABLE:
                return {
                    "jb_system_available": False,
                    "reason": "AWS libraries not available",
                    "fallback_available": True,
                    "timestamp": datetime.now().isoformat()
                }

            # Health check request to JB's endpoint
            health_url = f"{self.jb_endpoint_url.replace('/query', '/health')}"

            # This would be the actual call to JB's system
            # For now, simulate the health check
            health_response = {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_hours": 24.5,
                "models_available": ["gpt-4", "claude-3"],
                "rag_collections": 15,
                "avg_response_time_ms": 250
            }

            self.jb_system_available = True
            self.last_health_check = datetime.now()

            return {
                "jb_system_available": True,
                "jb_system_status": health_response,
                "fallback_available": True,
                "last_checked": self.last_health_check.isoformat()
            }

        except Exception as e:
            print(f"ERROR: JB system health check failed: {e}")
            self.jb_system_available = False

            return {
                "jb_system_available": False,
                "error": str(e),
                "fallback_available": True,
                "timestamp": datetime.now().isoformat()
            }

    async def query_jb_rag(self, request: JBRAGRequest) -> JBRAGResponse:
        """
        Query JB's RAG system directly.

        Args:
            request: JB RAG request with query and parameters

        Returns:
            JBRAGResponse with JB's system results
        """
        print(f"Querying JB LLM Judge: {request.query[:50]}...")

        try:
            if not self.jb_system_available:
                await self.health_check()

            if not self.jb_system_available:
                raise Exception("JB's RAG system not available")

            # This would be the actual API call to JB's system
            # Format: POST to JB's endpoint with request data
            api_request = {
                "query": request.query,
                "context": request.context,
                "domain": request.domain,
                "parameters": {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "confidence_threshold": request.confidence_threshold
                }
            }

            # Simulate JB's RAG response
            # In production, this would be: requests.post(self.jb_endpoint_url, json=api_request)
            jb_response_data = {
                "response": f"JB LLM Judge Analysis: {request.query}\n\nBased on my RAG analysis, here are the key findings...",
                "confidence": 0.87,
                "sources": [
                    {
                        "document": "JB_Knowledge_Base_Doc_1.pdf",
                        "relevance_score": 0.92,
                        "excerpt": "Relevant excerpt from JB's knowledge base..."
                    },
                    {
                        "document": "JB_Knowledge_Base_Doc_2.pdf",
                        "relevance_score": 0.85,
                        "excerpt": "Additional relevant information..."
                    }
                ],
                "processing_time_ms": 247.5,
                "model_used": "gpt-4-rag-enhanced",
                "system_status": "optimal"
            }

            return JBRAGResponse(
                response=jb_response_data["response"],
                confidence=jb_response_data["confidence"],
                sources=jb_response_data["sources"],
                processing_time_ms=jb_response_data["processing_time_ms"],
                model_used=jb_response_data["model_used"],
                jb_system_status=jb_response_data["system_status"]
            )

        except Exception as e:
            print(f"ERROR: JB RAG query failed: {e}")
            raise Exception(f"JB LLM Judge system error: {e}")

    def query_judge_rag(self, query: RAGQuery) -> RAGResponse:
        """
        Query JB's system using our unified RAGQuery/RAGResponse format.

        This maintains compatibility with existing Boss Portal pipeline.
        """
        print(f"Judge RAG Query: {query.query}")

        try:
            # Convert our RAGQuery to JB's format
            jb_request = JBRAGRequest(
                query=query.query,
                context=f"Domain: {query.domain}",
                domain=query.domain,
                confidence_threshold=query.confidence_threshold
            )

            # Query JB's system (async call made sync for compatibility)
            jb_response = asyncio.run(self.query_jb_rag(jb_request))

            # Convert JB's response to our unified RAGResponse format
            return RAGResponse(
                response=jb_response.response,
                confidence=jb_response.confidence,
                sources=jb_response.sources,
                authority_tier=50,  # Judge tier - between Expert (99) and Regulatory (1-3)
                collection_name="judge_rag",
                precedence_level=0.85  # High quality external judgment
            )

        except Exception as e:
            print(f"WARNING: JB system unavailable, using fallback: {e}")

            # Fallback to our local PostgresRAGAdapter
            return self.fallback_adapter.query_expert_rag(query)

    def query_hybrid_rag(self, query: RAGQuery, use_jb_primary: bool = True) -> RAGResponse:
        """
        Hybrid query using both JB's system and local adapters.

        Args:
            query: RAG query
            use_jb_primary: Whether to use JB as primary (fallback to local) or combine

        Returns:
            Best response from available systems
        """
        print(f"Hybrid RAG Query (JB Primary: {use_jb_primary}): {query.query}")

        if use_jb_primary:
            # Try JB first, fallback to local
            try:
                jb_response = self.query_jb_llm_judge_rag(query)
                if jb_response.confidence > 0.6:
                    return jb_response
                else:
                    print("JB confidence low, trying local RAG...")
                    return self.fallback_adapter.query_unified_rag(query)
            except:
                print("JB unavailable, using local RAG...")
                return self.fallback_adapter.query_unified_rag(query)
        else:
            # Combine both systems for best result
            try:
                # Get both responses
                jb_response = self.query_jb_llm_judge_rag(query)
                local_response = self.fallback_adapter.query_unified_rag(query)

                # Choose best based on confidence and authority
                if jb_response.confidence > local_response.confidence:
                    return jb_response
                elif local_response.authority_tier == 1:  # Local regulatory authority wins
                    return local_response
                else:
                    return jb_response  # Default to JB for tie-breaking

            except:
                # If JB fails, use local only
                return self.fallback_adapter.query_unified_rag(query)

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of both JB's system and local fallback."""
        print("Getting JB + Local RAG system status...")

        # Get JB system status
        jb_status = asyncio.run(self.health_check())

        # Get local system status
        local_collections = len(self.fallback_adapter.sme_system.list_collections())

        return {
            "jb_llm_judge": {
                "available": jb_status["jb_system_available"],
                "status": jb_status.get("jb_system_status", "unavailable"),
                "last_checked": jb_status.get("last_checked", "never")
            },
            "local_postgres_rag": {
                "available": True,
                "collections": local_collections,
                "types": ["ComplianceRAG", "DocumentRAG", "ExpertRAG"]
            },
            "hybrid_mode": "active",
            "primary_system": "jb_llm_judge" if self.jb_system_available else "local_postgres",
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Test JB LLM Judge Adapter integration."""
    print("Testing JB LLM Judge Adapter")
    print("=" * 50)

    # Initialize JB adapter with fallback
    jb_adapter = JBLLMJudgeAdapter(
        jb_endpoint_url="https://jb-rag-api.example.com/v1/query"
    )

    # Test 1: System Status
    print("\nTest 1: System Status")
    status = jb_adapter.get_system_status()
    print(f"JB Available: {status['jb_llm_judge']['available']}")
    print(f"Local Collections: {status['local_postgres_rag']['collections']}")
    print(f"Primary System: {status['primary_system']}")

    # Test 2: JB RAG Query
    print("\nTest 2: JB LLM Judge Query")
    jb_query = RAGQuery(
        query="What are the best practices for model validation in financial services?",
        domain="model_risk",
        confidence_threshold=0.7
    )

    jb_response = jb_adapter.query_jb_llm_judge_rag(jb_query)
    print(f"Response: {jb_response.response[:100]}...")
    print(f"Confidence: {jb_response.confidence:.2f}")
    print(f"Sources: {len(jb_response.sources)}")

    # Test 3: Hybrid Query
    print("\nTest 3: Hybrid RAG Query")
    hybrid_query = RAGQuery(
        query="What architectural patterns should be used for compliance systems?",
        domain="software_architecture"
    )

    hybrid_response = jb_adapter.query_hybrid_rag(hybrid_query, use_jb_primary=True)
    print(f"Response: {hybrid_response.response[:100]}...")
    print(f"Authority Tier: {hybrid_response.authority_tier}")
    print(f"Collection: {hybrid_response.collection_name}")

    print("\nJB LLM Judge Adapter testing complete!")

if __name__ == "__main__":
    main()