#!/usr/bin/env python3
"""
Postgres RAG Adapter - Uses Existing SME Tables
===============================================

Converts all 3 RAG types (Compliance, Document, Expert) to use the EXISTING
sme_collections, sme_documents, sme_document_chunks tables through adapter pattern.

RAG Types:
- ComplianceRAG: Authority-based regulatory decisions (was Hierarchical)
- DocumentRAG: Document search and retrieval (was Knowledge)
- ExpertRAG: Subject matter expert analysis (was SME)

NO new schemas - leverages existing postgres structure with:
- sme_collections.settings JSONB for authority tiers
- sme_documents.metadata JSONB for precedence info
- sme_document_chunks.embedding VECTOR for all search
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Use existing SME system imports (proper V2 paths)
try:
    from tidyllm.knowledge_systems.adapters.sme_rag.sme_rag_system import SMERAGSystem, EmbeddingModel
except ImportError:
    try:
        # Fallback to root-level import
        from _sme_rag_system import SMERAGSystem, EmbeddingModel
    except ImportError:
        # Final fallback for when SME system is not available
        class SMERAGSystem:
            def __init__(self): pass
            def list_collections(self): return []
            def get_collections(self): return []

        class EmbeddingModel:
            def __init__(self): pass

@dataclass
class RAGQuery:
    """Unified query for all RAG types."""
    query: str
    domain: str
    authority_tier: Optional[int] = None  # 1=Regulatory, 2=SOP, 3=Technical
    collection_name: Optional[str] = None
    confidence_threshold: float = 0.8

@dataclass
class RAGResponse:
    """Unified response from all RAG types."""
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    authority_tier: int
    collection_name: str
    precedence_level: float

class PostgresPort(Protocol):
    """Port interface for postgres operations."""
    async def vector_search(self, query: str, collection_id: str, **kwargs) -> List[Dict]: ...
    async def get_collections_by_authority(self, authority_tier: int) -> List[Dict]: ...
    async def store_document_with_authority(self, doc: Dict, authority_info: Dict) -> str: ...

class PostgresRAGAdapter:
    """
    Unified Postgres RAG Adapter using EXISTING sme_* tables.

    Supports all 3 RAG types through existing schema:
    - ComplianceRAG: Uses sme_collections.settings for authority tiers
    - DocumentRAG: Uses sme_documents.metadata for domain knowledge
    - ExpertRAG: Uses existing collection/document structure as-is
    """

    def __init__(self, sme_system: SMERAGSystem = None):
        """Initialize with existing SME system."""
        self.sme_system = sme_system or SMERAGSystem()

    def list_collections(self):
        """List all collections including legacy - wrapper for SME system compatibility."""
        collections = self.sme_system.get_all_collections_including_legacy()
        # Convert Collection objects to dicts for Streamlit compatibility
        result = []
        for col in collections:
            if hasattr(col, 'to_dict'):
                result.append(col.to_dict())
            elif hasattr(col, '__dict__'):
                col_dict = {
                    'collection_id': str(getattr(col, 'collection_id', str(col))),
                    'collection_name': getattr(col, 'name', getattr(col, 'collection_name', 'Unknown')),
                    'description': getattr(col, 'description', ''),
                    'settings': getattr(col, 'settings', {})
                }
                result.append(col_dict)
            else:
                result.append({'collection_id': str(col), 'collection_name': str(col)})
        return result

    def get_or_create_authority_collection(self,
                                         domain: str,
                                         authority_tier: int,
                                         description: str = "") -> str:
        """
        Get or create collection for specific authority tier using EXISTING tables.

        Uses sme_collections.settings JSONB to store authority metadata.
        """
        collection_name = f"{domain}_tier_{authority_tier}"

        # Check if collection exists
        collections = self.list_collections()  # Use the wrapper that handles conversion
        for collection in collections:
            if collection['collection_name'] == collection_name:
                return collection['collection_id']

        # Create new collection with authority metadata in settings
        authority_settings = {
            "authority_tier": authority_tier,
            "precedence_level": 1.0 if authority_tier == 1 else (0.8 if authority_tier == 2 else 0.6),
            "domain": domain,
            "rag_type": "compliance" if authority_tier else "document",
            "created_for": "v2_boss_portal_compliance"
        }

        collection_id = self.sme_system.create_collection(
            name=collection_name,
            description=description or f"{domain} Authority Tier {authority_tier}",
            embedding_model=EmbeddingModel.SENTENCE_BERT_LARGE,
            s3_bucket="dsai-2025-asu",
            s3_prefix=f"compliance/{domain}/tier_{authority_tier}",
            tags=[f"tier_{authority_tier}", domain, "compliance"],
            settings=authority_settings
        )

        return collection_id

    def query_compliance_rag(self, query: RAGQuery) -> RAGResponse:
        """
        ComplianceRAG using EXISTING sme_* tables with authority precedence.

        Searches collections by authority tier, resolves conflicts by precedence.
        Authority-based regulatory decisions with tiered compliance.
        """
        print(f"ComplianceRAG Query: {query.query}")

        # Get collections for domain, ordered by authority tier
        collections = self.list_collections()
        authority_collections = []

        for collection in collections:
            collection_settings = collection.get('settings', {})
            if (collection_settings.get('domain') == query.domain and
                collection_settings.get('authority_tier')):
                authority_collections.append({
                    'collection_id': collection['collection_id'],
                    'authority_tier': collection_settings['authority_tier'],
                    'precedence_level': collection_settings.get('precedence_level', 0.5),
                    'name': collection['collection_name']
                })

        # Sort by authority tier (1=highest precedence)
        authority_collections.sort(key=lambda x: x['authority_tier'])

        # Search each tier until we find authoritative answer
        best_response = None
        highest_precedence = 0.0

        for collection_info in authority_collections:
            # Skip if looking for specific tier and this isn't it
            if query.authority_tier and collection_info['authority_tier'] != query.authority_tier:
                continue

            # Search this authority tier
            search_results = self.sme_system.search_collection(
                collection_name=collection_info['name'],
                query=query.query,
                limit=5,
                similarity_threshold=query.confidence_threshold
            )

            if search_results and len(search_results) > 0:
                # Found results at this authority level
                precedence = collection_info['precedence_level']
                confidence = search_results[0].get('similarity_score', 0.0)

                # Use highest precedence response
                if precedence > highest_precedence:
                    best_response = {
                        'content': search_results[0]['content'],
                        'confidence': confidence,
                        'sources': search_results,
                        'authority_tier': collection_info['authority_tier'],
                        'precedence_level': precedence,
                        'collection_name': collection_info['name']
                    }
                    highest_precedence = precedence

                # If this is Tier 1 (regulatory), use it definitively
                if collection_info['authority_tier'] == 1:
                    break

        if best_response:
            return RAGResponse(
                response=best_response['content'],
                confidence=best_response['confidence'],
                sources=best_response['sources'],
                authority_tier=best_response['authority_tier'],
                collection_name=best_response['collection_name'],
                precedence_level=best_response['precedence_level']
            )
        else:
            return RAGResponse(
                response="No authoritative guidance found for this query.",
                confidence=0.0,
                sources=[],
                authority_tier=0,
                collection_name="none",
                precedence_level=0.0
            )

    def query_document_rag(self, query: RAGQuery) -> RAGResponse:
        """
        DocumentRAG using EXISTING sme_* tables for document retrieval.

        Simple semantic search across document collections for information discovery.
        """
        print(f"DocumentRAG Query: {query.query}")

        # Use specified collection or find domain collections
        if query.collection_name:
            search_results = self.sme_system.search_collection(
                collection_name=query.collection_name,
                query=query.query,
                limit=10,
                similarity_threshold=query.confidence_threshold
            )
            collection_name = query.collection_name
        else:
            # Search across all collections for domain
            collections = self.list_collections()
            all_results = []

            for collection in collections:
                if query.domain.lower() in collection['collection_name'].lower():
                    results = self.sme_system.search_collection(
                        collection_name=collection['collection_name'],
                        query=query.query,
                        limit=5,
                        similarity_threshold=query.confidence_threshold
                    )
                    for result in results:
                        result['source_collection'] = collection['collection_name']
                    all_results.extend(results)

            # Sort by similarity score
            search_results = sorted(all_results,
                                  key=lambda x: x.get('similarity_score', 0.0),
                                  reverse=True)[:10]
            collection_name = f"multi_collection_{query.domain}"

        if search_results:
            # Combine top results for comprehensive response
            combined_content = "\n\n".join([
                f"Source: {result.get('filename', 'unknown')}\n{result['content'][:500]}..."
                for result in search_results[:3]
            ])

            avg_confidence = sum(r.get('similarity_score', 0.0) for r in search_results) / len(search_results)

            return RAGResponse(
                response=combined_content,
                confidence=avg_confidence,
                sources=search_results,
                authority_tier=0,  # Knowledge doesn't have authority
                collection_name=collection_name,
                precedence_level=0.5
            )
        else:
            return RAGResponse(
                response="No relevant knowledge found for this query.",
                confidence=0.0,
                sources=[],
                authority_tier=0,
                collection_name=collection_name,
                precedence_level=0.0
            )

    def query_expert_rag(self, query: RAGQuery) -> RAGResponse:
        """
        ExpertRAG using EXISTING sme_* tables as-is for expert decisions.

        Leverages existing SME collections for specialized subject matter expertise.
        """
        print(f"ExpertRAG Query: {query.query}")

        # Use existing SME collections directly
        if query.collection_name:
            search_results = self.sme_system.search_collection(
                collection_name=query.collection_name,
                query=query.query,
                limit=5,
                similarity_threshold=query.confidence_threshold
            )
        else:
            # Find SME collections (those without authority tiers)
            collections = self.list_collections()
            sme_collections = []

            for collection in collections:
                settings = collection.get('settings', {})
                # Expert collections don't have authority_tier or have rag_type=expert
                if (not settings.get('authority_tier') or
                    settings.get('rag_type') == 'expert'):
                    sme_collections.append(collection)

            # Search relevant SME collections
            all_results = []
            for collection in sme_collections:
                if query.domain.lower() in collection['collection_name'].lower():
                    results = self.sme_system.search_collection(
                        collection_name=collection['collection_name'],
                        query=query.query,
                        limit=3,
                        similarity_threshold=query.confidence_threshold
                    )
                    all_results.extend(results)

            search_results = sorted(all_results,
                                  key=lambda x: x.get('similarity_score', 0.0),
                                  reverse=True)[:5]

        if search_results:
            # Expert provides specialized analysis
            expert_analysis = f"Expert Analysis: {search_results[0]['content']}"
            confidence = search_results[0].get('similarity_score', 0.0)

            return RAGResponse(
                response=expert_analysis,
                confidence=confidence,
                sources=search_results,
                authority_tier=99,  # Expert is specialized but not regulatory authority
                collection_name=query.collection_name or "expert_knowledge",
                precedence_level=0.9  # High expertise but not regulatory
            )
        else:
            return RAGResponse(
                response="No expert knowledge available for this query.",
                confidence=0.0,
                sources=[],
                authority_tier=99,
                collection_name="expert_knowledge",
                precedence_level=0.0
            )

    def query_unified_rag(self, query: RAGQuery) -> RAGResponse:
        """
        Unified query that determines best RAG type and uses existing tables.

        Logic:
        1. If authority_tier specified -> ComplianceRAG
        2. If collection_name specified -> DocumentRAG/ExpertRAG
        3. Otherwise -> Try ComplianceRAG first, fallback to DocumentRAG
        """
        print(f"Unified RAG Query: {query.query}")

        # Route to appropriate RAG type
        if query.authority_tier:
            return self.query_compliance_rag(query)
        elif query.collection_name:
            # Check if it's an Expert collection or Document collection
            collections = self.list_collections()
            target_collection = None
            for collection in collections:
                if collection['collection_name'] == query.collection_name:
                    target_collection = collection
                    break

            if target_collection:
                settings = target_collection.get('settings', {})
                if settings.get('rag_type') == 'expert' or not settings.get('authority_tier'):
                    return self.query_expert_rag(query)
                else:
                    return self.query_document_rag(query)
            else:
                return self.query_document_rag(query)
        else:
            # Try compliance first (authoritative), fallback to document
            compliance_response = self.query_compliance_rag(query)
            if compliance_response.confidence > 0.3:
                return compliance_response
            else:
                return self.query_document_rag(query)

def main():
    """Test the Postgres RAG Adapter using existing SME tables."""
    print("Testing Postgres RAG Adapter with Existing SME Tables")
    print("=" * 60)

    # Initialize adapter with existing SME system
    adapter = PostgresRAGAdapter()

    # Test 1: ComplianceRAG query
    print("\nTest 1: ComplianceRAG Query")
    compliance_query = RAGQuery(
        query="What are the architectural requirements for adapter patterns?",
        domain="software_architecture",
        authority_tier=1  # Regulatory level
    )

    response = adapter.query_compliance_rag(compliance_query)
    print(f"Authority Tier: {response.authority_tier}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Response: {response.response[:200]}...")

    # Test 2: DocumentRAG query
    print("\nTest 2: DocumentRAG Query")
    document_query = RAGQuery(
        query="How do I implement S3-first storage patterns?",
        domain="storage_architecture"
    )

    response = adapter.query_document_rag(document_query)
    print(f"Sources found: {len(response.sources)}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Response: {response.response[:200]}...")

    # Test 3: ExpertRAG query
    print("\nTest 3: ExpertRAG Query")
    expert_query = RAGQuery(
        query="What best practices should I follow for model validation?",
        domain="model_risk"
    )

    response = adapter.query_expert_rag(expert_query)
    print(f"Expert Authority: {response.authority_tier}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Response: {response.response[:200]}...")

    # Test 4: Unified query
    print("\nTest 4: Unified RAG Query")
    unified_query = RAGQuery(
        query="What compliance standards apply to model risk management?",
        domain="model_risk"
    )

    response = adapter.query_unified_rag(unified_query)
    print(f"Selected Authority Tier: {response.authority_tier}")
    print(f"Precedence Level: {response.precedence_level}")
    print(f"Response: {response.response[:200]}...")

    print("\nPostgres RAG Adapter testing complete!")

if __name__ == "__main__":
    main()