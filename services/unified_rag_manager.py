#!/usr/bin/env python3
"""
Unified RAG Manager - CRUD and HealthCheck for All 5 RAG Systems
================================================================

Provides unified interface for managing all 5 RAG systems:
1. AIPoweredRAGAdapter - AI-enhanced responses via CorporateLLMGateway
2. PostgresRAGAdapter - Authority-based precedence with SME infrastructure
3. JudgeRAGAdapter - External system integration with fallback
4. IntelligentRAGAdapter - Real content extraction with direct database
5. SMERAGSystem - Full document lifecycle with multi-model support

Features:
- Unified CRUD operations across all RAG systems
- Health monitoring and status checks
- Collection management and query routing
- Performance metrics and optimization tracking
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# USM Integration for credential management
try:
    from tidyllm.admin.credential_loader import set_aws_environment
    USM_AVAILABLE = True
except ImportError:
    USM_AVAILABLE = False

# RAG2DAG Accelerator Integration
try:
    from tidyllm.services.rag2dag import rag2dag_service, OptimizationResult, OptimizationSuggestion
    RAG2DAG_AVAILABLE = True
except ImportError:
    RAG2DAG_AVAILABLE = False

# Import all 5 RAG systems
try:
    from tidyllm.knowledge_systems.adapters.ai_powered.ai_powered_rag_adapter import AIPoweredRAGAdapter
    AI_POWERED_AVAILABLE = True
except ImportError:
    AI_POWERED_AVAILABLE = False

try:
    from tidyllm.knowledge_systems.adapters.postgres_rag.postgres_rag_adapter import PostgresRAGAdapter, RAGQuery, RAGResponse
    POSTGRES_RAG_AVAILABLE = True
except ImportError:
    POSTGRES_RAG_AVAILABLE = False

try:
    from tidyllm.knowledge_systems.adapters.judge_rag.judge_rag_adapter import JudgeRAGAdapter
    JUDGE_RAG_AVAILABLE = True
except ImportError:
    JUDGE_RAG_AVAILABLE = False

try:
    from tidyllm.knowledge_systems.adapters.intelligent.intelligent_rag_adapter import IntelligentRAGAdapter
    INTELLIGENT_RAG_AVAILABLE = True
except ImportError:
    INTELLIGENT_RAG_AVAILABLE = False

try:
    from tidyllm.knowledge_systems.adapters.sme_rag.sme_rag_system import SMERAGSystem, EmbeddingModel
    SME_RAG_AVAILABLE = True
except ImportError:
    SME_RAG_AVAILABLE = False

# DSPy Gateway Integration
try:
    # Import from the discovered DSPy gateway locations
    import dspy
    DSPY_FRAMEWORK_AVAILABLE = True
except ImportError:
    DSPY_FRAMEWORK_AVAILABLE = False

logger = logging.getLogger(__name__)


# DSPy Service Import
try:
    from .dspy_service import DSPyService
    DSPY_SERVICE_AVAILABLE = True
except ImportError:
    DSPY_SERVICE_AVAILABLE = False


class RAGSystemType(Enum):
    """Available RAG system types."""
    AI_POWERED = "ai_powered"
    POSTGRES = "postgres"
    JUDGE = "judge"
    INTELLIGENT = "intelligent"
    SME = "sme"
    DSPY = "dspy"  # DSPy Gateway for prompt optimization and signature engineering

@dataclass
class RAGSystemInfo:
    """Information about a RAG system instance."""
    system_type: RAGSystemType
    system_id: str
    name: str
    description: str
    status: str  # healthy, degraded, error, unavailable
    collections_count: int
    last_updated: datetime
    capabilities: List[str]
    health_score: float  # 0.0 - 1.0
    metadata: Dict[str, Any]

@dataclass
class UnifiedRAGQuery:
    """Unified query structure for all RAG systems."""
    query: str
    domain: str
    system_type: Optional[RAGSystemType] = None
    collection_id: Optional[str] = None
    authority_tier: Optional[int] = None
    confidence_threshold: float = 0.7
    max_results: int = 5
    enable_rag2dag_optimization: bool = True  # Enable RAG2DAG acceleration
    source_files: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class UnifiedRAGResponse:
    """Unified response structure from all RAG systems."""
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    system_type: RAGSystemType
    system_id: str
    collection_id: str
    authority_tier: int
    precedence_level: float
    processing_time_ms: float
    rag2dag_optimized: bool = False  # Whether RAG2DAG optimization was applied
    optimization_result: Optional[Dict[str, Any]] = None  # RAG2DAG optimization details
    metadata: Dict[str, Any] = None

class UnifiedRAGManager:
    """
    Unified manager for all 5 RAG systems with CRUD and HealthCheck operations.
    """

    def __init__(self, auto_load_credentials: bool = True):
        """Initialize unified RAG manager."""
        print("🚀 Initializing Unified RAG Manager...")

        # Auto-load USM credentials
        if auto_load_credentials and USM_AVAILABLE:
            try:
                set_aws_environment()
                logger.info("USM credentials loaded for RAG manager")
            except Exception as e:
                logger.warning(f"USM credential loading failed: {e}")

        # Initialize all available RAG systems
        self.rag_systems = {}
        self._initialize_rag_systems()

        # System registry
        self.system_registry = {
            RAGSystemType.AI_POWERED: {
                "name": "AI-Powered RAG",
                "description": "AI-enhanced responses via CorporateLLMGateway + Bedrock analysis",
                "available": AI_POWERED_AVAILABLE
            },
            RAGSystemType.POSTGRES: {
                "name": "Postgres RAG",
                "description": "Authority-based precedence with existing SME infrastructure",
                "available": POSTGRES_RAG_AVAILABLE
            },
            RAGSystemType.JUDGE: {
                "name": "Judge RAG",
                "description": "External system integration with transparent fallback mechanisms",
                "available": JUDGE_RAG_AVAILABLE
            },
            RAGSystemType.INTELLIGENT: {
                "name": "Intelligent RAG",
                "description": "Real content extraction with direct database operations",
                "available": INTELLIGENT_RAG_AVAILABLE
            },
            RAGSystemType.SME: {
                "name": "SME RAG System",
                "description": "Full document lifecycle with multi-model embedding support",
                "available": SME_RAG_AVAILABLE
            },
            RAGSystemType.DSPY: {
                "name": "DSPy Gateway",
                "description": "Prompt engineering and signature optimization with DSPy framework",
                "available": DSPY_FRAMEWORK_AVAILABLE
            }
        }

    def _initialize_rag_systems(self):
        """Initialize all available RAG systems."""

        # Initialize AI-Powered RAG
        if AI_POWERED_AVAILABLE:
            try:
                self.rag_systems[RAGSystemType.AI_POWERED] = AIPoweredRAGAdapter()
                logger.info("AI-Powered RAG adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AI-Powered RAG: {e}")

        # Initialize Postgres RAG
        if POSTGRES_RAG_AVAILABLE:
            try:
                self.rag_systems[RAGSystemType.POSTGRES] = PostgresRAGAdapter()
                logger.info("Postgres RAG adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Postgres RAG: {e}")

        # Initialize Judge RAG
        if JUDGE_RAG_AVAILABLE:
            try:
                self.rag_systems[RAGSystemType.JUDGE] = JudgeRAGAdapter()
                logger.info("Judge RAG adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Judge RAG: {e}")

        # Initialize Intelligent RAG
        if INTELLIGENT_RAG_AVAILABLE:
            try:
                self.rag_systems[RAGSystemType.INTELLIGENT] = IntelligentRAGAdapter()
                logger.info("Intelligent RAG adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Intelligent RAG: {e}")

        # Initialize SME RAG
        if SME_RAG_AVAILABLE:
            try:
                self.rag_systems[RAGSystemType.SME] = SMERAGSystem()
                logger.info("SME RAG system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SME RAG: {e}")

        # Initialize DSPy Service
        if DSPY_SERVICE_AVAILABLE:
            try:
                self.rag_systems[RAGSystemType.DSPY] = DSPyService()
                logger.info("DSPy Service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DSPy Service: {e}")

        logger.info(f"Initialized {len(self.rag_systems)}/6 RAG systems")

        # Initialize RAG2DAG acceleration if available
        self.rag2dag_enabled = RAG2DAG_AVAILABLE
        if self.rag2dag_enabled:
            logger.info("RAG2DAG acceleration service available")
        else:
            logger.warning("RAG2DAG acceleration service not available")

    # ============================================================================
    # HEALTH CHECK OPERATIONS
    # ============================================================================

    async def health_check_all(self) -> Dict[RAGSystemType, Dict[str, Any]]:
        """Perform health check on all RAG systems."""
        health_results = {}

        for system_type, system in self.rag_systems.items():
            try:
                health_results[system_type] = await self._health_check_system(system_type, system)
            except Exception as e:
                health_results[system_type] = {
                    "status": "error",
                    "error": str(e),
                    "health_score": 0.0,
                    "timestamp": datetime.now().isoformat()
                }

        return health_results

    async def _health_check_system(self, system_type: RAGSystemType, system) -> Dict[str, Any]:
        """Health check for individual RAG system."""
        start_time = datetime.now()

        try:
            if system_type == RAGSystemType.AI_POWERED:
                # Check AI-Powered RAG health
                result = {
                    "status": "healthy" if hasattr(system, 'corporate_llm_gateway') else "degraded",
                    "collections": 0,  # Would need to implement collection counting
                    "capabilities": ["ai_analysis", "corporate_context", "bedrock_integration"]
                }

            elif system_type == RAGSystemType.POSTGRES:
                # Check Postgres RAG health
                collections = system.list_collections()
                result = {
                    "status": "healthy",
                    "collections": len(collections),
                    "capabilities": ["authority_based", "compliance_rag", "expert_rag", "document_rag"]
                }

            elif system_type == RAGSystemType.JUDGE:
                # Check Judge RAG health (async)
                judge_health = await system.health_check()
                result = {
                    "status": "healthy" if judge_health.get("jb_system_available") else "fallback",
                    "collections": 0,  # External system
                    "capabilities": ["external_integration", "fallback_support", "hybrid_queries"],
                    "external_status": judge_health
                }

            elif system_type == RAGSystemType.INTELLIGENT:
                # Check Intelligent RAG health
                collections = system.list_collections()
                result = {
                    "status": "healthy",
                    "collections": len(collections),
                    "capabilities": ["pdf_extraction", "smart_chunking", "vector_similarity", "bedrock_embeddings"]
                }

            elif system_type == RAGSystemType.SME:
                # Check SME RAG health
                collections = system.get_collections()
                result = {
                    "status": "healthy",
                    "collections": len(collections),
                    "capabilities": ["document_lifecycle", "multi_model_embeddings", "s3_storage", "reindexing"]
                }

            else:
                result = {"status": "unknown", "collections": 0, "capabilities": []}

            # Calculate health score
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            health_score = min(1.0, max(0.0, (1000 - response_time) / 1000)) if result["status"] == "healthy" else 0.5

            result.update({
                "health_score": health_score,
                "response_time_ms": response_time,
                "timestamp": datetime.now().isoformat()
            })

            return result

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "health_score": 0.0,
                "response_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "timestamp": datetime.now().isoformat()
            }

    def get_system_status(self) -> List[RAGSystemInfo]:
        """Get status of all RAG systems."""
        system_infos = []

        for system_type in RAGSystemType:
            registry_info = self.system_registry[system_type]

            if system_type in self.rag_systems:
                # System is available and initialized
                try:
                    if system_type == RAGSystemType.POSTGRES:
                        collections = self.rag_systems[system_type].list_collections()
                        collections_count = len(collections)
                    elif system_type == RAGSystemType.INTELLIGENT:
                        collections = self.rag_systems[system_type].list_collections()
                        collections_count = len(collections)
                    elif system_type == RAGSystemType.SME:
                        collections = self.rag_systems[system_type].get_collections()
                        collections_count = len(collections)
                    else:
                        collections_count = 0  # Unknown for AI-Powered and Judge

                    status = "healthy"
                    health_score = 1.0
                except Exception as e:
                    collections_count = 0
                    status = "error"
                    health_score = 0.0
            else:
                collections_count = 0
                status = "unavailable" if registry_info["available"] else "disabled"
                health_score = 0.0

            system_info = RAGSystemInfo(
                system_type=system_type,
                system_id=f"rag_{system_type.value}",
                name=registry_info["name"],
                description=registry_info["description"],
                status=status,
                collections_count=collections_count,
                last_updated=datetime.now(),
                capabilities=self._get_system_capabilities(system_type),
                health_score=health_score,
                metadata={"available": registry_info["available"]}
            )

            system_infos.append(system_info)

        return system_infos

    def _get_system_capabilities(self, system_type: RAGSystemType) -> List[str]:
        """Get capabilities for a specific RAG system type."""
        capabilities_map = {
            RAGSystemType.AI_POWERED: ["ai_analysis", "corporate_context", "bedrock_integration", "session_continuity"],
            RAGSystemType.POSTGRES: ["authority_based", "compliance_rag", "expert_rag", "document_rag", "precedence_resolution"],
            RAGSystemType.JUDGE: ["external_integration", "fallback_support", "hybrid_queries", "zero_maintenance"],
            RAGSystemType.INTELLIGENT: ["pdf_extraction", "smart_chunking", "vector_similarity", "bedrock_embeddings", "real_content"],
            RAGSystemType.SME: ["document_lifecycle", "multi_model_embeddings", "s3_storage", "reindexing", "legacy_integration"]
        }
        return capabilities_map.get(system_type, [])

    # ============================================================================
    # CRUD OPERATIONS
    # ============================================================================

    def list_all_collections(self) -> Dict[RAGSystemType, List[Dict[str, Any]]]:
        """List collections from all RAG systems."""
        all_collections = {}

        for system_type, system in self.rag_systems.items():
            try:
                if system_type == RAGSystemType.POSTGRES:
                    collections = system.list_collections()
                elif system_type == RAGSystemType.INTELLIGENT:
                    collections = system.list_collections()
                elif system_type == RAGSystemType.SME:
                    collections = [asdict(col) for col in system.get_collections()]
                else:
                    collections = []  # AI-Powered and Judge don't have traditional collections

                all_collections[system_type] = collections

            except Exception as e:
                logger.error(f"Error listing collections for {system_type}: {e}")
                all_collections[system_type] = []

        return all_collections

    def create_collection(self,
                         system_type: RAGSystemType,
                         name: str,
                         description: str,
                         config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new collection in specified RAG system."""

        if system_type not in self.rag_systems:
            return {"success": False, "error": f"RAG system {system_type} not available"}

        system = self.rag_systems[system_type]
        config = config or {}

        try:
            if system_type == RAGSystemType.POSTGRES:
                # Create authority collection
                domain = config.get("domain", "general")
                authority_tier = config.get("authority_tier", 2)
                collection_id = system.get_or_create_authority_collection(domain, authority_tier, description)

            elif system_type == RAGSystemType.INTELLIGENT:
                # Create intelligent collection
                domain = config.get("domain", "general")
                authority_tier = config.get("authority_tier", 2)
                collection_id = system.get_or_create_authority_collection(domain, authority_tier, description)

            elif system_type == RAGSystemType.SME:
                # Create SME collection
                embedding_model = EmbeddingModel(config.get("embedding_model", "text-embedding-ada-002"))
                s3_bucket = config.get("s3_bucket", "dsai-2025-asu")
                s3_prefix = config.get("s3_prefix", f"collections/{name}")
                tags = config.get("tags", [])
                settings = config.get("settings", {})

                collection_id = system.create_collection(
                    name=name,
                    description=description,
                    embedding_model=embedding_model,
                    s3_bucket=s3_bucket,
                    s3_prefix=s3_prefix,
                    tags=tags,
                    settings=settings
                )

            else:
                return {"success": False, "error": f"Collection creation not supported for {system_type}"}

            return {
                "success": True,
                "collection_id": collection_id,
                "system_type": system_type.value,
                "message": f"Collection '{name}' created successfully in {system_type.value}"
            }

        except Exception as e:
            logger.error(f"Error creating collection in {system_type}: {e}")
            return {"success": False, "error": str(e)}

    def delete_collection(self,
                         system_type: RAGSystemType,
                         collection_id: str) -> Dict[str, Any]:
        """Delete a collection from specified RAG system."""

        if system_type not in self.rag_systems:
            return {"success": False, "error": f"RAG system {system_type} not available"}

        try:
            # Most systems don't have explicit delete methods, but SME does
            if system_type == RAGSystemType.SME:
                # SME system has delete capability
                system = self.rag_systems[system_type]
                # Would need to implement delete_collection method in SMERAGSystem
                return {"success": False, "error": "Delete not implemented in SME system yet"}
            else:
                return {"success": False, "error": f"Collection deletion not supported for {system_type}"}

        except Exception as e:
            logger.error(f"Error deleting collection from {system_type}: {e}")
            return {"success": False, "error": str(e)}

    # ============================================================================
    # RAG2DAG OPTIMIZATION OPERATIONS
    # ============================================================================

    def analyze_query_optimization(self, query: UnifiedRAGQuery) -> Optional[OptimizationResult]:
        """Analyze query for RAG2DAG optimization opportunities."""
        if not self.rag2dag_enabled or not query.enable_rag2dag_optimization:
            return None

        try:
            context = f"Domain: {query.domain}"
            if query.metadata:
                context += f", Metadata: {query.metadata}"

            optimization_result = rag2dag_service.analyze_request_optimization(
                request=query.query,
                context=context,
                source_files=query.source_files or []
            )

            logger.info(f"RAG2DAG Analysis: {optimization_result.optimization_reason}")
            return optimization_result

        except Exception as e:
            logger.error(f"RAG2DAG optimization analysis failed: {e}")
            return None

    def get_optimization_suggestions(self, workflow_description: str, expected_load: str = "medium") -> Optional[OptimizationSuggestion]:
        """Get RAG2DAG optimization suggestions for workflow design."""
        if not self.rag2dag_enabled:
            return None

        try:
            return rag2dag_service.suggest_workflow_optimization(workflow_description, expected_load)
        except Exception as e:
            logger.error(f"RAG2DAG suggestion failed: {e}")
            return None

    def execute_optimized_workflow(self, dag_workflow: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a RAG2DAG optimized workflow."""
        if not self.rag2dag_enabled:
            return {"error": "RAG2DAG service not available", "success": False}

        try:
            return rag2dag_service.execute_optimized_workflow(dag_workflow, input_data)
        except Exception as e:
            logger.error(f"RAG2DAG workflow execution failed: {e}")
            return {"error": str(e), "success": False}

    def get_rag2dag_stats(self) -> Dict[str, Any]:
        """Get RAG2DAG service statistics."""
        if not self.rag2dag_enabled:
            return {"available": False, "reason": "RAG2DAG service not available"}

        try:
            stats = rag2dag_service.get_service_stats()
            return {"available": True, **stats}
        except Exception as e:
            logger.error(f"Failed to get RAG2DAG stats: {e}")
            return {"available": False, "error": str(e)}

    def should_use_rag2dag_for_query(self, query: UnifiedRAGQuery) -> Dict[str, Any]:
        """Analyze whether RAG2DAG optimization would benefit this specific query."""
        if not self.rag2dag_enabled:
            return {
                "should_use": False,
                "reason": "RAG2DAG service not available",
                "suitable_systems": [],
                "optimization_potential": 0.0
            }

        # Define which systems benefit most from RAG2DAG
        rag2dag_benefits = {
            RAGSystemType.AI_POWERED: {
                "benefit_score": 0.9,
                "reason": "Complex AI workflows can be parallelized and optimized"
            },
            RAGSystemType.SME: {
                "benefit_score": 0.8,
                "reason": "Document lifecycle processing benefits from DAG optimization"
            },
            RAGSystemType.POSTGRES: {
                "benefit_score": 0.3,
                "reason": "Authority-based queries are already optimized for sequential processing"
            },
            RAGSystemType.JUDGE: {
                "benefit_score": 0.2,
                "reason": "External system integration doesn't benefit from internal DAG optimization"
            },
            RAGSystemType.INTELLIGENT: {
                "benefit_score": 0.4,
                "reason": "PDF extraction can benefit from parallel processing of multiple documents"
            }
        }

        # Calculate optimization potential
        optimization_factors = []

        # Multi-document processing
        if query.source_files and len(query.source_files) > 2:
            optimization_factors.append(("Multi-document processing", 0.7))

        # Complex queries (longer text = more complex analysis)
        if len(query.query.split()) > 20:
            optimization_factors.append(("Complex query analysis", 0.5))

        # Multi-system queries
        if query.system_type is None:
            optimization_factors.append(("Multi-system query routing", 0.6))

        # System-specific benefits
        if query.system_type and query.system_type in rag2dag_benefits:
            system_benefit = rag2dag_benefits[query.system_type]
            optimization_factors.append((f"{query.system_type.value} system optimization", system_benefit["benefit_score"]))

        # Calculate overall potential
        if optimization_factors:
            optimization_potential = sum(score for _, score in optimization_factors) / len(optimization_factors)
        else:
            optimization_potential = 0.0

        should_use = optimization_potential > 0.5

        return {
            "should_use": should_use,
            "optimization_potential": optimization_potential,
            "factors": optimization_factors,
            "suitable_systems": [sys.value for sys, info in rag2dag_benefits.items() if info["benefit_score"] > 0.5],
            "recommendation": (
                "RAG2DAG optimization recommended" if should_use else
                "Standard RAG processing recommended"
            ),
            "system_benefits": rag2dag_benefits
        }

    # ============================================================================
    # QUERY OPERATIONS
    # ============================================================================

    async def query_unified(self, query: UnifiedRAGQuery) -> UnifiedRAGResponse:
        """Query across RAG systems with unified interface and optional RAG2DAG optimization."""
        start_time = datetime.now()

        # Step 1: Analyze for RAG2DAG optimization if enabled (only for suitable systems)
        optimization_result = None
        rag2dag_optimized = False

        # RAG2DAG is most beneficial for:
        # - AI-Powered RAG (already uses complex workflows)
        # - SME RAG (document lifecycle with multiple models)
        # - DSPy Gateway (signature optimization workflows)
        # - Multi-system queries (can parallelize across systems)
        rag2dag_suitable_systems = {RAGSystemType.AI_POWERED, RAGSystemType.SME, RAGSystemType.DSPY}

        should_use_rag2dag = (
            query.enable_rag2dag_optimization and
            self.rag2dag_enabled and
            (query.system_type is None or  # Multi-system query
             query.system_type in rag2dag_suitable_systems or
             (query.source_files and len(query.source_files) > 2))  # Multi-document processing
        )

        if should_use_rag2dag:
            optimization_result = self.analyze_query_optimization(query)

            if (optimization_result and
                optimization_result.should_optimize and
                optimization_result.dag_workflow):

                # Execute optimized DAG workflow
                try:
                    input_data = {
                        "query": query.query,
                        "domain": query.domain,
                        "confidence_threshold": query.confidence_threshold,
                        "max_results": query.max_results,
                        "source_files": query.source_files or []
                    }

                    dag_result = self.execute_optimized_workflow(
                        optimization_result.dag_workflow,
                        input_data
                    )

                    if dag_result.get("success"):
                        processing_time = (datetime.now() - start_time).total_seconds() * 1000

                        return UnifiedRAGResponse(
                            response=dag_result.get("final_output", "RAG2DAG optimized response"),
                            confidence=dag_result.get("confidence", 0.9),
                            sources=dag_result.get("sources", []),
                            system_type=RAGSystemType.AI_POWERED,  # RAG2DAG uses AI processing
                            system_id="rag2dag_optimized",
                            collection_id="rag2dag_workflow",
                            authority_tier=optimization_result.pattern_detected.value if optimization_result.pattern_detected else 10,
                            precedence_level=0.95,  # High precedence for optimized results
                            processing_time_ms=processing_time,
                            rag2dag_optimized=True,
                            optimization_result={
                                "pattern_detected": optimization_result.pattern_detected.value if optimization_result.pattern_detected else None,
                                "estimated_speedup": optimization_result.estimated_speedup,
                                "optimization_reason": optimization_result.optimization_reason,
                                "confidence_score": optimization_result.confidence_score
                            },
                            metadata={"rag2dag_workflow_nodes": len(optimization_result.dag_workflow.get("nodes", []))}
                        )

                except Exception as e:
                    logger.warning(f"RAG2DAG optimization failed, falling back to standard RAG: {e}")

        # Step 2: Standard RAG system queries (fallback or when optimization not applied)
        # Determine which system(s) to query
        if query.system_type:
            # Query specific system
            systems_to_query = [query.system_type]
        else:
            # Query all available systems and return best result
            systems_to_query = list(self.rag_systems.keys())

        best_response = None
        best_confidence = 0.0

        for system_type in systems_to_query:
            if system_type not in self.rag_systems:
                continue

            try:
                response = await self._query_system(system_type, query)

                # Track best response by confidence and authority
                if (response.confidence > best_confidence or
                    (response.confidence == best_confidence and response.authority_tier < (best_response.authority_tier if best_response else 999))):
                    best_response = response
                    best_confidence = response.confidence

            except Exception as e:
                logger.error(f"Error querying {system_type}: {e}")
                continue

        if best_response:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            best_response.processing_time_ms = processing_time
            return best_response
        else:
            # Return empty response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return UnifiedRAGResponse(
                response="No results found across available RAG systems.",
                confidence=0.0,
                sources=[],
                system_type=RAGSystemType.POSTGRES,  # Default
                system_id="unified_manager",
                collection_id="none",
                authority_tier=0,
                precedence_level=0.0,
                processing_time_ms=processing_time,
                metadata={"systems_queried": [st.value for st in systems_to_query]}
            )

    async def _query_system(self, system_type: RAGSystemType, query: UnifiedRAGQuery) -> UnifiedRAGResponse:
        """Query individual RAG system."""
        system = self.rag_systems[system_type]

        if system_type == RAGSystemType.POSTGRES:
            # Convert to RAGQuery format
            rag_query = RAGQuery(
                query=query.query,
                domain=query.domain,
                authority_tier=query.authority_tier,
                confidence_threshold=query.confidence_threshold
            )
            response = system.query_unified_rag(rag_query)

        elif system_type == RAGSystemType.JUDGE:
            # Judge RAG query (async)
            rag_query = RAGQuery(
                query=query.query,
                domain=query.domain,
                confidence_threshold=query.confidence_threshold
            )
            response = system.query_judge_rag(rag_query)

        elif system_type == RAGSystemType.INTELLIGENT:
            # Intelligent RAG query
            rag_query = RAGQuery(
                query=query.query,
                domain=query.domain,
                authority_tier=query.authority_tier,
                confidence_threshold=query.confidence_threshold
            )
            response = system.query_unified_rag(rag_query)

        else:
            # Default response for unsupported systems
            response = RAGResponse(
                response=f"Query processed by {system_type.value} system: {query.query}",
                confidence=0.5,
                sources=[],
                authority_tier=50,
                collection_name=f"{system_type.value}_collection",
                precedence_level=0.5
            )

        # Convert to unified response
        return UnifiedRAGResponse(
            response=response.response,
            confidence=response.confidence,
            sources=response.sources,
            system_type=system_type,
            system_id=f"rag_{system_type.value}",
            collection_id=response.collection_name,
            authority_tier=response.authority_tier,
            precedence_level=response.precedence_level,
            processing_time_ms=0.0,  # Will be set by caller
            metadata={"original_system": system_type.value}
        )

    # ============================================================================
    # MANAGEMENT OPERATIONS
    # ============================================================================

    def get_available_systems(self) -> List[Dict[str, Any]]:
        """Get list of available RAG systems."""
        systems = []

        for system_type, registry_info in self.system_registry.items():
            is_initialized = system_type in self.rag_systems

            systems.append({
                "system_type": system_type.value,
                "name": registry_info["name"],
                "description": registry_info["description"],
                "available": registry_info["available"],
                "initialized": is_initialized,
                "capabilities": self._get_system_capabilities(system_type)
            })

        return systems

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics across all RAG systems."""
        metrics = {
            "total_systems": len(self.system_registry),
            "available_systems": len(self.rag_systems),
            "total_collections": 0,
            "system_breakdown": {}
        }

        for system_type, system in self.rag_systems.items():
            try:
                if system_type == RAGSystemType.POSTGRES:
                    collections = system.list_collections()
                    collection_count = len(collections)
                elif system_type == RAGSystemType.INTELLIGENT:
                    collections = system.list_collections()
                    collection_count = len(collections)
                elif system_type == RAGSystemType.SME:
                    collections = system.get_collections()
                    collection_count = len(collections)
                else:
                    collection_count = 0

                metrics["total_collections"] += collection_count
                metrics["system_breakdown"][system_type.value] = {
                    "collections": collection_count,
                    "status": "healthy"
                }

            except Exception as e:
                metrics["system_breakdown"][system_type.value] = {
                    "collections": 0,
                    "status": "error",
                    "error": str(e)
                }

        return metrics

def main():
    """Test the Unified RAG Manager."""
    print("🧪 Testing Unified RAG Manager")
    print("=" * 50)

    # Initialize manager
    manager = UnifiedRAGManager()

    # Test 1: System Status
    print("\n📊 Test 1: System Status")
    system_infos = manager.get_system_status()
    for info in system_infos:
        print(f"  {info.name}: {info.status} ({info.collections_count} collections)")

    # Test 2: Health Check
    print("\n🏥 Test 2: Health Check")
    health_results = asyncio.run(manager.health_check_all())
    for system_type, health in health_results.items():
        print(f"  {system_type.value}: {health['status']} (Score: {health['health_score']:.2f})")

    # Test 3: List Collections
    print("\n📋 Test 3: List All Collections")
    all_collections = manager.list_all_collections()
    for system_type, collections in all_collections.items():
        print(f"  {system_type.value}: {len(collections)} collections")

    # Test 4: Unified Query
    print("\n🔍 Test 4: Unified Query")
    query = UnifiedRAGQuery(
        query="What are the best practices for model validation?",
        domain="model_risk",
        confidence_threshold=0.7
    )

    response = asyncio.run(manager.query_unified(query))
    print(f"  Best Response from: {response.system_type.value}")
    print(f"  Confidence: {response.confidence:.2f}")
    print(f"  Response: {response.response[:100]}...")

    # Test 5: Performance Metrics
    print("\n📈 Test 5: Performance Metrics")
    metrics = manager.get_performance_metrics()
    print(f"  Total Systems: {metrics['total_systems']}")
    print(f"  Available: {metrics['available_systems']}")
    print(f"  Total Collections: {metrics['total_collections']}")

    print("\n✅ Unified RAG Manager testing complete!")

if __name__ == "__main__":
    main()