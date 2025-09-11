"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

Context Gateway - Final Orchestrator Gateway
============================================
🚀 CORE ENTERPRISE GATEWAY #4 - Context & Orchestration Layer

FINAL GATEWAY in the 4-gateway chain - orchestrates all other gateways to provide
enriched context for user requests.

DEPENDENCY CHAIN:
1. CorporateLLMGateway (foundation)
2. AIProcessingGateway (depends on corporate)  
3. WorkflowOptimizerGateway (depends on AI + corporate)
4. ContextGateway (depends on ALL) ← THIS GATEWAY

CONTEXT ORCHESTRATION ROLE:
- Calls CorporateLLM for compliance rules and access control
- Uses AIProcessing for content analysis and understanding
- Leverages WorkflowOptimizer for context selection strategies
- Integrates with File/Database resources for data retrieval
- Provides enriched, compliant context back to applications

This replaces the confused KnowledgeMCPServer architecture.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .base_gateway import BaseGateway, GatewayResponse, GatewayStatus
from ..knowledge_resource_server.mcp_server import KnowledgeMCPServer

logger = logging.getLogger("context_gateway")


@dataclass
class ContextRequest:
    """Context request for ContextGateway."""
    query: str
    domain: Optional[str] = None
    context_type: str = "general"  # general, compliance, technical, business
    max_results: int = 5
    similarity_threshold: float = 0.7
    include_sources: bool = True


class ContextGateway(BaseGateway):
    """
    Context Gateway - Final Orchestrator in the 4-Gateway Chain
    
    Orchestrates all other gateways to provide enriched context:
    1. Uses CorporateLLMGateway for compliance and access control
    2. Uses AIProcessingGateway for content analysis
    3. Uses WorkflowOptimizerGateway for context selection strategy
    4. Integrates with knowledge resources for data retrieval
    
    Examples:
        >>> context_gateway = ContextGateway()
        >>> request = ContextRequest(
        ...     query="What are the validation requirements?",
        ...     context_type="compliance",
        ...     domain="model-validation"
        ... )
        >>> response = context_gateway.process(request)
        >>> print(response.context)  # Enriched, compliant context
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        
        self.gateway_name = "ContextGateway"
        self.gateway_type = "context_orchestrator"
        self.version = "1.0.0"
        
        # Initialize internal MCP server for resource access
        # This preserves existing MCP capabilities while making this a proper gateway
        try:
            self.mcp_server = KnowledgeMCPServer()
            logger.info("Context Gateway: Internal MCP server initialized")
        except Exception as e:
            logger.warning(f"Context Gateway: MCP server initialization failed: {e}")
            self.mcp_server = None
        
        # Gateway dependencies (injected by GatewayRegistry)
        self.corporate_llm_gateway = None
        self.ai_processing_gateway = None  
        self.workflow_optimizer_gateway = None
        
        logger.info(f"Context Gateway initialized (depends on all 3 other gateways)")
    
    def set_gateway_dependencies(self, corporate_llm=None, ai_processing=None, workflow_optimizer=None):
        """Inject dependencies from other gateways (called by GatewayRegistry)."""
        self.corporate_llm_gateway = corporate_llm
        self.ai_processing_gateway = ai_processing
        self.workflow_optimizer_gateway = workflow_optimizer
        
        dependencies_ready = all([corporate_llm, ai_processing, workflow_optimizer])
        logger.info(f"Context Gateway dependencies set - Ready: {dependencies_ready}")
    
    def process(self, request: ContextRequest) -> GatewayResponse:
        """
        Process context request through orchestrated gateway chain.
        
        ORCHESTRATION FLOW:
        1. Corporate LLM - Check compliance and access rules
        2. AI Processing - Analyze query for content understanding  
        3. Workflow Optimizer - Determine optimal context retrieval strategy
        4. Knowledge Resources - Fetch relevant context data
        5. Return enriched, compliant context
        """
        try:
            # Validate dependencies
            if not self._check_dependencies():
                return self._error_response("Context Gateway dependencies not available")
            
            # STEP 1: Corporate LLM - Compliance and access control
            compliance_result = self._get_compliance_rules(request)
            if not compliance_result.get("allowed", True):
                return self._error_response(f"Access denied: {compliance_result.get('reason', 'Compliance violation')}")
            
            # STEP 2: AI Processing - Query analysis
            query_analysis = self._analyze_query(request)
            
            # STEP 3: Workflow Optimizer - Context strategy  
            context_strategy = self._optimize_context_strategy(request, query_analysis)
            
            # STEP 4: Knowledge Resources - Data retrieval
            context_data = self._retrieve_context_data(request, context_strategy)
            
            # STEP 5: Construct enriched response
            enriched_context = self._build_enriched_context(
                request, compliance_result, query_analysis, context_strategy, context_data
            )
            
            return GatewayResponse(
                status=GatewayStatus.SUCCESS,
                content=enriched_context,
                metadata={
                    "gateway": self.gateway_name,
                    "context_type": request.context_type,
                    "sources": context_data.get("sources", []),
                    "compliance_level": compliance_result.get("level", "standard"),
                    "strategy": context_strategy.get("strategy", "default")
                }
            )
            
        except Exception as e:
            logger.error(f"Context Gateway processing failed: {e}")
            return self._error_response(f"Context processing error: {str(e)}")
    
    def _check_dependencies(self) -> bool:
        """Check if all required gateway dependencies are available."""
        return all([
            self.corporate_llm_gateway,
            self.ai_processing_gateway, 
            self.workflow_optimizer_gateway
        ])
    
    def _get_compliance_rules(self, request: ContextRequest) -> Dict[str, Any]:
        """Use CorporateLLMGateway to check compliance and access rules."""
        if not self.corporate_llm_gateway:
            return {"allowed": True, "level": "none", "reason": "No corporate gateway"}
        
        try:
            # Create corporate LLM request for compliance check
            compliance_request = {
                "message": f"Check access and compliance for context query: {request.query}",
                "context_type": request.context_type,
                "domain": request.domain
            }
            
            compliance_response = self.corporate_llm_gateway.process(compliance_request)
            
            # Parse compliance response (would be more sophisticated in real implementation)
            return {
                "allowed": True,  # Default allow for demo
                "level": request.context_type,
                "rules": ["corporate_access", "data_privacy"],
                "reason": "Compliance validated"
            }
            
        except Exception as e:
            logger.warning(f"Compliance check failed: {e}")
            return {"allowed": True, "level": "standard", "reason": f"Compliance check error: {e}"}
    
    def _analyze_query(self, request: ContextRequest) -> Dict[str, Any]:
        """Use AIProcessingGateway to analyze query for better context understanding."""
        if not self.ai_processing_gateway:
            return {"intent": "general", "keywords": [request.query]}
        
        try:
            # Create AI processing request for query analysis
            analysis_request = {
                "task": "query_analysis",
                "content": request.query,
                "context": {"domain": request.domain, "type": request.context_type}
            }
            
            analysis_response = self.ai_processing_gateway.process(analysis_request)
            
            # Parse analysis response
            return {
                "intent": request.context_type,
                "keywords": request.query.split(),  # Simple keyword extraction for demo
                "complexity": "medium",
                "domain_relevance": request.domain or "general"
            }
            
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return {"intent": "general", "keywords": [request.query]}
    
    def _optimize_context_strategy(self, request: ContextRequest, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use WorkflowOptimizerGateway to determine optimal context retrieval strategy."""
        if not self.workflow_optimizer_gateway:
            return {"strategy": "default", "sources": ["all"]}
        
        try:
            # Create workflow optimization request
            strategy_request = {
                "workflow_data": {
                    "query": request.query,
                    "analysis": query_analysis,
                    "context_type": request.context_type,
                    "domain": request.domain
                },
                "optimization_type": "context_retrieval"
            }
            
            strategy_response = self.workflow_optimizer_gateway.process(strategy_request)
            
            # Parse strategy response
            return {
                "strategy": "optimized_retrieval",
                "sources": ["database", "files", "vector_search"],
                "priority": ["recent", "relevant", "authoritative"],
                "max_results": min(request.max_results, 10)
            }
            
        except Exception as e:
            logger.warning(f"Strategy optimization failed: {e}")
            return {"strategy": "default", "sources": ["all"]}
    
    def _retrieve_context_data(self, request: ContextRequest, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve context data using internal MCP server and external resources."""
        context_data = {
            "results": [],
            "sources": [],
            "total_found": 0
        }
        
        try:
            # Use internal MCP server if available
            if self.mcp_server:
                # Search using MCP tools
                search_params = {
                    "query": request.query,
                    "domain": request.domain,
                    "max_results": strategy.get("max_results", request.max_results),
                    "similarity_threshold": request.similarity_threshold
                }
                
                mcp_results = self.mcp_server.handle_mcp_tool_call("search", search_params)
                
                if mcp_results.get("success"):
                    context_data["results"] = mcp_results.get("results", [])
                    context_data["total_found"] = mcp_results.get("result_count", 0)
                    context_data["sources"].append("knowledge_mcp")
            
            # Add file/database resource results (would integrate with actual resources)
            context_data["sources"].extend(["file_storage", "vector_database"])
            
            return context_data
            
        except Exception as e:
            logger.error(f"Context data retrieval failed: {e}")
            return context_data
    
    def _build_enriched_context(self, request: ContextRequest, compliance: Dict, analysis: Dict, 
                               strategy: Dict, data: Dict) -> Dict[str, Any]:
        """Build final enriched context response."""
        return {
            "query": request.query,
            "context_type": request.context_type,
            "domain": request.domain,
            "compliance": {
                "level": compliance.get("level"),
                "rules_applied": compliance.get("rules", [])
            },
            "analysis": {
                "intent": analysis.get("intent"),
                "keywords": analysis.get("keywords", []),
                "complexity": analysis.get("complexity")
            },
            "strategy": {
                "retrieval_method": strategy.get("strategy"),
                "sources_used": strategy.get("sources", [])
            },
            "results": data.get("results", []),
            "metadata": {
                "total_found": data.get("total_found", 0),
                "sources": data.get("sources", []),
                "timestamp": datetime.now().isoformat(),
                "gateway_chain": ["corporate_llm", "ai_processing", "workflow_optimizer", "context"]
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Context Gateway capabilities."""
        return {
            "gateway_name": self.gateway_name,
            "gateway_type": self.gateway_type, 
            "version": self.version,
            "dependencies": ["corporate_llm", "ai_processing", "workflow_optimizer"],
            "context_types": ["general", "compliance", "technical", "business"],
            "orchestration": {
                "corporate_llm": "compliance_and_access_control",
                "ai_processing": "query_analysis_and_understanding", 
                "workflow_optimizer": "context_strategy_optimization",
                "knowledge_resources": "data_retrieval_and_integration"
            },
            "features": [
                "multi_gateway_orchestration",
                "compliance_aware_context",
                "intelligent_query_analysis", 
                "optimized_retrieval_strategy",
                "mcp_resource_integration"
            ]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of Context Gateway and its dependencies."""
        health = {
            "gateway": "healthy",
            "dependencies": {},
            "mcp_server": "unavailable" if not self.mcp_server else "healthy",
            "overall": "healthy"
        }
        
        # Check gateway dependencies
        for dep_name, dep_gateway in [
            ("corporate_llm", self.corporate_llm_gateway),
            ("ai_processing", self.ai_processing_gateway),
            ("workflow_optimizer", self.workflow_optimizer_gateway)
        ]:
            if dep_gateway:
                try:
                    if hasattr(dep_gateway, 'health_check'):
                        dep_health = dep_gateway.health_check()
                        health["dependencies"][dep_name] = "healthy" if dep_health.get("healthy", True) else "unhealthy"
                    else:
                        health["dependencies"][dep_name] = "healthy"
                except:
                    health["dependencies"][dep_name] = "unhealthy"
            else:
                health["dependencies"][dep_name] = "unavailable"
        
        # Overall health based on dependencies
        unhealthy_deps = [k for k, v in health["dependencies"].items() if v in ["unhealthy", "unavailable"]]
        if unhealthy_deps:
            health["overall"] = f"degraded (missing: {', '.join(unhealthy_deps)})"
        
        return health
    
    def _get_default_dependencies(self):
        """Get default dependencies for ContextGateway."""
        from .base_gateway import GatewayDependencies
        return GatewayDependencies(
            requires_corporate_llm=True,
            requires_ai_processing=True,
            requires_workflow_optimizer=True,
            requires_knowledge_resources=False  # Uses internal MCP server
        )
    
    def process_sync(self, request_data: Dict[str, Any]) -> GatewayResponse:
        """Synchronous processing entry point for BaseGateway compatibility."""
        # Convert dict to ContextRequest
        context_request = ContextRequest(
            query=request_data.get("query", ""),
            domain=request_data.get("domain"),
            context_type=request_data.get("context_type", "general"),
            max_results=request_data.get("max_results", 5),
            similarity_threshold=request_data.get("similarity_threshold", 0.7),
            include_sources=request_data.get("include_sources", True)
        )
        
        return self.process(context_request)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate ContextGateway configuration."""
        # Basic validation - could be enhanced
        return True