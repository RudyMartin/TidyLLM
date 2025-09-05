#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP (Model Control Protocol) Integration

Integration layer between TidyLLM Gateway and MCP orchestrators/workers.
Provides enterprise gateway services to MCP components with full governance.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
from dataclasses import dataclass

# Gateway imports
from ..core.base_gateway import BaseGateway, GatewayResponse
from ..gateways.llm_gateway import LLMGateway, LLMGatewayConfig
from ..gateways.database_gateway import DatabaseGateway, DatabaseGatewayConfig

# MCP system imports (if available)
try:
    from backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator
    from backend.mcp.workers.pdf_processing_worker import PDFProcessingWorker
    from backend.mcp.orchestrators.document_processing_orchestrator import DocumentProcessingOrchestrator
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


@dataclass
class MCPGatewayConfig:
    """Configuration for MCP gateway integration"""
    
    # Gateway configurations
    llm_gateway_config: Optional[LLMGatewayConfig] = None
    database_gateway_config: Optional[DatabaseGatewayConfig] = None
    
    # MCP-specific settings
    enable_orchestrator_integration: bool = True
    enable_worker_integration: bool = True
    
    # Enterprise controls
    require_audit_trails: bool = True
    enable_cost_tracking: bool = True
    enforce_quotas: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 300


class MCPGatewayProvider:
    """
    MCP Gateway Provider - Enterprise gateway services for MCP system
    
    Provides enterprise-grade gateway services to MCP orchestrators and workers:
    - Secure LLM access through corporate infrastructure
    - Governed database connectivity with audit trails
    - Cost tracking and quota management
    - Comprehensive monitoring and alerting
    - Failover and circuit breaking
    
    Integration Points:
    - AdvancedQAOrchestrator: Enterprise LLM access for Q&A workflows
    - DocumentProcessingOrchestrator: Secure document analysis
    - PDFProcessingWorker: Corporate file system integration
    - Database workers: Governed data access with audit trails
    """
    
    def __init__(self, config: MCPGatewayConfig):
        self.config = config
        
        # Initialize gateways
        self.llm_gateway = None
        self.database_gateway = None
        
        if config.llm_gateway_config:
            self.llm_gateway = LLMGateway(config.llm_gateway_config)
        
        if config.database_gateway_config:
            self.database_gateway = DatabaseGateway(config.database_gateway_config)
        
        # MCP integration tracking
        self.orchestrator_registry: Dict[str, Any] = {}
        self.worker_registry: Dict[str, Any] = {}
        self.integration_metrics = {
            "orchestrator_requests": 0,
            "worker_requests": 0,
            "total_cost_usd": 0.0,
            "avg_response_time_ms": 0.0
        }
        
        self.logger = logging.getLogger(f"{__name__}.MCPGatewayProvider")
        self.logger.info("🔗 MCP Gateway Provider initialized")
    
    def register_orchestrator(self, orchestrator_id: str, orchestrator: Any):
        """Register an MCP orchestrator for gateway services"""
        
        self.orchestrator_registry[orchestrator_id] = {
            "instance": orchestrator,
            "type": type(orchestrator).__name__,
            "registered_at": datetime.utcnow(),
            "request_count": 0,
            "total_cost_usd": 0.0
        }
        
        # Inject gateway services into orchestrator
        if hasattr(orchestrator, 'set_llm_gateway'):
            orchestrator.set_llm_gateway(self.llm_gateway)
        
        if hasattr(orchestrator, 'set_database_gateway'):
            orchestrator.set_database_gateway(self.database_gateway)
        
        self.logger.info(f"Registered orchestrator: {orchestrator_id} ({type(orchestrator).__name__})")
    
    def register_worker(self, worker_id: str, worker: Any):
        """Register an MCP worker for gateway services"""
        
        self.worker_registry[worker_id] = {
            "instance": worker,
            "type": type(worker).__name__,
            "registered_at": datetime.utcnow(),
            "request_count": 0,
            "total_cost_usd": 0.0
        }
        
        # Inject gateway services into worker
        if hasattr(worker, 'set_llm_gateway'):
            worker.set_llm_gateway(self.llm_gateway)
        
        if hasattr(worker, 'set_database_gateway'):
            worker.set_database_gateway(self.database_gateway)
        
        self.logger.info(f"Registered worker: {worker_id} ({type(worker).__name__})")
    
    def create_orchestrator_llm_client(self, orchestrator_id: str, user_id: str) -> 'MCPLLMClient':
        """Create LLM client for orchestrator with enterprise controls"""
        
        if orchestrator_id not in self.orchestrator_registry:
            raise ValueError(f"Orchestrator {orchestrator_id} not registered")
        
        return MCPLLMClient(
            gateway=self.llm_gateway,
            client_id=orchestrator_id,
            user_id=user_id,
            client_type="orchestrator",
            provider=self
        )
    
    def create_worker_database_client(self, worker_id: str, user_id: str) -> 'MCPDatabaseClient':
        """Create database client for worker with enterprise controls"""
        
        if worker_id not in self.worker_registry:
            raise ValueError(f"Worker {worker_id} not registered")
        
        return MCPDatabaseClient(
            gateway=self.database_gateway,
            client_id=worker_id,
            user_id=user_id,
            client_type="worker",
            provider=self
        )
    
    def execute_orchestrated_workflow(
        self,
        orchestrator_id: str,
        workflow_name: str,
        inputs: Dict[str, Any],
        user_id: str,
        audit_reason: str = None
    ) -> Dict[str, Any]:
        """Execute workflow through registered orchestrator with enterprise controls"""
        
        if orchestrator_id not in self.orchestrator_registry:
            raise ValueError(f"Orchestrator {orchestrator_id} not registered")
        
        orchestrator_info = self.orchestrator_registry[orchestrator_id]
        orchestrator = orchestrator_info["instance"]
        
        start_time = datetime.utcnow()
        
        try:
            # Execute workflow with audit context
            if hasattr(orchestrator, 'execute_workflow'):
                result = orchestrator.execute_workflow(
                    workflow_name=workflow_name,
                    inputs=inputs,
                    user_id=user_id,
                    audit_reason=audit_reason
                )
            else:
                # Fallback for orchestrators without standard interface
                result = {"error": "Orchestrator does not support workflow execution"}
            
            # Track metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            cost = result.get("cost_usd", 0.0)
            
            orchestrator_info["request_count"] += 1
            orchestrator_info["total_cost_usd"] += cost
            
            self.integration_metrics["orchestrator_requests"] += 1
            self.integration_metrics["total_cost_usd"] += cost
            
            # Update average response time
            total_requests = self.integration_metrics["orchestrator_requests"] + self.integration_metrics["worker_requests"]
            current_avg = self.integration_metrics["avg_response_time_ms"]
            self.integration_metrics["avg_response_time_ms"] = (
                (current_avg * (total_requests - 1) + execution_time) / total_requests
            )
            
            return {
                "success": True,
                "result": result,
                "orchestrator_id": orchestrator_id,
                "workflow_name": workflow_name,
                "execution_time_ms": execution_time,
                "cost_usd": cost,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "orchestrator_id": orchestrator_id,
                "workflow_name": workflow_name,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of MCP integration"""
        
        return {
            "mcp_available": MCP_AVAILABLE,
            "registered_orchestrators": len(self.orchestrator_registry),
            "registered_workers": len(self.worker_registry),
            "orchestrator_details": {
                orchestrator_id: {
                    "type": info["type"],
                    "request_count": info["request_count"],
                    "total_cost_usd": info["total_cost_usd"],
                    "registered_at": info["registered_at"].isoformat()
                }
                for orchestrator_id, info in self.orchestrator_registry.items()
            },
            "worker_details": {
                worker_id: {
                    "type": info["type"],
                    "request_count": info["request_count"],
                    "registered_at": info["registered_at"].isoformat()
                }
                for worker_id, info in self.worker_registry.items()
            },
            "integration_metrics": self.integration_metrics,
            "gateways_available": {
                "llm_gateway": self.llm_gateway is not None,
                "database_gateway": self.database_gateway is not None
            },
            "timestamp": datetime.utcnow().isoformat()
        }


class MCPLLMClient:
    """
    Enterprise LLM client for MCP orchestrators
    
    Provides governed LLM access with:
    - User attribution and audit trails
    - Cost tracking and quotas
    - Rate limiting and circuit breaking
    - Multi-provider abstraction
    """
    
    def __init__(
        self,
        gateway: LLMGateway,
        client_id: str,
        user_id: str,
        client_type: str,
        provider: MCPGatewayProvider
    ):
        self.gateway = gateway
        self.client_id = client_id
        self.user_id = user_id
        self.client_type = client_type
        self.provider = provider
        
        self.logger = logging.getLogger(f"{__name__}.MCPLLMClient")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        audit_reason: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute chat completion with enterprise governance"""
        
        if not audit_reason:
            audit_reason = f"{self.client_type} {self.client_id} - automated processing"
        
        try:
            response = self.gateway.chat(
                messages=messages,
                user_id=self.user_id,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                audit_reason=audit_reason,
                **kwargs
            )
            
            # Extract response content
            if response.success:
                result = {
                    "success": True,
                    "content": response.data.get("content", ""),
                    "model": response.data.get("model", "unknown"),
                    "cost_usd": response.data.get("cost_usd", 0.0),
                    "tokens_used": response.data.get("usage", {}).get("total_tokens", 0),
                    "processing_time_ms": response.processing_time_ms
                }
            else:
                result = {
                    "success": False,
                    "error": response.error,
                    "cost_usd": 0.0,
                    "tokens_used": 0
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM request failed for {self.client_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "cost_usd": 0.0,
                "tokens_used": 0
            }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models for this client"""
        
        providers = self.gateway.get_available_providers()
        models = []
        
        for provider in providers:
            provider_models = self.gateway.get_available_models(provider)
            for model in provider_models:
                models.append({
                    "provider": provider,
                    "model": model,
                    "available": True
                })
        
        return models


class MCPDatabaseClient:
    """
    Enterprise database client for MCP workers
    
    Provides governed database access with:
    - User attribution and audit trails
    - Query validation and security
    - Connection pooling and failover
    - Data classification handling
    """
    
    def __init__(
        self,
        gateway: DatabaseGateway,
        client_id: str,
        user_id: str,
        client_type: str,
        provider: MCPGatewayProvider
    ):
        self.gateway = gateway
        self.client_id = client_id
        self.user_id = user_id
        self.client_type = client_type
        self.provider = provider
        
        self.logger = logging.getLogger(f"{__name__}.MCPDatabaseClient")
    
    def execute_query(
        self,
        connection_name: str,
        query: str,
        parameters: List[Any] = None,
        audit_reason: str = None
    ) -> Dict[str, Any]:
        """Execute database query with enterprise governance"""
        
        if not audit_reason:
            audit_reason = f"{self.client_type} {self.client_id} - data processing"
        
        try:
            response = self.gateway.execute_query(
                connection_name=connection_name,
                query=query,
                parameters=parameters,
                user_id=self.user_id,
                audit_reason=audit_reason
            )
            
            # Extract response data
            if response.success:
                result = {
                    "success": True,
                    "data": response.data,
                    "row_count": response.data.get("row_count", 0),
                    "execution_time_ms": response.processing_time_ms
                }
            else:
                result = {
                    "success": False,
                    "error": response.error,
                    "row_count": 0
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Database query failed for {self.client_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "row_count": 0
            }
    
    def get_available_connections(self) -> List[Dict[str, Any]]:
        """Get available database connections for this client"""
        return self.gateway.get_available_connections()


# Convenience functions for MCP integration
def create_mcp_gateway_provider(
    llm_config: LLMGatewayConfig = None,
    database_config: DatabaseGatewayConfig = None
) -> MCPGatewayProvider:
    """Create MCP gateway provider with default configuration"""
    
    config = MCPGatewayConfig(
        llm_gateway_config=llm_config,
        database_gateway_config=database_config
    )
    
    return MCPGatewayProvider(config)


def integrate_with_advanced_qa_orchestrator(
    provider: MCPGatewayProvider,
    orchestrator: Any,
    user_id: str = "system@mcp"
) -> str:
    """Integrate TidyLLM Gateway with AdvancedQAOrchestrator"""
    
    orchestrator_id = f"advanced_qa_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Register orchestrator
    provider.register_orchestrator(orchestrator_id, orchestrator)
    
    # Create and inject LLM client
    llm_client = provider.create_orchestrator_llm_client(orchestrator_id, user_id)
    
    # Inject gateway services if orchestrator supports it
    if hasattr(orchestrator, 'set_enterprise_llm_client'):
        orchestrator.set_enterprise_llm_client(llm_client)
    
    return orchestrator_id


def integrate_with_pdf_processing_worker(
    provider: MCPGatewayProvider,
    worker: Any,
    user_id: str = "system@mcp"
) -> str:
    """Integrate TidyLLM Gateway with PDFProcessingWorker"""
    
    worker_id = f"pdf_worker_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Register worker
    provider.register_worker(worker_id, worker)
    
    # Workers typically don't need LLM access, but if they do:
    if hasattr(worker, 'set_llm_client'):
        llm_client = provider.create_orchestrator_llm_client(worker_id, user_id)
        worker.set_llm_client(llm_client)
    
    return worker_id


# Package exports for MCP integration
__all__ = [
    'MCPGatewayProvider',
    'MCPLLMClient', 
    'MCPDatabaseClient',
    'MCPGatewayConfig',
    'create_mcp_gateway_provider',
    'integrate_with_advanced_qa_orchestrator',
    'integrate_with_pdf_processing_worker'
]