"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

Gateway Registry - Unified Gateway Management
============================================

Central registry for all processing gateways with clear service purpose
and dependency management.

Purpose: Provides unified access to all gateways with automatic configuration,
dependency resolution, and service discovery.
"""

import logging
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .base_gateway import BaseGateway, GatewayResponse, GatewayStatus
from .ai_processing_gateway import AIProcessingGateway
from .workflow_optimizer_gateway import WorkflowOptimizerGateway
from .corporate_llm_gateway import CorporateLLMGateway
try:
    from ..knowledge_resource_server import KnowledgeMCPServer
    KNOWLEDGE_SERVER_AVAILABLE = True
except ImportError:
    KnowledgeMCPServer = None
    KNOWLEDGE_SERVER_AVAILABLE = False

logger = logging.getLogger("gateway_registry")


class ServiceType(Enum):
    """Service types available in the gateway registry."""
    AI_PROCESSING = "ai_processing"
    CORPORATE_LLM = "corporate_llm"
    WORKFLOW_OPTIMIZER = "workflow_optimizer"
    KNOWLEDGE_RESOURCES = "knowledge_resources"


@dataclass
class ServiceInfo:
    """Information about a registered service."""
    service_type: ServiceType
    service_class: Type[BaseGateway]
    description: str
    dependencies: List[ServiceType]
    instance: Optional[BaseGateway] = None
    initialized: bool = False
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_type": self.service_type.value,
            "service_class": self.service_class.__name__,
            "description": self.description,
            "dependencies": [dep.value for dep in self.dependencies],
            "initialized": self.initialized,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "capabilities": (
                self.instance.get_mcp_capabilities() 
                if hasattr(self.instance, 'get_mcp_capabilities') 
                else self.instance.get_capabilities() 
                if hasattr(self.instance, 'get_capabilities') 
                else None
            ) if self.instance else None
        }


class GatewayRegistry:
    """
    Central registry for CORE ENTERPRISE GATEWAYS ONLY.
    
    🚀 CORE ENTERPRISE WORKFLOW (4 Gateways):
    1. CorporateLLMGateway - Foundation access control layer (no dependencies)
    2. AIProcessingGateway - AI model orchestration layer (requires corporate_llm)
    3. WorkflowOptimizerGateway - Workflow intelligence layer (requires ai_processing + corporate_llm)
    4. KnowledgeResourceServer - Knowledge & context layer (independent MCP server)
    
    🔧 UTILITY SERVICES (Not registered here - used independently):
    - DatabaseUtilityService - Database access wrapper
    - FileStorageUtilityService - S3/file storage wrapper  
    - MVRDocumentService - Model validation report processor
    
    Purpose-Based Access:
    - gateway.get("corporate_llm") - Enterprise LLM access control
    - gateway.get("ai_processing") - Multi-model AI processing
    - gateway.get("workflow_optimizer") - Workflow intelligence
    - gateway.get("knowledge_resources") - Knowledge and context provision
    
    Examples:
        >>> registry = GatewayRegistry()
        >>> registry.auto_configure()  # Auto-detect and configure all gateways
        >>> 
        >>> # Use AI processing
        >>> ai = registry.get("ai_processing")
        >>> response = ai.process("Explain quantum computing")
        >>> 
        >>> # Optimize workflow
        >>> optimizer = registry.get("workflow_optimizer")
        >>> result = optimizer.process_workflow(WorkflowRequest(...))
        >>> 
        >>> # Query knowledge
        >>> knowledge = registry.get("knowledge_resources")
        >>> context = knowledge.search("validation criteria")
    """
    
    _instance = None
    _initialized_singleton = False
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(GatewayRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Gateway Registry (only once due to singleton pattern)."""
        # Only initialize once
        if self._initialized_singleton:
            return
            
        self._services: Dict[ServiceType, ServiceInfo] = {}
        self._initialized = False
        
        # INTEGRATION: Initialize UnifiedSessionManager for all gateways
        try:
            from ..infrastructure import UnifiedSessionManager
            self.session_manager = UnifiedSessionManager()
            logger.info("Gateway Registry: UnifiedSessionManager integrated")
        except ImportError as e:
            logger.debug(f"Gateway Registry: UnifiedSessionManager not available: {e}")
            self.session_manager = None
        
        # Register all available services
        self._register_core_services()
        
        logger.info("Gateway Registry initialized")
        self._initialized_singleton = True
    
    def _register_core_services(self):
        """Register core services with their dependencies."""
        
        # Corporate LLM Gateway - Foundation layer (no dependencies)
        self._services[ServiceType.CORPORATE_LLM] = ServiceInfo(
            service_type=ServiceType.CORPORATE_LLM,
            service_class=CorporateLLMGateway,
            description="Enterprise LLM access control with audit, cost tracking, and compliance",
            dependencies=[]  # Independent foundation
        )
        
        # AI Processing Gateway - Requires corporate LLM
        self._services[ServiceType.AI_PROCESSING] = ServiceInfo(
            service_type=ServiceType.AI_PROCESSING,
            service_class=AIProcessingGateway,
            description="Multi-model AI processing with backend selection and caching",
            dependencies=[ServiceType.CORPORATE_LLM]
        )
        
        # Workflow Optimizer Gateway - Requires AI processing and corporate LLM
        self._services[ServiceType.WORKFLOW_OPTIMIZER] = ServiceInfo(
            service_type=ServiceType.WORKFLOW_OPTIMIZER,
            service_class=WorkflowOptimizerGateway,
            description="Workflow analysis, optimization, and compliance validation",
            dependencies=[ServiceType.AI_PROCESSING, ServiceType.CORPORATE_LLM]
        )
        
        # Knowledge Resource Server - Independent MCP server
        if KNOWLEDGE_SERVER_AVAILABLE:
            self._services[ServiceType.KNOWLEDGE_RESOURCES] = ServiceInfo(
                service_type=ServiceType.KNOWLEDGE_RESOURCES,
                service_class=KnowledgeMCPServer,  # Note: Not a BaseGateway
                description="MCP resource server for knowledge, documents, and contexts",
                dependencies=[]  # Independent
            )
    
    def auto_configure(self, config: Dict[str, Any] = None) -> None:
        """
        Auto-detect and configure all available gateways.
        
        Args:
            config: Optional configuration dictionary with service-specific settings
        """
        config = config or {}
        
        try:
            # Initialize services in dependency order
            initialization_order = self._get_initialization_order()
            
            for service_type in initialization_order:
                service_info = self._services[service_type]
                service_config = config.get(service_type.value, {})
                
                try:
                    # Initialize the service
                    if service_type == ServiceType.KNOWLEDGE_RESOURCES:
                        # Special handling for MCP server
                        instance = service_info.service_class(config=service_config.get('mcp_config'))
                    else:
                        # Standard gateway initialization
                        instance = service_info.service_class(**service_config)
                    
                    # INTEGRATION: Inject UnifiedSessionManager into each gateway
                    if self.session_manager and hasattr(instance, 'set_session_manager'):
                        instance.set_session_manager(self.session_manager)
                        logger.info(f"UnifiedSessionManager injected into {service_type.value}")
                    
                    service_info.instance = instance
                    service_info.initialized = True
                    
                    logger.info(f"✅ Initialized {service_type.value}: {service_info.description}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {service_type.value}: {e}")
                    logger.info(f"Service will be unavailable: {service_info.description}")
                    continue
            
            self._initialized = True
            logger.info(f"🚀 Gateway Registry auto-configuration complete")
            self._log_status()
            
        except Exception as e:
            logger.error(f"Auto-configuration failed: {e}")
            raise
    
    def _get_initialization_order(self) -> List[ServiceType]:
        """Get services in dependency order for initialization."""
        ordered = []
        remaining = set(self._services.keys())
        
        while remaining:
            # Find services with satisfied dependencies
            ready = []
            for service_type in remaining:
                deps = self._services[service_type].dependencies
                if all(dep in [s.service_type for s in self._services.values() 
                              if s.initialized] or dep in [t for t in ordered] 
                      for dep in deps):
                    ready.append(service_type)
            
            if not ready:
                # No progress possible - circular dependencies or missing services
                logger.error(f"Cannot resolve dependencies for: {remaining}")
                break
            
            # Add ready services to order
            for service_type in ready:
                ordered.append(service_type)
                remaining.remove(service_type)
        
        return ordered
    
    def get(self, service_name: str) -> Optional[BaseGateway]:
        """
        Get gateway by service name.
        
        Args:
            service_name: Service name (ai_processing, corporate_llm, workflow_optimizer, knowledge_resources)
            
        Returns:
            Gateway instance if available, None otherwise
        """
        try:
            # Map string names to service types
            service_type_map = {
                "ai": ServiceType.AI_PROCESSING,
                "ai_processing": ServiceType.AI_PROCESSING,
                "llm": ServiceType.CORPORATE_LLM,
                "corporate_llm": ServiceType.CORPORATE_LLM,
                "workflow": ServiceType.WORKFLOW_OPTIMIZER,
                "workflow_optimizer": ServiceType.WORKFLOW_OPTIMIZER,
                "knowledge": ServiceType.KNOWLEDGE_RESOURCES,
                "knowledge_resources": ServiceType.KNOWLEDGE_RESOURCES
            }
            
            service_type = service_type_map.get(service_name)
            if not service_type:
                logger.warning(f"Unknown service name: {service_name}")
                return None
            
            service_info = self._services.get(service_type)
            if not service_info or not service_info.initialized:
                logger.warning(f"Service '{service_name}' not initialized")
                return None
            
            # Update access time
            service_info.last_accessed = datetime.now()
            
            return service_info.instance
            
        except Exception as e:
            logger.error(f"Error getting service '{service_name}': {e}")
            return None
    
    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific service."""
        service = self.get(service_name)
        if service:
            service_type_map = {
                "ai_processing": ServiceType.AI_PROCESSING,
                "corporate_llm": ServiceType.CORPORATE_LLM, 
                "workflow_optimizer": ServiceType.WORKFLOW_OPTIMIZER,
                "knowledge_resources": ServiceType.KNOWLEDGE_RESOURCES
            }
            service_type = service_type_map.get(service_name)
            if service_type:
                return self._services[service_type].to_dict()
        return None
    
    def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services with their status."""
        return [service.to_dict() for service in self._services.values()]
    
    def get_available_services(self) -> List[str]:
        """Get list of available (initialized) service names."""
        available = []
        for service_type, service_info in self._services.items():
            if service_info.initialized:
                available.append(service_type.value)
        return available
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        results = {}
        
        for service_type, service_info in self._services.items():
            service_name = service_type.value
            
            if not service_info.initialized:
                results[service_name] = {
                    "status": "not_initialized",
                    "healthy": False,
                    "error": "Service not initialized"
                }
                continue
            
            try:
                if hasattr(service_info.instance, 'health_check'):
                    health = service_info.instance.health_check()
                    results[service_name] = {
                        "status": "healthy",
                        "healthy": True,
                        "details": health
                    }
                else:
                    results[service_name] = {
                        "status": "initialized",
                        "healthy": True,
                        "note": "No health check method available"
                    }
                    
            except Exception as e:
                results[service_name] = {
                    "status": "unhealthy",
                    "healthy": False,
                    "error": str(e)
                }
        
        # Overall health
        all_healthy = all(result.get("healthy", False) for result in results.values())
        
        return {
            "overall_healthy": all_healthy,
            "timestamp": datetime.now().isoformat(),
            "services": results,
            "total_services": len(self._services),
            "healthy_services": sum(1 for r in results.values() if r.get("healthy", False))
        }
    
    def _log_status(self):
        """Log current registry status."""
        initialized_count = sum(1 for s in self._services.values() if s.initialized)
        total_count = len(self._services)
        
        logger.info(f"📊 Gateway Registry Status:")
        logger.info(f"   Total services: {total_count}")
        logger.info(f"   Initialized: {initialized_count}")
        logger.info(f"   Available services: {', '.join(self.get_available_services())}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total = len(self._services)
        initialized = sum(1 for s in self._services.values() if s.initialized)
        
        return {
            "total_services": total,
            "initialized_services": initialized,
            "available_services": self.get_available_services(),
            "registry_initialized": self._initialized,
            "last_health_check": None,  # Would track actual health checks
            "dependency_chain": {
                "foundation": ["corporate_llm", "knowledge_resources"],
                "processing": ["ai_processing"],
                "optimization": ["workflow_optimizer"]
            }
        }


# Global registry instance
_global_registry: Optional[GatewayRegistry] = None

def get_global_registry() -> GatewayRegistry:
    """Get or create global gateway registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = GatewayRegistry()
    return _global_registry

def init_gateways(config: Dict[str, Any] = None) -> GatewayRegistry:
    """
    Initialize gateways with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured GatewayRegistry instance
    """
    registry = get_global_registry()
    registry.auto_configure(config)
    return registry