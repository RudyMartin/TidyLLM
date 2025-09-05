"""
Base Gateway Interface for TidyLLM
===================================

Common interface that all gateways must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
from enum import Enum
import logging

if TYPE_CHECKING:
    from . import AIProcessingGateway, CorporateLLMGateway, WorkflowOptimizerGateway

logger = logging.getLogger(__name__)


class GatewayStatus(Enum):
    """Gateway processing status."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class GatewayDependencies:
    """
    Defines gateway dependency requirements.
    
    Dependency Chain:
    CorporateLLMGateway (base) → AIProcessingGateway → WorkflowOptimizerGateway → KnowledgeResourceServer
    """
    requires_ai_processing: bool = False    # Needs AI/ML model capabilities
    requires_corporate_llm: bool = False    # Needs corporate LLM access control
    requires_workflow_optimizer: bool = False  # Needs workflow optimization
    requires_knowledge_resources: bool = False  # Needs knowledge/context access
    
    def get_required_services(self) -> List[str]:
        """Return list of required service names."""
        services = []
        if self.requires_ai_processing: services.append("ai_processing")
        if self.requires_corporate_llm: services.append("corporate_llm")
        if self.requires_workflow_optimizer: services.append("workflow_optimizer")
        if self.requires_knowledge_resources: services.append("knowledge_resources")
        return services


@dataclass
class GatewayResponse:
    """
    Standard response from gateway operations.
    
    Attributes:
        status: Processing status
        data: Main response data
        metadata: Additional metadata (timing, tokens, etc.)
        errors: List of any errors encountered
        gateway_name: Name of the gateway that processed
        timestamp: When processing occurred
    """
    status: GatewayStatus
    data: Any
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    gateway_name: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def is_success(self) -> bool:
        """Check if operation succeeded."""
        return self.status == GatewayStatus.SUCCESS
    
    @property
    def is_partial(self) -> bool:
        """Check if operation partially succeeded."""
        return self.status == GatewayStatus.PARTIAL
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
    
    @property
    def success(self) -> bool:
        """Legacy property for backward compatibility."""
        return self.is_success
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "data": self.data,
            "metadata": self.metadata,
            "errors": self.errors,
            "gateway_name": self.gateway_name,
            "timestamp": self.timestamp.isoformat()
        }


class BaseGateway(ABC):
    """
    Abstract base class for all TidyLLM gateways.
    
    All gateways must implement this interface to ensure
    consistent behavior across different processing engines.
    """
    
    def __init__(self, **config):
        """
        Initialize gateway with configuration.
        
        Args:
            **config: Gateway-specific configuration
        """
        self.config = config
        self.name = self.__class__.__name__
        
        # Initialize dependency configuration
        self.dependencies = self._get_default_dependencies()
        self._resolve_dependencies()
        
        # Validate after dependencies are resolved
        self._validate_config()
    
    @abstractmethod
    def _get_default_dependencies(self) -> GatewayDependencies:
        """
        Get the default dependency configuration for this gateway.
        
        Each gateway must define its dependencies:
        - AIProcessingGateway requires CorporateLLMGateway
        - WorkflowOptimizerGateway requires AIProcessingGateway + CorporateLLMGateway
        - CorporateLLMGateway has no dependencies
        - KnowledgeResourceServer is independent
        """
        pass
    
    def _resolve_dependencies(self):
        """
        Resolve and auto-enable dependent gateways based on configuration.
        
        This implements the dependency chain:
        - If AIProcessingGateway is used → enable CorporateLLMGateway
        - If WorkflowOptimizerGateway is used → enable AIProcessingGateway + CorporateLLMGateway
        """
        logger.info(f"Resolving dependencies for {self.name}")
        
        # Log original dependencies
        logger.debug(f"Original dependencies: {self.dependencies.get_required_services()}")
        
        # Apply dependency rules (set by subclasses in _get_default_dependencies)
        if hasattr(self, '_dependency_rules_applied'):
            return  # Avoid infinite recursion
        
        self._dependency_rules_applied = True
        
        # Log resolved dependencies  
        logger.info(f"Resolved dependencies: {self.dependencies.get_required_services()}")
    
    def get_required_services(self) -> List[str]:
        """
        Get list of service names that this gateway depends on.
        
        Returns:
            List of service names that are required by this gateway
        """
        return self.dependencies.get_required_services()
    
    def get_required_gateways(self) -> List[str]:
        """Legacy method for backward compatibility."""
        return self.get_required_services()
    
    @abstractmethod
    async def process(self, 
                     input_data: Any,
                     **kwargs) -> GatewayResponse:
        """
        Main processing method for the gateway.
        
        Args:
            input_data: Data to process (text, dict, etc.)
            **kwargs: Additional processing parameters
            
        Returns:
            GatewayResponse with results
        """
        pass
    
    @abstractmethod
    def process_sync(self, 
                    input_data: Any,
                    **kwargs) -> GatewayResponse:
        """
        Synchronous processing method.
        
        Args:
            input_data: Data to process
            **kwargs: Additional processing parameters
            
        Returns:
            GatewayResponse with results
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate gateway configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get gateway capabilities and features.
        
        Returns:
            Dictionary describing what this gateway can do
            
        Example:
            {
                "models": ["claude-3", "gpt-4"],
                "max_tokens": 4096,
                "supports_streaming": True,
                "supports_async": True,
                "supports_batch": False
            }
        """
        pass
    
    def _validate_config(self):
        """Internal config validation."""
        try:
            self.validate_config()
        except Exception as e:
            raise ValueError(f"Invalid configuration for {self.name}: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check gateway health and availability.
        
        Returns:
            Health status dictionary
        """
        return {
            "gateway": self.name,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "capabilities": self.get_capabilities()
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}({self.config})"