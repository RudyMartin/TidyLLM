"""
Knowledge Resource Server Interfaces
===================================

Defines interfaces and data structures for MCP resource provision.
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .resource_manager import KnowledgeResourceManager


@dataclass
class KnowledgeResource:
    """Represents a knowledge resource (document, context, etc.)."""
    id: str
    title: str
    content: str
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_uri: str = ""
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "domain": self.domain,
            "metadata": self.metadata,
            "source_uri": self.source_uri,
            "last_updated": self.last_updated
        }


class MCPResourceInterface:
    """
    Interface for exposing resources via MCP protocol.
    
    Manages the mapping between internal knowledge resources
    and MCP resource URIs.
    """
    
    def __init__(self, resource_manager: "KnowledgeResourceManager"):
        """Initialize MCP resource interface."""
        self.resource_manager = resource_manager
        self.exposed_resources: Dict[str, Dict[str, Any]] = {}
    
    def expose_resource(self, resource_uri: str, resource_info: Dict[str, Any]) -> None:
        """
        Expose a resource via MCP URI.
        
        Args:
            resource_uri: MCP resource URI (e.g., "domains/legal-docs")
            resource_info: Resource metadata and capabilities
        """
        self.exposed_resources[resource_uri] = {
            **resource_info,
            "exposed_at": datetime.now().isoformat(),
            "access_count": 0
        }
    
    def get_resource(self, resource_uri: str) -> Optional[Dict[str, Any]]:
        """Get resource by MCP URI."""
        if resource_uri in self.exposed_resources:
            # Increment access counter
            self.exposed_resources[resource_uri]["access_count"] += 1
            return self.exposed_resources[resource_uri]
        return None
    
    def list_resources(self) -> List[str]:
        """List all exposed resource URIs."""
        return list(self.exposed_resources.keys())
    
    def get_resource_capabilities(self, resource_uri: str) -> List[str]:
        """Get capabilities for a specific resource."""
        resource = self.get_resource(resource_uri)
        return resource.get("capabilities", []) if resource else []