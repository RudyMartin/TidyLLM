"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

Knowledge MCP Server - MCP Resource Provider Implementation
==========================================================
🚀 CORE ENTERPRISE GATEWAY #4 - Knowledge & Context Layer
This is a core gateway in the main enterprise workflow processing chain.

Implements MCP (Model Context Protocol) server for providing knowledge resources
to LLMs and other services.

This server exposes knowledge resources and tools via the MCP protocol, allowing
gateways and applications to access structured knowledge, documents, and contexts.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .resource_manager import KnowledgeResourceManager, KnowledgeResource
from .interfaces import MCPResourceInterface
from .sources import KnowledgeSource

logger = logging.getLogger("knowledge_mcp_server")


@dataclass
class MCPServerConfig:
    """Configuration for Knowledge MCP Server."""
    server_name: str = "tidyllm-knowledge"
    server_version: str = "1.0.0"
    description: str = "TidyLLM Knowledge Resource Provider"
    
    # Resource configuration
    max_search_results: int = 10
    default_similarity_threshold: float = 0.7
    enable_caching: bool = True
    cache_ttl: int = 3600
    
    # Security and access control
    require_authentication: bool = False
    allowed_clients: List[str] = field(default_factory=list)
    
    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0


class KnowledgeMCPServer:
    """
    MCP Resource Server for knowledge provision.
    
    This server exposes knowledge resources via MCP protocol for consumption
    by gateways, applications, and other MCP clients.
    
    MCP Resources Exposed:
    - domains/{domain_name} - Domain-specific knowledge bases
    - documents/{doc_id} - Individual documents
    - contexts/{context_id} - Retrieved contexts for queries  
    - embeddings/{embedding_id} - Vector embeddings
    
    MCP Tools Provided:
    - search - Semantic similarity search
    - retrieve - Document retrieval  
    - embed - Generate embeddings
    - extract - Extract structured data
    - query - Natural language query
    
    Examples:
        >>> server = KnowledgeMCPServer()
        >>> 
        >>> # Register knowledge domain
        >>> server.register_domain(
        ...     "legal-docs",
        ...     S3KnowledgeSource(bucket="legal", prefix="contracts/")
        ... )
        >>> 
        >>> # Client queries via MCP
        >>> results = mcp_client.call_tool("search", {
        ...     "query": "contract termination clauses",
        ...     "domain": "legal-docs",
        ...     "max_results": 5
        ... })
    """
    
    def __init__(self, config: MCPServerConfig = None):
        """Initialize Knowledge MCP Server."""
        self.config = config or MCPServerConfig()
        self.resources = KnowledgeResourceManager()
        self.mcp_interface = MCPResourceInterface(self.resources)
        
        # Track registered domains and resources
        self.registered_domains: Dict[str, KnowledgeSource] = {}
        self.resource_cache: Dict[str, Any] = {}
        
        logger.info(f"Knowledge MCP Server '{self.config.server_name}' initialized")
    
    def register_domain(self, domain_name: str, source: KnowledgeSource) -> None:
        """
        Register a knowledge domain as MCP resource.
        
        Args:
            domain_name: Resource identifier (e.g., "legal-docs", "model-validation")
            source: Knowledge source to load data from
            
        Example:
            >>> server.register_domain(
            ...     "model-validation",
            ...     S3KnowledgeSource(bucket="docs", prefix="validation/")
            ... )
        """
        try:
            # Register with resource manager
            self.resources.register_domain(domain_name, source)
            self.registered_domains[domain_name] = source
            
            # Expose as MCP resource
            resource_uri = f"domains/{domain_name}"
            self.mcp_interface.expose_resource(resource_uri, {
                "name": domain_name,
                "description": f"Knowledge domain: {domain_name}",
                "type": "knowledge_domain",
                "source": source.get_info(),
                "capabilities": ["search", "retrieve", "query"]
            })
            
            logger.info(f"Registered knowledge domain '{domain_name}' at resource URI '{resource_uri}'")
            
        except Exception as e:
            logger.error(f"Failed to register domain '{domain_name}': {e}")
            raise
    
    def get_mcp_capabilities(self) -> Dict[str, Any]:
        """Get MCP server capabilities."""
        return {
            "server": {
                "name": self.config.server_name,
                "version": self.config.server_version,
                "description": self.config.description
            },
            "resources": {
                "domains": list(self.registered_domains.keys()),
                "resource_count": len(self.registered_domains),
                "supports_search": True,
                "supports_retrieval": True,
                "supports_embedding": True
            },
            "tools": [
                {
                    "name": "search",
                    "description": "Search knowledge bases using semantic similarity",
                    "parameters": {
                        "query": {"type": "string", "required": True},
                        "domain": {"type": "string", "required": False},
                        "max_results": {"type": "integer", "default": 5},
                        "similarity_threshold": {"type": "number", "default": 0.7}
                    }
                },
                {
                    "name": "retrieve",
                    "description": "Retrieve specific documents by ID or criteria", 
                    "parameters": {
                        "document_id": {"type": "string", "required": False},
                        "domain": {"type": "string", "required": False},
                        "criteria": {"type": "object", "required": False}
                    }
                },
                {
                    "name": "embed",
                    "description": "Generate embeddings for text or documents",
                    "parameters": {
                        "text": {"type": "string", "required": True},
                        "model": {"type": "string", "default": "sentence-transformers"}
                    }
                },
                {
                    "name": "extract",
                    "description": "Extract structured data from documents",
                    "parameters": {
                        "document_id": {"type": "string", "required": True},
                        "extraction_type": {"type": "string", "required": True}
                    }
                },
                {
                    "name": "query",
                    "description": "Natural language query against knowledge bases",
                    "parameters": {
                        "question": {"type": "string", "required": True},
                        "domain": {"type": "string", "required": False},
                        "context_length": {"type": "integer", "default": 2000}
                    }
                }
            ]
        }
    
    def handle_mcp_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP tool call requests.
        
        Args:
            tool_name: Name of the tool being called
            parameters: Tool parameters
            
        Returns:
            Tool execution results
        """
        try:
            if tool_name == "search":
                return self._handle_search_tool(parameters)
            elif tool_name == "retrieve":
                return self._handle_retrieve_tool(parameters)
            elif tool_name == "embed":
                return self._handle_embed_tool(parameters)
            elif tool_name == "extract":
                return self._handle_extract_tool(parameters)
            elif tool_name == "query":
                return self._handle_query_tool(parameters)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": ["search", "retrieve", "embed", "extract", "query"]
                }
                
        except Exception as e:
            logger.error(f"Error handling tool call '{tool_name}': {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "parameters": parameters
            }
    
    def _handle_search_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic search tool."""
        query = params.get("query")
        domain = params.get("domain")
        max_results = params.get("max_results", 5)
        similarity_threshold = params.get("similarity_threshold", 0.7)
        
        if not query:
            return {"success": False, "error": "Query parameter is required"}
        
        try:
            results = self.resources.search(
                query=query,
                domain=domain,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            
            return {
                "success": True,
                "tool": "search",
                "results": [result.to_dict() for result in results],
                "query": query,
                "domain": domain,
                "result_count": len(results)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_retrieve_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document retrieval tool."""
        document_id = params.get("document_id")
        domain = params.get("domain") 
        criteria = params.get("criteria", {})
        
        try:
            if document_id:
                document = self.resources.retrieve_document(document_id)
                return {
                    "success": True,
                    "tool": "retrieve",
                    "document": document.to_dict() if document else None,
                    "document_id": document_id
                }
            else:
                documents = self.resources.retrieve_by_criteria(domain, criteria)
                return {
                    "success": True,
                    "tool": "retrieve", 
                    "documents": [doc.to_dict() for doc in documents],
                    "domain": domain,
                    "criteria": criteria,
                    "result_count": len(documents)
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_embed_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle embedding generation tool."""
        text = params.get("text")
        model = params.get("model", "sentence-transformers")
        
        if not text:
            return {"success": False, "error": "Text parameter is required"}
        
        try:
            embedding = self.resources.generate_embedding(text, model)
            return {
                "success": True,
                "tool": "embed",
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "text_length": len(text),
                "model": model
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_extract_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle structured data extraction tool."""
        document_id = params.get("document_id")
        extraction_type = params.get("extraction_type")
        
        if not document_id or not extraction_type:
            return {"success": False, "error": "document_id and extraction_type are required"}
        
        try:
            extracted_data = self.resources.extract_structured_data(document_id, extraction_type)
            return {
                "success": True,
                "tool": "extract",
                "extracted_data": extracted_data,
                "document_id": document_id,
                "extraction_type": extraction_type
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_query_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle natural language query tool."""
        question = params.get("question")
        domain = params.get("domain")
        context_length = params.get("context_length", 2000)
        
        if not question:
            return {"success": False, "error": "Question parameter is required"}
        
        try:
            answer, context = self.resources.query_knowledge(question, domain, context_length)
            return {
                "success": True,
                "tool": "query",
                "answer": answer,
                "context": context,
                "question": question,
                "domain": domain,
                "context_length": len(context) if context else 0
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_resource_info(self, resource_uri: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific resource."""
        try:
            # Parse resource URI (e.g., "domains/legal-docs")
            parts = resource_uri.split("/")
            if len(parts) != 2:
                return None
            
            resource_type, resource_id = parts
            
            if resource_type == "domains" and resource_id in self.registered_domains:
                source = self.registered_domains[resource_id]
                stats = self.resources.get_domain_stats(resource_id)
                
                return {
                    "uri": resource_uri,
                    "type": "knowledge_domain",
                    "name": resource_id,
                    "source": source.get_info(),
                    "statistics": stats,
                    "capabilities": ["search", "retrieve", "query"],
                    "last_updated": datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting resource info for '{resource_uri}': {e}")
            return None
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available MCP resources."""
        resources = []
        
        for domain_name in self.registered_domains:
            resource_info = self.get_resource_info(f"domains/{domain_name}")
            if resource_info:
                resources.append(resource_info)
        
        return resources
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status and statistics."""
        return {
            "server_name": self.config.server_name,
            "version": self.config.server_version,
            "status": "running",
            "registered_domains": len(self.registered_domains),
            "total_resources": len(self.registered_domains),
            "uptime": "N/A",  # Would track actual uptime
            "resource_manager_status": self.resources.get_status(),
            "last_updated": datetime.now().isoformat()
        }