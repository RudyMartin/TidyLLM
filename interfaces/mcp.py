"""
TidyLLM MCP Protocol Server
==========================

Unified MCP (Model Context Protocol) server that exposes TidyLLM gateways
and knowledge resources via MCP protocol.

This consolidates the previous mixed protocol approach where KnowledgeMCPServer
tried to be both a gateway and protocol server.

Usage:
    python -m tidyllm.interfaces.mcp
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..gateways.gateway_registry import get_global_registry
from ..knowledge_resource_server.mcp_server import KnowledgeMCPServer

logger = logging.getLogger("tidyllm_mcp_interface")


@dataclass
class MCPInterfaceConfig:
    """Configuration for TidyLLM MCP interface."""
    server_name: str = "tidyllm"
    server_version: str = "1.0.0"
    enable_gateways: bool = True
    enable_knowledge_resources: bool = True
    port: int = 3000


class TidyLLMMCPInterface:
    """
    Unified MCP interface exposing all TidyLLM capabilities.
    
    Exposes:
    - Gateway tools (via GatewayRegistry)
    - Knowledge resources (via KnowledgeMCPServer)
    - Mixed protocol access to all TidyLLM features
    """
    
    def __init__(self, config: MCPInterfaceConfig = None):
        self.config = config or MCPInterfaceConfig()
        
        # Get gateway registry
        self.registry = get_global_registry()
        self.registry.auto_configure()
        
        # Setup knowledge MCP server for resource provision
        if self.config.enable_knowledge_resources:
            self.knowledge_server = KnowledgeMCPServer()
        else:
            self.knowledge_server = None
        
        logger.info(f"TidyLLM MCP Interface '{self.config.server_name}' initialized")
    
    def get_mcp_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive MCP capabilities for all TidyLLM features."""
        capabilities = {
            "server": {
                "name": self.config.server_name,
                "version": self.config.server_version,
                "description": "TidyLLM Unified MCP Interface"
            },
            "tools": [],
            "resources": []
        }
        
        # Add gateway tools
        if self.config.enable_gateways:
            capabilities["tools"].extend(self._get_gateway_tools())
        
        # Add knowledge resources
        if self.knowledge_server:
            knowledge_caps = self.knowledge_server.get_mcp_capabilities()
            capabilities["tools"].extend(knowledge_caps.get("tools", []))
            # Convert resources to proper format
            for domain in knowledge_caps.get("resources", {}).get("domains", []):
                capabilities["resources"].append({
                    "uri": f"knowledge/domains/{domain}",
                    "name": f"Knowledge Domain: {domain}",
                    "description": f"Knowledge resources for {domain} domain"
                })
        
        return capabilities
    
    def _get_gateway_tools(self) -> List[Dict[str, Any]]:
        """Get MCP tools for all available gateways."""
        tools = []
        
        for service_name in self.registry.get_available_services():
            gateway = self.registry.get(service_name)
            if not gateway:
                continue
            
            # Corporate LLM Gateway
            if service_name == "corporate_llm":
                tools.append({
                    "name": "corporate_chat",
                    "description": "Enterprise LLM chat with audit and compliance",
                    "parameters": {
                        "message": {"type": "string", "required": True},
                        "model": {"type": "string", "default": "claude"},
                        "compliance_level": {"type": "string", "default": "standard"}
                    }
                })
            
            # AI Processing Gateway
            elif service_name == "ai_processing":
                tools.append({
                    "name": "ai_process",
                    "description": "Multi-model AI processing with backend selection",
                    "parameters": {
                        "task": {"type": "string", "required": True},
                        "content": {"type": "string", "required": True},
                        "model_preference": {"type": "string", "default": "auto"}
                    }
                })
            
            # Workflow Optimizer Gateway  
            elif service_name == "workflow_optimizer":
                tools.append({
                    "name": "optimize_workflow",
                    "description": "Workflow analysis and optimization",
                    "parameters": {
                        "workflow_data": {"type": "object", "required": True},
                        "optimization_type": {"type": "string", "default": "efficiency"}
                    }
                })
        
        return tools
    
    def handle_mcp_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls for both gateways and knowledge resources."""
        try:
            # Knowledge resource tools
            if self.knowledge_server and tool_name in ["search", "retrieve", "embed", "extract", "query"]:
                return self.knowledge_server.handle_mcp_tool_call(tool_name, parameters)
            
            # Gateway tools
            elif tool_name == "corporate_chat":
                return self._handle_corporate_chat(parameters)
            elif tool_name == "ai_process":
                return self._handle_ai_process(parameters)
            elif tool_name == "optimize_workflow":
                return self._handle_optimize_workflow(parameters)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": [t["name"] for t in self.get_mcp_capabilities()["tools"]]
                }
                
        except Exception as e:
            logger.error(f"Error handling MCP tool call '{tool_name}': {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    def _handle_corporate_chat(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle corporate chat via CorporateLLMGateway."""
        gateway = self.registry.get("corporate_llm")
        if not gateway:
            return {"success": False, "error": "CorporateLLMGateway not available"}
        
        message = params.get("message")
        if not message:
            return {"success": False, "error": "Message parameter required"}
        
        try:
            # Use gateway's process method
            result = gateway.process({
                "message": message,
                "model": params.get("model", "claude"),
                "compliance_level": params.get("compliance_level", "standard")
            })
            
            return {
                "success": True,
                "tool": "corporate_chat",
                "response": result.content if hasattr(result, 'content') else str(result),
                "model": params.get("model", "claude")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_ai_process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AI processing via AIProcessingGateway."""
        gateway = self.registry.get("ai_processing")
        if not gateway:
            return {"success": False, "error": "AIProcessingGateway not available"}
        
        task = params.get("task")
        content = params.get("content")
        
        if not task or not content:
            return {"success": False, "error": "Task and content parameters required"}
        
        try:
            result = gateway.process({
                "task": task,
                "content": content,
                "model_preference": params.get("model_preference", "auto")
            })
            
            return {
                "success": True,
                "tool": "ai_process",
                "result": result.content if hasattr(result, 'content') else str(result),
                "task": task
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_optimize_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow optimization via WorkflowOptimizerGateway."""
        gateway = self.registry.get("workflow_optimizer")
        if not gateway:
            return {"success": False, "error": "WorkflowOptimizerGateway not available"}
        
        workflow_data = params.get("workflow_data")
        if not workflow_data:
            return {"success": False, "error": "workflow_data parameter required"}
        
        try:
            result = gateway.process({
                "workflow_data": workflow_data,
                "optimization_type": params.get("optimization_type", "efficiency")
            })
            
            return {
                "success": True,
                "tool": "optimize_workflow", 
                "optimization_result": result.content if hasattr(result, 'content') else str(result),
                "optimization_type": params.get("optimization_type", "efficiency")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_resource_info(self, resource_uri: str) -> Optional[Dict[str, Any]]:
        """Get information about MCP resources."""
        if resource_uri.startswith("knowledge/") and self.knowledge_server:
            # Strip knowledge/ prefix and delegate to knowledge server
            knowledge_uri = resource_uri[10:]  # Remove "knowledge/"
            return self.knowledge_server.get_resource_info(knowledge_uri)
        
        return None
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available MCP resources."""
        resources = []
        
        # Add knowledge resources
        if self.knowledge_server:
            knowledge_resources = self.knowledge_server.list_resources()
            for resource in knowledge_resources:
                # Add knowledge/ prefix to distinguish
                resource["uri"] = f"knowledge/{resource['uri']}"
                resources.append(resource)
        
        return resources
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get unified server status."""
        return {
            "server_name": self.config.server_name,
            "version": self.config.server_version,
            "status": "running",
            "gateway_registry": self.registry.get_registry_stats(),
            "knowledge_server": (
                self.knowledge_server.get_server_status() 
                if self.knowledge_server else None
            ),
            "available_gateways": self.registry.get_available_services(),
            "mcp_capabilities": len(self.get_mcp_capabilities()["tools"])
        }


def start_mcp_server(config: MCPInterfaceConfig = None):
    """Start the TidyLLM MCP server."""
    interface = TidyLLMMCPInterface(config)
    
    logger.info(f"TidyLLM MCP Server starting on port {interface.config.port}")
    logger.info(f"Available gateways: {', '.join(interface.registry.get_available_services())}")
    
    # In a real implementation, this would start the actual MCP protocol server
    # For now, just return the interface for testing
    return interface


if __name__ == "__main__":
    # Start MCP server
    server = start_mcp_server()
    print("TidyLLM MCP Server ready!")
    print(f"Capabilities: {json.dumps(server.get_mcp_capabilities(), indent=2)}")