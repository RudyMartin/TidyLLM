"""
MCP Protocol Implementation - JSON-RPC over stdio
==================================================

Implements the Model Context Protocol (MCP) specification for client-server
communication using JSON-RPC over stdin/stdout.

This allows the KnowledgeMCPServer to be used as a real MCP server that can
be integrated with MCP-compatible clients like Claude Code, VSCode extensions,
and other tools.
"""

import sys
import json
import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("mcp_protocol")


@dataclass
class MCPRequest:
    """MCP JSON-RPC request structure."""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: str = ""
    params: Optional[Dict[str, Any]] = None


@dataclass
class MCPResponse:
    """MCP JSON-RPC response structure."""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class MCPProtocolHandler:
    """
    Handles MCP protocol communication via JSON-RPC over stdio.
    
    This class implements the MCP specification for:
    - Server capabilities exchange
    - Resource listing and access
    - Tool execution
    - Error handling and logging
    
    Usage:
        handler = MCPProtocolHandler(knowledge_server)
        await handler.run()  # Starts stdio loop
    """
    
    def __init__(self, knowledge_server):
        """
        Initialize MCP protocol handler.
        
        Args:
            knowledge_server: KnowledgeMCPServer instance
        """
        self.knowledge_server = knowledge_server
        self.running = False
        self.request_handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "ping": self._handle_ping
        }
        
    async def run(self) -> None:
        """Start the MCP server stdio loop."""
        logger.info("Starting MCP protocol handler")
        self.running = True
        
        try:
            while self.running:
                try:
                    # Read line from stdin
                    line = sys.stdin.readline()
                    if not line:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Parse JSON-RPC request
                    try:
                        request_data = json.loads(line)
                        request = MCPRequest(
                            jsonrpc=request_data.get("jsonrpc", "2.0"),
                            id=request_data.get("id"),
                            method=request_data.get("method", ""),
                            params=request_data.get("params")
                        )
                        
                        # Handle request
                        response = await self._handle_request(request)
                        
                        # Send response
                        if response:
                            self._send_response(response)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON received: {e}")
                        error_response = MCPResponse(
                            id=None,
                            error={
                                "code": -32700,
                                "message": "Parse error",
                                "data": str(e)
                            }
                        )
                        self._send_response(error_response)
                        
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    
        finally:
            self.running = False
            logger.info("MCP protocol handler stopped")
    
    async def _handle_request(self, request: MCPRequest) -> Optional[MCPResponse]:
        """Handle incoming MCP request."""
        try:
            handler = self.request_handlers.get(request.method)
            if not handler:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {request.method}",
                        "data": {
                            "available_methods": list(self.request_handlers.keys())
                        }
                    }
                )
            
            result = await handler(request.params or {})
            return MCPResponse(id=request.id, result=result)
            
        except Exception as e:
            logger.error(f"Error handling request {request.method}: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            )
    
    def _send_response(self, response: MCPResponse) -> None:
        """Send MCP response to stdout."""
        try:
            response_data = {
                "jsonrpc": response.jsonrpc,
                "id": response.id
            }
            
            if response.result is not None:
                response_data["result"] = response.result
            elif response.error is not None:
                response_data["error"] = response.error
            
            json_response = json.dumps(response_data)
            print(json_response, flush=True)
            logger.debug(f"Sent response: {json_response}")
            
        except Exception as e:
            logger.error(f"Error sending response: {e}")
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request."""
        logger.info(f"Initializing MCP server with params: {params}")
        
        capabilities = self.knowledge_server.get_mcp_capabilities()
        
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "resources": {
                    "subscribe": False,
                    "listChanged": False
                },
                "tools": {
                    "listChanged": False
                }
            },
            "serverInfo": capabilities["server"],
            "instructions": "TidyLLM Knowledge Resource Server - Access legal documents, technical specs, and enterprise knowledge bases via MCP protocol."
        }
    
    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        capabilities = self.knowledge_server.get_mcp_capabilities()
        return {"tools": capabilities["tools"]}
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            raise ValueError("Tool name is required")
        
        # Call the knowledge server's tool handler
        result = self.knowledge_server.handle_mcp_tool_call(tool_name, arguments)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }
            ],
            "isError": not result.get("success", False)
        }
    
    async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        resources = self.knowledge_server.list_resources()
        
        return {
            "resources": [
                {
                    "uri": f"knowledge://{resource['name']}",
                    "name": resource['name'],
                    "description": f"Knowledge domain: {resource['name']}",
                    "mimeType": "application/json"
                }
                for resource in resources
            ]
        }
    
    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri", "")
        
        # Parse knowledge://domain_name URI
        if uri.startswith("knowledge://"):
            domain_name = uri[12:]  # Remove "knowledge://" prefix
            resource_info = self.knowledge_server.get_resource_info(f"domains/{domain_name}")
            
            if resource_info:
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(resource_info, indent=2)
                        }
                    ]
                }
        
        raise ValueError(f"Resource not found: {uri}")
    
    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "server": "tidyllm-knowledge"
        }


def run_mcp_server(knowledge_server) -> None:
    """
    Run MCP server with protocol handler.
    
    Args:
        knowledge_server: KnowledgeMCPServer instance
        
    Usage:
        from tidyllm.knowledge_resource_server import KnowledgeMCPServer, run_mcp_server
        
        server = KnowledgeMCPServer()
        server.register_domain("legal-docs", S3KnowledgeSource("legal", "contracts/"))
        run_mcp_server(server)
    """
    handler = MCPProtocolHandler(knowledge_server)
    
    try:
        asyncio.run(handler.run())
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        raise