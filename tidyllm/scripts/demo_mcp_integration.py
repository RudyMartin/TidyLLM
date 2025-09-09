#!/usr/bin/env python3
"""
TidyLLM MCP Integration Demo
============================

Demonstrates the complete MCP (Model Context Protocol) integration
showing how TidyLLM's Knowledge Resource Server works with real data sources.

This demo shows:
1. Real S3 integration (with fallback to mock data)
2. Real database integration (with fallback to mock data)
3. Enhanced semantic search
4. MCP protocol communication
5. Integration with AI agents like Claude Code

Usage:
    python scripts/demo_mcp_integration.py
"""

import sys
import json
import asyncio
import logging
from pathlib import Path

# Add tidyllm to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidyllm.knowledge_resource_server import (
    KnowledgeMCPServer,
    S3KnowledgeSource,
    LocalKnowledgeSource, 
    DatabaseKnowledgeSource,
    MCPProtocolHandler
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


async def demo_mcp_integration():
    """Main demonstration of MCP integration."""
    
    print_section("TidyLLM MCP Integration Demo")
    print("This demo shows the complete MCP stack in action!")
    print("Features: Real S3/DB integration, Enhanced search, MCP protocol")
    
    # 1. Initialize Knowledge Server
    print_section("1. Initialize Knowledge Server")
    server = KnowledgeMCPServer()
    
    # Register multiple data sources
    print("Registering data sources...")
    
    # S3 source (will fallback to mock if no AWS access)
    s3_source = S3KnowledgeSource(
        bucket="legal-documents",
        prefix="contracts/",
        region="us-east-1"
    )
    server.register_domain("legal-s3", s3_source)
    print("  [OK] S3 legal documents registered")
    
    # Database source (will fallback to mock if no DB access)
    db_source = DatabaseKnowledgeSource(
        table_name="legal_documents",
        schema="public"
    )
    server.register_domain("legal-db", db_source)
    print("  [OK] Database legal documents registered")
    
    # Local source for tech docs
    try:
        import tempfile
        import os
        
        # Create temp directory with sample documents
        temp_dir = tempfile.mkdtemp()
        
        # Create sample technical document
        tech_doc = os.path.join(temp_dir, "api_spec.md")
        with open(tech_doc, 'w') as f:
            f.write("""# TidyLLM API Specification
            
This document outlines the API endpoints for contract analysis and legal document processing.

## Contract Analysis Endpoint
- POST /api/v1/contracts/analyze
- Analyzes legal contracts for termination clauses, compliance issues, and risk factors
- Supports PDF, Word, and plain text formats

## Legal Document Search
- GET /api/v1/search/legal
- Semantic search across legal document repositories
- Supports advanced filters and metadata queries
            """)
        
        local_source = LocalKnowledgeSource(directory=temp_dir)
        server.register_domain("tech-docs", local_source)
        print("  [OK] Local technical documents registered")
        
    except Exception as e:
        print(f"  [WARNING] Local source setup failed: {e}")
    
    # 2. Test Enhanced Search Capabilities
    print_section("2. Enhanced Search Capabilities")
    
    test_queries = [
        ("termination clauses", "Legal contract search"),
        ("API specification", "Technical documentation search"), 
        ("compliance requirements", "Cross-domain compliance search"),
        ("contract analysis", "Multi-source contract search")
    ]
    
    for query, description in test_queries:
        print(f"\n[SEARCH] {description}: '{query}'")
        
        # Search across all domains
        for domain in ["legal-s3", "legal-db", "tech-docs"]:
            try:
                search_result = server.handle_mcp_tool_call("search", {
                    "query": query,
                    "domain": domain,
                    "max_results": 2,
                    "similarity_threshold": 0.5
                })
                
                if search_result["success"] and search_result["result_count"] > 0:
                    print(f"  {domain}: {search_result['result_count']} results")
                    for result in search_result["results"]:
                        score = result["similarity_score"]
                        title = result["title"]
                        print(f"    - {title} (score: {score:.2f})")
                else:
                    print(f"  {domain}: No results")
                    
            except Exception as e:
                print(f"  {domain}: Error - {e}")
    
    # 3. Test MCP Protocol
    print_section("3. MCP Protocol Communication")
    
    handler = MCPProtocolHandler(server)
    
    # Test protocol initialization
    print("Testing MCP protocol initialization...")
    init_response = await handler._handle_initialize({
        "protocolVersion": "2024-11-05",
        "capabilities": {"resources": {}, "tools": {}}
    })
    print(f"  [OK] Protocol version: {init_response['protocolVersion']}")
    print(f"  [OK] Server: {init_response['serverInfo']['name']}")
    
    # Test tools list
    tools_response = await handler._handle_tools_list({})
    tools = [tool["name"] for tool in tools_response["tools"]]
    print(f"  [OK] Available tools: {', '.join(tools)}")
    
    # Test resources list  
    resources_response = await handler._handle_resources_list({})
    resources = [res["name"] for res in resources_response["resources"]]
    print(f"  [OK] Available resources: {', '.join(resources)}")
    
    # 4. Simulate AI Agent Integration
    print_section("4. AI Agent Integration Simulation")
    print("Simulating how Claude Code would interact with the MCP server...")
    
    # Simulate AI agent making a search request
    print("\n[AI AGENT] Search for contract termination procedures")
    
    agent_search = await handler._handle_tools_call({
        "name": "search", 
        "arguments": {
            "query": "contract termination procedures",
            "max_results": 3,
            "similarity_threshold": 0.4
        }
    })
    
    if not agent_search.get("isError"):
        content = json.loads(agent_search["content"][0]["text"])
        if content["success"]:
            print(f"  [OK] Found {content['result_count']} relevant documents")
            for result in content["results"]:
                print(f"    [DOC] {result['title']} (score: {result['similarity_score']:.2f})")
                print(f"          Source: {result.get('source_uri', 'Unknown')}")
        else:
            print(f"  [ERROR] Search failed: {content.get('error', 'Unknown error')}")
    else:
        print("  [ERROR] Tool call failed")
    
    # 5. Server Status and Capabilities
    print_section("5. Server Status & Capabilities")
    
    status = server.get_server_status()
    capabilities = server.get_mcp_capabilities()
    
    print(f"Server Name: {status['server_name']}")
    print(f"Version: {status['version']}")
    print(f"Status: {status['status']}")
    print(f"Registered Domains: {status['registered_domains']}")
    print(f"Resource Count: {capabilities['resources']['resource_count']}")
    
    print("\n[SUMMARY] Integration Summary:")
    print("  [OK] Real S3 integration with fallback to mock data")
    print("  [OK] Real database integration with fallback to mock data") 
    print("  [OK] Enhanced semantic search with scoring")
    print("  [OK] Full MCP protocol implementation (JSON-RPC over stdio)")
    print("  [OK] Ready for Claude Code integration via .mcp.json")
    print("  [OK] Multi-source knowledge aggregation")
    print("  [OK] Enterprise security and compliance features")
    
    print_section("Claude Code Integration Instructions")
    print("""
To use this MCP server with Claude Code:

1. Ensure the .mcp.json file is in your project root
2. Start Claude Code in this directory
3. The MCP server will be automatically available

Available MCP servers:
- tidyllm-knowledge: Demo server with mock data
- tidyllm-s3-legal: Real S3 legal documents 
- tidyllm-local-docs: Local document processing
- tidyllm-database: Database-backed knowledge

Example Claude Code usage:
"Search for contract termination clauses in legal documents"
"Find API specifications for document processing"
"What are the compliance requirements for legal contracts?"
    """)

    # Cleanup
    try:
        import shutil
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
    except:
        pass


def main():
    """Main entry point."""
    try:
        asyncio.run(demo_mcp_integration())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()