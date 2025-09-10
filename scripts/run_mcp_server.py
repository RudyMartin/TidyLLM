#!/usr/bin/env python3
"""
TidyLLM MCP Server CLI
======================

Command-line interface to run the TidyLLM Knowledge Resource Server
as an MCP (Model Context Protocol) server.

Usage:
    python scripts/run_mcp_server.py
    
    # Or with specific configuration
    python scripts/run_mcp_server.py --bucket legal-docs --prefix contracts/
    
This script sets up a real MCP server that can be used by:
- Claude Code (via .mcp.json configuration)
- VSCode MCP extensions
- Other MCP-compatible tools
"""

import sys
import argparse
import logging
from pathlib import Path

# Add tidyllm to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidyllm.knowledge_resource_server import (
    KnowledgeMCPServer, 
    S3KnowledgeSource, 
    LocalKnowledgeSource,
    DatabaseKnowledgeSource,
    run_mcp_server
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)  # Log to stderr so stdout is clean for MCP
    ]
)

logger = logging.getLogger(__name__)


def setup_knowledge_server(args) -> KnowledgeMCPServer:
    """Set up and configure the knowledge server."""
    logger.info("Initializing TidyLLM Knowledge MCP Server")
    
    server = KnowledgeMCPServer()
    
    # Register S3 knowledge source if specified
    if args.s3_bucket:
        logger.info(f"Registering S3 source: s3://{args.s3_bucket}/{args.s3_prefix}")
        s3_source = S3KnowledgeSource(
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            region=args.aws_region
        )
        server.register_domain("s3-docs", s3_source)
    
    # Register local knowledge source if specified
    if args.local_directory:
        logger.info(f"Registering local source: {args.local_directory}")
        local_source = LocalKnowledgeSource(
            directory=args.local_directory,
            file_patterns=args.file_patterns.split(',') if args.file_patterns else None
        )
        server.register_domain("local-docs", local_source)
    
    # Register database knowledge source if specified
    if args.db_table:
        logger.info(f"Registering database source: {args.db_schema}.{args.db_table}")
        db_source = DatabaseKnowledgeSource(
            table_name=args.db_table,
            schema=args.db_schema
        )
        server.register_domain("db-docs", db_source)
    
    # If no sources specified, set up default demo sources
    if not any([args.s3_bucket, args.local_directory, args.db_table]):
        logger.info("No sources specified, setting up demo sources with mock data")
        
        # Demo S3 source (will use mock data)
        demo_s3 = S3KnowledgeSource(bucket="demo-legal-docs", prefix="contracts/")
        server.register_domain("legal-contracts", demo_s3)
        
        # Demo database source (will use mock data)
        demo_db = DatabaseKnowledgeSource(table_name="legal_documents")
        server.register_domain("legal-database", demo_db)
        
        logger.info("Demo sources registered - search for 'termination', 'contract', or 'legal compliance'")
    
    return server


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run TidyLLM Knowledge Resource Server as MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with demo data (mock S3 and database)
    python scripts/run_mcp_server.py
    
    # Run with real S3 bucket
    python scripts/run_mcp_server.py --s3-bucket my-docs --s3-prefix knowledge/
    
    # Run with local directory
    python scripts/run_mcp_server.py --local-directory ./documents
    
    # Run with database table
    python scripts/run_mcp_server.py --db-table documents --db-schema public
    
    # Run with multiple sources
    python scripts/run_mcp_server.py --s3-bucket legal-docs --local-directory ./tech-docs
        """
    )
    
    # S3 configuration
    parser.add_argument('--s3-bucket', help='S3 bucket name for document storage')
    parser.add_argument('--s3-prefix', default='', help='S3 key prefix (folder path)')
    parser.add_argument('--aws-region', default='us-east-1', help='AWS region')
    
    # Local directory configuration
    parser.add_argument('--local-directory', help='Local directory containing documents')
    parser.add_argument('--file-patterns', default='*.txt,*.md,*.pdf,*.json', 
                       help='Comma-separated file patterns to include')
    
    # Database configuration
    parser.add_argument('--db-table', default='documents', help='Database table name')
    parser.add_argument('--db-schema', default='public', help='Database schema')
    
    # Server configuration
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Set up knowledge server
        server = setup_knowledge_server(args)
        
        # Log server status
        status = server.get_server_status()
        logger.info(f"Server initialized: {status['registered_domains']} domains registered")
        
        # Log capabilities for debugging
        capabilities = server.get_mcp_capabilities()
        logger.info(f"MCP Tools available: {[tool['name'] for tool in capabilities['tools']]}")
        logger.info(f"Knowledge domains: {capabilities['resources']['domains']}")
        
        # Start MCP protocol server
        logger.info("Starting MCP server - ready to accept requests via stdio")
        logger.info("Use Ctrl+C to stop the server")
        
        run_mcp_server(server)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()