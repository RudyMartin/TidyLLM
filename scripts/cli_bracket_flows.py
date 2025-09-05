"""
TidyLLM CLI Chain Interface Solution
===================================

Extends existing CLI to expose the 7 core document chain operations.
Builds on existing RAG2DAG CLI but adds direct access to chain contracts.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Import existing document chains
from tidyllm.document_chains import (
    BackendDocumentPipeline, FrontendDocumentAPI,
    DocumentOperation, ChainExecutionMode
)
from tidyllm.gateways import get_gateway

class TidyLLMChainCLI:
    """CLI interface for TidyLLM chain operations."""
    
    def __init__(self):
        self.backend_pipeline = BackendDocumentPipeline()
        self.frontend_api = FrontendDocumentAPI()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create CLI argument parser with chain operations."""
        parser = argparse.ArgumentParser(
            prog='tidyllm',
            description='TidyLLM Document Chain Operations',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Backend operations (for data teams)
  tidyllm ingest ./docs --domain legal --bucket process-docs
  tidyllm embed legal --model tidyllm-sentence --target-dim 1024
  tidyllm index legal --vector-store s3://vectors-bucket/legal/
  tidyllm track legal --metrics accuracy,performance
  tidyllm report legal --format json --output s3://reports/legal.json
  
  # Frontend operations (for app teams)  
  tidyllm query legal "What are the compliance requirements?"
  tidyllm search legal --keywords "contract termination" --limit 10
  
  # Chaining operations
  tidyllm chain ingest embed index --domain legal --source ./docs
  tidyllm chain query --domain legal "summarize key findings"
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available operations')
        
        # Backend Layer Operations (Layer 1)
        self._add_ingest_parser(subparsers)
        self._add_embed_parser(subparsers)
        self._add_index_parser(subparsers)
        self._add_track_parser(subparsers)
        self._add_report_parser(subparsers)
        
        # Frontend Layer Operations (Layer 2)
        self._add_query_parser(subparsers)
        self._add_search_parser(subparsers)
        
        # Chain Operations
        self._add_chain_parser(subparsers)
        
        # Status and Monitoring
        self._add_status_parser(subparsers)
        
        return parser
    
    def _add_ingest_parser(self, subparsers):
        """Add ingest command parser."""
        parser = subparsers.add_parser('ingest', help='Ingest documents from source to S3')
        parser.add_argument('source', help='Source path or S3 URI')
        parser.add_argument('--domain', required=True, help='Knowledge domain')
        parser.add_argument('--bucket', help='Target S3 bucket')
        parser.add_argument('--format', choices=['pdf', 'docx', 'txt', 'auto'], default='auto')
        parser.add_argument('--batch-size', type=int, default=10, help='Batch processing size')
        parser.add_argument('--dry-run', action='store_true', help='Preview without execution')
    
    def _add_embed_parser(self, subparsers):
        """Add embed command parser.""" 
        parser = subparsers.add_parser('embed', help='Generate embeddings using tidyllm-sentence')
        parser.add_argument('domain', help='Knowledge domain to process')
        parser.add_argument('--model', default='tfidf', choices=['tfidf', 'word_avg', 'lsa', 'ngram'])
        parser.add_argument('--target-dim', type=int, default=1024, help='Target embedding dimension')
        parser.add_argument('--bucket', help='S3 bucket for embeddings storage')
        parser.add_argument('--parallel', type=int, default=3, help='Parallel processing workers')
    
    def _add_index_parser(self, subparsers):
        """Add index command parser."""
        parser = subparsers.add_parser('index', help='Create searchable indices using tlm')
        parser.add_argument('domain', help='Knowledge domain to index')
        parser.add_argument('--vector-store', help='S3 URI for vector storage')
        parser.add_argument('--index-type', choices=['faiss', 'simple'], default='simple')
        parser.add_argument('--cluster-count', type=int, help='Number of clusters for tlm.kmeans')
    
    def _add_track_parser(self, subparsers):
        """Add track command parser."""
        parser = subparsers.add_parser('track', help='Track processing metrics via PostgreSQL MLflow')
        parser.add_argument('domain', help='Knowledge domain to track')
        parser.add_argument('--metrics', help='Comma-separated metrics to track')
        parser.add_argument('--experiment', help='MLflow experiment name')
        parser.add_argument('--postgres-uri', help='PostgreSQL connection for MLflow')
    
    def _add_report_parser(self, subparsers):
        """Add report command parser."""
        parser = subparsers.add_parser('report', help='Generate processing reports')
        parser.add_argument('domain', help='Knowledge domain to report on')
        parser.add_argument('--format', choices=['json', 'markdown', 'html'], default='json')
        parser.add_argument('--output', help='Output S3 URI or local path')
        parser.add_argument('--include', help='Report sections to include')
    
    def _add_query_parser(self, subparsers):
        """Add query command parser."""
        parser = subparsers.add_parser('query', help='Natural language query (simple interface)')
        parser.add_argument('domain', help='Knowledge domain to query')
        parser.add_argument('question', help='Natural language question')
        parser.add_argument('--limit', type=int, default=5, help='Number of results')
        parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
        parser.add_argument('--format', choices=['text', 'json'], default='text')
    
    def _add_search_parser(self, subparsers):
        """Add search command parser."""
        parser = subparsers.add_parser('search', help='Keyword search (simple interface)')
        parser.add_argument('domain', help='Knowledge domain to search')
        parser.add_argument('--keywords', required=True, help='Search keywords')
        parser.add_argument('--limit', type=int, default=10, help='Number of results')
        parser.add_argument('--format', choices=['text', 'json'], default='text')
    
    def _add_chain_parser(self, subparsers):
        """Add chain command parser."""
        parser = subparsers.add_parser('chain', help='Execute chained operations')
        parser.add_argument('operations', nargs='+', 
                           choices=['ingest', 'embed', 'index', 'track', 'report', 'query', 'search'],
                           help='Operations to chain together')
        parser.add_argument('--domain', required=True, help='Knowledge domain')
        parser.add_argument('--source', help='Source for ingest operation')
        parser.add_argument('--question', help='Question for query operation')
        parser.add_argument('--mode', choices=['sequential', 'pipeline', 'parallel', 'auto'], 
                           default='auto', help='Execution mode')
    
    def _add_status_parser(self, subparsers):
        """Add status command parser."""
        parser = subparsers.add_parser('status', help='Check operation status')
        parser.add_argument('domain', nargs='?', help='Specific domain to check')
        parser.add_argument('--operation', help='Specific operation to check')
        parser.add_argument('--watch', action='store_true', help='Watch for changes')
    
    def handle_ingest(self, args) -> int:
        """Handle ingest command."""
        print(f"🔄 Ingesting documents from {args.source} to domain '{args.domain}'")
        
        if args.dry_run:
            print("DRY RUN: Would ingest documents with S3-first processing")
            return 0
        
        try:
            # Use backend pipeline for complex ingest operation
            result = self.backend_pipeline.ingest_documents(
                source=args.source,
                domain=args.domain,
                bucket=args.bucket,
                batch_size=args.batch_size,
                document_format=args.format
            )
            
            print(f"✅ Ingested {result['documents_processed']} documents")
            print(f"📁 Stored in S3: {result['s3_location']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            return 0
            
        except Exception as e:
            print(f"❌ Ingest failed: {e}")
            return 1
    
    def handle_query(self, args) -> int:
        """Handle query command (simple interface)."""
        print(f"🔍 Querying domain '{args.domain}': {args.question}")
        
        try:
            # Use frontend API for simple query operation
            results = self.frontend_api.query(
                domain=args.domain,
                question=args.question,
                limit=args.limit,
                similarity_threshold=args.threshold
            )
            
            if args.format == 'json':
                print(json.dumps(results, indent=2))
            else:
                print(f"\n📄 Found {len(results['matches'])} relevant results:\n")
                for i, match in enumerate(results['matches'], 1):
                    print(f"{i}. {match['title']} (score: {match['score']:.3f})")
                    print(f"   {match['content'][:150]}...")
                    print()
            
            return 0
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return 1
    
    def handle_chain(self, args) -> int:
        """Handle chained operations."""
        print(f"⛓️  Executing chain: {' → '.join(args.operations)}")
        print(f"📂 Domain: {args.domain}")
        
        try:
            # Build operation chain
            if 'ingest' in args.operations and not args.source:
                print("❌ --source required for ingest operation")
                return 1
            
            if 'query' in args.operations and not args.question:
                print("❌ --question required for query operation")
                return 1
            
            # Execute chain using document chains backend
            chain_config = {
                'domain': args.domain,
                'execution_mode': args.mode,
                'operations': args.operations
            }
            
            if args.source:
                chain_config['source'] = args.source
            if args.question:
                chain_config['question'] = args.question
            
            # Use both backend and frontend as needed
            backend_ops = {'ingest', 'embed', 'index', 'track', 'report'}
            frontend_ops = {'query', 'search'}
            
            results = {}
            for op in args.operations:
                if op in backend_ops:
                    result = getattr(self.backend_pipeline, f"{op}_operation")(chain_config)
                else:
                    result = getattr(self.frontend_api, op)(chain_config)
                results[op] = result
                print(f"✅ {op.upper()}: {result['status']}")
            
            print(f"\n🎉 Chain completed successfully!")
            return 0
            
        except Exception as e:
            print(f"❌ Chain execution failed: {e}")
            return 1
    
    def handle_status(self, args) -> int:
        """Handle status command."""
        if args.domain:
            print(f"📊 Status for domain '{args.domain}':")
        else:
            print("📊 Overall TidyLLM Status:")
        
        try:
            # Get status from gateways
            registry = get_gateway('workflow_optimizer')
            if registry:
                status = registry.get_domain_status(args.domain if args.domain else 'all')
                
                print(f"🟢 Active domains: {status['active_domains']}")
                print(f"📈 Documents processed: {status['total_documents']}")
                print(f"🔍 Queries served: {status['total_queries']}")
                print(f"⚡ Average response time: {status['avg_response_time']:.2f}s")
                
                if args.watch:
                    print("👀 Watching for changes... (Ctrl+C to stop)")
                    # Implement watch functionality
            
            return 0
            
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            return 1
    
    def run(self, args: List[str]) -> int:
        """Main CLI entry point."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        try:
            # Route to appropriate handler
            handler_name = f"handle_{parsed_args.command}"
            handler = getattr(self, handler_name, None)
            
            if handler:
                return handler(parsed_args)
            else:
                print(f"❌ Unknown command: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\n⏹️  Operation cancelled by user")
            return 130
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1


def main():
    """CLI entry point."""
    cli = TidyLLMChainCLI()
    return cli.run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())