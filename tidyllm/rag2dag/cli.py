"""
RAG2DAG CLI Interface
====================

Command-line interface for RAG2DAG workflow creation and management.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from .converter import RAG2DAGConverter, RAGPatternType
from .config import RAG2DAGConfig, BedrockModel
from .executor import DAGExecutor


class RAG2DAGCLI:
    """Command-line interface for RAG2DAG operations."""
    
    def __init__(self):
        self.config = RAG2DAGConfig.create_default_config()
        self.converter = RAG2DAGConverter(self.config)
        self.executor = DAGExecutor(self.config)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for CLI commands."""
        parser = argparse.ArgumentParser(
            prog='tidyllm rag2dag',
            description='Transform RAG workflows into optimized Bedrock DAG execution',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Simple workflow
  tidyllm rag2dag create --query "What are the main findings?" --files *.pdf
  
  # With configuration  
  tidyllm rag2dag create --query "Compare methods" --files study*.pdf --config speed
  
  # Interactive mode
  tidyllm rag2dag interactive
  
  # Check status
  tidyllm rag2dag status wf_20241207_104530
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Create workflow
        create_parser = subparsers.add_parser('create', help='Create new RAG2DAG workflow')
        create_parser.add_argument('--query', '-q', required=True, 
                                 help='Question or analysis goal')
        create_parser.add_argument('--files', '-f', nargs='+', required=True,
                                 help='Input files to analyze')
        create_parser.add_argument('--config', '-c', choices=['speed', 'balanced', 'quality'],
                                 default='balanced', help='Optimization configuration')
        create_parser.add_argument('--parallel', '-p', type=int,
                                 help='Maximum parallel nodes (overrides config)')
        create_parser.add_argument('--output', '-o', 
                                 help='Output directory for results')
        create_parser.add_argument('--dry-run', action='store_true',
                                 help='Preview workflow without execution')
        create_parser.add_argument('--wait', action='store_true',
                                 help='Wait for completion and show results')
        
        # Interactive mode
        interactive_parser = subparsers.add_parser('interactive', help='Interactive workflow builder')
        interactive_parser.add_argument('--config', choices=['speed', 'balanced', 'quality'],
                                       default='balanced')
        
        # Status and monitoring
        status_parser = subparsers.add_parser('status', help='Check workflow status')
        status_parser.add_argument('workflow_id', nargs='?', help='Workflow ID to check')
        
        # List workflows
        list_parser = subparsers.add_parser('list', help='List all workflows')
        list_parser.add_argument('--limit', type=int, default=10, help='Number of workflows to show')
        list_parser.add_argument('--status', choices=['running', 'completed', 'failed'],
                                help='Filter by status')
        
        # Get results
        results_parser = subparsers.add_parser('results', help='Get workflow results')
        results_parser.add_argument('workflow_id', help='Workflow ID')
        results_parser.add_argument('--format', choices=['json', 'text', 'markdown'], 
                                   default='text', help='Output format')
        
        # Stop workflow
        stop_parser = subparsers.add_parser('stop', help='Stop running workflow')
        stop_parser.add_argument('workflow_id', help='Workflow ID to stop')
        
        # Configuration
        config_parser = subparsers.add_parser('config', help='Manage configuration')
        config_group = config_parser.add_mutually_exclusive_group()
        config_group.add_argument('--show', action='store_true', help='Show current config')
        config_group.add_argument('--speed', action='store_true', help='Switch to speed config')
        config_group.add_argument('--balanced', action='store_true', help='Switch to balanced config')
        config_group.add_argument('--quality', action='store_true', help='Switch to quality config')
        config_group.add_argument('--custom', action='store_true', help='Interactive config editor')
        
        # Analysis tools
        analyze_parser = subparsers.add_parser('analyze', help='Preview workflow without execution')
        analyze_parser.add_argument('query', help='Analysis query')
        analyze_parser.add_argument('files', nargs='+', help='Files to analyze')
        analyze_parser.add_argument('--config', choices=['speed', 'balanced', 'quality'],
                                   default='balanced')
        
        # Information commands
        patterns_parser = subparsers.add_parser('patterns', help='List available RAG patterns')
        costs_parser = subparsers.add_parser('costs', help='Show cost estimates for different patterns')
        
        return parser
    
    def run(self, args: List[str]) -> int:
        """Main CLI entry point."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        try:
            # Route to appropriate handler
            handler_name = f"handle_{parsed_args.command.replace('-', '_')}"
            handler = getattr(self, handler_name, None)
            
            if handler:
                return handler(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            if parsed_args.command == 'interactive':
                # More detailed error in interactive mode
                import traceback
                traceback.print_exc()
            return 1
    
    def handle_create(self, args) -> int:
        """Handle create command."""
        # Load configuration
        self._load_config(args.config)
        
        if args.parallel:
            self.config.max_parallel_nodes = args.parallel
        
        # Validate files exist
        files = []
        for file_pattern in args.files:
            if '*' in file_pattern:
                import glob
                matched_files = glob.glob(file_pattern)
                if not matched_files:
                    print(f"No files found matching pattern: {file_pattern}")
                    return 1
                files.extend(matched_files)
            else:
                if not Path(file_pattern).exists():
                    print(f"File not found: {file_pattern}")
                    return 1
                files.append(file_pattern)
        
        print(f"Creating RAG2DAG workflow for {len(files)} files...")
        print(f"Query: \"{args.query}\"")
        print(f"Configuration: {args.config}")
        
        # Generate workflow
        workflow = self.converter.create_workflow_from_query(
            query=args.query,
            files=files
        )
        
        print(f"\nWorkflow Analysis:")
        print(f"  Pattern: {workflow['pattern_name']}")
        print(f"  Complexity: {workflow['complexity_score']}/10")
        print(f"  Estimated Cost: ${workflow['estimated_cost_factor'] * 1.50:.2f}")
        print(f"  Estimated Time: {workflow['execution_plan']['total_estimated_time_seconds']}s")
        
        print(f"\nDAG Structure ({len(workflow['dag_nodes'])} nodes):")
        for i, node in enumerate(workflow['dag_nodes'], 1):
            model_name = node['model_id'].split('.')[-1][:15]
            parallel_info = f" [Group: {node['parallel_group']}]" if node['parallel_group'] else ""
            dependencies = f" <- {', '.join(node['input_from'])}" if node['input_from'] else ""
            print(f"  {i}. {node['operation'].upper()}: {model_name}{parallel_info}")
            print(f"     \"{node['instruction'][:60]}...\"")
            if dependencies:
                print(f"     {dependencies}")
        
        if args.dry_run:
            print("\nDry run completed. No execution performed.")
            return 0
        
        # Save workflow metadata
        workflow_id = workflow['workflow_id']
        output_dir = Path(args.output) if args.output else Path(f"results/{workflow_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_file = output_dir / "workflow.json"
        with open(workflow_file, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        print(f"\nWorkflow created: {workflow_id}")
        print(f"Results will be saved to: {output_dir}")
        
        if args.wait:
            print("\nExecuting workflow...")
            # TODO: Implement actual execution
            print("Workflow execution not yet implemented")
            return 0
        else:
            print("\nUse 'tidyllm rag2dag status' to monitor progress")
            return 0
    
    def handle_interactive(self, args) -> int:
        """Handle interactive workflow builder."""
        print("RAG2DAG Interactive Workflow Builder")
        print("=" * 40)
        
        try:
            # File selection
            files = self._interactive_file_selection()
            if not files:
                return 1
            
            # Query input
            query = input("\n❓ What's your question/goal?\n> ").strip()
            if not query:
                print("Query is required")
                return 1
            
            # Load config
            self._load_config(args.config)
            
            # Preview workflow
            print(f"\n🔍 Analyzing query and {len(files)} files...")
            workflow = self.converter.create_workflow_from_query(query, files)
            
            print(f"\n📋 Analysis Results:")
            print(f"   Pattern: {workflow['pattern_name']}")
            print(f"   Complexity: {workflow['complexity_score']}/10")
            print(f"   Estimated Cost: ${workflow['estimated_cost_factor'] * 1.50:.2f}")
            print(f"   Estimated Time: ~{workflow['execution_plan']['total_estimated_time_seconds']//60} minutes")
            
            # Configuration selection
            config_choice = self._interactive_config_selection()
            if config_choice != args.config:
                self._load_config(config_choice)
                # Regenerate with new config
                workflow = self.converter.create_workflow_from_query(query, files)
            
            # Confirmation
            print(f"\n✅ Workflow Preview:")
            print(f"   {len(workflow['dag_nodes'])} nodes")
            exec_plan = workflow['execution_plan']
            print(f"   {exec_plan['max_parallel_nodes']} max parallel")
            print(f"   Streaming: {'Yes' if exec_plan['enable_streaming'] else 'No'}")
            
            confirm = input(f"\n🚀 Create this workflow? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Workflow creation cancelled")
                return 0
            
            # Create workflow
            workflow_id = workflow['workflow_id']
            output_dir = Path(f"results/{workflow_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "workflow.json", 'w') as f:
                json.dump(workflow, f, indent=2)
            
            print(f"\n✅ Workflow created: {workflow_id}")
            print(f"📁 Results directory: {output_dir}")
            print("🔄 Execution starting... (use 'tidyllm rag2dag status' to monitor)")
            
            return 0
            
        except (EOFError, KeyboardInterrupt):
            print("\nInteractive mode cancelled")
            return 130
    
    def handle_analyze(self, args) -> int:
        """Handle analyze command (preview only)."""
        self._load_config(args.config)
        
        # Validate files
        for file_path in args.files:
            if not Path(file_path).exists():
                print(f"File not found: {file_path}")
                return 1
        
        print(f"Analyzing query with {len(args.files)} files...")
        
        # Generate workflow preview
        workflow = self.converter.create_workflow_from_query(args.query, args.files)
        
        # Show analysis results
        print(f"\n📊 RAG Pattern Analysis:")
        print(f"   Detected Pattern: {workflow['pattern_name']}")
        print(f"   Description: {workflow['description']}")
        print(f"   Complexity Score: {workflow['complexity_score']}/10")
        print(f"   Cost Factor: {workflow['estimated_cost_factor']}x")
        
        print(f"\n🔗 Generated DAG Structure:")
        parallel_groups = {}
        for node in workflow['dag_nodes']:
            if node['parallel_group']:
                if node['parallel_group'] not in parallel_groups:
                    parallel_groups[node['parallel_group']] = []
                parallel_groups[node['parallel_group']].append(node)
        
        # Show execution flow
        printed_nodes = set()
        for i, node in enumerate(workflow['dag_nodes']):
            if node['node_id'] in printed_nodes:
                continue
                
            if node['parallel_group'] and node['parallel_group'] in parallel_groups:
                # Show parallel group
                group_nodes = parallel_groups[node['parallel_group']]
                print(f"   Parallel Group '{node['parallel_group']}':")
                for group_node in group_nodes:
                    model_name = group_node['model_id'].split('.')[-1][:20]
                    print(f"     • {group_node['operation'].upper()}: {model_name}")
                    print(f"       \"{group_node['instruction'][:50]}...\"")
                    printed_nodes.add(group_node['node_id'])
            else:
                # Show individual node
                model_name = node['model_id'].split('.')[-1][:20]
                deps = f" <- {', '.join(node['input_from'])}" if node['input_from'] else ""
                print(f"   • {node['operation'].upper()}: {model_name}{deps}")
                print(f"     \"{node['instruction'][:50]}...\"")
                printed_nodes.add(node['node_id'])
        
        exec_plan = workflow['execution_plan']
        print(f"\n⚡ Execution Plan:")
        print(f"   Max Parallel Nodes: {exec_plan['max_parallel_nodes']}")
        print(f"   Estimated Duration: {exec_plan['total_estimated_time_seconds']}s")
        print(f"   Streaming Results: {'Yes' if exec_plan['enable_streaming'] else 'No'}")
        
        return 0
    
    def handle_patterns(self, args) -> int:
        """Show available RAG patterns."""
        print("Available RAG2DAG Patterns:")
        print("=" * 40)
        
        for pattern_type, pattern in self.converter.patterns.items():
            print(f"\n📋 {pattern.name}")
            print(f"   Type: {pattern_type.value}")
            print(f"   Complexity: {pattern.complexity_score}/10")
            print(f"   Cost Factor: {pattern.estimated_cost_factor}x")
            print(f"   Description: {pattern.description}")
            print(f"   Triggers: {', '.join(pattern.intent_keywords)}")
            print(f"   File Types: {', '.join(pattern.file_type_hints)}")
        
        return 0
    
    def handle_config(self, args) -> int:
        """Handle configuration management."""
        if args.show or not any([args.speed, args.balanced, args.quality, args.custom]):
            # Show current config
            config_dict = self.config.to_dict()
            print("Current RAG2DAG Configuration:")
            print("=" * 35)
            print(json.dumps(config_dict, indent=2))
            return 0
        
        if args.speed:
            self.config = RAG2DAGConfig.create_speed_config()
            print("Switched to speed optimization configuration")
        elif args.balanced:
            self.config = RAG2DAGConfig.create_default_config()
            print("Switched to balanced optimization configuration")
        elif args.quality:
            self.config = RAG2DAGConfig.create_quality_config()
            print("Switched to quality optimization configuration")
        elif args.custom:
            return self._interactive_config_editor()
        
        return 0
    
    def _load_config(self, config_name: str):
        """Load configuration by name."""
        if config_name == 'speed':
            self.config = RAG2DAGConfig.create_speed_config()
        elif config_name == 'balanced':
            self.config = RAG2DAGConfig.create_default_config()
        elif config_name == 'quality':
            self.config = RAG2DAGConfig.create_quality_config()
        
        # Recreate converter with new config
        self.converter = RAG2DAGConverter(self.config)
    
    def _interactive_file_selection(self) -> List[str]:
        """Interactive file selection."""
        print("\n📁 Select files:")
        print("   1. Browse files")
        print("   2. Use current directory (*.pdf, *.docx, *.txt)")
        print("   3. Specify pattern")
        
        choice = input("> ").strip()
        
        if choice == "1":
            # Simple file browser
            print("Enter file paths (one per line, empty line to finish):")
            files = []
            while True:
                file_path = input("File: ").strip()
                if not file_path:
                    break
                if Path(file_path).exists():
                    files.append(file_path)
                else:
                    print(f"File not found: {file_path}")
            return files
            
        elif choice == "2":
            # Auto-detect common file types
            import glob
            patterns = ["*.pdf", "*.docx", "*.txt", "*.doc"]
            files = []
            for pattern in patterns:
                files.extend(glob.glob(pattern))
            
            if files:
                print(f"Found {len(files)} files: {', '.join(files[:3])}{'...' if len(files) > 3 else ''}")
                return files
            else:
                print("No compatible files found in current directory")
                return []
                
        elif choice == "3":
            # Pattern specification
            pattern = input("File pattern (e.g., '*.pdf', 'docs/*.txt'): ").strip()
            import glob
            files = glob.glob(pattern)
            if files:
                print(f"Found {len(files)} files matching pattern")
                return files
            else:
                print(f"No files found matching pattern: {pattern}")
                return []
        
        return []
    
    def _interactive_config_selection(self) -> str:
        """Interactive configuration selection."""
        print(f"\n🎛️ Optimization level:")
        
        configs = {
            "1": ("speed", "Speed (Haiku, 8 parallel)", "$2.50"),
            "2": ("balanced", "Balanced (Mixed models, 3 parallel)", "$4.20"),
            "3": ("quality", "Quality (Sonnet, 2 parallel)", "$8.10")
        }
        
        for key, (name, desc, cost) in configs.items():
            print(f"   {key}. {desc} - {cost}")
        
        choice = input("> ").strip()
        return configs.get(choice, ("balanced", "", ""))[0]
    
    def _interactive_config_editor(self) -> int:
        """Interactive configuration editor."""
        print("\nInteractive Configuration Editor")
        print("=" * 35)
        print("(Not yet implemented)")
        return 0


def main():
    """CLI entry point."""
    cli = RAG2DAGCLI()
    return cli.run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())